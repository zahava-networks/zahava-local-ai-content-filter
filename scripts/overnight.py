"""Autonomous overnight pipeline.

Runs end-to-end while you sleep:
  1. Switches NIM_MODEL to llama-3.2-11b-vision-instruct (8x faster than 90B)
  2. Finishes NSFW labeling on whatever's left
  3. Runs NIM labeling in parallel
  4. Backs up labels to HuggingFace every ~30 min (survives Colab crashes)
  5. After NIM done OR 6 hours, merges labels + trains + exports
  6. Uploads tflite + mlpackage + full checkpoint to HF model repo

Status file at /content/STATUS.txt updates throughout — read that file when
you wake up for a one-line summary.

Launch:
    cd /content/zahava-local-ai-content-filter && nohup python3 scripts/overnight.py > /content/overnight.log 2>&1 &
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path('/content/zahava-local-ai-content-filter')
sys.path.insert(0, str(REPO))
os.chdir(REPO)

LOG_FILE = Path('/content/overnight.log')
STATUS_FILE = Path('/content/STATUS.txt')

MAX_NIM_HOURS = 6
LABEL_BACKUP_EVERY_S = 1800  # 30 min


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def status(msg: str) -> None:
    STATUS_FILE.write_text(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n{msg}\n')
    log(f'STATUS: {msg}')


def _safe(name: str, fn):
    try:
        log(f'>>> {name}')
        fn()
        log(f'<<< {name} done')
        return True
    except Exception as e:
        log(f'!!! {name} failed: {type(e).__name__}: {e}')
        log(traceback.format_exc())
        return False


def main():
    status('overnight pipeline starting')

    # ------ Step 0: switch to 11B + install deps ------
    log('switching NIM_MODEL to 11B vision...')
    env_path = REPO / '.env'
    txt = env_path.read_text()
    txt = txt.replace(
        'NIM_MODEL=meta/llama-3.2-90b-vision-instruct',
        'NIM_MODEL=meta/llama-3.2-11b-vision-instruct',
    )
    env_path.write_text(txt)

    log('installing pip deps (one-time)...')
    subprocess.run(
        ['pip', 'install', '-q', 'timm', 'torchvision', 'coremltools', 'ai_edge_torch'],
        check=False,
    )

    from pipelines.common import load_env
    load_env()

    from huggingface_hub import HfApi, upload_file, upload_folder
    api = HfApi(token=os.environ['HF_TOKEN'])
    dataset_repo = os.environ['HF_DATASET_REPO']
    model_repo = os.environ['HF_MODEL_REPO']

    def backup(path: str, name: str) -> None:
        if not Path(path).exists():
            return
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=name,
                repo_id=dataset_repo,
                repo_type='dataset',
            )
            log(f'backup ok: {name}')
        except Exception as e:
            log(f'backup failed {name}: {e}')

    # ------ Step 1: start NIM in background ------
    status('starting NIM labeling (11B) in background')
    nim_cmd = [
        'python3', '-c',
        'import sys; sys.path.insert(0, "/content/zahava-local-ai-content-filter"); '
        'import os; os.chdir("/content/zahava-local-ai-content-filter"); '
        'from pipelines.common import load_env; load_env(); '
        'from pipelines.labeling import vlm_labeler; '
        'vlm_labeler.run(round_name="vlm_round_1")',
    ]
    nim_proc = subprocess.Popen(
        nim_cmd,
        stdout=open('/content/nim.log', 'a'),
        stderr=subprocess.STDOUT,
    )
    log(f'NIM started (pid {nim_proc.pid})')

    # ------ Step 2: finish NSFW oracle ------
    status('finishing NSFW oracle')
    def run_nsfw():
        from pipelines.labeling import nsfw_oracle
        nsfw_oracle.run()
    _safe('nsfw oracle', run_nsfw)
    backup('manifests/labels/nsfw_oracle.jsonl', 'labels/nsfw_oracle.jsonl')

    # ------ Step 3: wait for NIM (max 6h) with periodic backups ------
    status('waiting for NIM (max 6h)')
    start = time.time()
    last_backup = 0
    last_count = 0
    while True:
        elapsed = time.time() - start
        if nim_proc.poll() is not None:
            log(f'NIM exited (rc={nim_proc.returncode}) after {elapsed/60:.1f} min')
            break
        if elapsed > MAX_NIM_HOURS * 3600:
            log(f'NIM hit {MAX_NIM_HOURS}h limit, killing')
            try:
                nim_proc.send_signal(signal.SIGKILL)
            except Exception:
                pass
            break

        try:
            n = sum(1 for _ in open('manifests/labels/vlm_round_1.jsonl'))
            if n > last_count + 200:
                log(f'NIM progress: {n} labels')
                last_count = n
            if time.time() - last_backup > LABEL_BACKUP_EVERY_S:
                backup('manifests/labels/vlm_round_1.jsonl', 'labels/vlm_round_1.jsonl')
                last_backup = time.time()
            status(f'NIM labeling: {n}/10000 ({n/100:.1f}%), elapsed {elapsed/60:.1f} min')
        except FileNotFoundError:
            status(f'NIM labeling: starting, elapsed {elapsed/60:.1f} min')

        time.sleep(120)  # check every 2 min

    backup('manifests/labels/vlm_round_1.jsonl', 'labels/vlm_round_1.jsonl')

    # ------ Step 4: merge labels ------
    status('merging labels into single parquet')
    def merge():
        from pipelines.labeling.labels_store import merge_to_parquet
        out = merge_to_parquet('labels.parquet')
        import pandas as pd
        df = pd.read_parquet(out)
        log(f'merged: {len(df)} records, block_rate={df["block"].mean():.3f}')
    _safe('merge labels', merge)
    backup('manifests/labels.parquet', 'labels.parquet')

    # ------ Step 5: train ------
    status('training (CPU, ~2-3 hours)')
    def do_train():
        from pipelines.training import train
        train.train(epochs=3)
    _safe('training', do_train)

    # ------ Step 6: tune thresholds + calibrate ------
    status('threshold tuning + calibration')
    _safe('threshold tuner', lambda: __import__('pipelines.training.threshold_tuner', fromlist=['tune']).tune())
    _safe('calibrator', lambda: __import__('pipelines.training.calibrator', fromlist=['calibrate']).calibrate())

    # ------ Step 7: build calibration set + export ------
    status('exporting models')
    _safe('calibration set', lambda: __import__('pipelines.export.calibration_set', fromlist=['build']).build())
    _safe('TFLite export', lambda: __import__('pipelines.export.export_tflite', fromlist=['export']).export())
    _safe('CoreML export', lambda: __import__('pipelines.export.export_coreml', fromlist=['export']).export())
    _safe('Full PT export', lambda: __import__('pipelines.export.export_full', fromlist=['export']).export())

    # ------ Step 8: upload models to HF ------
    status('uploading models to HF')
    try:
        api.create_repo(model_repo, exist_ok=True)
    except Exception:
        pass

    files = [
        'models/tzniut.tflite',
        'models/tzniut.tflite.json',
        'models/tzniut_full.pt',
        'models/tzniut_full.json',
        'config/thresholds.yaml',
    ]
    for f in files:
        if Path(f).exists():
            try:
                upload_file(
                    path_or_fileobj=f,
                    path_in_repo=Path(f).name,
                    repo_id=model_repo,
                    token=os.environ['HF_TOKEN'],
                )
                log(f'uploaded {Path(f).name}')
            except Exception as e:
                log(f'upload {f} failed: {e}')

    if Path('models/tzniut.mlpackage').is_dir():
        try:
            upload_folder(
                folder_path='models/tzniut.mlpackage',
                path_in_repo='tzniut.mlpackage',
                repo_id=model_repo,
                token=os.environ['HF_TOKEN'],
            )
            log('uploaded tzniut.mlpackage')
        except Exception as e:
            log(f'mlpackage upload failed: {e}')

    status(f'COMPLETE — models in https://huggingface.co/{model_repo}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f'TOP-LEVEL ERROR: {e}')
        log(traceback.format_exc())
        status(f'FAILED: {e}')
