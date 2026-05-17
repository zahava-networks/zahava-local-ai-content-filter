"""Local CLI to orchestrate the pipeline.

  python orchestrator.py status            # summarize current state
  python orchestrator.py review            # start the review UI (with queue refill)
  python orchestrator.py pull-model        # download final model from HF
  python orchestrator.py setup-check       # verify env + connectivity
  python orchestrator.py pull-artifacts    # pull labels + manifests from HF
"""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from pipelines.common import REPO_ROOT, load_config, require_env

console = Console()


@click.group()
def cli() -> None:
    """Tzniut classifier orchestration."""


@cli.command()
def status() -> None:
    """Show what's been done so far."""
    cfg = load_config()
    t = Table(title="Pipeline state")
    t.add_column("Component"); t.add_column("State"); t.add_column("Details")
    p = REPO_ROOT / "manifests"
    for name in ("collection.parquet", "collection_deduped.parquet", "labels.parquet", "human_review.parquet"):
        f = p / name
        if f.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(f)
                t.add_row(name, "[green]ok[/]", f"{len(df)} rows · {f.stat().st_size//1024} KB")
            except Exception as e:
                t.add_row(name, "[yellow]err[/]", str(e))
        else:
            t.add_row(name, "[dim]—[/]", "not built yet")
    ck_best = REPO_ROOT / "models" / "checkpoints" / "best.pt"
    if ck_best.exists():
        t.add_row("best checkpoint", "[green]ok[/]", f"{ck_best.stat().st_size//(1024*1024)} MB")
    else:
        t.add_row("best checkpoint", "[dim]—[/]", "no checkpoint yet")
    for f in ("tzniut.tflite", "tzniut.mlpackage", "tzniut_full.pt"):
        path = REPO_ROOT / "models" / f
        exists = path.exists() if not path.suffix == ".mlpackage" else path.is_dir()
        t.add_row(f, "[green]ok[/]" if exists else "[dim]—[/]", str(path) if exists else "not exported")
    eval_metrics = REPO_ROOT / "models" / "eval" / "metrics.json"
    if eval_metrics.exists():
        m = json.loads(eval_metrics.read_text())
        t.add_row("eval", "[green]ok[/]",
                  f"recall={m['block_recall_not_acceptable']:.3f} precision={m['block_precision']:.3f}")
    console.print(t)


@cli.command()
@click.option("--port", type=int, default=None)
@click.option("--refill", type=int, default=None, help="populate N items into the queue first")
def review(port: int | None, refill: int | None) -> None:
    """Start the local review UI."""
    from review_ui import server as ui_server
    import sys

    argv = ["review_ui.server"]
    if port is not None:
        argv += ["--port", str(port)]
    if refill is not None:
        argv += ["--populate", str(refill)]
    sys.argv = argv
    ui_server.main()


@cli.command(name="pull-artifacts")
def pull_artifacts() -> None:
    """Pull labels.parquet, human_review.parquet, collection_deduped.parquet from HF."""
    from huggingface_hub import hf_hub_download
    repo = require_env("HF_DATASET_REPO")
    for f in ("collection_deduped.parquet", "labels.parquet", "human_review.parquet"):
        try:
            path = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir="manifests")
            console.print(f"[green]✓[/] {f} → {path}")
        except Exception as e:
            console.print(f"[yellow]skip[/] {f}: {e}")


@cli.command(name="pull-model")
def pull_model() -> None:
    """Pull final .tflite + .mlpackage + thresholds from HF model repo."""
    from huggingface_hub import hf_hub_download, snapshot_download
    repo = require_env("HF_MODEL_REPO")
    for f in ("tzniut.tflite", "tzniut.tflite.json", "tzniut_full.pt", "tzniut_full.json",
              "calibration_set.npy", "thresholds.yaml"):
        try:
            hf_hub_download(repo_id=repo, filename=f, local_dir="models")
            console.print(f"[green]✓[/] {f}")
        except Exception as e:
            console.print(f"[yellow]skip[/] {f}: {e}")
    try:
        snapshot_download(repo_id=repo, allow_patterns="tzniut.mlpackage/*", local_dir="models")
        console.print(f"[green]✓[/] tzniut.mlpackage")
    except Exception as e:
        console.print(f"[yellow]skip[/] tzniut.mlpackage: {e}")


@cli.command(name="setup-check")
def setup_check() -> None:
    """Verify .env, API keys, R2 reachability, HF reachability."""
    from scripts.smoke_test import run as smoke
    smoke()


if __name__ == "__main__":
    cli()
