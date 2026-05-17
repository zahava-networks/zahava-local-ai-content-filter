"""Top-level labeling pipeline: NSFW oracle → VLM → merge → push to HF."""
from __future__ import annotations

import argparse

from ..common import get_logger, require_env
from . import labels_store
from . import nsfw_oracle, vlm_labeler

log = get_logger(__name__)


def run(round_name: str = "vlm_round_1", push_to_hf: bool = True) -> None:
    log.info("=== Stage A: NSFW oracle ===")
    nsfw_oracle.run()
    log.info("=== Stage B: VLM labeling (NIM + Gemini QA) ===")
    vlm_labeler.run(round_name=round_name)
    log.info("=== merging labels ===")
    out = labels_store.merge_to_parquet("labels.parquet")
    log.info("labels parquet → %s", out)
    if push_to_hf:
        from huggingface_hub import HfApi
        api = HfApi(token=require_env("HF_TOKEN"))
        repo = require_env("HF_DATASET_REPO")
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo="labels.parquet",
            repo_id=repo,
            repo_type="dataset",
        )
        log.info("labels pushed to HF: %s", repo)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", default="vlm_round_1")
    p.add_argument("--no-push", action="store_true")
    a = p.parse_args()
    run(a.round, push_to_hf=not a.no_push)
