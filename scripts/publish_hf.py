"""Upload the dataset, eval logs and adapters to Hugging Face and build the two public collections
the Lexsi submission form asks for (models + datasets). Needs `hf auth login` with a write token.

  python scripts/publish_hf.py --user <hf_username> [--adapter runs/sft/adapter] [--dpo-adapter runs/dpo/adapter]
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--dpo-adapter")
    a = ap.parse_args()
    api = HfApi()
    u = a.user

    ds_repo = f"{u}/SafeToolCall"
    api.create_repo(ds_repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(ROOT / "data"), repo_id=ds_repo, repo_type="dataset",
                      allow_patterns=["*.jsonl", "*.json", "*.py", "README.md"])
    logs_repo = f"{u}/SafeToolCall-eval-logs"
    api.create_repo(logs_repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(ROOT / "results"), repo_id=logs_repo, repo_type="dataset",
                      allow_patterns=["**/metrics.json", "**/predictions.jsonl", "**/trajectories.jsonl", "*.md", "*.json"])

    models = []
    for name, path in (("tiny-aya-global-SafeToolCall-SFT-LoRA", a.adapter),
                       ("tiny-aya-global-SafeToolCall-DPO-LoRA", a.dpo_adapter)):
        if path and Path(path).exists():
            repo = f"{u}/{name}"
            api.create_repo(repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=path, repo_id=repo)
            models.append(repo)

    mc = api.create_collection(title="SafeToolCall models (Lexsi hackathon)", namespace=u, private=False, exists_ok=True,
                               description="LoRA adapters: Tiny Aya Global fine-tuned with AgentTune for safe tool use.")
    for m in models:
        api.add_collection_item(mc.slug, item_id=m, item_type="model", exists_ok=True)
    api.add_collection_item(mc.slug, item_id="CohereLabs/tiny-aya-global", item_type="model", exists_ok=True,
                            note="base model")
    dc = api.create_collection(title="SafeToolCall datasets (Lexsi hackathon)", namespace=u, private=False, exists_ok=True,
                               description="Tool-use safety train/test/DPO data and evaluation logs + trajectories.")
    for d in (ds_repo, logs_repo):
        api.add_collection_item(dc.slug, item_id=d, item_type="dataset", exists_ok=True)
    print("model collection:  https://huggingface.co/collections/" + mc.slug)
    print("dataset collection: https://huggingface.co/collections/" + dc.slug)


if __name__ == "__main__":
    main()
