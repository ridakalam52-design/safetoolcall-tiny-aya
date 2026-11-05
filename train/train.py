"""Fine-tune Tiny Aya Global for safe tool use with AgentTune.

Stage 1 (sft):  agentic distillation of gold trajectories with AgentTune's
                `create_distill_trainer` (LoRA + TRL SFTTrainer under the hood).
Stage 2 (dpo):  optional offline DPO with AgentTune's `create_agentic_trainer(algorithm="dpo")`
                on (safe = chosen, unsafe-or-over-refusing = rejected) pairs, starting from the SFT adapter.

Runs on one free Colab / Kaggle T4 (16 GB) in fp16. Example:
  python train/train.py --stage sft --out runs/sft
  python train/train.py --stage dpo --init-adapter runs/sft/adapter --out runs/dpo
  add --push <hf_user>/<repo> to upload the adapter (needs `hf auth login`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "CohereLabs/tiny-aya-global"


def load_rows(name):
    return [json.loads(l) for l in open(ROOT / "data" / name)]


def to_eventlog(item):
    """Gold conversation -> full-tier AgentTune EventLog.

    Tiny Aya's chat template has no tool role, so (exactly like AgentTune's
    tools_fallback_prompt path) tool calls are assistant TEXT holding the JSON list,
    and tool results are user turns ("Tool result for ...")."""
    from agenttune.agentic import Event, EventKind, EventLog

    log = EventLog(tier="full")
    for m in item["messages"]:
        kind = EventKind.OBSERVATION if m["role"] == "user" else EventKind.TEXT
        log.append(Event(kind, {"text": m["content"]}))
    log.append(Event(EventKind.TEXT, {"text": item["gold"]["completion"]}))
    return log


def dtype_and_flags():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, {"bf16": True}
    if torch.cuda.is_available():
        return torch.float16, {"fp16": True}
    return torch.float32, {}


def stage_sft(a):
    from agenttune import create_distill_trainer

    dtype, flags = dtype_and_flags()
    trajs = [to_eventlog(it) for it in load_rows("train.jsonl")]
    trainer = create_distill_trainer(
        student=a.model,
        trajectories=trajs,
        lora_r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        dtype=dtype,
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=2048,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=1105,
        **flags,
    )
    print("SFT rows:", len(trainer.sft_dataset()))
    trainer.train()
    sft = trainer.trainer_factory.sft_trainer
    adapter_dir = Path(a.out) / "adapter"
    sft.save_model(str(adapter_dir))
    trainer.trainer_factory.tokenizer.save_pretrained(str(adapter_dir))
    json.dump(sft.state.log_history, open(Path(a.out) / "train_log.json", "w"), indent=1)
    return adapter_dir


def stage_dpo(a):
    from datasets import Dataset
    from agenttune.core.backend_factory import create_agentic_trainer

    dtype, flags = dtype_and_flags()
    rows = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in load_rows("dpo_train.jsonl")]
    model = a.model
    if a.init_adapter:  # merge SFT adapter into the base so DPO starts from it
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto")
        merged = PeftModel.from_pretrained(base, a.init_adapter).merge_and_unload()
        model = str(Path(a.out) / "sft_merged")
        merged.save_pretrained(model)
        AutoTokenizer.from_pretrained(a.model).save_pretrained(model)
        del base, merged
        torch.cuda.empty_cache()
    trainer = create_agentic_trainer(
        algorithm="dpo",
        model=model,
        train_dataset=Dataset.from_list(rows),
        output_dir=a.out,
        beta=0.1,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_length=2048,
        peft_config=dict(r=a.lora_r, lora_alpha=2 * a.lora_r, target_modules="all-linear", task_type="CAUSAL_LM"),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        **flags,
    )
    trainer.train()
    adapter_dir = Path(a.out) / "adapter"
    inner = getattr(trainer, "trainer", trainer)
    inner.save_model(str(adapter_dir))
    return adapter_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--init-adapter")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--push", help="HF repo id to upload the adapter to")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    adapter = stage_sft(a) if a.stage == "sft" else stage_dpo(a)
    print("adapter saved to", adapter)
    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(adapter), repo_id=a.push)
        for extra in ("train_log.json", "lexsi_provenance.json"):
            p = Path(a.out) / extra
            if p.exists():
                api.upload_file(path_or_fileobj=str(p), path_in_repo=extra, repo_id=a.push)
        print("pushed to https://huggingface.co/" + a.push)


if __name__ == "__main__":
    main()
