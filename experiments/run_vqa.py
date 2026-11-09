"""
VQA v2 with BLIP-2. Official soft accuracy: min(#agreeing annotators / 3, 1).

    python experiments/run_vqa.py --method flowra --seed 0
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from methods import METHODS, attach, train_attached, trainable_count
from metrics import vqa_soft_accuracy
from flowra.utils.batching import move_to_device
from flowra.utils.logging_utils import ExperimentLogger


def _answers(ex) -> list:
    if "answers" in ex and isinstance(ex["answers"], list):
        out = []
        for a in ex["answers"]:
            out.append(a["answer"] if isinstance(a, dict) else str(a))
        return out
    if isinstance(ex.get("answers"), dict) and "answer" in ex["answers"]:
        return list(ex["answers"]["answer"])
    return [str(ex.get("multiple_choice_answer") or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "vqa"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    cfg = load_config(resolve_config(args.config, "vqa_blip2.yaml"))
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name="vqav2")
    logger.log_environment()

    proc = AutoProcessor.from_pretrained(cfg["model"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.get("dtype"), torch.float32)
    model = Blip2ForConditionalGeneration.from_pretrained(cfg["model"], torch_dtype=dtype).to(device)
    try:
        train_raw = load_dataset("HuggingFaceM4/VQAv2", split="train")
        val_raw = load_dataset("HuggingFaceM4/VQAv2", split="validation")
    except Exception:
        train_raw = load_dataset("lmms-lab/VQAv2", split="train")
        val_raw = load_dataset("lmms-lab/VQAv2", split="validation")
    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(train_raw))

    def collate(examples):
        images = [ex["image"].convert("RGB") for ex in examples]
        questions = [ex["question"] for ex in examples]
        golds = [_answers(ex)[0] for ex in examples]
        enc = proc(images=images, text=questions, padding=True, return_tensors="pt", truncation=True)
        labels = proc(text=golds, padding=True, return_tensors="pt", truncation=True).input_ids
        enc["labels"] = labels
        return enc

    g = torch.Generator().manual_seed(args.seed)
    train = DataLoader(train_raw, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    calib = DataLoader(train_raw.select(range(n_cal)), batch_size=1, collate_fn=collate)

    def evaluate(m):
        m.eval()
        scores = []
        with torch.no_grad():
            for ex in val_raw:
                batch = collate([ex])
                batch = move_to_device(batch, device)
                gen = m.generate(pixel_values=batch["pixel_values"],
                                 input_ids=batch.get("input_ids"),
                                 max_new_tokens=8)
                pred = proc.batch_decode(gen, skip_special_tokens=True)[0]
                scores.append(vqa_soft_accuracy(pred, _answers(ex)))
        acc = sum(scores) / max(len(scores), 1)
        logger.info(f"VQA v2 official acc={100 * acc:.1f} (n={len(scores)})")
        return {"accuracy": acc, "score": acc, "accuracy_pct": 100 * acc}

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    tcfg = training_config(cfg, args.seed, max_steps=cfg["training"].get("max_steps"), eval_every=500)
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model)
    result = {"benchmark": "vqav2", "method": args.method, "seed": args.seed, **final,
              "trainable_params": fw.parameter_counts()["adapter"] if fw else trainable_count(model)}
    logger.log("result", result, "RESULT " + json.dumps(result))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
