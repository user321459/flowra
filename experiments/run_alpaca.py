"""
    OPENAI_API_KEY=... python experiments/run_alpaca.py --method flowra --seed 0
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import causal_lm
from methods import METHODS, attach, train_attached, trainable_count
from metrics import alpaca_bucket_scores, alpaca_complexity, alpaca_score, generate_text
from flowra.utils.logging_utils import ExperimentLogger


def format_prompt(ex, with_response=True):
    inst = ex.get("instruction") or ""
    inp = ex.get("input") or ""
    out = ex.get("output") or ""
    if inp:
        head = f"### Instruction:\n{inst}\n### Input:\n{inp}\n### Response:\n"
    else:
        head = f"### Instruction:\n{inst}\n### Response:\n"
    return head + out if with_response else head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-model", default=os.environ.get("OPENAI_EVAL_MODEL", "gpt-4"))
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "alpaca"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling

    cfg = load_config(resolve_config(args.config, "alpaca_llama2_7b.yaml"))
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name="alpaca")
    logger.log_environment()

    tok, model = causal_lm(cfg["model"], device, cfg.get("dtype"))
    split = load_dataset("tatsu-lab/alpaca")["train"].shuffle(seed=args.seed)
    n_eval = int(cfg.get("alpaca_eval_n") or 805)
    try:
        val = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        if n_eval < len(val):
            val = val.select(range(n_eval))
    except Exception:
        val, split = split.select(range(min(n_eval, len(split) // 10))), split.select(range(min(n_eval, len(split) // 10), len(split)))
    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(split))
    max_new = int(cfg.get("max_eval_new_tokens") or 128)

    def tokenize(ex):
        return tok(format_prompt(ex, with_response=True), truncation=True, max_length=cfg["max_length"])

    data = split.map(tokenize, remove_columns=split.column_names)
    collate = DataCollatorForLanguageModeling(tok, mlm=False)
    g = torch.Generator().manual_seed(args.seed)
    train = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    calib = DataLoader(data.select(range(n_cal)), batch_size=1, collate_fn=collate)

    def evaluate(m):
        m.eval()
        rows = []
        for ex in val:
            pred = generate_text(m, tok, format_prompt(ex, with_response=False), device, max_new_tokens=max_new)[0]
            score = alpaca_score(ex.get("instruction") or "", ex.get("input") or "",
                                 ex.get("output") or "", pred, model=args.eval_model)
            rows.append({
                "complexity": alpaca_complexity(ex.get("instruction") or "", ex.get("input") or ""),
                "score": score,
            })
        scores = alpaca_bucket_scores(rows)
        logger.info(
            f"Alpaca %: Simple={scores['simple']:.1f} Moderate={scores['moderate']:.1f} "
            f"Complex={scores['complex']:.1f} Avg={scores['average']:.1f}"
        )
        return scores

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    tcfg = training_config(cfg, args.seed, max_steps=cfg["training"].get("max_steps"), eval_every=500)
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model)
    result = {"method": args.method, "seed": args.seed, "eval_model": args.eval_model, **final,
              "trainable_params": fw.parameter_counts()["adapter"] if fw else trainable_count(model)}
    logger.log("result", result, "RESULT " + json.dumps(result))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
