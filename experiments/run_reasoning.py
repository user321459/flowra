"""
Reasoning and code on LLaMA-2-13B.

GSM8K: 8-shot accuracy on the test split (final ``####`` number).
MATH: 4-shot accuracy on the test split.
HumanEval / MBPP: pass@1; GSM8K logs accuracy during training.

    python experiments/run_reasoning.py --benchmark gsm8k --method flowra
    python experiments/run_reasoning.py --benchmark math --method flowra
    python experiments/run_reasoning.py --benchmark openai_humaneval --method flowra
    python experiments/run_reasoning.py --benchmark mbpp --method flowra
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import causal_lm
from methods import METHODS, attach, train_attached, trainable_count
from metrics import fewshot_prompt, generate_text, gsm8k_gold, last_number, run_code_eval
from flowra.utils.logging_utils import ExperimentLogger


def _load_math():
    from datasets import load_dataset
    try:
        return load_dataset("hendrycks/competition_math")
    except Exception:
        return load_dataset("EleutherAI/hendrycks_math", "algebra")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", required=True, choices=("gsm8k", "math", "openai_humaneval", "mbpp"))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "reasoning"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling

    cfg = load_config(resolve_config(args.config, "reasoning_llama13b.yaml"))
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.benchmark}_{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name=args.benchmark)
    logger.log_environment()

    tok, model = causal_lm(cfg["model"], device, cfg.get("dtype"))

    if args.benchmark == "gsm8k":
        raw = load_dataset("openai/gsm8k", "main")
        train_raw, eval_raw = raw["train"], raw["test"]
        qk, ak, n_shot = "question", "answer", int(cfg.get("gsm8k_shots") or 8)
        gold_fn = gsm8k_gold
    elif args.benchmark == "math":
        raw = _load_math()
        train_raw, eval_raw = raw["train"], raw.get("test") or raw["train"]
        qk = "problem" if "problem" in train_raw.column_names else "question"
        ak = "solution" if "solution" in train_raw.column_names else "answer"
        n_shot = int(cfg.get("math_shots") or 4)
        gold_fn = last_number
    elif args.benchmark == "openai_humaneval":
        train_raw = load_dataset("google-research-datasets/mbpp", split="train")
        eval_raw = load_dataset("openai/openai_humaneval", split="test")
        qk, ak, n_shot, gold_fn = "prompt", "canonical_solution", 0, None
    else:
        raw = load_dataset("google-research-datasets/mbpp")
        train_raw, eval_raw = raw["train"], raw.get("test") or raw["validation"]
        qk, ak, n_shot, gold_fn = "text", "code", 0, None

    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(train_raw))
    n_sample = int(cfg.get("n_passk_samples") or 10)

    def tokenize(ex):
        if args.benchmark in ("openai_humaneval", "mbpp"):
            text = f"{ex.get(qk) or ex.get('text') or ''}\n{ex.get(ak) or ex.get('code') or ''}"
        else:
            text = f"Question: {ex.get(qk, '')}\nAnswer: {ex.get(ak, '')}"
        return tok(text, truncation=True, max_length=cfg["max_length"])

    data = train_raw.map(tokenize, remove_columns=train_raw.column_names)
    collate = DataCollatorForLanguageModeling(tok, mlm=False)
    g = torch.Generator().manual_seed(args.seed)
    train = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    calib = DataLoader(data.select(range(n_cal)), batch_size=1, collate_fn=collate)
    demos = [(train_raw[i][qk], train_raw[i][ak]) for i in range(min(n_shot, len(train_raw)))] if n_shot else []

    def evaluate(m):
        m.eval()
        if args.benchmark in ("openai_humaneval", "mbpp"):
            predictions, references = [], []
            for ex in eval_raw:
                prompt = ex.get("prompt") or ex.get("text") or ""
                gens = generate_text(m, tok, prompt, device, max_new_tokens=256, num_return=n_sample)
                if args.benchmark == "openai_humaneval":
                    predictions.append([prompt + g for g in gens])
                    references.append(ex["test"])
                else:
                    tests = ex.get("test_list") or []
                    references.append("\n".join(tests) if isinstance(tests, list) else str(tests))
                    predictions.append(gens)
            scores = run_code_eval(predictions, references, ks=(1, 10))
            logger.info(f"{args.benchmark} pass@1={100 * scores['pass@1']:.1f}")
            return {**scores, "pass@1_pct": 100 * scores["pass@1"]}

        correct, n = 0, len(eval_raw)
        max_new = 256 if args.benchmark == "math" else 128
        for ex in eval_raw:
            prompt = fewshot_prompt(demos, ex[qk], n_shot)
            pred = generate_text(m, tok, prompt, device, max_new_tokens=max_new)[0]
            gold = gold_fn(str(ex[ak]))
            got = gsm8k_gold(pred) if args.benchmark == "gsm8k" else last_number(pred)
            correct += int(got == gold and gold != "")
        acc = correct / max(n, 1)
        logger.info(f"{args.benchmark} {n_shot}-shot acc={100 * acc:.1f} (n={n})")
        return {"accuracy": acc, "score": acc, "shots": n_shot, "accuracy_pct": 100 * acc}

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    every = int(cfg.get("eval_every_steps") or 1000)
    tcfg = training_config(cfg, args.seed, max_steps=cfg["training"].get("max_steps"), eval_every=every)
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model)
    result = {"benchmark": args.benchmark, "method": args.method, "seed": args.seed, **final,
              "trainable_params": fw.parameter_counts()["adapter"] if fw else trainable_count(model)}
    logger.log("result", result, "RESULT " + json.dumps(result))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
