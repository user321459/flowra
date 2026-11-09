"""
Code generation: HumanEval and MBPP — pass@1 and pass@10 on LLaMA-2-13B.

    python experiments/run_code.py --benchmark openai_humaneval --method flowra --seed 0
    python experiments/run_code.py --benchmark mbpp --method flowra --seed 0
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import causal_lm
from methods import METHODS, attach, train_attached, trainable_count
from metrics import generate_text, run_code_eval
from flowra.utils.logging_utils import ExperimentLogger


def _load(benchmark: str):
    from datasets import load_dataset
    if benchmark == "openai_humaneval":
        train = load_dataset("google-research-datasets/mbpp", split="train")
        ev = load_dataset("openai/openai_humaneval", split="test")
        return train, ev, "text", "code", "prompt", "test", True
    train = load_dataset("google-research-datasets/mbpp", split="train")
    ev = load_dataset("google-research-datasets/mbpp", split="test")
    return train, ev, "text", "code", "text", "test_list", False


def _mbpp_test(ex) -> str:
    tests = ex.get("test_list") or []
    if isinstance(tests, str):
        tests = [tests]
    return "\n".join(tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", required=True, choices=("openai_humaneval", "mbpp"))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "code"))
    args = ap.parse_args()

    from transformers import DataCollatorForLanguageModeling

    cfg = load_config(resolve_config(args.config, "reasoning_llama13b.yaml"))
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.benchmark}_{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name=args.benchmark)
    logger.log_environment()

    tok, model = causal_lm(cfg["model"], device, cfg.get("dtype"))
    train_raw, eval_raw, tr_q, tr_a, ev_q, ev_test, is_humaneval = _load(args.benchmark)
    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(train_raw))
    n_sample = int(cfg.get("n_passk_samples") or 10)
    max_new = int(cfg.get("max_eval_new_tokens") or 256)

    def tokenize(ex):
        text = f"{ex.get(tr_q) or ''}\n{ex.get(tr_a) or ex.get('code') or ''}"
        return tok(text, truncation=True, max_length=cfg["max_length"])

    data = train_raw.map(tokenize, remove_columns=train_raw.column_names)
    collate = DataCollatorForLanguageModeling(tok, mlm=False)
    g = torch.Generator().manual_seed(args.seed)
    train = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    calib = DataLoader(data.select(range(n_cal)), batch_size=1, collate_fn=collate)

    def evaluate(m):
        m.eval()
        predictions, references = [], []
        for ex in eval_raw:
            prompt = ex[ev_q]
            gens = generate_text(m, tok, prompt, device, max_new_tokens=max_new, num_return=n_sample)
            if is_humaneval:
                predictions.append([prompt + g for g in gens])
                references.append(ex["test"])
            else:
                predictions.append(gens)
                references.append(_mbpp_test(ex))
        scores = run_code_eval(predictions, references, ks=(1, 10))
        logger.info(f"{args.benchmark}: pass@1={100 * scores['pass@1']:.1f} pass@10={100 * scores['pass@10']:.1f}")
        return {**scores, "pass@1_pct": 100 * scores["pass@1"], "pass@10_pct": 100 * scores["pass@10"]}

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    tcfg = training_config(cfg, args.seed, max_steps=cfg["training"].get("max_steps"), eval_every=500)
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
