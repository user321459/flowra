"""
Question answering: SQuAD v1.1 / v2.0, Natural Questions, TriviaQA — F1 and EM.

    python experiments/run_qa.py --dataset squad --method flowra --seed 0
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import causal_lm
from methods import METHODS, attach, train_attached, trainable_count
from metrics import em_f1, generate_text, gold_qa_answers, mean_em_f1
from flowra.utils.logging_utils import ExperimentLogger

DATASETS = {
    "squad": ("rajpurkar/squad", None, "context", "question", "validation"),
    "squad_v2": ("rajpurkar/squad_v2", None, "context", "question", "validation"),
    "natural_questions": ("google-research-datasets/natural_questions", "default", "document", "question", "validation"),
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc", "entity_pages", "question", "validation"),
}


def _context(ex, ctx_key):
    ctx = ex.get(ctx_key) or ex.get("context") or ""
    if isinstance(ctx, dict):
        ctx = ctx.get("text") or ctx.get("plain_text") or str(ctx)
    if isinstance(ctx, list):
        ctx = " ".join(str(c) for c in ctx[:3])
    return str(ctx)


def _question(ex, q_key):
    q = ex.get(q_key) or ex.get("question") or ""
    if isinstance(q, dict):
        q = q.get("text") or str(q)
    return str(q)


def _prompt(ex, ctx_key, q_key, with_answer=True):
    ctx, q = _context(ex, ctx_key), _question(ex, q_key)
    golds = gold_qa_answers(ex, "squad")
    gold = golds[0] if golds else ""
    body = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
    return f"{body} {gold}" if with_answer else body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "qa"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling

    cfg = load_config(resolve_config(args.config, "generation_llama.yaml"))
    device = device_str()
    set_seed(args.seed)
    ds_id, subset, ctx_key, q_key, eval_split = DATASETS[args.dataset]
    out = os.path.join(args.out, f"{args.dataset}_{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name=args.dataset)
    logger.log_environment()

    tok, model = causal_lm(cfg["model"], device, cfg.get("dtype"))
    raw = load_dataset(ds_id, subset) if subset else load_dataset(ds_id)
    split = raw["train"]
    val = raw[eval_split] if eval_split in raw else raw["validation"]
    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(split))

    def tokenize(ex):
        return tok(_prompt(ex, ctx_key, q_key, with_answer=True), truncation=True, max_length=cfg["max_length"])

    data = split.map(tokenize, remove_columns=split.column_names)
    collate = DataCollatorForLanguageModeling(tok, mlm=False)
    g = torch.Generator().manual_seed(args.seed)
    train = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    calib = DataLoader(data.select(range(n_cal)), batch_size=1, collate_fn=collate)
    max_new = int(cfg.get("max_eval_new_tokens") or 32)

    def evaluate(m):
        m.eval()
        rows = []
        n = len(val)
        for i in range(n):
            ex = val[i]
            prompt = _prompt(ex, ctx_key, q_key, with_answer=False)
            pred = generate_text(m, tok, prompt, device, max_new_tokens=max_new)[0]
            golds = gold_qa_answers(ex, args.dataset)
            rows.append(em_f1(pred, golds))
        scores = mean_em_f1(rows)
        logger.info(f"QA {args.dataset}: F1={100 * scores['f1']:.1f} EM={100 * scores['em']:.1f} (n={n})")
        return {**scores, "f1_pct": 100 * scores["f1"], "em_pct": 100 * scores["em"]}

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    tcfg = training_config(cfg, args.seed, max_steps=cfg["training"].get("max_steps"), eval_every=500)
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model)
    result = {"dataset": args.dataset, "method": args.method, "seed": args.seed, **final,
              "trainable_params": fw.parameter_counts()["adapter"] if fw else trainable_count(model)}
    logger.log("result", result, "RESULT " + json.dumps(result))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
