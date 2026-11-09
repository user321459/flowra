"""
Runtime and inference efficiency on LLaMA-2-7B.

    python experiments/run_efficiency.py --mode train --method flowra
    python experiments/run_efficiency.py --mode infer --method flowra
"""

import argparse
import json
import os
import time

import torch

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import GLUE_TASKS, causal_lm, encoded_glue_loaders, load_glue_raw, make_eval_fn, tokenizer_and_model
from methods import METHODS, attach, train_attached
from flowra.utils.logging_utils import ExperimentLogger


def train_mode(args, cfg, logger):
    device = device_str()
    set_seed(args.seed)
    task = "mnli"
    (k1, k2), num_labels, kind = GLUE_TASKS[task]
    tok, model = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
    raw = load_glue_raw(task)
    train, calib, dev = encoded_glue_loaders(raw, tok, (k1, k2), cfg, args.seed, task)
    evaluate = make_eval_fn(dev, device, kind, num_labels)
    t0 = time.time()
    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    t_cal = time.time() - t0
    tcfg = training_config(cfg, args.seed, epochs=cfg["epochs"][task], eval_every=len(train))
    t1 = time.time()
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    t_tr = time.time() - t1
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    result = {"method": args.method, "calibration_s": t_cal, "training_s": t_tr,
              "total_s": t_cal + t_tr, "memory_gb": mem, "steps": tcfg.max_steps or tcfg.epochs * len(train)}
    logger.log("result", result, "RESULT " + json.dumps(result))


def infer_mode(args, cfg, logger):
    device = device_str()
    set_seed(args.seed)
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    tok, model = causal_lm(cfg["model"], device, cfg.get("dtype"))
    prompt = tok("### Instruction:\nSummarise the following paragraph.\n### Response:\n",
                 return_tensors="pt").to(device)
    texts = ["### Instruction:\nHello\n### Response:\nHi there."] * 8
    enc = [{"input_ids": tok(t, truncation=True, max_length=64)["input_ids"]} for t in texts]
    collate = DataCollatorForLanguageModeling(tok, mlm=False)
    calib = DataLoader(enc, batch_size=1, collate_fn=collate)
    gen_cfg = dict(cfg)
    gen_cfg["flowra"] = dict(cfg.get("flowra") or {})
    gen_cfg["flowra"]["modules_to_save"] = []
    fw = None
    if args.method != "full_ft":
        model, fw, _ = attach(args.method, model, calib, gen_cfg, args.seed, device, logger)
        if args.merge and fw is not None:
            fw.merge()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_tok = 0
    with torch.no_grad():
        for _ in range(args.repeats):
            out = model.generate(**prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
            n_tok += int(out.shape[1] - prompt["input_ids"].shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    result = {"method": args.method, "latency_ms": 1000 * elapsed / args.repeats,
              "throughput_tok_s": n_tok / elapsed, "memory_gb": mem}
    logger.log("result", result, "RESULT " + json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", required=True, choices=("train", "infer"))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "efficiency"))
    args = ap.parse_args()

    cfg = load_config(resolve_config(args.config, "glue_llama2_7b.yaml"))
    out = os.path.join(args.out, f"{args.mode}_{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name="efficiency")
    logger.log_environment()
    if args.mode == "train":
        train_mode(args, cfg, logger)
    else:
        infer_mode(args, cfg, logger)
    logger.close()


if __name__ == "__main__":
    main()
