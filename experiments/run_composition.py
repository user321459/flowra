"""
Multi-task composition on eight GLUE tasks, excluding WNLI.

Train one adapter per task, then merge with FLowRA orthogonal composition,
naive merge, task arithmetic or TIES.

    python experiments/run_composition.py --composer flowra --seed 0
    python experiments/run_composition.py --composer naive --seed 0
    python experiments/run_composition.py --composer arithmetic --seed 0
    python experiments/run_composition.py --composer ties --seed 0
"""

import argparse
import json
import os

import torch

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import COMPOSITION_TASKS, GLUE_TASKS, encoded_glue_loaders, load_glue_raw, make_eval_fn, tokenizer_and_model
from methods import attach, train_attached
from flowra import (
    NaiveMergeComposer, OrthogonalComposer, TaskArithmeticComposer, TIESComposer,
    adapter_deltas, apply_deltas,
)
from flowra.utils.logging_utils import ExperimentLogger

COMPOSERS = {
    "flowra": lambda eta: OrthogonalComposer(eta=eta),
    "naive": lambda eta: NaiveMergeComposer(),
    "arithmetic": lambda eta: TaskArithmeticComposer(scaling=0.125),
    "ties": lambda eta: TIESComposer(density=0.2, scaling=0.125),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--composer", default="flowra", choices=sorted(COMPOSERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "composition"))
    args = ap.parse_args()

    cfg = load_config(resolve_config(args.config, "glue_llama2_7b.yaml"))
    tasks = list(cfg.get("composition_tasks") or COMPOSITION_TASKS)
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.composer}_seed{args.seed}")
    logger = ExperimentLogger(out, name="composition")
    logger.log_environment()

    deltas = []
    independent = {}
    for task in tasks:
        (k1, k2), num_labels, kind = GLUE_TASKS[task]
        tok, model = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
        raw = load_glue_raw(task)
        train, calib, dev = encoded_glue_loaders(raw, tok, (k1, k2), cfg, args.seed, task)
        evaluate = make_eval_fn(dev, device, kind, num_labels)
        model, fw, params_fn = attach("flowra", model, calib, cfg, args.seed, device, logger)
        tcfg = training_config(cfg, args.seed, epochs=cfg["epochs"][task], eval_every=len(train))
        train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
        independent[task] = evaluate(model.eval())
        deltas.append(adapter_deltas(fw.adapters))
        os.makedirs(os.path.join(out, "adapters"), exist_ok=True)
        fw.save(os.path.join(out, "adapters", f"{task}.pt"))
        logger.info(f"independent {task}: {independent[task]}")
        del model, fw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eta = float((cfg.get("flowra") or {}).get("eta", 0.3))
    merged = COMPOSERS[args.composer](eta).compose(deltas)
    scores = {}
    for task in tasks:
        (k1, k2), num_labels, kind = GLUE_TASKS[task]
        tok, model = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
        apply_deltas(model, merged)
        raw = load_glue_raw(task)
        _, _, dev = encoded_glue_loaders(raw, tok, (k1, k2), cfg, args.seed, task)
        scores[task] = make_eval_fn(dev, device, kind, num_labels)(model.eval())
        logger.info(f"merged {task}: {scores[task]}")

    avg = sum(v["score"] for v in scores.values()) / len(scores)
    result = {"composer": args.composer, "seed": args.seed, "tasks": tasks,
              "independent": independent, "merged": scores, "avg": avg}
    logger.log("result", result, "RESULT " + json.dumps(result))
    logger.close()


if __name__ == "__main__":
    main()
