"""
Sequential GLUE on LLaMA-7B.

Task order: MNLI → QQP → QNLI → SST-2.
After the sequence every task is re-evaluated (task-specific heads are kept).
Forgetting is the mean over tasks of
(score right after learning the task − score at the end).

    python experiments/run_sequential.py --method flowra --seed 0
    python experiments/run_sequential.py --method flowra --protocol composition --seed 0
"""

import argparse
import copy
import json
import os

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import GLUE_TASKS, SEQUENTIAL_TASKS, encoded_glue_loaders, load_glue_raw, make_eval_fn, tokenizer_and_model
from methods import attach, train_attached
from flowra import OrthogonalComposer, adapter_deltas, apply_deltas
from flowra.utils.logging_utils import ExperimentLogger


def _swap_head(model, head, num_labels: int):
    model.score = head
    model.config.num_labels = num_labels
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="flowra", choices=("flowra", "lora", "full_ft"))
    ap.add_argument("--protocol", default="sequential", choices=("sequential", "composition"),
                    help="sequential = one backbone; composition = merge 4 adapters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "sequential"))
    args = ap.parse_args()

    cfg = load_config(resolve_config(args.config, "glue_llama2_7b.yaml"))
    tasks = list(cfg.get("sequential_tasks") or SEQUENTIAL_TASKS)
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.protocol}_{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name="sequential")
    logger.log_environment()
    logger.info(f"sequential order: {tasks}")

    loaders = {}
    for task in tasks:
        (k1, k2), num_labels, kind = GLUE_TASKS[task]
        tok, _ = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
        raw = load_glue_raw(task)
        train, calib, dev = encoded_glue_loaders(raw, tok, (k1, k2), cfg, args.seed, task)
        loaders[task] = (tok, train, calib, dev, num_labels, kind, make_eval_fn(dev, device, kind, num_labels))

    if args.protocol == "composition":
        deltas, heads, independent = [], {}, {}
        for task in tasks:
            tok, train, calib, dev, num_labels, kind, evaluate = loaders[task]
            _, model = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
            model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
            tcfg = training_config(cfg, args.seed, epochs=cfg["epochs"][task], eval_every=len(train))
            train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
            independent[task] = evaluate(model.eval())
            heads[task] = copy.deepcopy(model.score)
            deltas.append(adapter_deltas(fw.adapters) if fw is not None else {})
            fw.save(os.path.join(out, f"{task}.pt")) if fw is not None else None
            del model, fw
        eta = float((cfg.get("flowra") or {}).get("eta", 0.3))
        merged = OrthogonalComposer(eta=eta).compose(deltas)
        final = {}
        for task in tasks:
            tok, train, calib, dev, num_labels, kind, evaluate = loaders[task]
            _, model = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
            apply_deltas(model, merged)
            model.score = heads[task]
            final[task] = evaluate(model.eval())
        avg = sum(v["score"] for v in final.values()) / len(final)
        forget = {t: independent[t]["score"] - final[t]["score"] for t in tasks}
        avg_forget = sum(forget.values()) / len(forget)
        result = {"protocol": "composition", "method": args.method, "order": tasks,
                  "independent": independent, "final": final, "avg": avg,
                  "forget": forget, "forget_pct": 100 * avg_forget}
        logger.log("result", result, "RESULT " + json.dumps(result))
        logger.close()
        return

    heads, peak, evals = {}, {}, {}
    model, fw, params_fn = None, None, None
    for i, task in enumerate(tasks):
        tok, train, calib, dev, num_labels, kind, evaluate = loaders[task]
        _, fresh = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
        if model is None:
            model, fw, params_fn = attach(args.method, fresh, calib, cfg, args.seed, device, logger)
        else:
            _swap_head(model, fresh.score, num_labels)
            for p in model.score.parameters():
                p.requires_grad_(True)
        tcfg = training_config(cfg, args.seed, epochs=cfg["epochs"][task], eval_every=len(train))
        train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
        heads[task] = copy.deepcopy(model.score)
        evals[task] = evaluate
        peak[task] = evaluate(model.eval())["score"]
        logger.info(f"after learning {task}: {peak[task]:.4f}")

    final = {}
    for task in tasks:
        tok, train, calib, dev, num_labels, kind, evaluate = loaders[task]
        _swap_head(model, heads[task], num_labels)
        final[task] = evaluate(model.eval())
    forget = {t: peak[t] - final[t]["score"] for t in tasks}
    avg_forget = sum(forget.values()) / len(forget)
    avg = sum(final[t]["score"] for t in tasks) / len(tasks)
    result = {
        "protocol": "sequential", "method": args.method, "seed": args.seed,
        "order": tasks, "peak": peak, "final": {t: final[t] for t in tasks},
        "forget": forget, "forget_pct": 100 * avg_forget, "avg": avg,
    }
    logger.info(f"final { {t: final[t]['score'] for t in tasks} } forget={100 * avg_forget:.1f}%")
    logger.log("result", result, "RESULT " + json.dumps({
        k: v for k, v in result.items() if k != "final"
    } | {"final_scores": {t: final[t]["score"] for t in tasks}}))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
