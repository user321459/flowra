"""
GLUE. Default config is LLaMA-3-8B. Pass glue_llama2_7b.yaml for LLaMA-2-7B.

    python experiments/run_glue.py --task mnli --method flowra --seed 0
    python experiments/run_glue.py --config experiments/configs/glue_llama2_7b.yaml --task mnli --method flowra
    python experiments/run_glue.py --config experiments/configs/glue_roberta.yaml --task rte --method flowra
    python experiments/run_glue.py --task cola --method flowra --variant no_flow
    python experiments/run_glue.py --task mnli --method flowra --lr 1e-3
    python experiments/run_glue.py --config experiments/configs/budget_0p05.yaml --task mnli --method flowra
"""

import argparse
import json
import os

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import GLUE_TASKS, encoded_glue_loaders, load_glue_raw, make_eval_fn, tokenizer_and_model
from methods import METHODS, attach, train_attached, trainable_count
from flowra.utils.logging_utils import ExperimentLogger

TASKS = GLUE_TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", required=True, choices=sorted(GLUE_TASKS))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="full", help="FLowRA ablation: full | no_flow | no_refinement | no_polymorphic | no_composition")
    ap.add_argument("--model", help="override Hugging Face model id")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "glue"))
    args = ap.parse_args()

    cfg = load_config(resolve_config(args.config, "glue_llama3_8b.yaml"))
    (k1, k2), num_labels, kind = GLUE_TASKS[args.task]
    model_id = args.model or cfg["model"]
    n_epochs = args.epochs if args.epochs is not None else cfg["epochs"][args.task]
    device = device_str()
    set_seed(args.seed)
    slug = model_id.split("/")[-1]
    out = os.path.join(args.out, f"{args.task}_{slug}_{args.method}_{args.variant}_seed{args.seed}")
    logger = ExperimentLogger(out, name=args.task)
    logger.log_environment()
    logger.info(f"model={model_id} task={args.task} method={args.method} variant={args.variant} "
                f"seed={args.seed} epochs={n_epochs}")

    tok, model = tokenizer_and_model(model_id, num_labels, device, cfg.get("dtype"))
    raw = load_glue_raw(args.task)
    train, calib, dev = encoded_glue_loaders(raw, tok, (k1, k2), cfg, args.seed, args.task)
    evaluate = make_eval_fn(dev, device, kind, num_labels)

    if args.method == "flowra" and args.variant != "full":
        cfg = dict(cfg)
        cfg["flowra"] = dict(cfg.get("flowra") or {})
        from methods import flow_config
        fcfg = flow_config(cfg, args.seed, device).variant(args.variant)
        cfg["flowra"].update(fcfg.to_dict())

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    tcfg = training_config(cfg, args.seed, epochs=n_epochs, lr=args.lr, eval_every=len(train))
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model.eval())
    result = {
        "task": args.task, "model": model_id, "method": args.method,
        "variant": args.variant, "seed": args.seed, **final,
        "trainable_params": fw.parameter_counts()["adapter"] if fw is not None else trainable_count(model),
    }
    if fw is not None:
        result["ranks"] = fw.current_ranks()
        fw.profile_result.save_csv(os.path.join(out, "layer_profile.csv")) if fw.profile_result else None
        fw.save(os.path.join(out, "adapter.pt"))
    logger.log("result", {k: v for k, v in result.items() if k != "ranks"},
               "RESULT " + json.dumps({k: v for k, v in result.items() if k != "ranks"}))
    logger.close()


if __name__ == "__main__":
    main()
