"""
Layer-level sensitivity analysis.

For every adaptation target of the pre-trained model it computes, on a
512-sample calibration set drawn from the training split of the target task,

* the flow descriptor ``D~_l = |D_l|``;
* the gradient term ``G_l = E ||grad_{h_l} L||_2``;
* AdaLoRA's SVD-based importance ``sum_i |s_i * u_i^T (grad_W L) v_i|``
  (``s_i * dL/ds_i`` for ``W = U diag(s) V^T``);
* the FLowRA criticality ``C_l = D~_l G_l ||W_l||_F``,

and reports Spearman correlations computed over individual layers, plus the
per-layer-type aggregation (scores normalised by their maximum, averaged per type).

    python experiments/sensitivity_correlation.py --hf-model meta-llama/Llama-2-7b-hf --task mnli \
        --out logs/sensitivity_llama7b_mnli
"""

import argparse
import csv
import json
import os

import torch
from scipy.stats import spearmanr

from common import ROOT
from flowra.core.analysis import FlowProfiler
from flowra.utils.batching import batch_size, forward_loss, slice_batch
from flowra.utils.logging_utils import ExperimentLogger

LAYER_TYPES = {
    "o_proj": "Attn Output (O)", "q_proj": "Query/Key (Q/K)", "k_proj": "Query/Key (Q/K)",
    "v_proj": "Value (V)", "down_proj": "FFN Final", "gate_proj": "FFN Intermediate", "up_proj": "FFN Intermediate",
    # BERT / RoBERTa naming
    "attention.output.dense": "Attn Output (O)", "query": "Query/Key (Q/K)", "key": "Query/Key (Q/K)",
    "value": "Value (V)", "intermediate.dense": "FFN Intermediate", "output.dense": "FFN Final",
}


def layer_type(name: str) -> str:
    for suffix in sorted(LAYER_TYPES, key=len, reverse=True):
        if name.endswith(suffix):
            return LAYER_TYPES[suffix]
    return "Other"


def hf_setup(args):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    from run_glue import TASKS

    (k1, k2), num_labels, _ = TASKS[args.task]
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_model, num_labels=num_labels, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    try:
        train = load_dataset("nyu-mll/glue", args.task)["train"]
    except Exception:
        train = load_dataset("glue", args.task)["train"]
    train = train.shuffle(seed=0).select(range(min(args.samples, len(train))))
    cols = [c for c in train.column_names if c != "label"]
    train = train.map(lambda ex: tok(ex[k1], ex[k2] if k2 else None, truncation=True, max_length=128),
                      batched=True, remove_columns=cols).rename_column("label", "labels")
    device = next(model.parameters()).device

    class OnDevice:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for b in self.loader:
                yield {k: v.to(device) for k, v in b.items()}

    loader = OnDevice(DataLoader(train, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tok)))
    return model, loader, f"{args.task} training split ({args.hf_model})"


def svd_importance(model, targets, loader, max_samples):
    weights = {n: t.module.weight for n, t in targets.items()}
    flags = {n: w.requires_grad for n, w in weights.items()}
    for w in weights.values():
        w.requires_grad_(True)
        w.grad = None
    model.eval()
    seen = 0
    for batch in loader:
        batch = slice_batch(batch, 0, max_samples - seen)
        n = batch_size(batch)
        loss, _ = forward_loss(model, batch)
        (loss * n / max_samples).backward()
        seen += n
        if seen >= max_samples:
            break
    scores = {}
    with torch.no_grad():
        for n, w in weights.items():
            u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
            ds = torch.einsum("mi,mn,in->i", u, w.grad.float(), vh)
            scores[n] = float((s * ds).abs().sum())
            w.grad = None
            w.requires_grad_(flags[n])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "sensitivity_llama7b_mnli"))
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--hf-model", required=True, help="Hugging Face model id")
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    logger = ExperimentLogger(args.out, name="sensitivity")
    logger.log_environment()

    model, calib, source = hf_setup(args)
    profiler = FlowProfiler(model, exclude_modules=[r"(^|\.)(score|classifier)(\.|$)"])
    prof = profiler.profile(calib, max_samples=args.samples)
    svd = svd_importance(model, profiler.targets, calib, args.samples)
    logger.info(f"calibration: {prof.num_samples} samples of {source} (training split)")

    rows = []
    for n, p in prof.modules.items():
        rows.append({
            "layer": n, "type": layer_type(n), "D": p.descriptor, "D_abs": p.descriptor_abs,
            "G": p.grad_norm, "W_norm": p.weight_norm, "C": p.criticality, "svd_importance": svd[n],
        })
    keys = ["D_abs", "G", "svd_importance", "C"]
    col = {k: [r[k] for r in rows] for k in keys}
    corr = {}
    for a in ("D_abs", "G", "C"):
        rho, pval = spearmanr(col[a], col["svd_importance"])
        corr[f"{a}_vs_SVD"] = {"rho": float(rho), "p": float(pval)}
    rho, pval = spearmanr(col["D_abs"], col["G"])
    corr["D_abs_vs_G"] = {"rho": float(rho), "p": float(pval)}

    logger.info(f"{'layer':<30} {'type':<18} {'D_l':>10} {'G_l':>9} {'SVD':>9} {'C_l':>9}")
    for r in rows:
        logger.info(f"{r['layer']:<30} {r['type']:<18} {r['D']:>10.3e} {r['G']:>9.3e} "
                    f"{r['svd_importance']:>9.3e} {r['C']:>9.3e}")
    for k, v in corr.items():
        logger.info(f"Spearman rho ({k}, n={len(rows)} layers) = {v['rho']:.3f} (p = {v['p']:.3g})")

    norm = {k: max(col[k]) or 1.0 for k in keys}
    by_type = {}
    for r in rows:
        agg = by_type.setdefault(r["type"], {k: [] for k in keys})
        for k in keys:
            agg[k].append(r[k] / norm[k])
    table = {t: {k: sum(v) / len(v) for k, v in agg.items()} for t, agg in by_type.items()}
    logger.info("per-type aggregation (scores normalised by their maximum over layers):")
    logger.info(f"{'Layer type':<20} {'Flow D_l':>9} {'Grad G_l':>9} {'SVD':>9} {'C_l':>9}")
    for t, v in table.items():
        logger.info(f"{t:<20} {v['D_abs']:>9.2f} {v['G']:>9.2f} {v['svd_importance']:>9.2f} {v['C']:>9.2f}")

    with open(os.path.join(args.out, "layer_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "correlations.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman": corr, "per_type": table, "num_layers": len(rows),
                   "num_samples": prof.num_samples}, f, indent=2)
    logger.log("correlations", {"spearman": corr, "per_type": table})
    logger.close()


if __name__ == "__main__":
    main()
