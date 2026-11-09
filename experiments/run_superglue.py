"""
SuperGLUE. Official task metrics:

* BoolQ / COPA / RTE / WiC / WSC — accuracy
* CB — mean of accuracy and macro-F1
* MultiRC — mean of F1a and exact match over questions
* ReCoRD — token F1 / exact match against entity answers

    python experiments/run_superglue.py --task boolq --method flowra --seed 0
    python experiments/run_superglue.py --task record --method flowra --seed 0
"""

import argparse
import json
import os

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from hf_utils import SUPERGLUE_TASKS, encoded_glue_loaders, load_superglue_raw, tokenizer_and_model
from methods import METHODS, attach, train_attached, trainable_count
from metrics import cb_score, em_f1, generate_text, mean_em_f1, multirc_scores
from flowra.utils.batching import move_to_device
from flowra.utils.logging_utils import ExperimentLogger
from hf_utils import causal_lm, glue_metric
import torch


def _qid(ex) -> tuple:
    idx = ex.get("idx")
    if isinstance(idx, dict):
        return (idx.get("paragraph"), idx.get("question"))
    return (idx,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", required=True, choices=sorted(SUPERGLUE_TASKS))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "superglue"))
    args = ap.parse_args()

    cfg = load_config(resolve_config(args.config, "superglue_llama.yaml"))
    (k1, k2), num_labels, kind = SUPERGLUE_TASKS[args.task]
    device = device_str()
    set_seed(args.seed)
    out = os.path.join(args.out, f"{args.task}_{args.method}_seed{args.seed}")
    logger = ExperimentLogger(out, name=args.task)
    logger.log_environment()
    raw = load_superglue_raw(args.task)
    val_raw = raw["validation"]

    if args.task == "record":
        tok, model = causal_lm(cfg["model"], device, cfg.get("dtype"))
        from transformers import DataCollatorForLanguageModeling
        from torch.utils.data import DataLoader

        def tokenize(ex):
            gold = (ex.get("answers") or [" "])[0]
            text = f"Passage: {ex['passage']}\nQuery: {ex['query']}\nAnswer: {gold}"
            return tok(text, truncation=True, max_length=cfg["max_length"])

        data = raw["train"].map(tokenize, remove_columns=raw["train"].column_names)
        collate = DataCollatorForLanguageModeling(tok, mlm=False)
        g = torch.Generator().manual_seed(args.seed)
        train = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
        n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(data))
        calib = DataLoader(data.select(range(n_cal)), batch_size=1, collate_fn=collate)

        def evaluate(m):
            m.eval()
            rows = []
            for ex in val_raw:
                prompt = f"Passage: {ex['passage']}\nQuery: {ex['query']}\nAnswer:"
                pred = generate_text(m, tok, prompt, device, max_new_tokens=16)[0]
                golds = list(ex.get("answers") or [""])
                rows.append(em_f1(pred, golds))
            scores = mean_em_f1(rows)
            logger.info(f"ReCoRD F1={100 * scores['f1']:.1f} EM={100 * scores['em']:.1f}")
            return {**scores, "f1_pct": 100 * scores["f1"], "em_pct": 100 * scores["em"]}
    elif args.task == "copa":
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        tok, model = tokenizer_and_model(cfg["model"], 2, device, cfg.get("dtype"))

        def _copa_pair(ex, choice: str):
            q = "because" if ex.get("question") == "cause" else "so"
            return tok(ex["premise"], f"{q} {choice}", truncation=True, max_length=cfg["max_length"])

        rows = []
        for ex in raw["train"]:
            for i, ch in enumerate((ex["choice1"], ex["choice2"])):
                enc = _copa_pair(ex, ch)
                item = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
                item["labels"] = int(ex["label"] == i)
                rows.append(item)
        collate = DataCollatorWithPadding(tok)
        g = torch.Generator().manual_seed(args.seed)
        train = DataLoader(rows, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
        n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(rows))
        calib = DataLoader(rows[:n_cal], batch_size=1, collate_fn=collate)

        def evaluate(m):
            m.eval()
            correct, n = 0, 0
            for ex in val_raw:
                scores = []
                for ch in (ex["choice1"], ex["choice2"]):
                    enc = _copa_pair(ex, ch)
                    enc = {k: torch.tensor(enc[k]).unsqueeze(0).to(device)
                           for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
                    logits = m(**enc).logits
                    scores.append(float(logits[0, 1] if logits.shape[-1] > 1 else logits[0, 0]))
                pred = int(scores[1] > scores[0])
                correct += int(pred == int(ex["label"]))
                n += 1
            acc = correct / max(n, 1)
            logger.info(f"COPA two-choice acc={100 * acc:.1f}")
            return {"accuracy": acc, "score": acc}
    else:
        tok, model = tokenizer_and_model(cfg["model"], num_labels, device, cfg.get("dtype"))
        train, calib, dev = encoded_glue_loaders(raw, tok, (k1, k2), cfg, args.seed, args.task, suite="super_glue")

        def evaluate(m):
            m.eval()
            preds, labels, qids = [], [], []
            i = 0
            for batch in dev:
                batch = move_to_device(batch, device)
                logits = m(**{k: v for k, v in batch.items() if k != "labels"}).logits
                p = logits.argmax(-1).detach().cpu().tolist()
                y = batch["labels"].detach().cpu().tolist()
                preds.extend(p if isinstance(p, list) else [p])
                labels.extend(y if isinstance(y, list) else [y])
                for _ in range(len(y) if isinstance(y, list) else 1):
                    if i < len(val_raw):
                        qids.append(_qid(val_raw[i]))
                    i += 1
            if kind == "cb":
                scores = cb_score(preds, labels)
            elif kind == "multirc":
                scores = multirc_scores(preds, labels, qids)
            else:
                t = torch.tensor(preds).float()
                scores = glue_metric("acc", t, torch.tensor(labels).float())
            logger.info(f"{args.task}: {scores}")
            return scores

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    tcfg = training_config(cfg, args.seed, epochs=cfg["epochs"][args.task], eval_every=len(train))
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model.eval() if hasattr(model, "eval") else model)
    result = {"task": args.task, "method": args.method, "seed": args.seed, **final,
              "trainable_params": fw.parameter_counts()["adapter"] if fw else trainable_count(model)}
    logger.log("result", result, "RESULT " + json.dumps(result))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
