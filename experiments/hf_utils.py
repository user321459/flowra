"""Hugging Face loaders shared by the NLU / generation experiment scripts."""

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from flowra.utils.batching import move_to_device

GLUE_TASKS = {
    "cola": (("sentence", None), 2, "mcc"),
    "sst2": (("sentence", None), 2, "acc"),
    "mrpc": (("sentence1", "sentence2"), 2, "acc_f1"),
    "qqp": (("question1", "question2"), 2, "acc_f1"),
    "stsb": (("sentence1", "sentence2"), 1, "spearman"),
    "mnli": (("premise", "hypothesis"), 3, "acc"),
    "qnli": (("question", "sentence"), 2, "acc"),
    "rte": (("sentence1", "sentence2"), 2, "acc"),
    "wnli": (("sentence1", "sentence2"), 2, "acc"),
}

SUPERGLUE_TASKS = {
    "boolq": (("question", "passage"), 2, "acc"),
    "cb": (("premise", "hypothesis"), 3, "cb"),
    "copa": (("premise", "choice1"), 2, "acc"),
    "multirc": (("paragraph", "question"), 2, "multirc"),
    "record": (("passage", "query"), 1, "record"),
    "rte": (("premise", "hypothesis"), 2, "acc"),
    "wic": (("sentence1", "sentence2"), 2, "acc"),
    "wsc": (("text", None), 2, "acc"),
}

GLUE_DEV_TASKS = ("mrpc", "rte", "cola", "sst2", "qnli", "mnli")
COMPOSITION_TASKS = ("mnli", "qqp", "qnli", "sst2", "cola", "stsb", "mrpc", "rte")
SEQUENTIAL_TASKS = ("mnli", "qqp", "qnli", "sst2")


def load_glue_raw(task: str):
    from datasets import load_dataset
    try:
        return load_dataset("nyu-mll/glue", task)
    except Exception:
        return load_dataset("glue", task)


def load_superglue_raw(task: str):
    from datasets import load_dataset
    return load_dataset("super_glue", task)


def tokenizer_and_model(model_id: str, num_labels: int, device: str, dtype: Optional[str] = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        dtype or "float32", torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, torch_dtype=torch_dtype)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    return tok, model.to(device)


def causal_lm(model_id: str, device: str, dtype: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        dtype or "bfloat16", torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.config.pad_token_id = tok.pad_token_id
    return tok, model.to(device)


def _superglue_pair(ex: dict, task: str, keys: Tuple[str, Optional[str]]):
    k1, k2 = keys
    if task == "copa":
        q = "because" if ex.get("question") == "cause" else "so"
        choice = ex.get("hypothesis") or ex.get("choice1")
        return ex["premise"], f"{q} {choice}"
    if task == "multirc":
        return ex["paragraph"], f"{ex['question']} {ex['answer']}"
    if task == "wic":
        return f"{ex['sentence1']} ({ex.get('word', '')})", ex["sentence2"]
    if task == "wsc":
        return ex["text"], f"{ex.get('span1_text', '')} / {ex.get('span2_text', '')}"
    if task == "record":
        return ex["passage"], ex["query"]
    first = ex[k1]
    second = ex[k2] if k2 else None
    return first, second


def encoded_glue_loaders(raw, tok, keys: Tuple[str, Optional[str]], cfg: Dict[str, Any], seed: int,
                         task: str, suite: str = "glue"):
    from transformers import DataCollatorWithPadding

    def encode(ex):
        if suite == "super_glue":
            first, second = _superglue_pair(ex, task, keys)
        else:
            first, second = ex[keys[0]], ex[keys[1]] if keys[1] else None
        return tok(first, second, truncation=True, max_length=cfg["max_length"])

    cols = [c for c in raw["train"].column_names if c != "label"]
    data = raw.map(encode, batched=False, remove_columns=cols).rename_column("label", "labels")
    collate = DataCollatorWithPadding(tok)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(data["train"], batch_size=cfg["batch_size"], shuffle=True,
                       collate_fn=collate, generator=g)
    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(data["train"]))
    calib = DataLoader(data["train"].shuffle(seed=seed).select(range(n_cal)),
                       batch_size=min(16, cfg["batch_size"]), collate_fn=collate)
    split = "validation_matched" if task == "mnli" else "validation"
    if split not in data:
        split = "validation"
    dev = DataLoader(data[split], batch_size=cfg.get("eval_batch_size", cfg["batch_size"]),
                     collate_fn=collate)
    return train, calib, dev


def glue_metric(kind: str, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    if kind in ("pearson", "spearman"):
        p = float(pearsonr(preds, labels)[0])
        s = float(spearmanr(preds, labels)[0])
        return {"pearson": p, "spearman": s, "score": s if kind == "spearman" else p}
    preds, labels = preds.long(), labels.long()
    acc = float((preds == labels).float().mean())
    if kind == "acc":
        return {"accuracy": acc, "score": acc}
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    if kind == "acc_f1":
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return {"accuracy": acc, "f1": f1, "score": f1}
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
    mcc = (tp * tn - fp * fn) / denom
    return {"mcc": float(mcc), "score": float(mcc)}


def make_eval_fn(dev: DataLoader, device: str, kind: str, num_labels: int):
    def evaluate(m):
        preds, labels = [], []
        for batch in dev:
            batch = move_to_device(batch, device)
            logits = m(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds.append(logits.squeeze(-1) if num_labels == 1 else logits.argmax(-1))
            labels.append(batch["labels"])
        return glue_metric(kind, torch.cat(preds).float().cpu(), torch.cat(labels).float().cpu())
    return evaluate
