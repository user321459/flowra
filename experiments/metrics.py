"""
Evaluation metrics.

SQuAD / NQ / TriviaQA: token F1 and exact match
Alpaca: 1–10 → % in Simple / Moderate / Complex
HumanEval / MBPP: pass@1 and pass@10
GSM8K 8-shot and MATH 4-shot accuracy (final numeric answer)
VQA v2: min(#agreeing annotators / 3, 1); ADE20K: mIoU
SuperGLUE: CB (acc+macro-F1)/2, MultiRC F1a/EM, ReCoRD F1/EM
"""

import math
import os
import re
import string
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

import torch


# ---------------------------------------------------------------------------
# SQuAD-style F1 / EM
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, golds: Sequence[str]) -> float:
    n = normalize_answer(pred)
    return float(any(n == normalize_answer(g) for g in golds))


def f1_against(pred: str, golds: Sequence[str]) -> float:
    if not golds:
        return float(normalize_answer(pred) == "")
    return max(_token_f1(pred, g) for g in golds)


def em_f1(pred: str, golds: Sequence[str]) -> Dict[str, float]:
    golds = [g for g in golds if g is not None]
    if not golds:
        golds = [""]
    em, f1 = exact_match(pred, golds), f1_against(pred, golds)
    return {"em": em, "f1": f1, "score": f1}


def mean_em_f1(pairs: Iterable) -> Dict[str, float]:
    rows = list(pairs)
    if not rows:
        return {"em": 0.0, "f1": 0.0, "score": 0.0}
    em = sum(r["em"] for r in rows) / len(rows)
    f1 = sum(r["f1"] for r in rows) / len(rows)
    return {"em": em, "f1": f1, "score": f1}


def gold_qa_answers(ex: dict, dataset: str) -> List[str]:
    if dataset == "squad_v2" and (ex.get("is_impossible") or not (ex.get("answers") or {}).get("text")):
        return [""]
    if dataset in ("squad", "squad_v2"):
        return list((ex.get("answers") or {}).get("text") or [""])
    if dataset == "trivia_qa":
        ans = ex.get("answer") or {}
        aliases = list(ans.get("aliases") or [])
        if ans.get("value"):
            aliases = [ans["value"]] + aliases
        return aliases or [""]
    if dataset == "natural_questions":
        anns = ex.get("annotations") or []
        texts = []
        for a in anns if isinstance(anns, list) else []:
            for s in a.get("short_answers") or []:
                t = s.get("text") if isinstance(s, dict) else s
                if t:
                    texts.append(t if isinstance(t, str) else " ".join(t))
            if a.get("yes_no_answer") in ("YES", "NO"):
                texts.append(a["yes_no_answer"].title())
        if ex.get("short_answers"):
            texts.extend(ex["short_answers"])
        return texts or [""]
    texts = (ex.get("answers") or {}).get("text") if isinstance(ex.get("answers"), dict) else ex.get("answers")
    if isinstance(texts, list):
        return [str(t) for t in texts] or [""]
    return [str(texts or "")]


# ---------------------------------------------------------------------------
# Alpaca: instruction complexity (1–10 → percent)
# ---------------------------------------------------------------------------
_COMPLEX_HINTS = re.compile(
    r"\b(step[- ]?by[- ]?step|first|then|finally|list|compare|analyze|implement|prove|debug)\b", re.I)

_ALPACA_SYSTEM = (
    "You are an expert rater of instruction-following quality. "
    "Compare the model response to the reference. "
    "Score from 1 (unusable) to 10 (matches or exceeds the reference). "
    "Reply with JSON only: {\"score\": <integer 1-10>}."
)


def alpaca_complexity(instruction: str, inp: str = "") -> str:
    """Simple / Moderate / Complex from constraint load."""
    n = len((instruction or "").split())
    has_input = bool((inp or "").strip())
    long_input = len((inp or "").split()) >= 40
    if long_input or n >= 40 or _COMPLEX_HINTS.search(instruction or ""):
        return "Complex"
    if has_input or n >= 16:
        return "Moderate"
    return "Simple"


def alpaca_score(instruction: str, inp: str, reference: str, prediction: str,
                 model: str = "gpt-4") -> float:
    """
    Returns 1–10.
    Requires ``OPENAI_API_KEY``; optional ``OPENAI_BASE_URL``.
    """
    import json
    import os
    import urllib.request

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Alpaca eval requires OPENAI_API_KEY; optionally OPENAI_EVAL_MODEL."
        )
    user = (
        f"Instruction:\n{instruction}\n"
        + (f"Input:\n{inp}\n" if inp else "")
        + f"Reference:\n{reference}\n"
        + f"Model response:\n{prediction}\n"
    )
    payload = json.dumps({
        "model": os.environ.get("OPENAI_EVAL_MODEL", model),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _ALPACA_SYSTEM},
            {"role": "user", "content": user},
        ],
    }).encode()
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    data = json.loads(match.group(0) if match else content)
    return float(max(1, min(10, int(data["score"]))))


def alpaca_bucket_scores(rows: Sequence[dict]) -> Dict[str, float]:
    """``rows`` items: ``{"complexity", "score"}`` with score in 1–10. Reported percent = 10×score."""
    by = defaultdict(list)
    for r in rows:
        by[r["complexity"]].append(r["score"])
    out = {}
    for k in ("Simple", "Moderate", "Complex"):
        vals = by.get(k) or [0.0]
        out[k.lower()] = 10.0 * sum(vals) / len(vals)
    out["average"] = sum(out[k] for k in ("simple", "moderate", "complex")) / 3.0
    out["score"] = out["average"]
    return out


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(n_correct: Sequence[int], n_sample: int, ks: Sequence[int] = (1, 10)) -> Dict[str, float]:
    out = {}
    for k in ks:
        out[f"pass@{k}"] = sum(pass_at_k(n_sample, c, k) for c in n_correct) / max(len(n_correct), 1)
    out["score"] = out.get("pass@1", 0.0)
    return out


def run_code_eval(predictions: List[List[str]], references: List[str], ks=(1, 10)) -> Dict[str, float]:
    """Execute generated programs against unit tests (HumanEval / MBPP)."""
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from evaluate import load
    metric = load("code_eval")
    scores, _ = metric.compute(predictions=predictions, references=references, k=list(ks), num_workers=1)
    return {f"pass@{k}": float(scores[f"pass@{k}"]) for k in ks} | {"score": float(scores["pass@1"])}


# ---------------------------------------------------------------------------
# GSM8K / MATH
# ---------------------------------------------------------------------------
def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return normalize_answer(answer.split("####")[-1])
    nums = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    return nums[-1] if nums else ""


def last_number(text: str) -> str:
    nums = re.findall(r"-?\d+\.?\d*", str(text).replace(",", ""))
    return nums[-1] if nums else ""


def fewshot_prompt(demos: Sequence[tuple], question: str, n_shot: int) -> str:
    parts = []
    for q, a in demos[:n_shot]:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# VQA v2
# ---------------------------------------------------------------------------
def vqa_soft_accuracy(pred: str, answers: Sequence[str]) -> float:
    """Official VQA acc: min(# annotators with this answer / 3, 1)."""
    pred_n = normalize_answer(pred)
    n = sum(1 for a in answers if normalize_answer(a) == pred_n)
    return min(n / 3.0, 1.0)


# ---------------------------------------------------------------------------
# ADE20K mIoU
# ---------------------------------------------------------------------------
def confusion_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                   ignore_index: int = 255) -> torch.Tensor:
    """Accumulate a ``num_classes x num_classes`` confusion matrix (rows = gold)."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = target != ignore_index
    pred, target = pred[mask], target[mask]
    idx = (target.clamp(0, num_classes - 1) * num_classes + pred.clamp(0, num_classes - 1)).long()
    conf = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()
    return conf


def miou_from_confusion(conf: torch.Tensor) -> float:
    inter = conf.diag()
    union = conf.sum(1) + conf.sum(0) - inter
    valid = union > 0
    if not valid.any():
        return 0.0
    return float((inter[valid] / union[valid]).mean())


# ---------------------------------------------------------------------------
# SuperGLUE helpers
# ---------------------------------------------------------------------------
def cb_score(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    from sklearn.metrics import f1_score, accuracy_score
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro"))
    return {"accuracy": acc, "f1": f1, "score": 0.5 * (acc + f1)}


def multirc_scores(preds: Sequence[int], labels: Sequence[int], question_ids: Sequence) -> Dict[str, float]:
    """F1a over answer-options and EM over questions (all options of a question correct)."""
    tp = fp = fn = 0
    by_q = defaultdict(list)
    for p, y, qid in zip(preds, labels, question_ids):
        tp += int(p == 1 and y == 1)
        fp += int(p == 1 and y == 0)
        fn += int(p == 0 and y == 1)
        by_q[qid].append(int(p == y))
    f1a = 2 * tp / max(2 * tp + fp + fn, 1)
    em = sum(all(v) for v in by_q.values()) / max(len(by_q), 1)
    return {"f1a": f1a, "em": em, "score": 0.5 * (f1a + em)}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(model, tok, prompt: str, device, max_new_tokens: int = 64,
                  num_return: int = 1, temperature: float = 0.8) -> List[str]:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    sample = num_return > 1
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if sample:
        kw.update(do_sample=True, temperature=temperature, num_return_sequences=num_return)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    prompt_len = enc["input_ids"].shape[1]
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]
