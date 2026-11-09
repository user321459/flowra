# Layer-level sensitivity (LLaMA-7B, MNLI)

Calibration: 512 examples from the MNLI **training** split.
160 observations: 32 transformer blocks × 5 layer types
(Attn Output, Query/Key, Value, FFN Final, FFN Intermediate).

Machine-readable copies: `layer_scores.csv`, `layer_scores.xlsx`.

Spearman correlations are computed on the 160 individual rows of `layer_scores.csv`,
not on type-level means.

```bash
python -c "import csv; from scipy.stats import spearmanr
rows=list(csv.DictReader(open('logs/sensitivity_llama7b_mnli/layer_scores.csv')))
d=[float(r['D_abs']) for r in rows]; c=[float(r['C']) for r in rows]; s=[float(r['svd_importance']) for r in rows]
print(round(spearmanr(d,s).correlation, 2), round(spearmanr(c,s).correlation, 2))"
```
