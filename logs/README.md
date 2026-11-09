# Experiment logs

Reported scores for the launch scripts are in `logs/results/` (`summary.md`, `metrics.json`).

Live runs write under `logs/glue`, `logs/superglue`, `logs/qa`, `logs/alpaca`,
`logs/code`, `logs/reasoning`, `logs/vision`, `logs/vqa`, `logs/sequential`,
`logs/composition`, `logs/efficiency`. Those directories are created when a job
starts and are not committed. FLowRA runs write `layer_profile.csv` and `adapter.pt`.

`sensitivity_llama7b_mnli/` holds layer-level flow / gradient / SVD scores
from a 512-example MNLI training-split calibration (32 blocks × 5 types).

```bash
python experiments/sensitivity_correlation.py --hf-model meta-llama/Llama-2-7b-hf --task mnli \
    --out logs/sensitivity_llama7b_mnli
```
