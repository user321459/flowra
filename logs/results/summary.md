# Experiment results

Scores for the launch scripts in `experiments/`. Machine-readable copy: `metrics.json`.
Default budget `B = 0.0087`, `η = 0.3`, 512 training-split calibration examples, 3 seeds.

## GLUE on LLaMA-3-8B

`experiments/run_glue.py` + `glue_llama3_8b.yaml`

| Method   | Params (%) | GLUE |
| -------- | ---------- | ---- |
| LoRA     | 0.84       | 85.9 |
| AdaLoRA  | 0.92       | 86.4 |
| DoRA     | 0.92       | 86.7 |
| FLowRA   | 0.87       | 88.1 |
| Full FT  | 100        | 89.4 |

## NLU on LLaMA-2-7B

`experiments/run_glue.py`, `experiments/run_superglue.py` + `glue_llama2_7b.yaml`

| Method   | Params (%) | GLUE | SuperGLUE |
| -------- | ---------- | ---- | --------- |
| Frozen   | 0.00       | 68.2 | 64.5      |
| LoRA     | 0.84       | 84.7 | 78.9      |
| AdaLoRA  | 1.21       | 85.2 | 79.5      |
| DoRA     | 0.92       | 85.8 | 80.1      |
| FLowRA   | 0.87       | 87.4 | 82.7      |
| Full FT  | 100        | 89.1 | 84.5      |

GLUE tasks (FLowRA): MNLI 88.85, QQP 91.79, QNLI 94.30, SST-2 95.89, CoLA 67.89, STS-B 91.49, MRPC 91.24, RTE 82.53, WNLI 83.04.

## Question answering

`experiments/run_qa.py` + `generation_llama.yaml` (LLaMA-3-8B)

| Dataset | Full FT F1/EM | LoRA | FLowRA |
| ------- | ------------- | ---- | ------ |
| SQuAD v1.1 | 91.2/84.3 | 88.5/81.2 | 90.1/82.9 |
| SQuAD v2.0 | 84.7/81.9 | 81.2/78.1 | 83.5/80.6 |
| NQ | 79.3/68.7 | 75.8/64.2 | 78.1/67.3 |
| TriviaQA | 82.4/76.8 | 78.9/72.1 | 81.2/75.4 |

## Alpaca

`experiments/run_alpaca.py` + `alpaca_llama2_7b.yaml`

| Bucket   | Full FT | LoRA | DoRA | FLowRA |
| -------- | ------- | ---- | ---- | ------ |
| Simple   | 92.4    | 89.1 | 90.8 | 91.7   |
| Moderate | 85.7    | 80.2 | 82.5 | 84.9   |
| Complex  | 76.3    | 65.8 | 69.7 | 75.9   |
| Average  | 84.8    | 78.4 | 81.0 | 84.2   |

## Code and reasoning (LLaMA-2-13B)

`experiments/run_code.py`, `experiments/run_reasoning.py` + `reasoning_llama13b.yaml`

| Method  | GSM8K | MATH | HumanEval | MBPP | Avg  |
| ------- | ----- | ---- | --------- | ---- | ---- |
| Full FT | 54.8  | 18.2 | 32.5      | 41.2 | 36.7 |
| LoRA    | 48.3  | 14.1 | 28.1      | 36.8 | 31.8 |
| AdaLoRA | 50.9  | 15.8 | 29.8      | 38.4 | 33.7 |
| DoRA    | 51.7  | 16.4 | 30.2      | 38.9 | 34.3 |
| FLowRA  | 53.6  | 17.5 | 31.5      | 40.2 | 35.7 |

HumanEval/MBPP pass@10 (FLowRA): 53.1 / 61.5.

## Vision and VQA

`experiments/run_vision.py`, `experiments/run_vqa.py`

| Method | ImageNet | CUB-200 | VQA v2 | ADE20K |
| ------ | -------- | ------- | ------ | ------ |
| Full FT | 81.8 | 85.2 | 72.1 | 48.3 |
| LoRA    | 80.1 | 82.4 | 69.8 | 45.1 |
| Adapter | 79.8 | 82.9 | 68.9 | 44.7 |
| Prefix  | 78.4 | 80.2 | 67.3 | 42.9 |
| FLowRA  | 81.3 | 84.6 | 71.4 | 47.5 |

CUB few-shot (FLowRA): 1-shot 44.16, 4-shot 62.30, 16-shot 75.41, 32-shot 78.43.

## Ablations, budget, efficiency, sequential, composition

See `metrics.json`: GLUE-dev ablations (`--variant`), matched 0.05% budget (`budget_0p05.yaml`), MNLI train / Alpaca decode (`run_efficiency.py`), MNLI→QQP→QNLI→SST-2 (`run_sequential.py`), eight-task merge (`run_composition.py`).
