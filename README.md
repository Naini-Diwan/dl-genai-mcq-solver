# IIT Madras DL Gen-AI Project [2026]

## Milestone 4: Formulating MCQ Task & Fine-Tuning

**Short Summary**

*peft RoBERTa-base + LoRA* (parameter-efficient fine-tuning of RoBERTa via Low-Rank Adaptation)

* Data formatting for MCQ: Concatenating the question with options.
* Introduced LoRA-finetuning, which is advantageous over Full-finetuning.
* Setting up a training loop to fine-tune the model weights on the dataset.
* Managing GPU memory and batch sizes.
* Training efficiency strategies with Training arguments.

**Description**

Fine-tuned `RoBERTa` base with LoRA to answer five-option multiple-choice questions for the Kaggle "Smart MCQ Solver Challenge" competition, and generated top-3 ranked predictions scored by MAP@3.

Notebook: `roberta-lora-fine-tuning.ipynb`

## What it does

1. Loads `train.csv` / `test.csv` from the competition data.
2. Cleans the data: drops duplicate prompts, lower-cases text, strips boilerplate instruction phrases (e.g. "pick the best possible answer:") from prompts.
3. Splits train data 80/20 into train/validation, stratified on the answer label.
4. Wraps each question + its 5 options into a Hugging Face `AutoModelForMultipleChoice`-compatible dataset.
5. Fine-tunes `roberta-base` using LoRA (PEFT) instead of full fine-tuning.
6. Tracks accuracy, macro F1, and MAP@3 during training (MAP@3 is the model-selection metric, matching the competition's scoring).
7. Logs runs to Weights & Biases.
8. Runs inference on the test set and builds top-3 ranked answer strings (e.g. `"B D A"`).

## Data

| Column | Description |
|---|---|
| `id` | Row identifier |
| `prompt` | Question text |
| `A`–`E` | Five candidate answers |
| `answer` | Correct option (train only), one of A–E |

Train set: 2,000 rows, no missing values, 242 duplicate prompts removed before splitting.

## Requirements

```
torch
torchao
pandas
numpy
scikit-learn
transformers
peft
wandb
```

Also expects a Kaggle Secrets entry named for W&B logging (via `kaggle_secrets.UserSecretsClient`).

## Key config

**LoRA:** `r=8`, `lora_alpha=16`, `target_modules=["query","value"]`, `lora_dropout=0.1`, `bias="none"`, `task_type=SEQ_CLS`

**Training:** `learning_rate=5e-4`, batch size 4 with gradient accumulation 4 (effective 16), 12 epochs, cosine LR schedule with 3% warmup, eval/save every epoch, best checkpoint selected by `eval_map@3`.

Random seed fixed at `42` across Python, NumPy, and PyTorch (including CUDA) for reproducibility.

## Results

| **Model** | **Accuracy** | **F1 Score** | **MAP@3** |
|---|---|---|---|
| **RoBERTa L(ORA finetuned)** | 0.8409 | 0.8422 | 0.8929 |

## Output

Per-test-question top-3 predicted labels, ranked by predicted probability, in the format the competition's MAP@3 scoring expects (space-separated letters, e.g. `"B D A"`).

## Data

Expects the competition files at:

```
/kaggle/input/competitions/smart-mcq-solver-challenge/
├── train.csv     # id, prompt, A, B, C, D, E, answer (2000 rows)
├── test.csv      # id, prompt, A, B, C, D, E
└── sample_submission.csv
```

Outside Kaggle, update `TRAIN_PATH` / `TEST_PATH` in the setup cell to point at your local copy.

## Environment

- Python 3.12
- PyTorch 2.10 (cu128)
- `transformers`, `peft`, `torchao`, `wandb`, `scikit-learn`, `pandas`, `numpy`

Requires a Weights & Biases API key, retrieved via Kaggle Secrets (`WandB-API`) in the notebook. Outside Kaggle, set the `WANDB_API_KEY` environment variable or call `wandb.login()` directly instead.

GPU is recommended; the notebook auto-detects CUDA and falls back to CPU otherwise.


---
By: [Naini Diwan](https://naini-diwan.github.io/Hello-Naini/)

Roll No.: 23f3001480
