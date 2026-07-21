# IIT Madras DL Gen-AI Project [2026]
# Kaggle - Smart MCQ Solver Challenge

## Milestone-2
**Transformers**

Utilized the Hugging Face transformers and datasets libraries. Learnt architecture of BERT/RoBERTa and the concept of attention mechanisms. Used pre-trained embedding models to generate context-aware embeddings.

Fine-tuned `microsoft/deberta-v3-small` as a multiple-choice question answering model using Hugging Face `Trainer`.

## Overview

Each question has a prompt and five answer options (A–E). The model scores every `(prompt, option)` pair and picks the option with the highest score, using the `AutoModelForMultipleChoice` head. Training and evaluation are logged to Weights & Biases.

## Pipeline

1. **Setup** — seeds all RNGs (`seed=42`) for reproducibility; detects CUDA device.
2. **Data** — loads `train.csv` / `test.csv`, creates an 80/20 stratified train/validation split on the `answer` column.
3. **Dataset** — `HuggingFaceMCQDataset` tokenizes each `(prompt, option)` pair for all 5 options per row, producing tensors shaped `[num_choices, max_length]`.
4. **Model** — `microsoft/deberta-v3-small` loaded via `AutoModelForMultipleChoice` (classification head is randomly initialized, since the base checkpoint has no MC head).
5. **Training** — `Trainer` with cosine LR schedule, 3 epochs, gradient accumulation, evaluated every epoch on accuracy and macro F1.
6. **Inference** — runs the trained model on the test set, takes the softmax over the 5 options, and writes the top-3 predicted labels per question to `submission.csv`.

## Requirements

```
torch
transformers
peft
scikit-learn
pandas
numpy
wandb
```

`peft` and `wandb` require a Kaggle/Colab environment or local installation (`pip install torch transformers peft scikit-learn pandas numpy wandb`). A Weights & Biases account/API key is needed to log runs (or set `report_to="none"` in `TrainingArguments` to skip it).

## Configuration

| Parameter | Value |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Max sequence length | 128 |
| Epochs | 3 |
| Learning rate | 1e-5 (cosine schedule, 10% warmup) |
| Batch size | 8 (train/eval), grad accumulation ×2 |
| Max grad norm | 0.5 |
| Seed | 42 |

## Results

*Validation performance*:

| Epoch | Training Loss | Validation Loss | Accuracy | F1 (macro) |
|---|---|---|---|---|
| 1 | 4.695 | 2.740 | 0.235 | 0.232 |
| 2 | 4.430 | 2.613 | 0.435 | 0.439 |
| 3 | 5.199 | 2.354 | 0.510 | 0.511 |

Best checkpoint (epoch 3): **51.0% accuracy**, **0.511 macro F1**.



### Reference - [Vaswani et. al. : Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)


---
Hey! I'm [Naini Diwan](https://naini-diwan.github.io/Hello-Naini/) :)

Roll No.: 23f3001480