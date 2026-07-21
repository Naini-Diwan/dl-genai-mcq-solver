# IIT Madras DL Gen-AI Project [2026]
## Kaggle - Smart MCQ Solver Challenge

### Milestone-1
**NLP Foundation & Semantic Similarity**

Performed preprocessing of data: Text cleaning, normalization, and handling punctuation.

Generated baseline model TF-IDF (Text Frequency-Inverse Document Frequency):

TfidfVectorizer is chosen over CountVectorizer because it penalizes highly frequent, uninformative words. cosine_similarity is chosen over Euclidean distance because it measures the semantic direction of the text vectors, independent of their length.

Computed cosine similarity between a 'prompt' and 'options' and understood its concepts.

Evaluation of the model: Calculated Accuracy, F1 score and mAP@3. Learnt the importance of mAP.

**MAP@3 MechanicsPrecision@3 -**

Calculates the ratio of relevant items among the top 3 recommended results.
* Average Precision at 3 (AP@3): Averages the precision scores at each rank position up to 3, but only for the ranks where a relevant item was actually retrieved.
* Mean Average Precision (MAP@3): Takes the mean (average) of the AP@3 scores across all evaluation queries, users, or test cases in your dataset.

Scoring Range: The final metric value falls strictly between 0 (completely irrelevant top 3 suggestions) and 1 (perfect top 3 relevant predictions).

### Milestone-5
**Ensembling**

An ensemble pipeline for the Kaggle **Smart MCQ Solver Challenge**: predicting the correct answer to five-option (A–E) multiple-choice questions, scored by MAP@3.

Four models are built, in order of increasing complexity:

1. **TF-IDF cosine similarity** — no training, similarity between prompt and option vectors
2. **DeBERTa-v3-small** — fully fine-tuned with Hugging Face `AutoModelForMultipleChoice`
3. **RoBERTa-base + LoRA** — parameter-efficient fine-tuning via Low-Rank Adaptation
4. **Weighted ensemble** — 0.70 × DeBERTa + 0.30 × RoBERTa softmax probabilities

## Results

| Model | Accuracy | Macro F1 | MAP@3 |
|---|---|---|---|
| TF-IDF (cosine similarity) | 0.1250 | 0.1231 | 0.2712 |
| **Ensemble (0.7 DeBERTa + 0.3 RoBERTa)** | **0.9750** | **0.9746** | **0.9833** |

Standalone validation metrics for DeBERTa and RoBERTa are logged to Weights & Biases during training but not printed as a final summary in the notebook; only the ensemble's combined metrics are reported directly.


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

## Running

Open and run `dl-23f3001480-ensemble.ipynb` top to bottom. It:

1. Loads and lowercases the training/test prompts
2. Splits train into an 80/20 stratified train/validation split (`random_state=42`)
3. Fits and evaluates the TF-IDF baseline
4. Fine-tunes DeBERTa-v3-small (4 epochs, lr 1e-5, cosine schedule)
5. Fine-tunes RoBERTa-base with LoRA (r=8, alpha=16, target modules `query`/`value`, 10 epochs, lr 5e-4)
6. Combines DeBERTa and RoBERTa softmax probabilities (0.7 / 0.3) and evaluates the ensemble on validation
7. Runs inference on the test set and writes `submission.csv`



---
By: [Naini Diwan](https://naini-diwan.github.io/Hello-Naini/)

Roll No.: 23f3001480
