# IIT Madras DL Gen-AI Project [2026]
# Kaggle - Smart MCQ Solver Challenge

## Milestone-1
**NLP Foundation & Semantic Similarity**

Performed preprocessing of data: dropping rows with duplicate questions, lower-case conversion, removing the repititive parts from questions and encoding the answer-options.

Generated baseline model TF-IDF (Text Frequency-Inverse Document Frequency):

TfidfVectorizer is chosen over CountVectorizer because it penalizes highly frequent, uninformative words. cosine_similarity is chosen over Euclidean distance because it measures the semantic direction of the text vectors, independent of their length.

Computed cosine similarity between a 'prompt' and 'options' and understood its concepts.

Evaluation of the model: Calculated Accuracy, F1 score and mAP@3. Learnt the importance of mAP.

**MAP@3 MechanicsPrecision@3 -**

Calculates the ratio of relevant items among the top 3 recommended results.
* Average Precision at 3 (AP@3): Averages the precision scores at each rank position up to 3, but only for the ranks where a relevant item was actually retrieved.
* Mean Average Precision (MAP@3): Takes the mean (average) of the AP@3 scores across all evaluation queries, users, or test cases in your dataset.

Scoring Range: The final metric value falls strictly between 0 (completely irrelevant top 3 suggestions) and 1 (perfect top 3 relevant predictions).

## Milestone-2
**Transformers**

Utilized the Hugging Face transformers and datasets libraries. Learnt architecture of BERT/RoBERTa and the concept of attention mechanisms. Used pre-trained embedding models to generate context-aware embeddings.

Fine-tuned `microsoft/deberta-v3-small` as a multiple-choice question answering model using Hugging Face `Trainer`.

## Milestone-3
**Context Augmentation with RAG Pipelines**

Retrieved relevant Wikipedia passages, made an index and prompted a local instruction-tuned LLM (Phi-3-mini) to rank the three most likely answers for any given MCQ. This is a retrieval + prompting pipeline, not a fine-tuned classifier so the weights of the model were not trained.

* Understood the limitations of general LLMs
* Learnt the RAG pipeline
* Loaded a simple pre-built vector database.
* Retrieved external context based on the question prompt.
* Fed the retrieved context + prompt + choices into the model to improve reasoning.

## Milestone-4
**Formulating MCQ Task & Fine-Tuning**

*peft RoBERTa-base + LoRA* (parameter-efficient fine-tuning of RoBERTa via Low-Rank Adaptation)

* Data formatting for MCQ: Concatenating the question with options.
* LoRA-finetuning was utilized, which is advantageous over Full-finetuning.
* Setting up a training loop to fine-tune the model weights on the dataset.
* Managed GPU memory and batch sizes.
* Training efficiency strategies with Training arguments.

## Milestone-5
**Ensembling**

An ensemble pipeline for predicting the correct answer to five-option (A–E) multiple-choice questions, scored by MAP@3.

* **DeBERTa-v3-small** — fully fine-tuned with Hugging Face `AutoModelForMultipleChoice`
* **RoBERTa-base + LoRA** — parameter-efficient fine-tuning via Low-Rank Adaptation
* **Weighted ensemble** — 0.70 × DeBERTa + 0.30 × RoBERTa softmax probabilities

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
