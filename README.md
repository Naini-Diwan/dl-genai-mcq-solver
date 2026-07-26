# IIT Madras DL Gen-AI Project 
This repository contains 6 branches, which constitute (most of) my progress for the Kaggle Smart MCQ Solver Challenge (2026), the go-to branch being `main`. It contains in the *notebooks* folder, two notebooks built for the multiple-choice question (MCQ) answering competition scored with **MAP@3** (Mean Average Precision at 3). Each item presents a question prompt and five candidate answers (`A`–`E`); a submission ranks the top three most likely correct options.
 
**dl-23f3001480-notebook-t22026.ipynb** — four supervised/lexical approaches, trained and evaluated end-to-end, culminating in a weighted ensemble.

| Model | Approach | Accuracy | Macro F1 | MAP@3 |
|---|---|---|---|---|
| 1 | TF-IDF + cosine similarity (unsupervised baseline) | 0.1761 | 0.1715 | 0.2973 |
| 2 | Pretrained **DeBERTa-v3-small**, full fine-tune | 0.9972 | 0.9972 | 0.9986 |
| 3 | Base **RoBERTa** with **LoRA** (PEFT fine-tune) | 0.9631 | 0.9609 | 0.9778 |
| 4 | Ensemble (0.70 × DeBERTa + 0.30 × RoBERTa-LoRA) | 0.9972 | 0.9970 | 0.9986 |

**rag-23f3001480.ipynb** — a standalone retrieval-augmented generation (RAG) pipeline using dense retrieval over a Wikipedia corpus plus a small instruction-tuned LLM.
Both notebooks read from the same competition data (`train.csv`, `test.csv`) and independently produce a `submission.csv`.

| Evaluation Metric | Score |
|---|---|
| Accuracy | 0.6100 |
| Macro F1 | 0.5996 |
| MAP@3 | 0.7625 |

**Note**: A detailed report for the project is present in the *reports* folder (along with milestone reports).

---
### How to Run (I mean, not etymologically!)

1. Attach the competition dataset (and, for Notebook 2, the `mbanaei` dataset) as inputs.
2. Install dependencies mentioned in `requirements.txt`.
3. Run `dl-23f3001480-notebook-t22026.ipynb` top to bottom to train Models 1–4 and generate a submission.
4. Run `rag-23f3001480.ipynb` top to bottom to build the FAISS index, run RAG inference, and generate a separate submission.
Each notebook is independent and can be run without the other.


## Milestone-1
**NLP Foundation & Semantic Similarity**

* Performed preprocessing of data: dropping rows with duplicate questions, lower-case conversion, removing the repititive parts from questions and encoding the answer-options (This preprocessing is uniform accross all the milestones, with [Milestone-3](#milestone-3) having different kind of preprocessing methodology).
* TF-IDF (Text Frequency-Inverse Document Frequency) was utilized
* Computed cosine similarity between a 'prompt' and 'options' and understood its concepts.
* Evaluation of the model: Calculated Accuracy, F1 score and mAP@3. 

TfidfVectorizer is chosen over CountVectorizer because it penalizes highly frequent, uninformative words. cosine_similarity is chosen over Euclidean distance because it measures the semantic direction of the text vectors, independent of their length.

Learnt the importance of **MAP@3**

* Average Precision at 3 (AP@3): Averages the precision scores at each rank position up to 3, but only for the ranks where a relevant item was actually retrieved.
* Mean Average Precision (MAP@3): Takes the mean (average) of the AP@3 scores across all evaluation queries, users, or test cases in your dataset.


## Milestone-2
**Transformers**

Utilized the Hugging Face transformers and datasets libraries. Learnt architecture of BERT/RoBERTa and the concept of attention mechanisms. Used pre-trained embedding models to generate context-aware embeddings.

Fine-tuned `microsoft/deberta-v3-small` as a multiple-choice question answering model using Hugging Face `Trainer`.

## Milestone-3
**Context Augmentation with RAG Pipelines**

Retrieved relevant Wikipedia passages, made an index and prompted a local instruction-tuned LLM to rank the 3 most likely answers for any given MCQ. This is a retrieval-augmented pipeline that answers questions without any task-specific fine-tuning:
 
1. **Corpus**: a 5,000-passage sample of the `mbanaei/stem-wiki-cohere-no-emb` Wikipedia dataset (downloaded via `kagglehub`).
2. **Retriever**: `all-MiniLM-L6-v2` sentence embeddings, indexed with a FAISS flat inner-product index (`IndexFlatIP`), retrieving the top-3 passages per question.
3. **Reasoner**: `microsoft/Phi-3-mini-4k-instruct`, prompted with the retrieved context, the question, and the five options, and asked to output the three most likely correct letters in ranked order.
4. **Parsing**: ranked letters are extracted from the generated text; any missing options are backfilled in `A`–`E` order to guarantee exactly 3 predictions.

Understood the limitations of general LLMs and learnt the RAG pipeline

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

---
### Data

Expects the competition files at:

```
/kaggle/input/competitions/smart-mcq-solver-challenge/
├── train.csv     # id, prompt, A, B, C, D, E, answer (2000 rows)
├── test.csv      # id, prompt, A, B, C, D, E
└── sample_submission.csv
```
Outside Kaggle, update `TRAIN_PATH` / `TEST_PATH` in the setup cell to point at your local copy.

---
### Environment
Do refer `requirements.txt`.

A Weights & Biases API key would be required, retrieved via Kaggle Secrets (`WandB-API`) in the notebook. Outside Kaggle, set the `WANDB_API_KEY` environment variable or call `wandb.login()` directly instead.

> `faiss-gpu` wheels are only published for a limited set of Python/CUDA combinations on PyPI. If installation fails in your environment, use `faiss-cpu` instead (slower, but works everywhere), or install FAISS via `conda-forge`.


---
*Phew, that was big work (and I'm so proud of it)! By the way I'm [Naini Diwan](https://naini-diwan.github.io/Hello-Naini/)* :)

Roll No.: DS23F3001480
