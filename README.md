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

---
By: [Naini Diwan](https://naini-diwan.github.io/Hello-Naini/)

Roll No.: 23f3001480
