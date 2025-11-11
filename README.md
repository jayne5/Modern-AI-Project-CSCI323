# Modern-AI-Project-CSCI323
Sentiment Analysis on Twitter Data
This project performs sentiment analysis on tweets using multiple machine learning and deep learning models:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC
- CNN (Convolutional Neural Network)

We compare these models using both CountVectorizer and TF-IDF Vectorizer, along with Random Over Sampling and Random Under Sampling to handle class imbalance.

Repository Structure:
Modern-AI-Project-CSCI323/  
- Grp28.ipynb                # Google Colab notebook with full code and output
- README.md                  # This file

Dataset
Dataset used: Twitter Sentiment Dataset (Negative, Neutral, Positive).
Date set link : https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset

Environment Setup (Google Colab):
1. Open the notebook in Google Colab.
2. Make sure the runtime type is set to GPU (optional, for CNN).
-Requirements::
    - !pip install
    - scikit-learn
    - imbalanced-learn
    - tensorflow
    - scikeras
    - matplotlib
    - NumPy & Pandas
3.Run all cells sequentially.

Code Workflow:

Data Cleaning & Preparation:
- Removed duplicates, special symbols, and short tweets.
- Added a custom stopword list (including political terms and slang like “rt”, “plz”).
- Applied lemmatization for consistent word representation.

Feature Extraction:
- CountVectorizer: Counts how many times each word appears.
- TF-IDF Vectorizer: Assigns more weight to unique or meaningful words and downweights common ones.
- Tokenizer + Padding (for CNN): Converts tweets to sequences while preserving word order.

Data Balancing:
- Used RandomOverSampler and RandomUnderSampler to balance positive, neutral, and negative tweet counts.

Model Training & Tuning:
- MultinomialNB, Logistic Regression, and LinearSVC trained using TF-IDF.
- CNN model built using Keras Sequential API with:
        -  Embedding Layer -> Conv1D -> GlobalMaxPooling -> Dropout > Dense
- Performed GridSearchCV / RandomizedSearchCV for hyperparameter tuning and best configuration selection.
- Evaluated using Stratified K-Fold Cross Validation to prevent data leakage.

Evaluation & Visualization:
- Generated Confusion Matrices, ROC Curves, Macro-F1 Scores, and Misclassified Example Analysis.
- Analyzed confidence levels of CNN predictions to understand uncertainty in ambiguous tweets.

RESULT:
-------------------------------------------------------------------------------
| Model               | Feature Extraction | Sampling             | Macro-F1   |
| ------------------- | ------------------ | -------------------- | ---------- |
| MultinomialNB       | TF-IDF             | Random Over Sampling | 0.6937     |
| Logistic Regression | TF-IDF             | Random Over Sampling | 0.8205     |
| Linear SVC          | TF-IDF             | Random Over Sampling | 0.8212     |
| CNN                 | Tokenizer + Pad    | Random Over Sampling | 0.8277     |
--------------------------------------------------------------------------------

Best Model: CNN
Reason: CNN captures complex patterns and word context better than classical models.

Conclusion:
- TF-IDF outperformed CountVectorizer for most models.
- RandomOverSampler improved balance across sentiment classes.
- CNN achieved the best overall performance (Macro-F1 = 0.8277).












