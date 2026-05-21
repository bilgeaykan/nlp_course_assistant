import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocessing import clean_text


def load_dataset(dataset_path="questions_dataset.csv"):
    """
    Loads the question dataset and applies text preprocessing.

    Returns:
        DataFrame with original questions, category labels,
        and cleaned question texts.
    """

    data = pd.read_csv(dataset_path)

    # The model will be trained on the cleaned version of each question.
    data["clean_question"] = data["question"].apply(clean_text)

    return data


def train_topic_classifier(dataset_path="questions_dataset.csv", ngram_range=(1, 2)):
    """
    Trains a topic classification model using
    TF-IDF features and Multinomial Naive Bayes.

    Returns:
        Trained model and fitted TF-IDF vectorizer.
    """

    data = load_dataset(dataset_path)

    X = data["clean_question"]
    y = data["category"]

    # TF-IDF converts text into numerical vectors.
    # ngram_range=(1, 2) allows the model to use both
    # single words and two-word expressions.
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range
    )

    X_vectorized = vectorizer.fit_transform(X)

    # Multinomial Naive Bayes is a simple and effective
    # classification method for text-based features.
    model = MultinomialNB()
    model.fit(X_vectorized, y)

    return model, vectorizer


def predict_topic(question, model, vectorizer):
    """
    Predicts the NLP topic category of a new user question.

    Returns:
        Predicted topic label.
    """

    # The new question must pass through the same preprocessing
    # and TF-IDF transformation used during training.
    cleaned_question = clean_text(question)
    question_vector = vectorizer.transform([cleaned_question])

    predicted_topic = model.predict(question_vector)[0]

    return predicted_topic