import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocessing import clean_text


DATASET_PATH = "questions_dataset.csv"
OUTPUT_FOLDER = "outputs"


def load_and_prepare_data(dataset_path=DATASET_PATH):
    """
    Loads the dataset and applies the same preprocessing
    steps used in the classification module.
    """

    data = pd.read_csv(dataset_path)
    data["clean_question"] = data["question"].apply(clean_text)

    return data


def save_dataset_distribution(data):
    """
    Creates and saves a bar chart showing
    the number of questions in each topic category.
    """

    category_counts = data["category"].value_counts().sort_index()

    plt.figure(figsize=(11, 6))
    category_counts.plot(kind="bar")

    plt.title("Number of Questions per Topic Category")
    plt.xlabel("Topic Category")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, "dataset_category_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_and_evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test,
    ngram_range,
    model_name
):
    """
    Trains and evaluates one TF-IDF + Naive Bayes model.

    Returns:
        Dictionary containing the model name and accuracy score.
    """

    # TF-IDF converts cleaned text into numerical features.
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Multinomial Naive Bayes is used for topic classification.
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    # Save classification report as a text file
    report_file_name = model_name.lower().replace(" ", "_").replace("+", "plus")
    report_path = os.path.join(
        OUTPUT_FOLDER,
        f"{report_file_name}_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"{model_name}\n")
        file.write("=" * 70 + "\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report:\n\n")
        file.write(report)

    # Create and save confusion matrix
    labels = sorted(y_test.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        OUTPUT_FOLDER,
        f"{report_file_name}_confusion_matrix.png"
    )

    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4)
    }


def save_accuracy_comparison(results_df):
    """
    Creates and saves a bar chart comparing
    the accuracy scores of tested models.
    """

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    # Write the exact accuracy value above each bar
    # to make the comparison easier to read in the presentation.
    for bar, accuracy in zip(bars, results_df["Accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.015,
            f"{accuracy:.4f}",
            ha="center",
            va="bottom"
        )

    plt.title("Accuracy Comparison of Feature Settings")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_FOLDER, "accuracy_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_evaluation():
    """
    Runs the full evaluation process:
    - loads and preprocesses the dataset
    - splits data into training and test sets
    - evaluates unigram and unigram-bigram settings
    - saves visualizations and summary results
    """

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    data = load_and_prepare_data()

    save_dataset_distribution(data)

    X = data["clean_question"]
    y = data["category"]

    # Stratified splitting keeps category distribution balanced
    # across the training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    results = []

    unigram_result = train_and_evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test,
        ngram_range=(1, 1),
        model_name="Unigram TF-IDF"
    )

    results.append(unigram_result)

    unigram_bigram_result = train_and_evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test,
        ngram_range=(1, 2),
        model_name="Unigram + Bigram TF-IDF"
    )

    results.append(unigram_bigram_result)

    results_df = pd.DataFrame(results)

    results_csv_path = os.path.join(
        OUTPUT_FOLDER,
        "model_accuracy_comparison.csv"
    )

    results_df.to_csv(results_csv_path, index=False)

    save_accuracy_comparison(results_df)

    print("\n" + "=" * 70)
    print("MODEL ACCURACY COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print("\nAll evaluation outputs were saved in the 'outputs' folder.")


if __name__ == "__main__":
    run_evaluation()