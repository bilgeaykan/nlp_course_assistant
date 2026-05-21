import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize


# Download stopwords only if they are not already available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Applies basic text preprocessing:
    - lowercase conversion
    - URL removal
    - number removal
    - punctuation removal
    - tokenization
    - stopword removal

    Returns:
        Cleaned text as a single string.
    """

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = wordpunct_tokenize(text)

    cleaned_tokens = []

    for token in tokens:
        if token not in stop_words and len(token) > 1:
            cleaned_tokens.append(token)

    cleaned_text = " ".join(cleaned_tokens)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def show_preprocessing_steps(text):
    """
    Returns preprocessing steps one by one.
    This function is useful for observing how the input text
    changes during preprocessing.

    Returns:
        Dictionary containing each preprocessing step.
    """

    original_text = str(text)

    lowercase_text = original_text.lower()

    without_urls = re.sub(
        r"http\S+|www\S+",
        "",
        lowercase_text
    )

    without_numbers = re.sub(
        r"\d+",
        "",
        without_urls
    )

    without_punctuation = without_numbers.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = wordpunct_tokenize(without_punctuation)

    tokens_without_stopwords = []

    for token in tokens:
        if token not in stop_words and len(token) > 1:
            tokens_without_stopwords.append(token)

    final_cleaned_text = " ".join(tokens_without_stopwords)
    final_cleaned_text = re.sub(r"\s+", " ", final_cleaned_text).strip()

    return {
        "Original Text": original_text,
        "Lowercase Text": lowercase_text,
        "Without URLs": without_urls,
        "Without Numbers": without_numbers,
        "Without Punctuation": without_punctuation,
        "Tokens": tokens,
        "After Stopword Removal": tokens_without_stopwords,
        "Final Cleaned Text": final_cleaned_text
    }