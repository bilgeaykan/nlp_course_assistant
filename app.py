import streamlit as st

from classifier import train_topic_classifier, predict_topic
from preprocessing import show_preprocessing_steps


st.set_page_config(
    page_title="NLP-Based Smart Course Assistant",
    layout="centered"
)


st.title("NLP-Based Smart Course Assistant")
st.subheader("Topic Classification Module")

st.write(
    """
    This prototype classifies NLP-related student questions into course topics.
    The system uses text preprocessing, TF-IDF feature extraction,
    and a Multinomial Naive Bayes classifier for topic prediction.
    """
)


# Cache the trained model so that it is not retrained
# every time the Streamlit page is refreshed.
@st.cache_resource
def load_classifier():
    return train_topic_classifier(
        dataset_path="questions_dataset.csv",
        ngram_range=(1, 2)
    )


model, vectorizer = load_classifier()


question = st.text_input(
    "Enter an NLP-related question:",
    placeholder="Example: What is Masked Language Modeling?"
)


if question:
    predicted_topic = predict_topic(
        question,
        model,
        vectorizer
    )

    st.markdown("### Predicted Topic")
    st.success(predicted_topic)

    with st.expander("Show preprocessing steps"):
        preprocessing_steps = show_preprocessing_steps(question)

        for step_name, step_value in preprocessing_steps.items():
            st.markdown(f"**{step_name}:**")
            st.write(step_value)
else:
    st.info("Please enter a question to see the predicted NLP topic.")