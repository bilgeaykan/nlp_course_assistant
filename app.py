import streamlit as st

from classifier import train_topic_classifier, predict_topic
from preprocessing import show_preprocessing_steps
from rag_module import retrieve_relevant_chunks, generate_context_based_answer

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
    retrieved_chunks = retrieve_relevant_chunks(
        question,
        predicted_topic,
        top_k=3
    )

    answer = generate_context_based_answer(
        question,
        retrieved_chunks
    )

    st.markdown("### Retrieved Context")

    if retrieved_chunks:
        for chunk in retrieved_chunks:
            st.write(f"**Source:** {chunk['source']}")
            st.write(f"**Similarity Score:** {chunk['score']}")
            st.write(chunk["text"])
    else:
        st.warning("No relevant context was found.")

    st.markdown("### Context-Based Answer")
    st.info(answer)
    with st.expander("Show preprocessing steps"):
        preprocessing_steps = show_preprocessing_steps(question)

        for step_name, step_value in preprocessing_steps.items():
            st.markdown(f"**{step_name}:**")
            st.write(step_value)
else:
    st.info("Please enter a question to see the predicted NLP topic.")