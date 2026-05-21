import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENT_FOLDER = "pdfs"


def read_text_file(file_path):
    """
    Reads a text file and returns its content.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    """
    Splits long text into smaller overlapping chunks.
    This helps the retrieval system search inside smaller text parts.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def load_documents(document_folder=DOCUMENT_FOLDER):
    """
    Loads all text documents and stores each chunk with metadata.
    Metadata helps us show the topic and source file in the interface.
    """

    documents = []

    for file_name in os.listdir(document_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(document_folder, file_name)
            text = read_text_file(file_path)

            topic = file_name.replace(".txt", "").upper()
            chunks = split_text_into_chunks(text)

            for chunk in chunks:
                documents.append({
                    "topic": topic,
                    "source": file_name,
                    "text": chunk
                })

    return documents


def build_retrieval_index():
    """
    Creates the retrieval index by converting document chunks
    into sentence embeddings.
    """

    documents = load_documents()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in documents]
    embeddings = embedding_model.encode(texts)

    return documents, embeddings, embedding_model


def retrieve_relevant_chunks(question, predicted_topic=None, top_k=3):
    """
    Retrieves the most relevant chunks for a user question.
    If a predicted topic is available, the search is first focused
    on documents related to that topic.
    """

    documents, embeddings, embedding_model = build_retrieval_index()

    if predicted_topic:
        filtered_indices = [
            index for index, doc in enumerate(documents)
            if doc["topic"].lower() == predicted_topic.lower().replace("-", "")
        ]

        if filtered_indices:
            filtered_documents = [documents[i] for i in filtered_indices]
            filtered_embeddings = [embeddings[i] for i in filtered_indices]
        else:
            filtered_documents = documents
            filtered_embeddings = embeddings
    else:
        filtered_documents = documents
        filtered_embeddings = embeddings

    question_embedding = embedding_model.encode([question])

    similarity_scores = cosine_similarity(
        question_embedding,
        filtered_embeddings
    )[0]

    ranked_indices = similarity_scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = filtered_documents[index].copy()
        result["score"] = round(float(similarity_scores[index]), 4)
        results.append(result)

    return results


def generate_context_based_answer(question, retrieved_chunks):
    """
    Creates a short answer based on the retrieved context.
    This is a retrieval-based answer, not a full generative LLM system.
    """

    if not retrieved_chunks:
        return "No relevant context was found."

    best_context = retrieved_chunks[0]["text"]

    answer = (
        "Based on the retrieved course material, "
        "this topic can be explained as follows: "
        + best_context
    )

    return answer