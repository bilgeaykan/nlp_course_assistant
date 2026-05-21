\# NLP-Based Smart Course Assistant



\## Project Overview



This project is an NLP-based smart course assistant developed for educational purposes.



The system classifies NLP-related student questions into course topics and retrieves relevant course material using a simple Retrieval-Augmented Generation (RAG) approach.



The project combines:



\- Text preprocessing

\- TF-IDF feature extraction

\- Multinomial Naive Bayes classification

\- Semantic retrieval with embeddings

\- Cosine similarity search

\- Context-based answer generation

\- Streamlit user interface



\---



\## Features



\- NLP question classification

\- Text preprocessing visualization

\- TF-IDF vectorization

\- Topic prediction using Naive Bayes

\- Retrieval of relevant course material

\- Embedding-based semantic search

\- Context-based answer generation

\- Interactive Streamlit interface



\---



\## Technologies Used



\- Python

\- Streamlit

\- Scikit-learn

\- NLTK

\- SentenceTransformers

\- Pandas

\- Matplotlib



\---



\## Project Pipeline



The system works with the following pipeline:



1\. User enters an NLP-related question

2\. Text preprocessing is applied

3\. TF-IDF features are extracted

4\. Naive Bayes predicts the topic

5\. Relevant documents are retrieved

6\. Sentence embeddings are generated

7\. Cosine similarity is calculated

8\. Most relevant context is selected

9\. Context-based answer is generated



\---



\## Dataset Topics



The dataset contains NLP-related questions from topics such as:



\- TF-IDF

\- BERT

\- RAG

\- Word2Vec

\- Naive Bayes

\- N-Gram

\- Text Preprocessing

\- BPE



\---



\## Example Questions



\- What is Masked Language Modeling?

\- What is inverse document frequency?

\- What is a bigram?

\- How does retrieval work in RAG?



\---



\## How to Run



\### 1. Create virtual environment



```bash

python -m venv .venv

```



\### 2. Activate virtual environment



Windows:



```bash

.venv\\Scripts\\activate

```



\### 3. Install requirements



```bash

pip install -r requirements.txt

```



\### 4. Run Streamlit application



```bash

streamlit run app.py

```



\---



\## Project Structure



```text

app.py

classifier.py

preprocessing.py

rag\_module.py

evaluation.py

questions\_dataset.csv

pdfs/

outputs/

requirements.txt

```



\---



\## Evaluation



The classification system was evaluated using accuracy score, classification report, and confusion matrix visualization.



The project achieved high classification accuracy using TF-IDF and Multinomial Naive Bayes.



\---



\## Authors



\- Student 1 — Topic Classification Module

\- Student 2 — Retrieval and RAG Module

