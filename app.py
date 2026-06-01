import streamlit as st
from transformers import pipeline
import re
from collections import Counter

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Mini AI Toolkit",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini AI Toolkit")
st.write("Sentiment Analysis | Question Answering | Text Generation | Text Summarization")

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    return sentiment_classifier, qa_pipeline, generator

sentiment_classifier, qa_pipeline, generator = load_models()

# =====================================
# TEXT SUMMARIZER
# =====================================

def summarize(text, num_sentences=3):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    words = re.findall(
        r'\w+',
        text.lower()
    )

    freq = Counter(words)

    sentence_scores = {}

    for sentence in sentences:

        sentence_words = re.findall(
            r'\w+',
            sentence.lower()
        )

        score = sum(
            freq[word]
            for word in sentence_words
        )

        sentence_scores[sentence] = score

    top_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    summary = []

    for sentence in sentences:
        if sentence in top_sentences:
            summary.append(sentence)

    return " ".join(summary)

# =====================================
# USER INPUT
# =====================================

paragraph = st.text_area(
    "Enter Paragraph",
    height=250
)

question = st.text_input(
    "Enter Question Based On Paragraph"
)

# =====================================
# PROCESS BUTTON
# =====================================

if st.button("Run AI Toolkit"):

    if paragraph.strip() == "":
        st.warning("Please enter a paragraph.")
        st.stop()

    # =====================================
    # SENTIMENT ANALYSIS
    # =====================================

    st.subheader("😊 Sentiment Analysis")

    sentiment_result = sentiment_classifier(
        paragraph
    )

    st.success(
        f"Sentiment: {sentiment_result[0]['label']}"
    )

    st.write(
        f"Confidence: {sentiment_result[0]['score']:.4f}"
    )

    # =====================================
    # QUESTION ANSWERING
    # =====================================

    if question:

        st.subheader("❓ Question Answering")

        answer = qa_pipeline(
            question=question,
            context=paragraph
        )

        st.write(
            f"Answer: {answer['answer']}"
        )

        st.write(
            f"Confidence: {answer['score']:.4f}"
        )

    # =====================================
    # TEXT GENERATION
    # =====================================

    st.subheader("✍️ Text Generation")

    generated = generator(
        paragraph,
        max_length=120,
        num_return_sequences=1
    )

    st.write(
        generated[0]["generated_text"]
    )

    # =====================================
    # SUMMARY
    # =====================================

    st.subheader("📄 Text Summary")

    summary = summarize(paragraph)

    st.write(summary)

    st.success("Mini AI Toolkit Completed Successfully!")