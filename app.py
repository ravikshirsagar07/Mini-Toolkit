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
st.write(
    "Sentiment Analysis | Question Answering | NER | Summarization"
)

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    qa_model = pipeline(
        task="question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )

    ner_model = pipeline(
        "ner",
        aggregation_strategy="simple"
    )

    return sentiment_model, qa_model, ner_model


sentiment_model, qa_model, ner_model = load_models()

# =====================================
# SUMMARIZATION FUNCTION
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
    "Enter Question"
)

# =====================================
# RUN BUTTON
# =====================================

if st.button("Analyze"):

    if paragraph.strip() == "":
        st.warning("Please enter a paragraph.")
        st.stop()

    # =====================================
    # SENTIMENT ANALYSIS
    # =====================================

    st.subheader("😊 Sentiment Analysis")

    sentiment_result = sentiment_model(paragraph)

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

        answer = qa_model(
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
    # NER
    # =====================================

    st.subheader("🏷 Named Entity Recognition")

    entities = ner_model(paragraph)

    if entities:

        for entity in entities:

            st.write(
                f"Entity: {entity['word']} | Type: {entity['entity_group']}"
            )

    else:
        st.warning("No entities found.")

    # =====================================
    # SUMMARY
    # =====================================

    st.subheader("📄 Summary")

    summary = summarize(paragraph)

    st.write(summary)

    st.success("Analysis Completed Successfully!")
