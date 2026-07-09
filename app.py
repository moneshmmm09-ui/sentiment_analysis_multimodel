import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analysis GenAI Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sentiment Analysis GenAI Bot")
st.write("Compare sentiment predictions from multiple Hugging Face models.")

# Cache models so they load only once
@st.cache_resource
def load_models():

    models = {
        "DistilBERT":
            pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            ),

        "RoBERTa":
            pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            ),

        "SieBERT":
            pipeline(
                "sentiment-analysis",
                model="siebert/sentiment-roberta-large-english"
            ),

        "BERT Multilingual":
            pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
    }

    return models


models = load_models()

review = st.text_area(
    "Enter Review",
    placeholder="Example: The product quality is amazing!",
    height=180
)

selected_models = st.multiselect(
    "Select Models",
    list(models.keys()),
    default=list(models.keys())
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    for model_name in selected_models:

        st.subheader(model_name)

        try:

            result = models[model_name](review)

            label = result[0]["label"]
            score = result[0]["score"]

            st.success(f"Sentiment: {label}")
            st.info(f"Confidence: {score:.2%}")

        except Exception as e:

            st.error(f"Error: {e}")
