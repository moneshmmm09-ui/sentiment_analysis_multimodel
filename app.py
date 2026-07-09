import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analysis GenAI Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sentiment Analysis GenAI Bot")
st.write("Compare sentiment using multiple Hugging Face models.")

# -------------------------
# Load Models (cached)
# -------------------------
@st.cache_resource
def load_models():
    return {
        "DistilBERT": pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        ),

        "RoBERTa": pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        ),

        "BERT Multilingual": pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        ),

        # Text-generation model (prompted for sentiment)
        "Mistral 7B": pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2"
        )
    }

models = load_models()

review = st.text_area(
    "Enter Customer Review",
    height=180
)

selected = st.multiselect(
    "Choose Models",
    list(models.keys()),
    default=["DistilBERT", "RoBERTa"]
)

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    for model_name in selected:

        st.subheader(model_name)

        if model_name == "Mistral 7B":

            prompt = f"""
Analyze the sentiment of the following review.

Return:
Sentiment:
Confidence:
Explanation:

Review:
{review}
"""

            output = models[model_name](
                prompt,
                max_new_tokens=120,
                do_sample=False
            )

            st.write(output[0]["generated_text"])

        else:

            result = models[model_name](review)

            st.success(f"Sentiment : {result[0]['label']}")
            st.info(f"Confidence : {result[0]['score']:.4f}")

st.divider()
st.caption("Powered by Hugging Face Transformers")
