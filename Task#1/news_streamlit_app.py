import streamlit as st 
from transformers import pipeline

MODEL_DIR = "./news_classifier_model"

st.title("News Headline Topic Classifier")
st.write("Fine-tuned BERT model to classify AG News headlines into topic categories.")

st.markdown(
    """
    Enter a news headline below and the app will predict the topic category using the saved Transformer model.
    """
)

headline = st.text_area("News headline", value="Stocks rallied after the earnings report.", height=120)

if st.button("Predict"):
    with st.spinner("Loading model and predicting..."):
        classifier = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            return_all_scores=True,
        )
        results = classifier(headline)[0]
        predictions = {
            item["label"]: item["score"] for item in results
        }

        st.subheader("Predicted topic scores")
        for label, score in predictions.items():
            st.write(f"**{label}**: {score:.4f}")

        best_label = max(predictions, key=predictions.get)
        st.success(f"Top predicted category: {best_label}")

st.markdown("---")
st.write("Model directory:", MODEL_DIR)
