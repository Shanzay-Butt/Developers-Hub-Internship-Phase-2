# app.py
# Auto Tagging Support Tickets Using LLM (Zero-shot + Few-shot)

import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("customer_support_tickets.csv")

# Keep only required columns
df = df[["ticket_text", "true_tag"]]

# Clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["ticket_text"] = df["ticket_text"].str.lower().str.strip()

# Use small sample for fast execution (important for assignment)
df = df.sample(min(200, len(df)), random_state=42)

# -----------------------------
# Define Labels
# -----------------------------
labels = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Refund Request",
    "General Inquiry"
]

# -----------------------------
# Load Zero-Shot Model
# -----------------------------
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# -----------------------------
# Few-shot examples
# -----------------------------
few_shot_examples = """
Ticket: I cannot log into my account
Category: Account Access

Ticket: My payment was charged twice
Category: Billing

Ticket: The app is crashing again and again
Category: Technical Issue

Ticket: I want a refund for my order
Category: Refund Request
"""

# -----------------------------
# Prediction Functions
# -----------------------------
def zero_shot_predict(text):
    result = classifier(text, labels)
    return result["labels"][0], result["scores"]

def few_shot_predict(text):
    prompt = few_shot_examples + f"\nTicket: {text}\nCategory:"
    result = classifier(prompt, labels)
    return result["labels"][0], result["scores"]

def top_3_tags(text):
    result = classifier(text, labels)
    top3 = list(zip(result["labels"][:3], result["scores"][:3]))
    return top3

# -----------------------------
# Run Predictions
# -----------------------------
zero_preds = []
few_preds = []
top3_list = []

for text in df["ticket_text"]:
    z_pred, _ = zero_shot_predict(text)
    f_pred, _ = few_shot_predict(text)
    top3 = top_3_tags(text)

    zero_preds.append(z_pred)
    few_preds.append(f_pred)
    top3_list.append(top3)

df["zero_shot_pred"] = zero_preds
df["few_shot_pred"] = few_preds
df["top_3_tags"] = top3_list

# -----------------------------
# Evaluation
# -----------------------------
print("\n📊 Zero-Shot Accuracy:")
print(accuracy_score(df["true_tag"], df["zero_shot_pred"]))

print("\n📊 Few-Shot Accuracy:")
print(accuracy_score(df["true_tag"], df["few_shot_pred"]))

print("\n📄 Classification Report (Zero-shot):")
print(classification_report(df["true_tag"], df["zero_shot_pred"]))

# -----------------------------
# Save Results
# -----------------------------
df.to_csv("results.csv", index=False)

print("\n✅ results.csv saved successfully!")