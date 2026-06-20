# 📰 News Topic Classifier Using BERT

## 📌 Project Overview
This project builds a machine learning model that can automatically classify news headlines into different categories using a transformer model (BERT).  
We fine-tune a pre-trained model so it can understand and predict the topic of a news headline.

---

## 🎯 Objective
- Classify news headlines into topics like:
  - World
  - Sports
  - Business
  - Science/Technology
- Use a transformer model (BERT) for better accuracy
- Evaluate performance using accuracy and F1-score
- Build a simple web app for live predictions

---

## 📊 Dataset
We use the **AG News Dataset** from Hugging Face:

- Contains news headlines and their labels
- 4 main categories:
  - World
  - Sports
  - Business
  - Science/Technology

---

## 🧠 Model Used
- `bert-base-uncased` (pre-trained transformer model)
- Fine-tuned using Hugging Face Transformers library

---

## ⚙️ Workflow

1. Load dataset from Hugging Face
2. Tokenize text using BERT tokenizer
3. Fine-tune BERT model on training data
4. Evaluate model using:
   - Accuracy
   - F1-score
5. Build a simple web app for predictions

---

## 📈 Evaluation Metrics
We measure model performance using:
- **Accuracy** → How many predictions are correct
- **F1-score** → Balance between precision and recall

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install transformers datasets torch scikit-learn streamlit
