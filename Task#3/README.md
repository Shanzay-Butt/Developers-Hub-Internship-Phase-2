# 🏷️ Auto Tagging Support Tickets Using LLM

## 📌 Project Overview
This project classifies customer support tickets into categories using a Large Language Model (LLM).  
It compares Zero-Shot and Few-Shot learning without training any model.

---

## 🎯 Objective
- Automatically classify support tickets
- Compare Zero-Shot vs Few-Shot learning
- Predict top 3 most likely categories
- Evaluate model performance using accuracy

---

## 📊 Dataset
File used:  
`customer_support_tickets.csv`

Columns:
- ticket_text → customer complaint text
- true_tag → actual category label

Categories:
- Billing
- Technical Issue
- Account Access
- Refund Request
- General Inquiry

---

## 🧠 Model Used
- facebook/bart-large-mnli (Hugging Face)
- Used for Zero-Shot classification

---

## ⚙️ Workflow

1. Load dataset using pandas
2. Clean text (remove duplicates, missing values)
3. Apply Zero-Shot classification
4. Apply Few-Shot classification (with examples)
5. Generate Top 3 predictions
6. Evaluate results using accuracy
7. Save results to CSV

---

## 🚀 How to Run

### 1. Install dependencies
```bash id="install5"
pip install -r requirements.txt
