import os
import pandas as pd
import joblib

MODEL_PATH = 'best_churn_pipeline.joblib'
DATA_PATH = 'cutomer_churn.csv'

assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
assert os.path.exists(DATA_PATH), f"Dataset not found: {DATA_PATH}"

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['Churn']).copy()
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype(str)

predictions = pipeline.predict(X)
proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, 'predict_proba') else None
print('First 10 predictions:')
print(predictions[:10])
if proba is not None:
    print('First 10 probabilities:')
    print(proba[:10])
