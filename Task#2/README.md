# Customer Churn Prediction Pipeline

This task contains a full end-to-end customer churn pipeline using scikit-learn.

## Files

- `customer_churn.ipynb` — main notebook with preprocessing, GridSearchCV training, evaluation, and pipeline export.
- `validate_pipeline.py` — helper script to run the full pipeline outside the notebook.
- `best_churn_pipeline.joblib` — exported trained pipeline model.
- `requirements.txt` — Python dependencies.
- `cutomer_churn.csv` — input dataset.

## How to run

From the `Task#2` directory in your activated virtual environment:

```powershell
python validate_pipeline.py
```

This will train the pipeline, evaluate it on a test split, and save `best_churn_pipeline.joblib`.

## Inference

You can use the exported pipeline in Python with:

```python
import joblib
import pandas as pd

pipeline = joblib.load('best_churn_pipeline.joblib')
df = pd.read_csv('cutomer_churn.csv')
X = df.drop(columns=['Churn'])
preds = pipeline.predict(X)
```

