import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

DATA_PATH = 'cutomer_churn.csv'
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert Churn target to numeric binary values
# and keep the pipeline ready for sklearn
assert 'Churn' in df.columns, 'Churn column missing'
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)

target = 'Churn'
X = df.drop(columns=[target]).copy()
y = df[target].to_numpy()

# Convert object-like categoricals to strings for consistent preprocessing
X = X.reset_index(drop=True)
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype(str)

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='drop')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
param_grid = [
    {
        'clf': [LogisticRegression(max_iter=1000, random_state=42)],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l2']
    },
    {
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0)
print('Fitting GridSearchCV...')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print('Best CV ROC AUC:', grid.best_score_)
print('Best params:', grid.best_params_)

preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
print('Classification report:')
print(classification_report(y_test, preds))
print('Confusion matrix:')
print(confusion_matrix(y_test, preds))
if probs is not None:
    print('Test ROC AUC:', roc_auc_score(y_test, probs))

OUTPUT_PATH = 'best_churn_pipeline.joblib'
joblib.dump(best_model, OUTPUT_PATH)
print('Saved pipeline to', OUTPUT_PATH)
