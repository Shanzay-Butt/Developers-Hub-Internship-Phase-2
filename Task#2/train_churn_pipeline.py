import os
import pandas as pd
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
OUTPUT_PATH = 'best_churn_pipeline.joblib'

assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

# Load data
print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Data cleaning
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

assert 'Churn' in df.columns, 'Churn column missing in dataset'
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)

# Features and target
X = df.drop(columns=['Churn']).copy()
y = df['Churn'].to_numpy()

# Convert object-like categorical columns to strings for consistent preprocessing
X = X.reset_index(drop=True)
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype(str)

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
print('Numeric columns:', numeric_cols)
print('Categorical columns:', cat_cols)

# Build preprocessing pipeline
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

# Hyperparameter search
param_grid = [
    {
        'clf': [LogisticRegression(max_iter=1000, random_state=42)],
        'clf__C': [0.01, 0.1, 1, 10]
    },
    {
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('Starting GridSearchCV...')
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print('Best CV ROC AUC:', grid.best_score_)
print('Best params:', grid.best_params_)

# Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
print('Classification report:')
print(classification_report(y_test, y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
if y_proba is not None:
    print('Test ROC AUC:', roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(best_model, OUTPUT_PATH)
print('Saved best pipeline to', OUTPUT_PATH)
