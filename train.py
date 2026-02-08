"""
Training script executed by GitHub Actions
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

print("Starting CI training...")

X, y = make_classification(
    n_samples=200,
    n_features=10,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "ci_model.pkl")

print("Model trained and saved successfully.")
