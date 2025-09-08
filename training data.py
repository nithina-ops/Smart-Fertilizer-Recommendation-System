# train_baseline.py
# Baseline Random Forest Model for Smart Fertilizer Recommendation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Step 1: Load dataset
# -------------------------
# Replace this with your actual dataset path
# Example dataset should have features like soil_type, crop_type, pH, N, P, K etc.
data = pd.read_csv("../data/fertilizer_dataset.csv")

# Separate features (X) and target (y)
X = data.drop("fertilizer", axis=1)   # fertilizer = target column
y = data["fertilizer"]

# -------------------------
# Step 2: Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Step 3: Train Random Forest
# -------------------------
model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------
# Step 4: Make predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Step 5: Evaluate performance
# -------------------------
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Step 6: Save the model (optional)
# -------------------------
import joblib
joblib.dump(model, "../models/random_forest_baseline.pkl")
print("Model saved as ../models/random_forest_baseline.pkl")
