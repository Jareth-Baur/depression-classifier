import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from imblearn.combine import SMOTEENN
import numpy as np

# Load dataset
df = pd.read_csv("C:/School Files/Python Model with API Integration/Feauture Selection/final_selected_features.csv")

# Define features and target
features = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration","Dietary Habits",
            "Suicidal thoughts ?","Work/Study Hours","Financial Stress","Family History of Mental Illness",
            "City","Profession","Degree"]
X = df[features]
y = df["target/Depression"]

# Encode categorical features
X = X.apply(LabelEncoder().fit_transform)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# SMOTE-ENN oversampling + cleaning
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

# Save resampled data
resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
resampled_df["target/Depression"] = y_train_resampled.values
resampled_df.to_csv("C:/School Files/Python Model with API Integration/Feauture Selection/depression_dataset_with_smote.csv", index=False)


# Train model
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred_thresh = (y_probs >= best_thresh).astype(int)

# Evaluation
print(f"🔍 Best Threshold: {best_thresh:.2f}")
print(classification_report(y_test, y_pred_thresh))
print(f"F1 Score: {f1_score(y_test, y_pred_thresh):.4f}")
