import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("C:/School Files/Python Model with API Integration/Model Training and Testing/depression_dataset_with_smote.csv")

# Features and target
features = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration", "Dietary Habits",
            "Suicidal thoughts", "Work/Study Hours", "Financial Stress", "Family History of Mental Illness",
            "City", "Profession", "Degree"]
X = df[features]
y = df["target/Depression"]

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Stacking Classifier (RF + LR + NB)
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('nb', GaussianNB())
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42)
)

stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

print("=== Stacking Classifier Performance ===")
print(classification_report(y_test, y_pred_stack))

print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_stack):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_stack):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_stack):.4f}")


# =========================================================================================
import joblib

joblib.dump(stacking_clf, 'C:/School Files/Python Model with API Integration/stacking_classifier_model.pkl')
