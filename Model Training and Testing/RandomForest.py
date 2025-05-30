import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load data from Excel file
df = pd.read_csv("C:/School Files/Python Model with API Integration/Model Training and Testing/depression_dataset_with_smote.csv")  # Replace with your actual file path


# Features and target
features = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration","Dietary Habits",
            "Suicidal thoughts ?","Work/Study Hours","Financial Stress","Family History of Mental Illness",
            "City","Profession","Degree"]
X = df[features]
y = df["target/Depression"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Display the predicted vs actual values
results_df = pd.DataFrame({
    "Actual": y_test.values if hasattr(y_test, "values") else y_test,
    "Predicted": y_pred
})

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Optional: detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: confusion matrix plot
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
