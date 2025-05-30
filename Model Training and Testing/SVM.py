import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/School Files/Python Model with API Integration/Model Training and Testing/depression_dataset_with_smote.csv")  # Replace with your actual file path

# Features and target
features = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration", "Dietary Habits",
            "Suicidal thoughts ?", "Work/Study Hours", "Financial Stress", "Family History of Mental Illness",
            "City", "Profession", "Degree"]
X = df[features]
y = df["target/Depression"]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Encode target variable
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')  # You can try 'linear' or 'poly' kernels too
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Evaluation
print("📊 SVM Evaluation Metrics:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()
