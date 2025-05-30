import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:/School Files/Python Model with API Integration/Integration/cleaned_depression_dataset.csv")

# Split into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize model
model = LogisticRegression(max_iter=1000, solver='liblinear')

# RFECV for automatic feature selection with cross-validation
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy'  # You can change to 'f1', 'roc_auc', etc.
)

# Fit RFECV
rfecv.fit(X, y_encoded)

# Get selected features
selected_features = X.columns[rfecv.support_]

# Output results
print("✅ Optimal number of features:", rfecv.n_features_)
print("\n📌 Selected Features:")
print(selected_features.tolist())

# Plot feature selection results (fixed for modern sklearn)
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (accuracy)")
plt.plot(
    range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
    rfecv.cv_results_['mean_test_score'],
    marker='o'
)
plt.title("Feature Selection using RFECV")
plt.grid(True)
plt.tight_layout()
plt.show()

