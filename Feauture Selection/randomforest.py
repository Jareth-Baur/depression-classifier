import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway

# Load dataset
data = pd.read_csv("C:/School Files/Python Model with API Integration/Integration/cleaned_depression_dataset.csv")

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Drop 'id' if it exists (not a true feature)
if 'id' in X.columns:
    X = X.drop(columns=['id'])

# Encode target if needed
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------
# 1. ANOVA Feature Selection
# -------------------------
anova_features = []
for col in X.columns:
    groups = [X[col][y_encoded == label] for label in np.unique(y_encoded)]
    try:
        f_stat, p_val = f_oneway(*groups)
        if p_val < 0.05:
            anova_features.append(col)
    except:
        continue  # skip columns that can't be tested (e.g., all same values in group)

# -------------------------
# 2. RFECV Recursive Feature Selection
# -------------------------
model = LogisticRegression(max_iter=1000, solver='liblinear')
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y_encoded)
rfecv_features = X.columns[rfecv.support_].tolist()

# -------------------------
# 3. Random Forest Importance
# -------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_encoded)
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
rf_features = rf_importances[rf_importances['Importance'] > 0.01]['Feature'].tolist()

# -------------------------
# 4. Combine all selections
# -------------------------
# Count appearances
all_selected = pd.Series(anova_features + rfecv_features + rf_features)
final_features = all_selected.value_counts()
final_selected_features = final_features[final_features >= 2].index.tolist()  # appear in at least 2

# Show results
print("✅ Final Selected Features (in at least 2 methods):")
print(final_selected_features)

print("\n❌ Features to Drop:")
features_to_drop = [col for col in X.columns if col not in final_selected_features]
print(features_to_drop)

# Optional: create a cleaned dataset
cleaned_data = X[final_selected_features].copy()
cleaned_data['target/Depression'] = y_encoded
cleaned_data.to_csv("C:/School Files/Python Model with API Integration/Feauture Selection/final_selected_features.csv", index=False)
