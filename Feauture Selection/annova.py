import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("C:/School Files/Python Model with API Integration/Integration/cleaned_depression_dataset.csv")


# Extract numerical features and categorical target
X = data.iloc[:, :-1]  # Features (Sepal/Petal measurements)
y = data.iloc[:, -1]   # Target (Species)

# Encode target variable (if categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Perform One-Way ANOVA for each feature
f_values = []
p_values = []
feature_names = X.columns
significance = []

for feature in feature_names:
    groups = [X[feature][y_encoded == label] for label in np.unique(y_encoded)]
    f_stat, p_value = f_oneway(*groups)
    f_values.append(f_stat)
    p_values.append(p_value)
    
    # Mark feature as Significant or Not Significant
    significance.append("✅ Significant" if p_value < 0.05 else "❌ Not Significant")

# Store results in a DataFrame for easy viewing
anova_results = pd.DataFrame({
    "Feature": feature_names,
    "F-Value": f_values,
    "P-Value": p_values,
    "Significance": significance
})

# Display the results
print(anova_results)

# Show only selected features
selected_features = anova_results[anova_results["P-Value"] < 0.05]["Feature"].tolist()
print("\nSelected Features for Prediction:", selected_features)
