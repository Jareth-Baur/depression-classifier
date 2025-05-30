import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/School Files/Python Model with API Integration/Data Processing/depression_without_duplicates2.csv')
df.replace('?', np.nan, inplace=True)

# Count missing values per column before filling
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

#Fill in City using Mode
df['Sleep Duration'].fillna(df['Sleep Duration'].mode().iloc[0], inplace=True)

# Check again for missing values for City
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

#Fill in Work Pressure using Forward Fill
df['Work/Study Hours'] = df['Work/Study Hours'].fillna(method='ffill')

#Fill in Work Pressure using Forward Fill
df['Financial Stress'] = df['Financial Stress'].fillna(method='bfill')

# Check again for missing values for Work Pressure
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

#Regression

#Convert categorical values to numerical values
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Sleep Duration'] = encoder.fit_transform(df['Sleep Duration'])
df['Dietary Habits'] = encoder.fit_transform(df['Dietary Habits'])
df['Degree'] = encoder.fit_transform(df['Degree'])
df['Suicidal thoughts ?'] = encoder.fit_transform(df['Suicidal thoughts ?'])
df['Family History of Mental Illness'] = encoder.fit_transform(df['Family History of Mental Illness'])
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].replace(2, np.nan)

# Select features for prediction / tanang 0 ang value
features = ['Gender', 'Age', 'Sleep Duration', 'Dietary Habits', 
            'Degree', 'Suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 'Depression']

target = df['Family History of Mental Illness']

# Separate data into known and missing / diri mailhan nga 0 dapat tanan predictors
df_known = df.dropna(subset=['Family History of Mental Illness']) 
df_missing = df[df['Family History of Mental Illness'].isnull()]

# Define X (features) and y (target)
X_train = df_known[features]
y_train = df_known['Family History of Mental Illness']
X_missing = df_missing[features]

# Train a Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing values
predicted_values = model.predict(X_missing)

df.loc[df['Family History of Mental Illness'].isnull(), 'Family History of Mental Illness'] = np.round(predicted_values)

# Check again for missing values
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)


df.to_csv('C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/cleaned_dataset_with_outliers2.csv', index=False)

