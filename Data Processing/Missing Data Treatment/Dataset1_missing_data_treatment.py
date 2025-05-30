import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/School Files/Python Model with API Integration/Data Processing/depression_without_duplicates1.csv')


# Count missing values per column before filling
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

#Fill in City using Mode
df['City'].fillna(df['City'].mode().iloc[0], inplace=True)

# Check again for missing values for City
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

#Fill in Work Pressure using Forward Fill
df['Work Pressure'] = df['Work Pressure'].fillna(method='ffill')

# Check again for missing values for Work Pressure
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

#Regression

#Convert categorical values to numerical values
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['City'] = encoder.fit_transform(df['City'])
df['Profession'] = encoder.fit_transform(df['Profession'])

# Select features for prediction / tanang 0 ang value
features = ['Gender', 'Age', 'City', 'Profession', 
            'Academic Pressure', 'Work Pressure', 'CGPA', 'Job Satisfaction']

target = df['Study Satisfaction']

# Separate data into known and missing / diri mailhan nga 0 dapat tanan predictors
df_known = df.dropna(subset=['Study Satisfaction']) 
df_missing = df[df['Study Satisfaction'].isnull()]

# Define X (features) and y (target)
X_train = df_known[features]
y_train = df_known['Study Satisfaction']
X_missing = df_missing[features]

# Train a Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict missing values
predicted_values = model.predict(X_missing)

df.loc[df['Study Satisfaction'].isnull(), 'Study Satisfaction'] = predicted_values

# Check again for missing values 
missing_values_before = df.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)


df.to_csv('C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/cleaned_dataset_with_outliers1.csv', index=False)

