import pandas as pd

# Load dataset
df = pd.read_csv("C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/Outlier Detection/dataset2_without_outliers.csv")

# Remove quotation marks from 'id' column
df['id'] = df['id'].astype(str).str.replace('"', '', regex=False)

# Drop the 'Outlier_IQR' column if it exists
cols_to_drop = ['Outlier_IQR', 'Age', 'Gender']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

if existing_cols_to_drop:
    df.drop(columns=existing_cols_to_drop, inplace=True)

# Display updated dataframe info
print(df.info())
print(df.head())

df2 = pd.read_csv("C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/Outlier Detection/dataset1_without_outliers.csv")

# Drop the 'Outlier_IQR' column if it exists
if 'Outlier_IQR' in df2.columns:
    df2.drop(columns=['Outlier_IQR'], inplace=True)
    
    # Round the 'study_satisfaction' column to the nearest integer
df2['Study Satisfaction'] = df2['Study Satisfaction'].round(0).astype(int)

# Display updated dataframe info
print(df2.info())
print(df2.head())


df.to_csv('C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/Outlier Detection/Data Transformation/cleaned_dataset1.csv', index=False)

df2.to_csv('C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/Outlier Detection/Data Transformation/cleaned_dataset2.csv', index=False)


