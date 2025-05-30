import pandas as pd

# === Step 1: Load Data ===
df_survey1 = pd.read_csv('depression_survey1_uncleaned.csv')
df_survey2 = pd.read_csv('depression_survey2_uncleaned.csv')  # Update path if needed

# Count duplicate rows
df_survey1_duplicate_count = df_survey1.duplicated().sum()
df_survey2_duplicate_count = df_survey2.duplicated().sum()

print("Duplicate Count in dataset1: ") 
print(df_survey1_duplicate_count)
print("Duplicate Count in dataset2: " ) 
print(df_survey2_duplicate_count)

# Display duplicate rows
duplicate_rows1 = df_survey1[df_survey1.duplicated()]
duplicate_rows2 = df_survey2[df_survey2.duplicated()]


# Remove duplicates
df_no_duplicates1 = df_survey1.drop_duplicates()
df_no_duplicates2 = df_survey2.drop_duplicates()

# Check again for duplicates
duplicate_rows1 = df_survey1[df_survey1.duplicated()]
duplicate_rows2 = df_survey2[df_survey2.duplicated()]


# Count missing values in survey 1
missing_values_before = df_no_duplicates1.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)

# Count missing values per column before filling
missing_values_before = df_no_duplicates2.isnull().sum()
print("Missing Values per Column (Before Filling):")
print(missing_values_before)


df_no_duplicates1.to_csv('depression_without_duplicates1.csv', index=False)
df_no_duplicates2.to_csv('depression_without_duplicates2.csv', index=False)

