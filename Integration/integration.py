import pandas as pd

depression_survey1 = pd.read_csv("C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/Outlier Detection/Data Transformation/cleaned_dataset1.csv")
depression_survey2 = pd.read_csv("C:/School Files/Python Model with API Integration/Data Processing/Missing Data Treatment/Outlier Detection/Data Transformation/cleaned_dataset2.csv")


final_data = pd.merge(depression_survey2, depression_survey1, on='id', how='inner')



final_data.to_csv("C:/School Files/Python Model with API Integration/Integration/cleaned_depression_dataset.csv", index=False)