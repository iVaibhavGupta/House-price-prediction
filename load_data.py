import pandas as pd
import numpy as np

df = pd.read_csv("house_data.csv")

print("check for the missing values in eaach columns:")
print(df.isnull().sum())

# ("fill the missing value") only for numerical value
numeric_cols= df.select_dtypes(include = ['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

#check for categorical data if presnt
df = pd.get_dummies(df,drop_first=True)

print("Cleaned dataset Preview")
print(df.head())

df.to_csv("cleaned_house_data.csv", index=False)
print("cleaned data set is saved as 'cleaned_house_data_csv'")

