import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_house_data.csv")

#check of dataset shape and columns type
print("\nprint Dataset Shape:", df.shape)
print("\nData set Info:")
print(df.info())

print("\n stastics summary")
print(df.describe())

print("\n check for null values")
print(df.isnull().sum().sum())  #summ of all missing values

#visulation price destribution
plt.figure(figsize=(8,3))
#sns.histplot(df['SalePrice'], bin= 30, kde = True)
sns.histplot(df['SalePrice'], bins=30, kde=True)
plt.title("Destribution of House Price")
plt.xlabel("Frequency")
plt.ylabel("Price")
plt.show()

#check for corelation matrix top 10 corelation freatures
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot = False, cmap = 'coolwarm')
plt.title("Feature corelation HeatMap")
plt.show()

#check relationship b/w living area & price
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'])
plt.title("Living are v/s SalePrice")
plt.xlabel("Above Ground Living Area (sq.ft)")
plt.ylabel("Sale Price")
plt.show()

