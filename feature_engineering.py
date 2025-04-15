import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cleaned_house_data.csv')

#select important feature
important_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
    'FullBath', 'YearBuilt', 'SalePrice'
]

df =df[important_features]

#spliting data into features (X) and Target Variable(Y)
X = df.drop(columns=['SalePrice'])  #feature
y = df['SalePrice'] #target variable

#train  and test split (80%,20%)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#scale numerical feature
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#convert scaled data back to data frames
X_train  = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

#save processed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)


print("Feature Engineering Complete! Processed Saved data")