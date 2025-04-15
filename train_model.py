import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib #for saving the model

#load processed data
X_train =  pd.read_csv("X_train.csv")
X_test= pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel() #convert it into 1D array
y_test = pd.read_csv("y_test.csv").values.ravel()

#train the model
model = LinearRegression()
model.fit(X_train, y_train)

#model prediction
y_pred = model.predict(X_test)

#evaluate model porfermance
mae =  mean_absolute_error(y_test, y_pred)
mse =  mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,  y_pred)

print(" Model Training Complete!")
print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f" R² Score: {r2:.2f}")

#save the trained model
joblib.dump(model,'house_price_model.pkl')
print("model save AS house_price_mode.pkl")
