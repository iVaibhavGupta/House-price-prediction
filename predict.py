import numpy  as np
import pandas as pd
import joblib #for loading trained model

model = joblib.load("house_price_model.pkl")
print("model load sucessfully")

#load new data
new_data = pd.read_csv("X_test.csv")

#make predictions
predictions = model.predict(new_data)

#save the prediction
output = pd.DataFrame(predictions, columns=["Predict Price"])
output.to_csv("house_price_predictions.csv", index = False)

#output someof the predictions
print("\nOutput some Sample Predictions")
print(output.head())
