import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("car_price_dataset.csv")


#Filter out  cars less than $6646 as that is the bottom 25% of cars, this prevents considering outliers of old or damaged cars.
#Filter out  cars with more than 200,000 miles as that prevents outliers that break down and have been heavily used.

#This brings df from 10,000 to roughly 5,800
reviewed = df.query("Price > 6646 and Mileage < 200000")

# Compute correlation of all int features with Price
feature_correlation = reviewed[["Year", "Mileage", "Owner_Count", "Engine_Size"]].corrwith(reviewed["Price"])

# Display the correlation values
print(feature_correlation)

#Year           0.646377 : correlates the most to pricing of vehicles (newer cars = higher price)
#Mileage       -0.308777 : correlates 30% to pricing (lower mileage = higher price)
#Owner_Count    0.017468 : correlates minimally to pricing (# of owners does not largely affect pricing)
#Engine_Size    0.301200 : correlates 30% to pricing (larger engine size = higher pricing)

#features
#X = reviewed[["Year", "Mileage", "Owner_Count", "Engine_Size"]]
X = reviewed[["Year", "Mileage", "Engine_Size"]]

#label
y = reviewed["Price"]

reg = LinearRegression().fit(X, y)

print("--------Scikitlearn---------")
print(f"R^2 score : {reg.score(X, y)}") #R^2 score = .776
print(f"MSE score : {np.sqrt(np.average((y - reg.predict(X))**2.0))}") # RMSE = on average the regression is off by 1095.96 : as 2000 < Price < 18301, I consider this a reasonable error
print(reg.coef_)
