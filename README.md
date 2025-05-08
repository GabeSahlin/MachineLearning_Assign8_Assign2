### MachineLearning_Assignment8

CarPricesANN is an ANN created to predit the prices of cars given a dataset of features such as the car brand, model, year, engine size, etc.
It utilizes two hidden layers going from 48 (features) to 64 weights, through a ReLU activation, then down from 64 to 32 values, through another 
ReLU activation, and finally down from 32 to 1 value, or the predicted car price. I also normalized the data when I prepped it, as without it the 
MSE became quite inaccurate by tens of thousands of dollars.

An example output of the ANN is :
Epoch 25 / 100 Average loss: 0.007929
Epoch 50 / 100 Average loss: 0.000771
Epoch 75 / 100 Average loss: 0.000397
Epoch 100 / 100 Average loss: 0.000265
MSE (fully trained): 5.098549081594683e-05
Unnormalized MSE: 493.96

In which case as the prices of cars in the dataset range roughly from 2000 to 18301, this is a fairly accurate model.

CarPricesReg uses linear regression from scikit-learn to complete a similar task. I first cleaned my data slightly to 
prevent considering outliers such as old and/or possibly damaged cars, which might have a skewed pricing compared to a "normal" car. 
I computed the featured correlation of each of the features with the label (Pricing), and found that, surprisingly, the number of previous 
owners seemed to have little to no impact on the predicted pricing of the model. As a result, after computing the regression the MSE was roughly 1095.82.

Example output :

Feature correlation:
Year           0.646377
Mileage       -0.308777
Owner_Count    0.017468
Engine_Size    0.301200
dtype: float64
Scoring:
R^2 score : 0.7763212713512162
MSE score : 1095.8211960353788


Overall, it seemed that with the normalization and layers selected, the ANN did better than sklearn's linear regression 
(with the cost of programming an ANN).
