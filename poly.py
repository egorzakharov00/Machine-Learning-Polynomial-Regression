# Example of polynomial regression in python simulating throwing a ball
# Code from polynomial-regression.py is used

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[0.32], [1.19], [2.3], [3.57], [4.2], [5.12], [6.05],
           [7.21], [8.73], [11.464], [12.7], [13.91], [14.535], [15]]  # distance in metres
y_train = [[4.698], [16.343], [29.21], [40.805], [45.36], [50.586], [56.166],
           [54.148], [54.737], [40.537], [29.21], [15.162], [6.759], [0]]  # height in metres

# Testing set
x_test = [[0], [1.79], [2.22], [6], [10], [14.89]]  # distance in metres
y_test = [[1],  [23.646], [28.372], [45], [44], [1.638]]  # height in metres

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 60, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_featurizer.fit_transform(x_train)
X_test_quadratic = quadratic_featurizer.transform(x_test)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Height (m) vs Distance (m) reached by throwing a ball')
plt.xlabel('Distance in metres')
plt.ylabel('Height in metres')
plt.axis([0, 15, 0, 60])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()

