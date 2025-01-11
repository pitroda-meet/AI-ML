# Import libraries
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])  # Quadratic relationship

# Transform features to polynomial
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Predict and plot
y_pred = model.predict(X_poly)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Prediction')
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Output coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
