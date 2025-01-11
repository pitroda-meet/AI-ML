# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Bengaluru Housing dataset
file_path = '/content/drive/MyDrive/workshop/Bengaluru_House_Data.csv'
data = pd.read_csv(file_path)

# Display dataset information
print("Dataset shape:", data.shape)
print(data.head())

# Handle missing data (drop rows with NaN or perform imputation if needed)
data = data.dropna()

# Select features and target variable
# Assuming "Price" is the target column, and other numeric columns are features
X = data.drop(columns=["price"])  # Replace "Price" with the actual target column
y = data["price"]  # Replace "Price" with the actual target column

# Ensure that features are numeric
X = X.select_dtypes(include=[np.number])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Housing Prices")
plt.legend()
plt.show()
