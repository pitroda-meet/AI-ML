import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.5, size=x.shape)

# Model Complexity vs Error
train_errors = []
test_errors = []

# Varying model complexity (polynomial degrees)
for degree in range(1, 20):
    # Fit a polynomial of degree `degree`
    p = np.polyfit(x, y, degree)
    y_train_pred = np.polyval(p, x)

    # Compute training error (Bias) and testing error (Variance)
    train_error = np.mean((y - y_train_pred)**2)
    test_error = np.mean((np.sin(x) - y_train_pred)**2)

    train_errors.append(train_error)
    test_errors.append(test_error)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), train_errors, label="Training Error (Bias)", color='blue', linestyle='--')
plt.plot(range(1, 20), test_errors, label="Testing Error (Variance)", color='red')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()

# cExplanation of the Graph:
# X-axis (Model Complexity): Represents the complexity of the model (in this case, the polynomial degree). As model complexity increases, the model can fit the data better.
# Y-axis (Error): Represents the error of the model. This is a combination of bias and variance.
# Training Error (Bias): As complexity increases, the model can fit the training data better, reducing bias (the model captures more patterns).
# Testing Error (Variance): At first, the model improves as complexity increases but after a certain point, the variance increases and the model starts to overfit, resulting in a higher testing error.
# At low complexity (underfitting), the model has high bias and low variance. As complexity increases, bias decreases and variance increases. The goal is to find the sweet spot where both bias and variance are balanced, minimizing total error.

# This visualizes how underfitting (high bias) and overfitting (high variance) can affect model performance, helping to illustrate the bias-variance tradeoff.
