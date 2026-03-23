Linear Regression — Ordinary Least Squares (OLS)

Overview
Linear Regression is one of the most fundamental algorithms in machine learning. It models the relationship between input variables (X) and a target variable (y) using a linear function.

For a single feature:
y = θ₀ + θ₁x

For multiple features:
y = Xθ

Ordinary Least Squares (OLS)
Ordinary Least Squares (OLS) is a method used to estimate the parameters θ by minimizing the sum of squared errors between predicted values and actual values.

The objective function is:
J(θ) = Σ (yᵢ − ŷᵢ)²

The closed-form solution is:
θ = (XᵀX)⁻¹Xᵀy

Intuition
OLS finds the line (or hyperplane) that best fits the data by minimizing the squared vertical distances between observed values and predicted values.

Implementation (NumPy)

import numpy as np

Add bias term

X_b = np.c_[np.ones((X.shape[0], 1)), X]

Compute OLS solution

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

Advantages

Simple and easy to implement
Has a closed-form solution
Works well when the relationship is linear

Limitations

Sensitive to outliers
Requires XᵀX to be invertible
Not efficient for very large datasets

OLS vs Gradient Descent
OLS is a closed-form solution that is fast for small datasets but does not scale well.
Gradient Descent is an iterative method that is slower but works better for large datasets.

Summary
OLS is a widely used method for solving linear regression problems. It provides an exact solution but may not be suitable for large or complex datasets.
