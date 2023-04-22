"""
polynominal Regression
"""
import numpy as np
from .linear_reg import LinearRegression


class PolynominalRegression(LinearRegression):
    """
    polynominal Regression
    """

    def __init__(self, reg=0.0):
        super().__init__(reg)

    def polynomial_basis(self, X, degree=1):
        """
        generate polynomial basis function

        input：
        - X：samples，shape = [n_samples, 1]
        - degree： the degree of polynomial

        output：
        - phi: polynomial basis function，shape = [n_samples,degree+1]
        """
        n_samples = X.shape[0]
        phi = np.ones((n_samples, degree))
        for i in range(0, degree):
            phi[:, i] = np.power(X[:, 0], i+1)
        return phi

    def fit(self, X, y, degree=1):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values.
        degree : int    the degree of polynomial

        Returns
        -------
        self : object
            Fitted model with predicted self.coef_ and self.intercept_.
        """
        phi = self.polynomial_basis(X, degree)
        super().fit(phi, y)

    def predict(self, X, degree=1):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array of shape (n_samples,)
            Returns predicted values.
        """
        phi = self.polynomial_basis(X, degree)
        return super().predict(phi)
