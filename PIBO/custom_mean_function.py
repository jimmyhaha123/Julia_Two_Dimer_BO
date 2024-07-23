# custom_mean_functions.py
import torch
from gpytorch.means import Mean

class CustomMean(Mean):
    def __init__(self, x, y):
        super(CustomMean, self).__init__()
        # Perform linear regression to find the best fit parameters
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # Add a column of ones to x for the intercept term
        x = torch.cat([torch.ones(x.shape[0], 1), x], dim=1)
        # Compute the coefficients using the normal equation: (X^T X)^-1 X^T y
        coefficients = torch.linalg.lstsq(x, y.unsqueeze(-1)).solution.squeeze()
        # Separate intercept and slopes
        self.bias = torch.nn.Parameter(torch.tensor(coefficients[0]))
        self.coefficients = torch.nn.Parameter(coefficients[1:])
        
        # Calculate the Mean Squared Error (MSE)
        predictions = torch.matmul(x, coefficients)
        mse = torch.mean((predictions - y) ** 2).item()
        print(f"Initial linear fit MSE: {mse}")

    def forward(self, X):
        # Add a column of ones to X for the intercept term
        X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
        # Combine bias and coefficients
        combined_params = torch.cat([self.bias.unsqueeze(0), self.coefficients])
        # Compute the mean as the dot product of the input X and the coefficients
        return torch.matmul(X, combined_params)
