# custom_mean_functions.py
import torch
from botorch.models import FixedNoiseGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import Mean

class CustomMean(Mean):
    def __init__(self, coefficients):
        super(CustomMean, self).__init__()
        self.bias = torch.nn.Parameter(torch.tensor(coefficients[0]))
        self.coefficients = torch.nn.Parameter(torch.tensor(coefficients[1:]))

    def forward(self, X):
        # Add a column of ones to X for the intercept term
        X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
        # Combine bias and coefficients
        combined_params = torch.cat([self.bias.unsqueeze(0), self.coefficients])
        # Compute the mean as the dot product of the input X and the coefficients
        return torch.matmul(X, combined_params)

class CustomMeanGP(FixedNoiseGP):
    def __init__(self, train_x, train_y, coefficients):
        noise = torch.zeros_like(train_y)  # Noiseless observations
        mean_module = CustomMean(coefficients)
        covar_module = ScaleKernel(RBFKernel())
        super(CustomMeanGP, self).__init__(train_x, train_y, noise, mean_module=mean_module, covar_module=covar_module)
