import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from PIBO.custom_mean_functions import CustomMean

class CustomMeanGP(SingleTaskGP):
    def __init__(self, train_x, train_y):
        likelihood = GaussianLikelihood()
        mean_module = CustomMean(train_x.numpy(), train_y.numpy())
        covar_module = ScaleKernel(RBFKernel())
        super().__init__(train_x, train_y, likelihood=likelihood, mean_module=mean_module, covar_module=covar_module)
