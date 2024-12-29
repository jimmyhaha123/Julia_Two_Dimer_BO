import math
import os
import warnings
from dataclasses import dataclass
import datetime

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np

from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import normalize, unnormalize
from botorch.optim.optimize import optimize_acqf
from loss_functions import *
from stability import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

import matplotlib.pyplot as plt

device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}
max_cholesky_size = float("inf")

def generate_bounds(val):
    # diff = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ub = []
    lb = []
    for i in range(len(val)):
        mag = np.abs(val[i])
        ub.append(val[i] + 0.3*mag)
        lb.append(val[i] - 0.3*mag)
    return ub, lb

ub, lb = generate_bounds([0.5348356988269656, 0.9539459131310281, 1.3706409241248039, 1.0075697873925655, -0.9582053739454646, 0.3533010453018266, 0.08243808576203537, -0.9173040619954651, 0.9737381475497143, 0.21232344561167762, 0.7559953794996148])
bounds = [(lb[i], ub[i]) for i in range(len(lb))]

fun = TwoDimerCMTLoss(bounds=bounds).to(**tkwargs)
lb, ub = fun.bounds
dim = fun.dim

def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""  
    return fun(unnormalize(x, fun.bounds))

def c1(x):  # The stability constraint, >0 for fixed points
    return -julia_stability_constraint(x)

# We assume c1, c2 have same bounds as the Ackley function above
def eval_c1(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return -julia_stability_constraint(unnormalize(x, fun.bounds))

def eval_c1_binary(x):
    val = -julia_stability_constraint(unnormalize(x, fun.bounds))
    return np.sign(val)

def get_initial_points(dim, n_pts):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def fit_eigenvalues(n_pts=100, use_dataset=True, validation_split=0.3, n_epochs=50, learning_rate=0.001):
    # Step 1: Load or generate data
    if not use_dataset:
        print("Start C1 evaluations.")
        dim = 10  # Set your dimensionality here
        from torch.quasirandom import SobolEngine

        sobol = SobolEngine(dimension=dim, scramble=True)
        X_init = sobol.draw(n=n_pts).to(torch.float32)  # Shape: (n_pts, dim)

        train_C1 = np.array([eval_c1_binary(torch.tensor(x, dtype=torch.float32)) for x in X_init.numpy()])
        train_C1 = torch.tensor(train_C1, dtype=torch.float32)
        print("C1 evaluations complete.")
    else:
        data = pd.read_csv('datasets/eigenvalues_data_domain1.csv')
        X_init = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)  # All columns except 'train_C1'
        train_C1 = torch.tensor(data['train_C1'].values, dtype=torch.float32)

    # Map labels from {-1, 1} to {0, 1} for binary classification
    train_C1 = (train_C1 + 1) / 2

    # Step 2: Split data into training and validation sets
    total_points = len(X_init)
    val_size = int(total_points * validation_split)
    train_size = total_points - val_size

    train_data, val_data = random_split(
        list(zip(X_init, train_C1)), [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_X, train_y = zip(*train_data)
    val_X, val_y = zip(*val_data)

    train_X = torch.stack(train_X)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_X = torch.stack(val_X)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    # Step 3: Define a neural network
    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim):
            super(BinaryClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Ensures the output is a probability between 0 and 1
            )

        def forward(self, x):
            return self.model(x)

    # Initialize the network
    input_dim = X_init.shape[1]
    model = BinaryClassifier(input_dim)

    # Step 4: Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Train the network
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X).squeeze()
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X).squeeze()
            val_loss = criterion(val_outputs, val_y)

            # Calculate validation accuracy
            val_predictions = (val_outputs >= 0.5).float()
            val_acc = (val_predictions == val_y).float().mean().item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")


        # Save the model if it has the best validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # Load the best model state
    model.load_state_dict(best_model_state)
    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")

    # Step 6: Return the trained model
    return model


def physics_informed_initial_points(dim, n_pts, seed=0, hyperparameter=0.6):
    model_dict = fit_eigenvalues()
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    selected_points = []
    
    while len(selected_points) < n_pts:
        # Generate a batch of points using Sobol
        X_batch = sobol.draw(n=n_pts).to(dtype=torch.float32)
        probs = predict_normalized(model_dict, X_batch, hyperparameter=hyperparameter)
        # Iterate over generated points and apply biased sampling with normalized probabilities
        for point, prob in zip(X_batch, probs):
            # print(f"Normalized Acceptance prob: {prob.item()}")

            if torch.rand(1).item() < prob.item():  # Keep the point with normalized probability
                selected_points.append(point)
                if len(selected_points) == n_pts:
                    break
    return torch.stack(selected_points).to(**tkwargs)

def get_fitted_model(X, Y):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)

    return model

def opt_BO(X, Y, max_its=50):
    train_X = X
    train_Y = Y

    best_loss_history = []
    current_best_loss = float('inf')
    for y in (-train_Y).flatten():  # Use -train_Y to account for the negation
        current_best_loss = min(current_best_loss, y.item())
        best_loss_history.append(current_best_loss)

    its = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    while its < max_its:
        its += 1
        print(f"Current: {its}")
        model = get_fitted_model(train_X, train_Y)

        acq_func = ExpectedImprovement(model=model, best_f=-current_best_loss)

        l = [0] * dim
        u = [1] * dim
        b = torch.tensor([l, u]).to(**tkwargs)
        X_next, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=b,
            q=1,
            num_restarts=20,
            raw_samples=512,  # initialization samples
        )

        Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        train_X = torch.cat((train_X, X_next), dim=0)
        train_X_unnormalized = train_X
        train_Y = torch.cat((train_Y, Y_next), dim=0)

        current_best_loss = min(current_best_loss, (-Y_next).min().item())
        best_loss_history.append(current_best_loss)

        # Save the DataFrame to a CSV file
        train_X_np = train_X_unnormalized.numpy()
        train_Y_np = train_Y.numpy().flatten()
        train_X_columns = [f'train_X_{i + 1}' for i in range(train_X_np.shape[1])]

        df = pd.DataFrame(train_X_np, columns=train_X_columns)
        df['train_Y'] = train_Y_np

        file_name = f"datasets/PI_{False}_{timestamp}.csv"
        df.to_csv(file_name, index=False)

        print(f"{len(train_X)}) Best value: {current_best_loss}")

    best_loss, best_x = min(zip(best_loss_history, train_X.tolist()), key=lambda pair: pair[0])
    if its < max_its:
        print("Trust region too small. The model has converged.")
    else:
        print("Reached maximum iteration. Optimization ended.")
    return best_loss_history, train_X, best_loss, best_x