import math
import os
import warnings
from dataclasses import dataclass

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
import numpy as np

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import normalize, unnormalize
from loss_functions import *
from stability import *
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

import matplotlib.pyplot as plt

class EigenvalueNet(nn.Module):
    def __init__(self, input_dim):
        super(EigenvalueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0, nu1
lb = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 0.5, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711]
ub = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 1.5, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711]
ub = [ub[i] + 0.0001 for i in range(11)]
bounds = [(lb[i], ub[i]) for i in range(len(lb))]

fun = TwoDimerCMTLoss(bounds=bounds).to(**tkwargs)
lb, ub = fun.bounds
dim = fun.dim


batch_size = 4
max_cholesky_size = float("inf")  # Always use Cholesky

# When evaluating the function, we must first unnormalize the inputs since
# we will use normalized inputs x in the main optimizaiton loop
def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""  
    return fun(unnormalize(x, fun.bounds))

def c1(x):  # The stability constraint, >0 for fixed points
    return -julia_stability_constraint(x)

# We assume c1, c2 have same bounds as the Ackley function above
def eval_c1(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return -julia_stability_constraint(unnormalize(x, fun.bounds))


@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(1, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = 10 * math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def update_tr_length(state: ScboState):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state



def eval_constraint(x):
    return max(0, -eval_c1(x))

def fit_eigenvalues(n_pts=50, seed=0, save_model=False, model_path='neural_model.pkl', 
                    n_epochs=1000, lr=0.001, patience=100, use_dataset=True):
    """
    Fits a neural network model with early stopping and computes theoretical min and max outputs over [0,1]^dim.
    
    Args:
        n_pts (int): Number of Sobol points to generate.
        dim (int): Input dimensionality.
        seed (int): Seed for reproducibility.
        save_model (bool): Whether to save the model and normalization parameters.
        model_path (str): Path to save the model.
        n_epochs (int): Maximum number of training epochs for the neural network.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait with no improvement on validation loss before stopping.
        
    Returns:
        dict: Contains the fitted model, min_output, and max_output.
    """


    if not use_dataset:
        # 1. Generate Sobol Samples
        print("Start C1 evaluations. ")
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(torch.float32)  # Shape: (n_pts, dim)

        # 2. Evaluate train_C1
        train_C1 = np.array([eval_constraint(torch.tensor(x, dtype=torch.float32)) for x in X_init.numpy()])  # Shape: (n_pts,)
        train_C1 = torch.tensor(train_C1, dtype=torch.float32)
        print("C1 evaluations complete. ")
    else:
        data = pd.read_csv('eigenvalues_data.csv')
        X_init = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)  # All columns except 'train_C1'
        train_C1 = torch.tensor(data['train_C1'].values, dtype=torch.float32)

    # Split into training and validation sets
    val_split = int(0.8 * len(X_init))  # Use the length of X_init instead of n_pts

    # Split into training and validation sets
    X_train, X_val = X_init[:val_split], X_init[val_split:]
    y_train, y_val = train_C1[:val_split], train_C1[val_split:]

    # 3. Initialize Neural Network Model
    model = EigenvalueNet(input_dim=dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 4. Train Neural Network Model with Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        model.train()
        optimizer.zero_grad()
        train_predictions = model(X_train)
        train_loss = loss_fn(train_predictions, y_train)
        train_loss.backward()
        optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = loss_fn(val_predictions, y_val).item()

        # Early stopping check\
        print(f"Val loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)  # Restore best model
                break

    print("Training complete. ")
    # 5. Compute Theoretical Min and Max Output
    model.eval()
    with torch.no_grad():
        # Evaluating at corners of [0, 1]^dim to get min and max possible values
        corners = torch.tensor(np.array(np.meshgrid(*[[0, 1]] * dim)).T.reshape(-1, dim), dtype=torch.float32)
        corner_outputs = model(corners)
        min_output = corner_outputs.min().item()
        max_output = corner_outputs.max().item()

    if max_output - min_output == 0:
        raise ValueError("Theoretical max and min outputs are equal; cannot perform normalization.")
    print("Normalization complete. Eigenvalue model training complete.")
    # 6. Save Model and Normalization Parameters if Required
    if save_model:
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model.state_dict(), 'min_output': min_output, 'max_output': max_output}, f)
        print(f"Model saved to {model_path}")

    return {'model': model.state_dict(), 'min_output': min_output, 'max_output': max_output, 'X_init':X_init, 'train_C1':train_C1}

def predict_normalized(model_dict, X_new):
    """
    Makes normalized predictions ensuring outputs are within [0, 1].
    
    Args:
        model_dict (dict): Contains the fitted model, min_output, and max_output.
        X_new (torch.Tensor): New input samples of shape (n_samples, dim).
        
    Returns:
        np.ndarray: Normalized predictions within [0, 1].
        
    Raises:
        ValueError: If any prediction is outside the [0, 1] range.
    """
    model = EigenvalueNet(input_dim=X_new.shape[1])
    model.load_state_dict(model_dict['model'])
    model.eval()
    
    min_output = model_dict['min_output']
    max_output = model_dict['max_output']

    with torch.no_grad():
        predictions = model(X_new).numpy()
    
    # Min-Max Normalization
    predictions_normalized = (predictions - min_output) / (max_output - min_output)
    
    # Check for out-of-bounds predictions
    if np.any(predictions_normalized < 0) or np.any(predictions_normalized > 1):
        out_indices = np.where((predictions_normalized < 0) | (predictions_normalized > 1))[0]
        out_values = predictions_normalized[out_indices]
        raise ValueError(f"Predictions out of [0, 1] range at indices {out_indices}: {out_values}")
    
    return predictions_normalized


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


def physics_informed_initial_points(dim, n_pts, seed=0):
    model_dict = fit_eigenvalues()
    model = EigenvalueNet(input_dim=model_dict['X_init'].shape[1])
    model.load_state_dict(model_dict['model'])
    model.eval()
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    selected_points = []
    
    while len(selected_points) < n_pts:
        # Generate a batch of points using Sobol
        X_batch = sobol.draw(n=n_pts).to(dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_batch).cpu()

        # Perform min-max normalization on the batch probabilities
        min_prob, max_prob = probs.min(), probs.max()
        normalized_probs = (probs - min_prob) / (max_prob - min_prob + 1e-6)  # Avoid division by zero

        # Iterate over generated points and apply biased sampling with normalized probabilities
        for point, prob in zip(X_batch, normalized_probs):
            # print(f"Normalized Acceptance prob: {prob.item()}")

            if torch.rand(1).item() < prob.item():  # Keep the point with normalized probability
                selected_points.append(point)
                if len(selected_points) == n_pts:
                    break

    return torch.stack(selected_points)


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraint values
    batch_size,
    n_candidates,  # Number of candidates for Thompson sampling
    constraint_model,
    sobol: SobolEngine,
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the TR bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

def opt(n_init=200, physics_informed=True):
    # Define example state
    state = ScboState(dim=dim, batch_size=batch_size)
    print(state)
    # Generate initial data
    print("Sampling initial points.")
    train_X = physics_informed_initial_points(dim, n_init) if physics_informed else get_initial_points(dim, n_init)
    print("Initial sampling complete.")
    train_Y = torch.tensor([eval_objective(x) for x in train_X], **tkwargs).unsqueeze(-1)
    C1 = torch.tensor([eval_c1(x) for x in train_X], **tkwargs).unsqueeze(-1)

    # Initialize TuRBO state
    state = ScboState(dim, batch_size=batch_size)

    # Note: We use 2000 candidates here to make the tutorial run faster.
    # SCBO actually uses min(5000, max(2000, 200 * dim)) candidate points by default.
    N_CANDIDATES = 2000 if not SMOKE_TEST else 4
    sobol = SobolEngine(dim, scramble=True, seed=1)


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


    while not state.restart_triggered:  # Run until TuRBO converges
        # Fit GP models for objective and constraints
        model = get_fitted_model(train_X, train_Y)
        c1_model = get_fitted_model(train_X, C1)

        # Generate a batch of candidates
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            X_next = generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=torch.cat((C1, torch.empty(0)), dim=-1),
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                constraint_model=ModelListGP(c1_model),
                sobol=sobol,
            )

        # Evaluate both the objective and constraints for the selected candidaates
        Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C1_next = torch.tensor([eval_c1(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C_next = torch.cat([C1_next, torch.empty(0)], dim=-1)

        # Update TuRBO state
        state = update_state(state=state, Y_next=Y_next, C_next=C_next)

        # Append data. Note that we append all data, even points that violate
        # the constraints. This is so our constraint models can learn more
        # about the constraint functions and gain confidence in where violations occur.
        train_X = torch.cat((train_X, X_next), dim=0)
        train_X_unnormalized = train_X
        train_Y = torch.cat((train_Y, Y_next), dim=0)
        C1 = torch.cat((C1, C1_next), dim=0)

        # Save the DataFrame to a CSV file
        train_X_np = train_X_unnormalized.numpy()
        train_Y_np = train_Y.numpy().flatten()
        C1_np = C1.numpy().flatten()
        train_X_columns = [f'train_X_{i+1}' for i in range(train_X_np.shape[1])]

        # Combine the data into a DataFrame
        df = pd.DataFrame(train_X_np, columns=train_X_columns)
        df['train_Y'] = train_Y_np
        df['C1'] = C1_np

        df.to_csv("training_data.csv", index=False)

        # Print current status. Note that state.best_value is always the best
        # objective value found so far which meets the constraints, or in the case
        # that no points have been found yet which meet the constraints, it is the
        # objective value of the point with the minimum constraint violation.
        if (state.best_constraint_values <= 0).all():
            print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
        else:
            violation = state.best_constraint_values.clamp(min=0).sum()
            print(
                f"{len(train_X)}) No feasible point yet! Smallest total violation: "
                f"{violation:.2e}, TR length: {state.length:.2e}"
            )

    print("Trust region too small, model has converged.")


# Eigenvalue test

# model_dict = fit_eigenvalues(n_pts=10, seed=42, save_model=False)

# # Generate new samples
# sobol_new = SobolEngine(dimension=dim, scramble=True, seed=1)
# X_new = sobol_new.draw(n=5000).to(**tkwargs)

# # Make normalized predictions with error handling
# try:
#     preds_normalized = predict_normalized(model_dict, X_new)
#     print("Normalized Predictions:", preds_normalized)
# except ValueError as e:
#     print("Error:", e)

# opt(n_init=100)


# an20_range = np.linspace(0.5, 1.5, 50)
# loss_values = []
# constraint_values = []
# for an20 in an20_range:
#     p = torch.tensor([0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, an20, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711], dtype=torch.float32)
#     loss_values.append(fun(p).item())
#     constraint_values.append(c1(p))

# data = pd.DataFrame({
#     "an20": an20_range,
#     "loss_value": loss_values,
#     "constraint_value": constraint_values
# })

# # Save DataFrame to a local CSV file
# file_path = 'an20_loss_constraint_values.csv'
# data.to_csv(file_path, index=False)



X_init = physics_informed_initial_points(11, 5000)
X_init = unnormalize(X_init, fun.bounds)
reduced_tensor = X_init[:, 5]
# print(reduced_tensor)

plt.figure(figsize=(10, 6))
plt.hist(reduced_tensor.numpy(), bins=20, edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Reduced Tensor Values")
plt.show()



# model_dict = fit_eigenvalues(n_pts=50, seed=42, save_model=False)
# an20_range = np.linspace(0.5, 1.5, 5)
# prob = []
# for an20 in an20_range:
#     p = torch.tensor([0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, an20, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711], dtype=torch.float32)
#     prob.append(predict_normalized(model_dict, p.reshape([1, -1])))
# prob = np.array(prob)
# plt.plot(an20_range, prob)
# plt.show()


# Plotting 

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rc


# fig, ax = plt.subplots(figsize=(8, 6))

# score = train_Y.clone()
# # Set infeasible to -inf
# score[~(torch.cat((C1, torch.empty(0)), dim=-1) <= 0).all(dim=-1)] = float("-inf")
# fx = np.maximum.accumulate(score.cpu())
# plt.plot(fx, marker="", lw=3)

# plt.plot([0, len(train_Y)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
# plt.ylabel("Function value", fontsize=18)
# plt.xlabel("Number of evaluations", fontsize=18)
# plt.title("10D Ackley with 2 outcome constraints", fontsize=20)
# plt.xlim([0, len(train_Y)])
# plt.ylim([-15, 1])

# plt.grid(True)
# plt.show()

# df = pd.read_csv("n20_vs_loss.csv")

# plt.plot(df["n20"], df["loss"])
# plt.show()

# Neural network eigenvalue fitting testing code
# def plot_training_and_model(model_dict):
#     """
#     Plots the training points and the fitted model predictions for visualization.
    
#     Args:
#         model_dict (dict): Dictionary containing the trained model, min_output, max_output, X_init, and train_C1.
#     """
#     # Unpack model and training data
#     model = EigenvalueNet(input_dim=model_dict['X_init'].shape[1])
#     model.load_state_dict(model_dict['model'])
#     model.eval()
#     X_train = model_dict['X_init']
#     y_train = model_dict['train_C1']

#     # Unnormalize x[5] for training data
#     x5_train = X_train[:, 5] * (1.5 - 0.5) + 0.5

#     # Plot the training data with unnormalized x[5]
#     plt.scatter(x5_train.cpu(), y_train.cpu(), color='blue', label='Training Data', alpha=0.6)

#     # Generate normalized x[5] values from 0 to 1 for model evaluation
#     x5_range_normalized = torch.linspace(0, 1, 100).unsqueeze(1).to(X_train.device)
#     X_fixed = X_train[0].repeat(100, 1)  # Use the first training sample as a template
#     X_fixed[:, 5] = x5_range_normalized.squeeze()  # Vary only x[5] for evaluation

#     # Predict using the fitted model
#     with torch.no_grad():
#         y_model = model(X_fixed).cpu()

#     # Unnormalize x5 for plotting purposes
#     x5_range_unnormalized = x5_range_normalized * (1.5 - 0.5) + 0.5

#     # Plot the fitted model predictions
#     plt.plot(x5_range_unnormalized.cpu(), y_model, color='red', label='Fitted Model')

#     # Formatting the plot
#     plt.xlabel('x[5] (6th Entry, Unnormalized)')
#     plt.ylabel('Output')
#     plt.title('Training Data and Fitted Model')
#     plt.legend()
#     plt.show()

# # Set up parameters for training
# n_pts = 1000
# seed = 0
# n_epochs = 1000
# lr = 0.001
# patience = 100

# # Train the model and obtain training data and model parameters
# model_dict = fit_eigenvalues(n_pts=n_pts, seed=seed, n_epochs=n_epochs, lr=lr, patience=patience, use_dataset=True)

# # Call plot function with trained model dictionary
# plot_training_and_model(model_dict)





# Generating dataset for C1 (in the 1d domain that only an10 varies from 0.5 to 1.5)
# sobol = SobolEngine(dimension=dim, scramble=True)
# X_init = sobol.draw(n=300).to(torch.float32)  # Shape: (n_pts, dim)

# # 2. Evaluate train_C1
# train_C1 = np.array([eval_constraint(torch.tensor(x, dtype=torch.float32)) for x in X_init.numpy()])
# train_C1 = torch.tensor(train_C1, dtype=torch.float32)

# X_init_np = X_init.cpu().numpy()
# train_C1_np = train_C1.cpu().numpy()

# # Combine X_init and train_C1 into a single DataFrame
# data = pd.DataFrame(X_init_np, columns=[f'x{i}' for i in range(X_init_np.shape[1])])
# data['train_C1'] = train_C1_np

# # Save to CSV
# data.to_csv('eigenvalues_data.csv', index=False)
# print("Data saved to 'eigenvalues_data.csv'")



# Generating dataset for fun
# sobol = SobolEngine(dimension=dim, scramble=True)
# X_init = sobol.draw(n=200).to(torch.float32)  # Shape: (n_pts, dim)

# # 2. Evaluate train_C1
# train_fun = np.array([eval_objective(torch.tensor(x, dtype=torch.float32)) for x in X_init.numpy()])
# train_fun = torch.tensor(train_fun, dtype=torch.float32)

# X_init_np = X_init.cpu().numpy()
# train_fun_np = train_fun.cpu().numpy()

# # Combine X_init and train_C1 into a single DataFrame
# data = pd.DataFrame(X_init_np, columns=[f'x{i}' for i in range(X_init_np.shape[1])])
# data['train_fun'] = train_fun_np

# # Save to CSV
# data.to_csv('fun_data.csv', index=False)
# print("Data saved to 'fun_data.csv'")



