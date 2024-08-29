import torch
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Normalize, Standardize
import subprocess
import matplotlib.pyplot as plt

# Define the objective function (without inversion)
def objective(gainr1):
    input_args = ['ngspice'] + [str(gainr1), '5', '543.8022037767349', '1156.0876792090266', '0.4027742682261706', '0.27689905702227396']
    result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
    result = float(result.decode("utf-8").strip())
    return result

# Wrapper to make it compatible with BoTorch (inversion for minimization)
def objective_wrapper(x):
    gainr1 = x.item()  # Convert tensor to scalar
    return torch.tensor([-objective(gainr1)], dtype=torch.float64)  # Invert the objective here for minimization

# Define the bounds of the gainr1 parameter
bounds = torch.tensor([[50.0], [5000.0]], dtype=torch.float64)  # Example bounds; modify according to your needs

# Number of initial samples and optimization iterations
n_initial_samples = 1
n_iterations = 60

# Generate initial data using Sobol sampling
train_x = draw_sobol_samples(bounds=bounds, n=n_initial_samples, q=1).squeeze(1).to(torch.float64)
train_obj = torch.stack([objective_wrapper(x) for x in train_x]).to(torch.float64)

# Fit a GP model with input normalization and output standardization
noise_variance = torch.full_like(train_obj, 1e-6)

# Fit a FixedNoiseGP model with input normalization and output standardization
gp = FixedNoiseGP(
    train_x, 
    train_obj, 
    train_Yvar=noise_variance, 
    input_transform=Normalize(d=train_x.shape[-1]), 
    outcome_transform=Standardize(m=train_obj.shape[-1])
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Optimization loop
for iteration in range(n_iterations):
    # Define the acquisition function (Expected Improvement)
    EI = ExpectedImprovement(gp, best_f=train_obj.max())  # Use the max of the inverted (minimized) objective

    # Optimize the acquisition function to get the next point
    candidate, acq_value = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=1,  # Number of candidates to propose
        num_restarts=5,
        raw_samples=20,
    )

    # Evaluate the objective function at the new candidate
    new_x = candidate.detach().to(torch.float64)
    new_obj = torch.tensor([objective(new_x.item())], dtype=torch.float64)  # Evaluate the original (non-inverted) objective

    # Ensure that new_obj has the same number of dimensions as train_obj
    new_obj = new_obj.view(1, -1)

    # Update training data
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, -new_obj])  # Store the negative value for GP model fitting
    
    print(train_x)

    # Refit the GP model with the new data, using normalization and standardization
    gp = SingleTaskGP(
        train_x, 
        train_obj, 
        input_transform=Normalize(d=train_x.shape[-1]), 
        outcome_transform=Standardize(m=train_obj.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Plot the results
    f, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Create a grid of points for plotting the model predictions
    x_grid = torch.linspace(bounds[0].item(), bounds[1].item(), 101, dtype=torch.float64).unsqueeze(-1)

    # Compute the posterior over the grid
    with torch.no_grad():
        posterior = gp.posterior(x_grid)
        lower, upper = posterior.mvn.confidence_region()

    # Plot the observed data points
    ax.plot(train_x.cpu().numpy(), -train_obj.cpu().numpy(), "k*")  # Invert back for plotting

    # Plot the posterior mean
    ax.plot(x_grid.cpu().numpy(), -posterior.mean.cpu().numpy(), "b")  # Invert back for plotting

    # Shade the area between the lower and upper confidence bounds
    ax.fill_between(
        x_grid.cpu().numpy().flatten(), 
        -upper.cpu().numpy().flatten(),  # Invert back for plotting
        -lower.cpu().numpy().flatten(),  # Invert back for plotting
        alpha=0.5
    )

    ax.legend(["Observed Data", "Mean", "Confidence"])
    plt.tight_layout()
    
    # if iteration % 20 == 0 and iteration != 0:
    plt.show()

    # Print the progress
    print(f"Iteration {iteration + 1}/{n_iterations}: gainr1 = {new_x.item()}, Objective value = {new_obj.item()}")

# Extract the best solution
best_idx = torch.argmin(train_obj)  # Minimize, so use argmin
best_x = train_x[best_idx]
best_obj_value = -train_obj[best_idx]  # Invert back to original value

print(f"\nBest gainr1: {best_x.item()}")
print(f"Best objective value: {best_obj_value.item()}")
