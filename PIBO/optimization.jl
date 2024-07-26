include("util.jl")
using PyCall
using LinearAlgebra

# Import Python modules
pushfirst!(PyVector(pyimport("sys")."path"), "/Users/jimmy/Documents/GitHub/Julia_Two_Dimer_BO/PIBO")
botorch = pyimport("botorch")
torch = pyimport("torch")
np = pyimport("numpy")
optimize_acqf = pyimport("botorch.optim").optimize_acqf
custom_mean_functions = pyimport("custom_mean_functions.py")

# Initial data for fitting the mean function
init_x = [-1.0; 0.0; 1.0]  # Replace with your initialization data
init_y = [-1.0; 0.0; 1.0]  # Replace with your initialization data

# Fit the initial linear model using Julia
X = hcat(ones(length(init_x)), init_x)
coefficients = X \ init_y

println("Initial linear fit coefficients: ", coefficients)

# Convert coefficients to a numpy array and then to a Python object
coefficients_py = np.array(coefficients)

# Function to evaluate the actual objective (replace with your actual objective function)
function evaluate_objective(x)
    # This is a placeholder objective function. Replace it with the actual function you want to optimize.
    return sum(x.^2)
end

# Define the bounds of the optimization problem
bounds = torch.tensor([[0.0], [1.0]])  # Replace with the actual bounds of your problem

# Number of iterations
num_iterations = 10  # Set the number of iterations for Bayesian optimization

# Initialize actual training data
train_x = torch.empty(0, 1)
train_y = torch.empty(0)

for i in 1:num_iterations
    # Define the custom mean GP model using the saved coefficients
    model = custom_mean_functions.CustomMeanGP(train_x, train_y, coefficients_py)

    # Add the actual training data (if any) to the model
    if train_x.size(0) > 0
        model.set_train_data(inputs=train_x, targets=train_y, strict=false)
    end

    # Set model to training mode and fit the model
    model.train()
    likelihood = model.likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = botorch.fit_gpytorch_model(model=model, mll=likelihood)
    
    # Define the acquisition function
    best_f = train_y.size(0) > 0 ? train_y.max() : torch.tensor(0.0)
    qEI = botorch.acquisition.qExpectedImprovement(model=model, best_f=best_f)
    
    # Optimize the acquisition function
    candidate, acq_value = optimize_acqf(qEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
    
    # Evaluate the candidate
    new_x = candidate.detach().numpy()
    new_y = evaluate_objective(new_x)
    
    # Append new observations to the training data
    global train_x = torch.cat([train_x, torch.tensor([new_x]).unsqueeze(0)])
    global train_y = torch.cat([train_y, torch.tensor([new_y])])
    
    println("Iteration $i:")
    println("Suggested candidate: ", new_x)
    println("Objective value: ", new_y)
end

println("Final suggested candidate: ", train_x[train_y.argmax()].numpy())
