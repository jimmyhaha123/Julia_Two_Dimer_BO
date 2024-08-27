from model import *

from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate

import os
from contextlib import contextmanager, nullcontext

from ax.utils.testing.mock import fast_botorch_optimize_context_manager
import plotly.io as pio

from ax.utils.notebook.plotting import render
import subprocess


pio.renderers.default = "png"

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_EVALS = 10 if SMOKE_TEST else 3000


ax_model = BoTorchModel(
    surrogate=Surrogate(
        # The model class to use
        botorch_model_class=SimpleCustomGP,
        # Optional, MLL class with which to optimize model parameters
        # mll_class=ExactMarginalLogLikelihood,
        # Optional, dictionary of keyword arguments to model constructor
        # model_options={}
    ),
    # Optional, acquisition function class to use - see custom acquisition tutorial
    # botorch_acqf_class=qExpectedImprovement,
)

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


gs = GenerationStrategy(
    steps=[
        # Quasi-random initialization step
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
        ),
        # Bayesian optimization step using the custom acquisition function
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
            model_kwargs={
                "surrogate": Surrogate(SimpleCustomGP),
            },
        ),
    ]
)

import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import Branin

# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name="single_dimer_ngspice_experiment",
    parameters=[
        {
            "name": "gainr1",
            "type": "range",
            # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
            # Otherwise, the parameter would be inferred as an integer range.
            "bounds": [float(5.0), float(2000.0)],
        },
        {
            "name": "gainr2",
            "type": "range",
            "bounds": [float(5.0), float(10000.0)],
        },
    ],
    objectives={
        "single_dimer_ngspice": ObjectiveProperties(minimize=True),
    },
)


# Setup a function to evaluate the trials
branin = Branin()
fast_smoke_test = nullcontext

def evaluate(parameters):
    optim_var = ['gainr1', 'gainr2']
    x = torch.tensor([[parameters.get(var) for var in optim_var]])

    # Prepare the input array with fixed values and dynamic parameters
    input_values = [x[0, 0].item(), x[0, 1].item(), 543.8022037767349, 1156.0876792090266, 0.4027742682261706, 0.27689905702227396]
    
    # Convert the input list to strings (as command-line arguments)
    input_args = ['ngspice'] + [str(val) for val in input_values]

    # Call the Julia script via subprocess and capture the output
    result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
    
    # Convert the result from bytes to a float
    result = float(result.decode("utf-8").strip())
    
    # Return the result, assuming noiseless observation
    return {"single_dimer_ngspice": (result, float(1e-6))}


with fast_smoke_test():
    for i in range(NUM_EVALS):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

parameters, values = ax_client.get_best_parameters()
print(f"Best parameters: {parameters}")
print(f"Corresponding mean: {values[0]}, covariance: {values[1]}")

df = ax_client.get_trials_data_frame()
print(df)


