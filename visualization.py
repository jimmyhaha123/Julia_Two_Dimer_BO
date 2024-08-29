import numpy as np
import matplotlib.pyplot as plt
import subprocess

def objective(gainr1):
    input_args = ['cmt'] + ['1.4', '1.0', '-1.0', '3.0', str(gainr1), '0.0']
    result = subprocess.check_output(["julia", "single_dimer.jl"] + input_args)
    result = float(result.decode("utf-8").strip())
    return result

# Generate a linspace from 50 to 3000
x_values = np.linspace(0, 1, 200)
y_values = []

# Evaluate the objective function for each value in the linspace
for x in x_values:
    print("Current: " + str(x))
    y = objective(x)
    print("Loss: " + str(y))
    y_values.append(y)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Objective Function')
plt.xlabel('gainr1')
plt.ylabel('Objective Value')
plt.title('Objective Function Plot')
plt.legend()
plt.grid(True)
plt.show()
