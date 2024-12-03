import matplotlib.pyplot as plt
import numpy as np

def piecewise_plot(partitions, behaviors):
    """
    Plot a piecewise function based on specified partitions and behaviors.
    
    :param partitions: List of tuples specifying the intervals [(start1, end1), (start2, end2), ...]
    :param behaviors: List of behaviors for each interval ("constant" or "slope").
                      For "slope", the slope of 2 is applied.
    """
    if len(partitions) != len(behaviors):
        raise ValueError("Each partition must have a corresponding behavior.")
    
    x_values = []
    y_values = []
    
    # Initialize starting x and y
    x_start = partitions[0][0]
    y_start = 6
    
    for i, (start, end) in enumerate(partitions):
        x = np.linspace(start, end, 100)
        if behaviors[i] == "constant":
            y = np.full_like(x, y_start)
        elif behaviors[i] == "slope":
            y = y_start + 2 * (x - start)
        else:
            raise ValueError(f"Invalid behavior: {behaviors[i]}. Use 'constant' or 'slope'.")
        
        x_values.extend(x)
        y_values.extend(y)
        y_start = y[-1]  # Update y_start for the next partition
    
    # Plot the piecewise function
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.xlabel("time")
    plt.ylabel("Collision point (x)")
    plt.ylim(0, 30)
    plt.title("Collision point vs t")
    plt.grid()
    plt.legend()
    plt.show()

# Example Usage
partitions = [(100, 280.4), (280.4, 282.8), (282.8, 307.9), (307.9, 346)]
behaviors = ["constant", "slope", "constant", "slope"]

piecewise_plot(partitions, behaviors)