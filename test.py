from opt_lib import *


def visualize_model():

    model = fit_eigenvalues(learning_rate=0.05, n_epochs=500)

    model.eval()  # Set the model to evaluation mode

    # Generate a grid of values for x[5] and x[6]
    x5 = np.linspace(0, 1, 100)  # 100 points from 0 to 1
    x6 = np.linspace(0, 1, 100)
    x5_grid, x6_grid = np.meshgrid(x5, x6)

    # Flatten the grid for model input
    x5_flat = x5_grid.ravel()
    x6_flat = x6_grid.ravel()

    # Create a fixed random vector for the other features
    input_dim = model.model[0].in_features  # Get the input dimension from the model
    fixed_features = np.array([0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.17825681156415515,0.0,0.04013017315107373])  # Fixed random values between 0 and 1
    # fixed_features = [0.5] * 11
    fixed_features[5] = 0  # Placeholder to be replaced
    fixed_features[6] = 0  # Placeholder to be replaced

    # Create the input data by replicating fixed features and updating x[5] and x[6]
    input_data = np.tile(fixed_features, (len(x5_flat), 1))  # Replicate fixed features
    input_data[:, 5] = x5_flat  # Replace x[5] with grid values
    input_data[:, 6] = x6_flat  # Replace x[6] with grid values

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.double)

    # Get model predictions
    with torch.no_grad():
        predictions = model(input_tensor).numpy().reshape(x5_grid.shape)

    # Plot the surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x5_grid, x6_grid, predictions, cmap='viridis')

    # Label the axes
    ax.set_xlabel("x[5]")
    ax.set_ylabel("x[6]")
    # ax.set_zlabel("Model Output (Probability)")
    ax.set_title("2D Surface of Model Output with Fixed Random Features")
    plt.show()


def optimization_comparison(num_trials=10, n_init=10, max_its=20):
    loss_histories_physics = []
    loss_histories_random = []

    for trial in range(num_trials):

        print(f"Running trial {trial + 1} with physics_informed={True}")
        best_loss_history, _, best_loss, best_X = opt_cEI(
                n_init=n_init, max_its=max_its
            )
        loss_histories_physics.append(best_loss_history)

        print(f"Running trial {trial + 1} with physics_informed={False}")
        best_loss_history, _, best_loss, best_X = opt_BO(
                n_init=n_init, max_its=max_its
            )
        loss_histories_random.append(best_loss_history)

    # Compute the average loss histories
    average_loss_history_physics = np.mean(loss_histories_physics, axis=0)
    average_loss_history_random = np.mean(loss_histories_random, axis=0)

    # Plotting individual trajectories and average loss descent
    plt.figure(figsize=(12, 8))

    # Plot individual trajectories for physics-informed initialization
    for trial_loss in loss_histories_physics:
        plt.plot(trial_loss, color='blue', alpha=0.3, label='Physics-Informed (Individual)' if 'Physics-Informed (Individual)' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot average trajectory for physics-informed initialization
    plt.plot(average_loss_history_physics, color='blue', label='Physics-Informed (Average)', linewidth=2)

    # Plot individual trajectories for random initialization
    for trial_loss in loss_histories_random:
        plt.plot(trial_loss, color='orange', alpha=0.3, label='Physics-Agnostic (Individual)' if 'Physics-Agnostic (Individual)' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot average trajectory for random initialization
    plt.plot(average_loss_history_random, color='orange', label='Physics-Agnostic (Average)', linewidth=2)

    # Labeling the plot
    plt.xlabel("Iteration")
    plt.ylabel("Best Loss")
    plt.title("Convergence Comparison")
    plt.legend()

    # Create a directory to save plots if it doesn't exist
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique and informative file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{output_dir}/loss_trajectories_trials{num_trials}_ninit{n_init}_"
        f"maxits{max_its}_{timestamp}.png"
    )

    # Save the plot locally
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {file_name}")


optimization_comparison(50, 10, 20)