from SCBO_two_dimer import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

tkwargs = {"device": device, "dtype": dtype}


def visualize_fun_and_eig():
    # Load and extract data
    data = pd.read_csv("datasets/fun_data.csv")
    X = data.iloc[:, 5].values
    y = -data['train_fun'].values

    # Fit the eigenvalues model (assuming fit_eigenvalues is defined elsewhere)
    model_dict = fit_eigenvalues(n_pts=1000, seed=0, n_epochs=1000, lr=0.01, patience=50, use_dataset=True)
    model = EigenvalueNet(input_dim=model_dict['X_init'].shape[1])
    model.load_state_dict(model_dict['model'])
    model.eval()

    # Extract training data
    X_train = model_dict['X_init']
    y_train = model_dict['train_C1']
    y_model = predict_normalized(model_dict, X_train, hyperparameter=0.6)
    X_train = X_train[:, 5]

    # Plot the data
    fig, ax1 = plt.subplots()

    # Plot the first scatter plot
    ax1.scatter(X, y, color='blue', label='Function Data')
    ax1.set_ylabel('Function Data (y)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.scatter(X_train, y_model.detach().numpy(), color='red', label='Smaple distribution')
    ax2.set_ylabel('Sample distribution', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax3 = ax2
    ax3.scatter(X_train, y_train, color='green', label='Eigenvalues')

    # Add legends
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.show()


def generate_fun_value(n_pts=100):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(**tkwargs)  # Shape: (n_pts, dim)

    # 2. Evaluate fun
    train_fun = np.array([eval_objective(torch.tensor(x, **tkwargs)) for x in X_init.numpy()])
    train_fun = torch.tensor(train_fun, **tkwargs)

    X_init_np = X_init.cpu().numpy()
    train_fun_np = train_fun.cpu().numpy()

    # Combine X_init and train_C1 into a single DataFrame
    data = pd.DataFrame(X_init_np, columns=[f'x{i}' for i in range(X_init_np.shape[1])])
    data['train_fun'] = train_fun_np

    # Save to CSV
    file_name = 'fun_data_fp.csv'
    data.to_csv(f'datasets/{file_name}', index=False)
    print(f"Data saved to '{file_name}'")


def visualize_biased_sampling():
    X_init = physics_informed_initial_points(11, 5000)
    reduced_tensor = X_init[:, 5]
    # print(reduced_tensor)

    plt.figure(figsize=(10, 6))
    plt.hist(reduced_tensor.numpy(), bins=20, edgecolor='black')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Reduced Tensor Values")
    plt.show()


def optimization_comparison(num_trials=10, n_init=15, max_its=4):
    loss_histories_physics = []
    loss_histories_random = []

    for trial in range(num_trials):
        X_init = get_initial_points(dim, n_init)
        Y_init = torch.tensor([eval_objective(x) for x in X_init], **tkwargs).unsqueeze(-1)

        print(f"Running trial {trial + 1} with physics_informed={True}")
        best_loss_history, _, best_loss, best_X = opt_SCBO(
                X=X_init, Y=Y_init, max_its=max_its
            )
        loss_histories_physics.append(best_loss_history)

        print(f"Running trial {trial + 1} with physics_informed={False}")
        best_loss_history, _, best_loss, best_X = opt_BO(
                X=X_init, Y=Y_init, max_its=max_its
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
        plt.plot(trial_loss, color='orange', alpha=0.3, label='Random (Individual)' if 'Random (Individual)' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot average trajectory for random initialization
    plt.plot(average_loss_history_random, color='orange', label='Random (Average)', linewidth=2)

    # Labeling the plot
    plt.xlabel("Iteration")
    plt.ylabel("Best Loss")
    plt.title("Loss Function Descent: Physics-Informed vs Random Initialization")
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



def generate_eigenvalues(n_pts=100):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(**tkwargs)  # Shape: (n_pts, dim)

    # 2. Evaluate fun
    train_C1 = np.array([eval_c1_binary(torch.tensor(x, **tkwargs)) for x in X_init.numpy()])
    train_C1 = torch.tensor(train_C1, **tkwargs)

    X_init_np = X_init.cpu().numpy()
    train_fun_np = train_C1.cpu().numpy()

    # Combine X_init and train_C1 into a single DataFrame
    data = pd.DataFrame(X_init_np, columns=[f'x{i}' for i in range(X_init_np.shape[1])])
    data['train_C1'] = train_fun_np

    # Save to CSV
    file_name = 'eigenvalues_data_domain3.csv'
    data.to_csv(f'datasets/{file_name}', index=False)
    print(f'datasets/{file_name}')


def plot_from_csv_files(data_dir="datasets", output_dir="plots"):
    """
    Plot individual and average trajectories of optimization results from pre-created CSV files.
    """
    def process_csv(file_path, physics_informed):
        """
        Process a single CSV file to compute the current minimum at each iteration.
        """
        data = pd.read_csv(file_path)
        if physics_informed:
            sampled_values = -data.iloc[:, -2]  # Second to last column for physics-informed
        else:
            sampled_values = -data.iloc[:, -1]  # Last column for random initialization
        current_min = sampled_values.cummin()  # Compute the running minimum
        return current_min.values

    def load_and_process_files(physics_informed):
        """
        Load and process all CSV files for a given `physics_informed` status.
        """
        pattern = os.path.join(data_dir, f"PI_{physics_informed}_*.csv")
        file_paths = glob.glob(pattern)
        trajectories = []
        for file_path in file_paths:
            trajectory = process_csv(file_path, physics_informed=(physics_informed == "True"))
            trajectories.append(trajectory)
        return np.array(trajectories)

    # Load and process files for physics-informed and random cases
    physics_trajectories = load_and_process_files("True")
    random_trajectories = load_and_process_files("False")

    # Compute averages
    avg_physics_trajectory = np.mean(physics_trajectories, axis=0)
    avg_random_trajectory = np.mean(random_trajectories, axis=0)

    # Plot individual trajectories and average trajectory
    plt.figure(figsize=(12, 8))

    # Plot individual trajectories for physics-informed initialization
    for trajectory in physics_trajectories:
        plt.plot(trajectory, color='blue', alpha=0.3, label='Physics-Informed (Individual)' if 'Physics-Informed (Individual)' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot average trajectory for physics-informed initialization
    plt.plot(avg_physics_trajectory, color='blue', label='Physics-Informed (Average)', linewidth=2)

    # Plot individual trajectories for random initialization
    for trajectory in random_trajectories:
        plt.plot(trajectory, color='orange', alpha=0.3, label='Random (Individual)' if 'Random (Individual)' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot average trajectory for random initialization
    plt.plot(avg_random_trajectory, color='orange', label='Random (Average)', linewidth=2)

    # Labeling the plot
    plt.xlabel("Iteration")
    plt.ylabel("Current Minimum Function Value")
    plt.title("Loss Function Descent: Physics-Informed vs Random Initialization")
    plt.legend()

    # Create a directory to save plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir, f"loss_trajectories_{timestamp}.png")

    # Save the plot
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {file_name}")


generate_eigenvalues(n_pts=1000)
