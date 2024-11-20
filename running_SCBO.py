from SCBO_two_dimer import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    y_model = predict_normalized(model_dict, X_train)
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
    data.to_csv('datasets/fun_data.csv', index=False)
    print("Data saved to 'fun_data.csv'")

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
    def get_average_loss_history(physics_informed):
        all_loss_histories = []
        for trial in range(num_trials):
            print(f"Running trial {trial + 1} with physics_informed={physics_informed}")
            best_loss_history, _, best_loss, best_X = opt(
                n_init=n_init, max_its=max_its, physics_informed=physics_informed
            )
            all_loss_histories.append(best_loss_history)
        return np.mean(all_loss_histories, axis=0)

    # Get average loss history for physics-informed and random initial points
    average_loss_history_physics = get_average_loss_history(physics_informed=True)
    average_loss_history_random = get_average_loss_history(physics_informed=False)

    # Plotting the average loss descent for both methods
    plt.figure(figsize=(10, 6))
    plt.plot(
        average_loss_history_physics, label="Physics-Informed Initialization"
    )
    plt.plot(average_loss_history_random, label="Random Initialization")
    plt.xlabel("Iteration")
    plt.ylabel("Average Best Loss")
    plt.title(
        "Average Loss Function Descent: Physics-Informed vs Random Initialization"
    )
    plt.legend()

    # Create a directory to save plots if it doesn't exist
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique and informative file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{output_dir}/avg_loss_descent_trials{num_trials}_ninit{n_init}_"
        f"maxits{max_its}_{timestamp}.png"
    )

    # Save the plot locally
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {file_name}")



visualize_fun_and_eig()
