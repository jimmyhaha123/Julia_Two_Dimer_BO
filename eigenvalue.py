import sympy as sp
import numpy as np
import nlopt

def symbolic_jacobian(p):
    w2, k, n11, n10, n20 = p
    # Define the symbolic variables
    I1, I2, theta = sp.symbols('I1 I2 theta')

    # Define the system of equations symbolically
    eq1 = 2 * n11 * (I1 ** 2) + 2 * k * n10 * I1 + 2 * sp.sqrt(I1 * I2) * sp.sin(theta)
    eq2 = 2 * n20 * I2 - 2 * k * sp.sqrt(I1 * I2) * sp.sin(theta)
    eq3 = (1 - w2) + k * (sp.sqrt(I2 / I1) - sp.sqrt(I1 / I2)) * sp.cos(theta)

    # Create the system vector
    system = sp.Matrix([eq1, eq2, eq3])

    # Define the vector of variables
    variables = sp.Matrix([I1, I2, theta])

    # Compute the Jacobian matrix symbolically
    jacobian_matrix = system.jacobian(variables)

    # Simplify the Jacobian matrix (optional)
    jacobian_matrix_simplified = sp.simplify(jacobian_matrix)

    return jacobian_matrix_simplified, variables


def find_fixed_points(p):
    w2, k, n11, n10, n20 = p
    # Define the objective function as the sum of squared residuals
    def objective(x, grad):
        I1, I2, theta = x
        eq1 = 2 * n11 * (I1 ** 2) + 2 * k * n10 * I1 + 2 * np.sqrt(I1 * I2) * np.sin(theta)
        eq2 = 2 * n20 * I2 - 2 * k * np.sqrt(I1 * I2) * np.sin(theta)
        eq3 = (1 - w2) + k * (np.sqrt(I2 / I1) - np.sqrt(I1 / I2)) * np.cos(theta)
        return eq1 ** 2 + eq2 ** 2 + eq3 ** 2

    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)  # Using Nelder-Mead algorithm
    opt.set_min_objective(objective)
    opt.set_xtol_rel(1e-8)

    # Set bounds for I1 and I2 (theta is unbounded)
    lower_bounds = [0.1, 0.1, 0]  # Small positive numbers for I1 and I2
    upper_bounds = [np.inf, np.inf, 2 * np.pi]

    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)

    # Generate random initial conditions
    num_random_conditions = 10000
    initial_conditions = np.random.uniform(
        low=[0.1, 0.1, 0],
        high=[10.0, 10.0, 2 * np.pi],
        size=(num_random_conditions, 3)
    )

    results = []

    for x0 in initial_conditions:
        x = opt.optimize(x0)
        minf = opt.last_optimum_value()
        results.append((x, minf))

    # Combine solutions that are close and filter by a threshold
    tolerance = 1e-2
    min_value_threshold = 1e-4  # Define the threshold for minimum values
    combined_results = []

    for x, minf in results:
        # Check if this result is close to an already found solution
        found_close = False
        if minf > min_value_threshold:
            continue  # Skip solutions that exceed the minimum value threshold
        for combined_x, combined_minf in combined_results:
            if np.linalg.norm(np.array(x) - np.array(combined_x)) < tolerance:
                found_close = True
                break
        if not found_close:
            combined_results.append((x, minf))

    return combined_results

def jacobian_eigenvalues(p):
    w2, k, n11, n10, n20 = p
    # Compute the symbolic Jacobian matrix
    jacobian_matrix, variables = symbolic_jacobian(w2, k, n11, n10, n20)
    
    # Find the fixed points
    fixed_points = find_fixed_points(w2, k, n11, n10, n20)
    eigenvalues_list = []

    for x, _ in fixed_points:
        # Substitute the fixed point values into the Jacobian matrix
        subs = {variables[i]: x[i] for i in range(len(variables))}
        jacobian_numeric = jacobian_matrix.subs(subs).evalf()

        # Convert the symbolic Jacobian to a numerical numpy array
        jacobian_numeric_array = np.array(jacobian_numeric.tolist(), dtype=np.float64)

        # Compute the eigenvalues of the numerical Jacobian matrix
        eigenvalues = np.linalg.eigvals(jacobian_numeric_array)
        eigenvalues_list.append(eigenvalues)

    return eigenvalues_list

def stability_constraint(p):
    w2, k, n11, n10, n20 = p
    eigenvalues = jacobian_eigenvalues(w2, k, n11, n10, n20)
    eig_list = []
    eig_list.extend(eigenvalues)
    min_eigenvalue = min(eig_list, key=lambda x: x.real)
    return min_eigenvalue

# Example usage
eigenvalues = jacobian_eigenvalues(n11=-0.5, n10=2.88, n20=0.4, w2=1.4, k=1)

for i, eig in enumerate(eigenvalues):
    print(f"Eigenvalues for fixed point {i + 1}: {eig}")
    print("-" * 50)
