import sympy as sp
import numpy as np
import nlopt
from autograd import grad
import autograd.numpy as anp  # autograd-compatible numpy

def symbolic_jacobian(p, dimer=1, sim_method='cmt'):
    if dimer == 1:
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
    elif dimer == 2:
        w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, v = p  # Unpacking variables
        I1, I2, I3, I4, theta1, theta2, theta3 = sp.symbols('I1 I2 I3 I4 theta1 theta2 theta3')

        eq1 = 2 * (an11 * I1 + an10) * I1 + 2 * sp.sqrt(I1 * I2) * sp.sin(theta1)
        eq2 = 2 * an20 * I2 - 2 * sp.sqrt(I1 * I2) * sp.sin(theta1) + 2 * v * sp.sqrt(I2 * I3) * sp.sin(theta2)
        eq3 = 2 * (bn11 * I3 + bn10) * I3 - 2 * v * sp.sqrt(I2 * I3) * sp.sin(theta2) + 2 * k * sp.sqrt(I3 * I4) * sp.sin(theta3)
        eq4 = 2 * bn20 * I4 - 2 * k * sp.sqrt(I3 * I4) * sp.sin(theta3)
        eq5 = (1 - w2) + (sp.sqrt(I2 / I1) - sp.sqrt(I1 / I2)) * sp.cos(theta1) - v * (sp.sqrt(I3 / I2)) * sp.cos(theta2)
        eq6 = (w2 - w3) + (v * (sp.sqrt(I3 / I2) - sp.sqrt(I2 / I3))) * sp.cos(theta2) + (sp.sqrt(I1 / I2)) * sp.cos(theta1) - k * (sp.sqrt(I4 / I3)) * sp.cos(theta3)
        eq7 = (w3 - w4) + (k * (sp.sqrt(I4 / I3) - sp.sqrt(I3 / I4))) * sp.cos(theta3) + v * (sp.sqrt(I2 / I3)) * sp.cos(theta2)

        system = sp.Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
        variables = sp.Matrix([I1, I2, I3, I4, theta1, theta2, theta3])
        jacobian_matrix = system.jacobian(variables)
        jacobian_matrix_simplified = sp.simplify(jacobian_matrix)

    return jacobian_matrix_simplified, variables

def find_fixed_points(p, dimer=1, sim_method='cmt'):
    if dimer == 1:
        w2, k, n11, n10, n20 = p
        # Define the objective function as the sum of squared residuals
        def objective(x, grad):
            I1, I2, theta = x
            eq1 = 2 * n11 * (I1 ** 2) + 2 * k * n10 * I1 + 2 * np.sqrt(I1 * I2) * np.sin(theta)
            eq2 = 2 * n20 * I2 - 2 * k * np.sqrt(I1 * I2) * np.sin(theta)
            eq3 = (1 - w2) + k * (np.sqrt(I2 / I1) - np.sqrt(I1 / I2)) * np.cos(theta)
            result = eq1 ** 2 + eq2 ** 2 + eq3 ** 2
            # print(f"Curr :{result}")
            return result

        opt = nlopt.opt(nlopt.LN_NELDERMEAD, 3)  # Using Nelder-Mead algorithm
        opt.set_min_objective(objective)
        opt.set_xtol_rel(1e-8)

        # Set bounds for I1 and I2 (theta is unbounded)
        lower_bounds = [0.1, 0.1, 0]  # Small positive numbers for I1 and I2
        upper_bounds = [10, 10, 2 * np.pi]

        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)

        # Generate random initial conditions
        num_random_conditions = 2000
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
    elif dimer == 2:
        w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, v = p  # Unpacking variables
        # Define the objective function as the sum of squared residuals
        def objective(x, grad_out):
            I1, I2, I3, I4, theta1, theta2, theta3 = x
            
            # Define the equations for the objective
            eq1 = 2 * (an11 * I1 + an10) * I1 + 2 * np.sqrt(I1 * I2) * np.sin(theta1)
            eq2 = 2 * an20 * I2 - 2 * np.sqrt(I1 * I2) * np.sin(theta1) + 2 * v * np.sqrt(I2 * I3) * np.sin(theta2)
            eq3 = 2 * (bn11 * I3 + bn10) * I3 - 2 * v * np.sqrt(I2 * I3) * np.sin(theta2) + 2 * k * np.sqrt(I3 * I4) * np.sin(theta3)
            eq4 = 2 * bn20 * I4 - 2 * k * np.sqrt(I3 * I4) * np.sin(theta3)
            eq5 = (1 - w2) + (np.sqrt(I2 / I1) - np.sqrt(I1 / I2)) * np.cos(theta1) - v * (np.sqrt(I3 / I2)) * np.cos(theta2)
            eq6 = (w2 - w3) + (v * (np.sqrt(I3 / I2) - np.sqrt(I2 / I3))) * np.cos(theta2) + (np.sqrt(I1 / I2)) * np.cos(theta1) - k * (np.sqrt(I4 / I3)) * np.cos(theta3)
            eq7 = (w3 - w4) + (k * (np.sqrt(I4 / I3) - np.sqrt(I3 / I4))) * np.cos(theta3) + v * (np.sqrt(I2 / I3)) * np.cos(theta2)
            
            # Sum of squared errors (objective function)
            result = eq1 ** 2 + eq2 ** 2 + eq3 ** 2 + eq4 ** 2 + eq5 ** 2 + eq6 ** 2 + eq7 ** 2
            # print(result)

            # If gradient output is requested, provide the manually computed analytical gradients
            if grad_out.size > 0:
                # These are placeholders, you need to replace these with your actual analytical gradient expressions
                grad_out[0] = 2*(-np.sqrt(I2/I1)/(2*I1) - np.sqrt(I1/I2)/(2*I1))*(-v*np.sqrt(I3/I2)*np.cos(theta2) - w2 + (np.sqrt(I2/I1) - np.sqrt(I1/I2))*np.cos(theta1) + 1)*np.cos(theta1) + (I1*(2*I1*an11 + 2*an10) + 2*np.sqrt(I1*I2)*np.sin(theta1))*(8*I1*an11 + 4*an10 + 2*np.sqrt(I1*I2)*np.sin(theta1)/I1) + np.sqrt(I1/I2)*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1))*np.cos(theta1)/I1 - 2*np.sqrt(I1*I2)*(2*I2*an20 + 2*v*np.sqrt(I2*I3)*np.sin(theta2) - 2*np.sqrt(I1*I2)*np.sin(theta1))*np.sin(theta1)/I1  # Replace with your derivative of objective wrt I1
                
                
                grad_out[1] = (2*(np.sqrt(I2/I1)/(2*I2) + np.sqrt(I1/I2)/(2*I2))*np.cos(theta1) + v*np.sqrt(I3/I2)*np.cos(theta2)/I2)*(-v*np.sqrt(I3/I2)*np.cos(theta2) - w2 + (np.sqrt(I2/I1) - np.sqrt(I1/I2))*np.cos(theta1) + 1) + (2*v*(-np.sqrt(I3/I2)/(2*I2) - np.sqrt(I2/I3)/(2*I2))*np.cos(theta2) - np.sqrt(I1/I2)*np.cos(theta1)/I2)*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1)) + (4*an20 + 2*v*np.sqrt(I2*I3)*np.sin(theta2)/I2 - 2*np.sqrt(I1*I2)*np.sin(theta1)/I2)*(2*I2*an20 + 2*v*np.sqrt(I2*I3)*np.sin(theta2) - 2*np.sqrt(I1*I2)*np.sin(theta1)) + v*np.sqrt(I2/I3)*(k*(np.sqrt(I4/I3) - np.sqrt(I3/I4))*np.cos(theta3) + v*np.sqrt(I2/I3)*np.cos(theta2) + w3 - w4)*np.cos(theta2)/I2 - 2*v*np.sqrt(I2*I3)*(I3*(2*I3*bn11 + 2*bn10) + 2*k*np.sqrt(I3*I4)*np.sin(theta3) - 2*v*np.sqrt(I2*I3)*np.sin(theta2))*np.sin(theta2)/I2 + 2*np.sqrt(I1*I2)*(I1*(2*I1*an11 + 2*an10) + 2*np.sqrt(I1*I2)*np.sin(theta1))*np.sin(theta1)/I2  # Replace with your derivative of objective wrt I2
                
                
                grad_out[2] = (2*k*(-np.sqrt(I4/I3)/(2*I3) - np.sqrt(I3/I4)/(2*I3))*np.cos(theta3) - v*np.sqrt(I2/I3)*np.cos(theta2)/I3)*(k*(np.sqrt(I4/I3) - np.sqrt(I3/I4))*np.cos(theta3) + v*np.sqrt(I2/I3)*np.cos(theta2) + w3 - w4) + (2*v*(np.sqrt(I3/I2)/(2*I3) + np.sqrt(I2/I3)/(2*I3))*np.cos(theta2) + k*np.sqrt(I4/I3)*np.cos(theta3)/I3)*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1)) + (I3*(2*I3*bn11 + 2*bn10) + 2*k*np.sqrt(I3*I4)*np.sin(theta3) - 2*v*np.sqrt(I2*I3)*np.sin(theta2))*(8*I3*bn11 + 4*bn10 + 2*k*np.sqrt(I3*I4)*np.sin(theta3)/I3 - 2*v*np.sqrt(I2*I3)*np.sin(theta2)/I3) - 2*k*np.sqrt(I3*I4)*(2*I4*bn20 - 2*k*np.sqrt(I3*I4)*np.sin(theta3))*np.sin(theta3)/I3 - v*np.sqrt(I3/I2)*(-v*np.sqrt(I3/I2)*np.cos(theta2) - w2 + (np.sqrt(I2/I1) - np.sqrt(I1/I2))*np.cos(theta1) + 1)*np.cos(theta2)/I3 + 2*v*np.sqrt(I2*I3)*(2*I2*an20 + 2*v*np.sqrt(I2*I3)*np.sin(theta2) - 2*np.sqrt(I1*I2)*np.sin(theta1))*np.sin(theta2)/I3

                grad_out[3] = (2*k*(np.sqrt(I4/I3)/(2*I4) + np.sqrt(I3/I4)/(2*I4))*(k*(np.sqrt(I4/I3) - np.sqrt(I3/I4))*np.cos(theta3) + v*np.sqrt(I2/I3)*np.cos(theta2) + w3 - w4)*np.cos(theta3) + (4*bn20 - 2*k*np.sqrt(I3*I4)*np.sin(theta3)/I4)*(2*I4*bn20 - 2*k*np.sqrt(I3*I4)*np.sin(theta3)) - k*np.sqrt(I4/I3)*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1))*np.cos(theta3)/I4 + 2*k*np.sqrt(I3*I4)*(I3*(2*I3*bn11 + 2*bn10) + 2*k*np.sqrt(I3*I4)*np.sin(theta3) - 2*v*np.sqrt(I2*I3)*np.sin(theta2))*np.sin(theta3)/I4)  # Replace with your derivative of objective wrt I4

                grad_out[4] = (-2*np.sqrt(I1/I2)*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1))*np.sin(theta1) + 4*np.sqrt(I1*I2)*(I1*(2*I1*an11 + 2*an10) + 2*np.sqrt(I1*I2)*np.sin(theta1))*np.cos(theta1) - 4*np.sqrt(I1*I2)*(2*I2*an20 + 2*v*np.sqrt(I2*I3)*np.sin(theta2) - 2*np.sqrt(I1*I2)*np.sin(theta1))*np.cos(theta1) - 2*(np.sqrt(I2/I1) - np.sqrt(I1/I2))*(-v*np.sqrt(I3/I2)*np.cos(theta2) - w2 + (np.sqrt(I2/I1) - np.sqrt(I1/I2))*np.cos(theta1) + 1)*np.sin(theta1))  # Replace with your derivative of objective wrt theta1

                grad_out[5] = (2*v*np.sqrt(I3/I2)*(-v*np.sqrt(I3/I2)*np.cos(theta2) - w2 + (np.sqrt(I2/I1) - np.sqrt(I1/I2))*np.cos(theta1) + 1)*np.sin(theta2) - 2*v*np.sqrt(I2/I3)*(k*(np.sqrt(I4/I3) - np.sqrt(I3/I4))*np.cos(theta3) + v*np.sqrt(I2/I3)*np.cos(theta2) + w3 - w4)*np.sin(theta2) + 4*v*np.sqrt(I2*I3)*(2*I2*an20 + 2*v*np.sqrt(I2*I3)*np.sin(theta2) - 2*np.sqrt(I1*I2)*np.sin(theta1))*np.cos(theta2) - 4*v*np.sqrt(I2*I3)*(I3*(2*I3*bn11 + 2*bn10) + 2*k*np.sqrt(I3*I4)*np.sin(theta3) - 2*v*np.sqrt(I2*I3)*np.sin(theta2))*np.cos(theta2) - 2*v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1))*np.sin(theta2))

                grad_out[6] = (2*k*np.sqrt(I4/I3)*(-k*np.sqrt(I4/I3)*np.cos(theta3) + v*(np.sqrt(I3/I2) - np.sqrt(I2/I3))*np.cos(theta2) + w2 - w3 + np.sqrt(I1/I2)*np.cos(theta1))*np.sin(theta3) - 4*k*np.sqrt(I3*I4)*(2*I4*bn20 - 2*k*np.sqrt(I3*I4)*np.sin(theta3))*np.cos(theta3) + 4*k*np.sqrt(I3*I4)*(I3*(2*I3*bn11 + 2*bn10) + 2*k*np.sqrt(I3*I4)*np.sin(theta3) - 2*v*np.sqrt(I2*I3)*np.sin(theta2))*np.cos(theta3) - 2*k*(np.sqrt(I4/I3) - np.sqrt(I3/I4))*(k*(np.sqrt(I4/I3) - np.sqrt(I3/I4))*np.cos(theta3) + v*np.sqrt(I2/I3)*np.cos(theta2) + w3 - w4)*np.sin(theta3))  # Replace with your derivative of objective wrt theta3

            return result
        
    

        opt = nlopt.opt(nlopt.LD_LBFGS, 7)  # Using L-BFGS
        opt.set_min_objective(objective)
        opt.set_xtol_rel(1e-8)
        # opt.set_maxeval(500)

        # Set bounds for I1 and I2 (theta is unbounded)
        lower_bounds = [0.1] * 4 + [0] * 3  # Small positive numbers for I1 and I2
        upper_bounds = [np.inf] * 4 + [2 * np.pi] * 3

        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)

        # Generate random initial conditions
        num_random_conditions = 2000
        initial_conditions = np.random.uniform(
            low=lower_bounds,
            high=[10] * 4 + [2 * np.pi] * 3,
            size=(num_random_conditions, 7)
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

def jacobian_eigenvalues(p, dimer=1, sim_method='cmt'):
    # Compute the symbolic Jacobian matrix
    jacobian_matrix, variables = symbolic_jacobian(p, dimer=dimer)
    
    # Find the fixed points
    fixed_points = find_fixed_points(p, dimer=dimer)
    print(fixed_points)
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

    return eigenvalues_list, fixed_points

def stability_constraint(p, dimer=1, sim_method='cmt'):

    # p = [t.item() for t in p]
    temp, fixed_points = jacobian_eigenvalues(p, dimer=dimer)
    eigenvalues = [arr.tolist() for arr in temp]
    
    # Flatten the list of arrays and convert them to a single list of complex numbers
    flattened_eigenvalues = np.array(eigenvalues).flatten().tolist()
    print(eigenvalues)
    
    # Extract the eigenvalue with the largest real part
    max_eigenvalue = max(flattened_eigenvalues, key=lambda x: x.real)

    def find_original_index(nested_list, value):
        for i, sublist in enumerate(nested_list):
            if value in sublist:
                return (i, sublist.index(value))
            
    def random_ic():
        if dimer == 1: return
        elif dimer == 0: return 


    # Get the original index in the nested list
    max_idx = find_original_index(eigenvalues, max_eigenvalue)
    # print(f"Fixed point number: {max_idx[0]}")
    if max_eigenvalue == []:
        return 1.0, random_ic() # When no fixed points are found; assuem strong limit cycle
    
    return max_eigenvalue.real, fixed_points[max_idx[0]][0]  # Returns the max real eigenvalue and also corresponding fixed point


# Example usage
# print(stability_constraint([1, 1, -0.5, 2.88, 0.3]))

p = [1, 1, -0.5, 2.88, 0.3]
# p = [1, 1, 1, 1, -0.5, 2.88, 0.2, -0.5, 2.88, 0.2, 1]
p = [1, 1.1, 0.9, 1, -1, 0.5, 0.1, -1, 0.4, 0.1, 1]
print(stability_constraint(p, dimer=2))
