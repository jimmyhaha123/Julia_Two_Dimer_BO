import sympy as sp

def compute_jacobian_dimer1():
    # Define symbols for parameters
    w2, k, n11, n10, n20 = sp.symbols('w2 k n11 n10 n20')
    I1, I2, theta = sp.symbols('I1 I2 theta')

    # Define the equations
    eq1 = 2 * n11 * I1**2 + 2 * k * n10 * I1 + 2 * sp.sqrt(I1 * I2) * sp.sin(theta)
    eq2 = 2 * n20 * I2 - 2 * k * sp.sqrt(I1 * I2) * sp.sin(theta)
    eq3 = (1 - w2) + k * (sp.sqrt(I2 / I1) - sp.sqrt(I1 / I2)) * sp.cos(theta)

    # Construct the system and compute the Jacobian
    system = sp.Matrix([eq1, eq2, eq3])
    variables = sp.Matrix([I1, I2, theta])
    jacobian_matrix = system.jacobian(variables)

    return jacobian_matrix

def compute_jacobian_dimer2():
    # Define symbols for parameters
    w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, v = sp.symbols('w2 w3 w4 k an11 an10 an20 bn11 bn10 bn20 v')
    I1, I2, I3, I4, theta1, theta2, theta3 = sp.symbols('I1 I2 I3 I4 theta1 theta2 theta3')

    # Define the equations
    eq1 = 2 * (an11 * I1 + an10) * I1 + 2 * sp.sqrt(I1 * I2) * sp.sin(theta1)
    eq2 = 2 * an20 * I2 - 2 * sp.sqrt(I1 * I2) * sp.sin(theta1) + 2 * v * sp.sqrt(I2 * I3) * sp.sin(theta2)
    eq3 = 2 * (bn11 * I3 + bn10) * I3 - 2 * v * sp.sqrt(I2 * I3) * sp.sin(theta2) + 2 * k * sp.sqrt(I3 * I4) * sp.sin(theta3)
    eq4 = 2 * bn20 * I4 - 2 * k * sp.sqrt(I3 * I4) * sp.sin(theta3)
    eq5 = (1 - w2) + (sp.sqrt(I2 / I1) - sp.sqrt(I1 / I2)) * sp.cos(theta1) - v * (sp.sqrt(I3 / I2)) * sp.cos(theta2)
    eq6 = (w2 - w3) + (v * (sp.sqrt(I3 / I2) - sp.sqrt(I2 / I3))) * sp.cos(theta2) + (sp.sqrt(I1 / I2)) * sp.cos(theta1) - k * (sp.sqrt(I4 / I3)) * sp.cos(theta3)
    eq7 = (w3 - w4) + (k * (sp.sqrt(I4 / I3) - sp.sqrt(I3 / I4))) * sp.cos(theta3) + v * (sp.sqrt(I2 / I3)) * sp.cos(theta2)

    # Construct the system and compute the Jacobian
    system = sp.Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
    variables = sp.Matrix([I1, I2, I3, I4, theta1, theta2, theta3])
    jacobian_matrix = system.jacobian(variables)

    return jacobian_matrix

# Print the Jacobians
jacobian_dimer1 = compute_jacobian_dimer1()
print("Jacobian Matrix for dimer == 1:")
print(jacobian_dimer1)

jacobian_dimer2 = compute_jacobian_dimer2()
print("\nJacobian Matrix for dimer == 2:")
print(jacobian_dimer2)