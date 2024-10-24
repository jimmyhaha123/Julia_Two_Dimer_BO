import sympy as sp

# Define the symbolic variables
I1, I2, I3, I4, theta1, theta2, theta3 = sp.symbols('I1 I2 I3 I4 theta1 theta2 theta3')
an11, an10, an20, bn11, bn10, bn20, v, k, w2, w3, w4 = sp.symbols('an11 an10 an20 bn11 bn10 bn20 v k w2 w3 w4')

# Define the equations
eq1 = 2 * (an11 * I1 + an10) * I1 + 2 * sp.sqrt(I1 * I2) * sp.sin(theta1)
eq2 = 2 * an20 * I2 - 2 * sp.sqrt(I1 * I2) * sp.sin(theta1) + 2 * v * sp.sqrt(I2 * I3) * sp.sin(theta2)
eq3 = 2 * (bn11 * I3 + bn10) * I3 - 2 * v * sp.sqrt(I2 * I3) * sp.sin(theta2) + 2 * k * sp.sqrt(I3 * I4) * sp.sin(theta3)
eq4 = 2 * bn20 * I4 - 2 * k * sp.sqrt(I3 * I4) * sp.sin(theta3)
eq5 = (1 - w2) + (sp.sqrt(I2 / I1) - sp.sqrt(I1 / I2)) * sp.cos(theta1) - v * (sp.sqrt(I3 / I2)) * sp.cos(theta2)
eq6 = (w2 - w3) + (v * (sp.sqrt(I3 / I2) - sp.sqrt(I2 / I3))) * sp.cos(theta2) + (sp.sqrt(I1 / I2)) * sp.cos(theta1) - k * (sp.sqrt(I4 / I3)) * sp.cos(theta3)
eq7 = (w3 - w4) + (k * (sp.sqrt(I4 / I3) - sp.sqrt(I3 / I4))) * sp.cos(theta3) + v * (sp.sqrt(I2 / I3)) * sp.cos(theta2)

# Define the objective function as the sum of squared equations
objective = eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2 + eq7**2

# Compute the gradient (partial derivatives with respect to I1, I2, I3, I4, theta1, theta2, theta3)
grad_I1 = sp.diff(objective, I1)
grad_I2 = sp.diff(objective, I2)
grad_I3 = sp.diff(objective, I3)
grad_I4 = sp.diff(objective, I4)
grad_theta1 = sp.diff(objective, theta1)
grad_theta2 = sp.diff(objective, theta2)
grad_theta3 = sp.diff(objective, theta3)

# Display the results
print("Gradient with respect to I1:")
print(grad_I1)

print("\nGradient with respect to I2:")
print(grad_I2)

print("\nGradient with respect to I3:")
print(grad_I3)

print("\nGradient with respect to I4:")
print(grad_I4)

print("\nGradient with respect to theta1:")
print(grad_theta1)

print("\nGradient with respect to theta2:")
print(grad_theta2)

print("\nGradient with respect to theta3:")
print(grad_theta3)