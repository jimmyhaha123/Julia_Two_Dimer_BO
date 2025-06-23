using SymPy
using NLopt
using LinearAlgebra
using Random

function symbolic_jacobian(p, state_variables, dimer=1)
    if dimer == 1
        w2, k, n11, n10, n20 = p
        I1, I2, theta = state_variables

        # Numerical Jacobian for dimer == 1
        jacobian_matrix = [
            4*I1*n11 + 2*k*n10 + sqrt(I1*I2)*sin(theta)/I1                sqrt(I1*I2)*sin(theta)/I2                 2*sqrt(I1*I2)*cos(theta);
           -k*sqrt(I1*I2)*sin(theta)/I1                                   2*n20 - k*sqrt(I1*I2)*sin(theta)/I2      -2*k*sqrt(I1*I2)*cos(theta);
            k*(-sqrt(I2/I1)/(2*I1) - sqrt(I1/I2)/(2*I1))*cos(theta)       k*(sqrt(I2/I1)/(2*I2) + sqrt(I1/I2)/(2*I2))*cos(theta)   -k*(sqrt(I2/I1) - sqrt(I1/I2))*sin(theta)
        ]

    elseif dimer == 2
        w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, v = p
        I1, I2, I3, I4, theta1, theta2, theta3 = state_variables

        # Numerical Jacobian for dimer == 2
        jacobian_matrix = [
            4*I1*an11 + 2*an10 + sqrt(I1*I2)*sin(theta1)/I1                  sqrt(I1*I2)*sin(theta1)/I2                   0                                       0                           2*sqrt(I1*I2)*cos(theta1)   0                             0;
           -sqrt(I1*I2)*sin(theta1)/I1                                       2*an20 + v*sqrt(I2*I3)*sin(theta2)/I2 - sqrt(I1*I2)*sin(theta1)/I2   v*sqrt(I2*I3)*sin(theta2)/I3   0                          -2*sqrt(I1*I2)*cos(theta1)   2*v*sqrt(I2*I3)*cos(theta2)   0;
            0                                                                 -v*sqrt(I2*I3)*sin(theta2)/I2               4*I3*bn11 + 2*bn10 + k*sqrt(I3*I4)*sin(theta3)/I3 - v*sqrt(I2*I3)*sin(theta2)/I3 k*sqrt(I3*I4)*sin(theta3)/I4   0                             -2*v*sqrt(I2*I3)*cos(theta2)   2*k*sqrt(I3*I4)*cos(theta3);
            0                                                                 0                                           -k*sqrt(I3*I4)*sin(theta3)/I3        2*bn20 - k*sqrt(I3*I4)*sin(theta3)/I4   0  0 -2*k*sqrt(I3*I4)*cos(theta3);
            (-sqrt(I2/I1)/(2*I1) - sqrt(I1/I2)/(2*I1))*cos(theta1)           (sqrt(I2/I1)/(2*I2) + sqrt(I1/I2)/(2*I2))*cos(theta1) + v*sqrt(I3/I2)*cos(theta2)/(2*I2)  -v*sqrt(I3/I2)*cos(theta2)/(2*I3)   0   -(sqrt(I2/I1) - sqrt(I1/I2))*sin(theta1)   v*sqrt(I3/I2)*sin(theta2)  0;
            sqrt(I1/I2)*cos(theta1)/(2*I1)                                   v*(-sqrt(I3/I2)/(2*I2) - sqrt(I2/I3)/(2*I2))*cos(theta2) - sqrt(I1/I2)*cos(theta1)/(2*I2) v*(sqrt(I3/I2)/(2*I3) + sqrt(I2/I3)/(2*I3))*cos(theta2) + k*sqrt(I4/I3)*cos(theta3)/(2*I3) -k*sqrt(I4/I3)*cos(theta3)/(2*I4)  -sqrt(I1/I2)*sin(theta1)  -v*(sqrt(I3/I2) - sqrt(I2/I3))*sin(theta2)  k*sqrt(I4/I3)*sin(theta3);
            0                                                                 v*sqrt(I2/I3)*cos(theta2)/(2*I2)           k*(-sqrt(I4/I3)/(2*I3) - sqrt(I3/I4)/(2*I3))*cos(theta3) - v*sqrt(I2/I3)*cos(theta2)/(2*I3) k*(sqrt(I4/I3)/(2*I4) + sqrt(I3/I4)/(2*I4))*cos(theta3)  0 -v*sqrt(I2/I3)*sin(theta2) -k*(sqrt(I4/I3) - sqrt(I3/I4))*sin(theta3)
        ]
    end

    return jacobian_matrix
end

function find_fixed_points(p, dimer=1, sim_method="cmt")
    if dimer == 1
        w2, k, n11, n10, n20 = p

        # Objective function for minimization
        function objective_d1(x::Vector, grad::Vector)
            I1, I2, theta = x
            eq1 = 2 * n11 * I1^2 + 2 * k * n10 * I1 + 2 * sqrt(I1 * I2) * sin(theta)
            eq2 = 2 * n20 * I2 - 2 * k * sqrt(I1 * I2) * sin(theta)
            eq3 = (1 - w2) + k * (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta)
            result = eq1^2 + eq2^2 + eq3^2
            return result
        end

        opt = Opt(:LN_NELDERMEAD, 3)
        lower_bounds!(opt, [0.1, 0.1, 0])
        upper_bounds!(opt, [10, 10, 2 * π])
        xtol_rel!(opt, 1e-8)
        min_objective!(opt, objective_d1)

        num_random_conditions = 2000
        initial_conditions = [rand(3) .* [9.9, 9.9, 2 * π] .+ [0.1, 0.1, 0] for i in 1:num_random_conditions]

        results = []
        for x0 in initial_conditions
            minf, xopt = optimize(opt, x0)
            push!(results, (xopt, minf))
        end

        tolerance = 1e-2
        min_value_threshold = 1e-4

        combined_results = []
        for (x, minf) in results
            if minf > min_value_threshold
                continue
            end

            found_close = any(norm(x - x_exist[1]) < tolerance for x_exist in combined_results)
            if !found_close
                push!(combined_results, (x, minf))
            end
        end

        return combined_results
    elseif dimer == 2
        # The dimer 2 case is handled similarly but with 7 variables
        w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, v = p

        function objective_d2(x, grad_out)
            # Unpack parameters
            w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, v = p
            # Unpack state variables
            I1, I2, I3, I4, theta1, theta2, theta3 = x
        
            # Define equations
            eq1 = 2 * (an11 * I1 + an10) * I1 + 2 * sqrt(I1 * I2) * sin(theta1)
            eq2 = 2 * an20 * I2 - 2 * sqrt(I1 * I2) * sin(theta1) + 2 * v * sqrt(I2 * I3) * sin(theta2)
            eq3 = 2 * (bn11 * I3 + bn10) * I3 - 2 * v * sqrt(I2 * I3) * sin(theta2) + 2 * k * sqrt(I3 * I4) * sin(theta3)
            eq4 = 2 * bn20 * I4 - 2 * k * sqrt(I3 * I4) * sin(theta3)
            eq5 = (1 - w2) + (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta1) - v * sqrt(I3 / I2) * cos(theta2)
            eq6 = (w2 - w3) + (v * (sqrt(I3 / I2) - sqrt(I2 / I3))) * cos(theta2) + (sqrt(I1 / I2)) * cos(theta1) - k * sqrt(I4 / I3) * cos(theta3)
            eq7 = (w3 - w4) + (k * (sqrt(I4 / I3) - sqrt(I3 / I4))) * cos(theta3) + v * sqrt(I2 / I3) * cos(theta2)
        
            # Sum of squared errors (objective function)
            result = eq1^2 + eq2^2 + eq3^2 + eq4^2 + eq5^2 + eq6^2 + eq7^2
        
            # Compute gradients if requested
            if !isempty(grad_out)
                grad_out[1] = 2 * (-sqrt(I2 / I1) / (2 * I1) - sqrt(I1 / I2) / (2 * I1)) * (-v * sqrt(I3 / I2) * cos(theta2) - w2 + (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta1) + 1) * cos(theta1) + (I1 * (2 * I1 * an11 + 2 * an10) + 2 * sqrt(I1 * I2) * sin(theta1)) * (8 * I1 * an11 + 4 * an10 + 2 * sqrt(I1 * I2) * sin(theta1) / I1) + sqrt(I1 / I2) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) * cos(theta1) / I1 - 2 * sqrt(I1 * I2) * (2 * I2 * an20 + 2 * v * sqrt(I2 * I3) * sin(theta2) - 2 * sqrt(I1 * I2) * sin(theta1)) * sin(theta1) / I1
        
                grad_out[2] = (2 * (sqrt(I2 / I1) / (2 * I2) + sqrt(I1 / I2) / (2 * I2)) * cos(theta1) + v * sqrt(I3 / I2) * cos(theta2) / I2) * (-v * sqrt(I3 / I2) * cos(theta2) - w2 + (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta1) + 1) + (2 * v * (-sqrt(I3 / I2) / (2 * I2) - sqrt(I2 / I3) / (2 * I2)) * cos(theta2) - sqrt(I1 / I2) * cos(theta1) / I2) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) + (4 * an20 + 2 * v * sqrt(I2 * I3) * sin(theta2) / I2 - 2 * sqrt(I1 * I2) * sin(theta1) / I2) * (2 * I2 * an20 + 2 * v * sqrt(I2 * I3) * sin(theta2) - 2 * sqrt(I1 * I2) * sin(theta1)) + v * sqrt(I2 / I3) * (k * (sqrt(I4 / I3) - sqrt(I3 / I4)) * cos(theta3) + v * sqrt(I2 / I3) * cos(theta2) + w3 - w4) * cos(theta2) / I2 - 2 * v * sqrt(I2 * I3) * (I3 * (2 * I3 * bn11 + 2 * bn10) + 2 * k * sqrt(I3 * I4) * sin(theta3) - 2 * v * sqrt(I2 * I3) * sin(theta2)) * sin(theta2) / I2 + 2 * sqrt(I1 * I2) * (I1 * (2 * I1 * an11 + 2 * an10) + 2 * sqrt(I1 * I2) * sin(theta1)) * sin(theta1) / I2
        
                grad_out[3] = (2 * k * (-sqrt(I4 / I3) / (2 * I3) - sqrt(I3 / I4) / (2 * I3)) * cos(theta3) - v * sqrt(I2 / I3) * cos(theta2) / I3) * (k * (sqrt(I4 / I3) - sqrt(I3 / I4)) * cos(theta3) + v * sqrt(I2 / I3) * cos(theta2) + w3 - w4) + (2 * v * (sqrt(I3 / I2) / (2 * I3) + sqrt(I2 / I3) / (2 * I3)) * cos(theta2) + k * sqrt(I4 / I3) * cos(theta3) / I3) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) + (I3 * (2 * I3 * bn11 + 2 * bn10) + 2 * k * sqrt(I3 * I4) * sin(theta3) - 2 * v * sqrt(I2 * I3) * sin(theta2)) * (8 * I3 * bn11 + 4 * bn10 + 2 * k * sqrt(I3 * I4) * sin(theta3) / I3 - 2 * v * sqrt(I2 * I3) * sin(theta2) / I3) - 2 * k * sqrt(I3 * I4) * (2 * I4 * bn20 - 2 * k * sqrt(I3 * I4) * sin(theta3)) * sin(theta3) / I3 - v * sqrt(I3 / I2) * (-v * sqrt(I3 / I2) * cos(theta2) - w2 + (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta1) + 1) * cos(theta2) / I3 + 2 * v * sqrt(I2 * I3) * (2 * I2 * an20 + 2 * v * sqrt(I2 * I3) * sin(theta2) - 2 * sqrt(I1 * I2) * sin(theta1)) * sin(theta2) / I3
        
                grad_out[4] = (2 * k * (sqrt(I4 / I3) / (2 * I4) + sqrt(I3 / I4) / (2 * I4)) * (k * (sqrt(I4 / I3) - sqrt(I3 / I4)) * cos(theta3) + v * sqrt(I2 / I3) * cos(theta2) + w3 - w4) * cos(theta3) + (4 * bn20 - 2 * k * sqrt(I3 * I4) * sin(theta3) / I4) * (2 * I4 * bn20 - 2 * k * sqrt(I3 * I4) * sin(theta3)) - k * sqrt(I4 / I3) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) * cos(theta3) / I4 + 2 * k * sqrt(I3 * I4) * (I3 * (2 * I3 * bn11 + 2 * bn10) + 2 * k * sqrt(I3 * I4) * sin(theta3) - 2 * v * sqrt(I2 * I3) * sin(theta2)) * sin(theta3) / I4)
        
                grad_out[5] = (-2 * sqrt(I1 / I2) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) * sin(theta1) + 4 * sqrt(I1 * I2) * (I1 * (2 * I1 * an11 + 2 * an10) + 2 * sqrt(I1 * I2) * sin(theta1)) * cos(theta1) - 4 * sqrt(I1 * I2) * (2 * I2 * an20 + 2 * v * sqrt(I2 * I3) * sin(theta2) - 2 * sqrt(I1 * I2) * sin(theta1)) * cos(theta1) - 2 * (sqrt(I2 / I1) - sqrt(I1 / I2)) * (-v * sqrt(I3 / I2) * cos(theta2) - w2 + (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta1) + 1) * sin(theta1))
        
                grad_out[6] = (2 * v * sqrt(I3 / I2) * (-v * sqrt(I3 / I2) * cos(theta2) - w2 + (sqrt(I2 / I1) - sqrt(I1 / I2)) * cos(theta1) + 1) * sin(theta2) - 2 * v * sqrt(I2 / I3) * (k * (sqrt(I4 / I3) - sqrt(I3 / I4)) * cos(theta3) + v * sqrt(I2 / I3) * cos(theta2) + w3 - w4) * sin(theta2) + 4 * v * sqrt(I2 * I3) * (2 * I2 * an20 + 2 * v * sqrt(I2 * I3) * sin(theta2) - 2 * sqrt(I1 * I2) * sin(theta1)) * cos(theta2) - 4 * v * sqrt(I2 * I3) * (I3 * (2 * I3 * bn11 + 2 * bn10) + 2 * k * sqrt(I3 * I4) * sin(theta3) - 2 * v * sqrt(I2 * I3) * sin(theta2)) * cos(theta2) - 2 * v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) * sin(theta2))
        
                grad_out[7] = (2 * k * sqrt(I4 / I3) * (-k * sqrt(I4 / I3) * cos(theta3) + v * (sqrt(I3 / I2) - sqrt(I2 / I3)) * cos(theta2) + w2 - w3 + sqrt(I1 / I2) * cos(theta1)) * sin(theta3) - 4 * k * sqrt(I3 * I4) * (2 * I4 * bn20 - 2 * k * sqrt(I3 * I4) * sin(theta3)) * cos(theta3) + 4 * k * sqrt(I3 * I4) * (I3 * (2 * I3 * bn11 + 2 * bn10) + 2 * k * sqrt(I3 * I4) * sin(theta3) - 2 * v * sqrt(I2 * I3) * sin(theta2)) * cos(theta3) - 2 * k * (sqrt(I4 / I3) - sqrt(I3 / I4)) * (k * (sqrt(I4 / I3) - sqrt(I3 / I4)) * cos(theta3) + v * sqrt(I2 / I3) * cos(theta2) + w3 - w4) * sin(theta3))
            end
        
            return result
        end

        opt = Opt(:LD_LBFGS, 7)
        lower_bounds!(opt, [0.1, 0.1, 0.1, 0.1, 0, 0, 0])
        upper_bounds!(opt, [Inf, Inf, Inf, Inf, 2 * π, 2 * π, 2 * π])
        xtol_rel!(opt, 1e-8)
        min_objective!(opt, objective_d2)

        num_random_conditions = 2000
        initial_conditions = [rand(7) .* [9.9, 9.9, 9.9, 9.9, 2 * π, 2 * π, 2 * π] .+ [0.1, 0.1, 0.1, 0.1, 0, 0, 0] for i in 1:num_random_conditions]

        results = []
        for x0 in initial_conditions
            minf, xopt = optimize(opt, x0)
            push!(results, (xopt, minf))
        end

        tolerance = 1e-2
        min_value_threshold = 1e-4

        combined_results = []
        for (x, minf) in results
            if minf > min_value_threshold
                continue
            end

            found_close = any(norm(x - x_exist[1]) < tolerance for x_exist in combined_results)
            if !found_close
                push!(combined_results, (x, minf))
            end
        end

        return combined_results
    end
end

function jacobian_eigenvalues(p, dimer=1, sim_method="cmt")
    # Find the fixed points
    fixed_points = find_fixed_points(p, dimer)
    # println("Fixed Points:")
    # println(fixed_points)

    eigenvalues_list = []

    # Iterate over each fixed point
    for (fixed_point, _) in fixed_points
        # println("\nEvaluating at Fixed Point:", fixed_point)

        # Call symbolic_jacobian with `p` and the current fixed point (state variables)
        jacobian_matrix = symbolic_jacobian(p, fixed_point, dimer)
        # println("Numerical Jacobian Matrix:\n", jacobian_matrix)

        # Compute the eigenvalues of the numerical Jacobian matrix
        eigenvalues = eigvals(jacobian_matrix)
        # println("Eigenvalues:", eigenvalues)

        # Store the eigenvalues for each fixed point
        push!(eigenvalues_list, eigenvalues)
    end

    return eigenvalues_list, fixed_points
end

function stability_constraint(p, dimer=1, sim_method="cmt", multiple_eig=false)
    function convert_cartesian(x)
        if dimer == 2
            I1, I2, I3, I4, theta1, theta2, theta3 = x
            # Calculate the phases
            phi1 = 0
            phi2 = phi1 - theta1
            phi3 = phi2 - theta2
            phi4 = phi3 - theta3
            # Convert to complex Cartesian coordinates
            a1 = sqrt(I1) * exp(im * phi1)
            a2 = sqrt(I2) * exp(im * phi2)
            b1 = sqrt(I3) * exp(im * phi3)
            b2 = sqrt(I4) * exp(im * phi4)
            return a1, a2, b1, b2
        elseif dimer == 1
            I1, I2, theta = x
            # Define initial phases
            phi1 = 0
            phi2 = theta
            # Convert to complex Cartesian coordinates
            a1 = sqrt(I1) * exp(im * phi1)
            a2 = sqrt(I2) * exp(im * phi2)
            return a1, a2
        else
            error("Invalid dimer value; only dimer == 1 or dimer == 2 are supported.")
        end
    end

    # Compute the Jacobian eigenvalues for all fixed points
    eigenvalues_list, fixed_points = jacobian_eigenvalues(p, dimer, sim_method)
    # println("Eigenvalues List:", eigenvalues_list)

    if eigenvalues_list == []
        if multiple_eig
            return 10
        else
            return 1.0
        end
        # return 1.0, convert_cartesian([0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand()])
    end
    # Flatten the eigenvalues to locate the maximum real part
    flattened_eigenvalues = vcat(eigenvalues_list...)
    # println("Flattened Eigenvalues:", flattened_eigenvalues)

    # Find the maximum real part of the eigenvalues
    if ~multiple_eig
        result = maximum(real(flattened_eigenvalues))
    else
        result = sum(real(eig) for eig in flattened_eigenvalues if real(eig) > 0)
    end

    # Locate the sublist that contains the max eigenvalue
    # sublist_index = findfirst(x -> max_eigenvalue in real.(x), eigenvalues_list)
    # corresponding_fixed_point = fixed_points[sublist_index][1]  # Get the fixed point for that sublist

    # println("Max Eigenvalue:", max_eigenvalue)
    # println("Corresponding Fixed Point:", corresponding_fixed_point)

    return result
    # return result, convert_cartesian(corresponding_fixed_point)
end

function get_max_fixed_point(p, dimer=2, sim_method="cmt")
    function convert_cartesian(x)
        if dimer == 2
            I1, I2, I3, I4, theta1, theta2, theta3 = x
            # Calculate the phases
            phi1 = 0
            phi2 = phi1 - theta1
            phi3 = phi2 - theta2
            phi4 = phi3 - theta3
            # Convert to complex Cartesian coordinates
            a1 = sqrt(I1) * exp(im * phi1)
            a2 = sqrt(I2) * exp(im * phi2)
            b1 = sqrt(I3) * exp(im * phi3)
            b2 = sqrt(I4) * exp(im * phi4)
            return [a1, a2, b1, b2]
        elseif dimer == 1
            I1, I2, theta = x
            # Define initial phases
            phi1 = 0
            phi2 = theta
            # Convert to complex Cartesian coordinates
            a1 = sqrt(I1) * exp(im * phi1)
            a2 = sqrt(I2) * exp(im * phi2)
            return [a1, a2]
        else
            error("Invalid dimer value; only dimer == 1 or dimer == 2 are supported.")
        end
    end

    # Compute the Jacobian eigenvalues for all fixed points
    eigenvalues_list, fixed_points = jacobian_eigenvalues(p, dimer, sim_method)
    # println("Eigenvalues List:", eigenvalues_list)

    if eigenvalues_list == []
        if dimer == 2
            return [0.1 + 0.1*im, 0.1 + 0.1*im, 0.1 + 0.1*im, 0.1 + 0.1*im]
        else
            return [0.1 + 0.1*im, 0.1 + 0.1*im]
        end
        # return 1.0, convert_cartesian([0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand(), 0.1*rand()])
    end
    # Flatten the eigenvalues to locate the maximum real part
    flattened_eigenvalues = vcat(eigenvalues_list...)
    max_eigenvalue = maximum(real(flattened_eigenvalues))
    sublist_index = findfirst(x -> max_eigenvalue in real.(x), eigenvalues_list)
    corresponding_fixed_point = fixed_points[sublist_index][1]  # Get the fixed point for that sublist
    result = convert_cartesian(corresponding_fixed_point)
    rounded_result = [complex(round(real(x), digits=3), round(imag(x), digits=3)) for x in result]
    return rounded_result
end

# Example usage
# p = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 1.5, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711]
# println(jacobian_eigenvalues(p, 2)[1])


if abspath(PROGRAM_FILE) == @__FILE__
    func = ARGS[1]
    x = [parse(Float64, ARGS[i]) for i in 1:length(ARGS)]

    local result = nothing

    old_stdout = stdout
    old_stderr = stderr
    try
        redirect_stdout(devnull)
        redirect_stderr(devnull)
        result = stability_constraint(x, 2)
    finally
        # Restore stdout and stderr
        redirect_stdout(old_stdout)
        redirect_stderr(old_stderr)
    end

    println(result)
end