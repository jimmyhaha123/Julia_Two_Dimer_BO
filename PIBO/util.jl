# import Pkg; Pkg.add("DifferentialEquations"); Pkg.add("Plots"); Pkg.add("FFTW"); Pkg.add("Statistics"); Pkg.add("BayesianOptimization");
# Pkg.add("GaussianProcesses"); Pkg.add("Distributions"); Pkg.add("Peaks"); Pkg.add("Interpolations");Pkg.add("DSP");Pkg.add("ForwardDiff")
# import Pkg; Pkg.add("BenchmarkTools")
using DifferentialEquations, Plots, FFTW, Statistics, BayesianOptimization, GaussianProcesses, Distributions, Peaks, Interpolations, DSP, LinearAlgebra, ForwardDiff
using Base: redirect_stdout
using BenchmarkTools

num = 40
min = 0
replication = 1


function highest_peak_deviation(freqs, vals)
    min = num - 1
    if length(vals) < num + 1
        println("No enough peaks. ")
        return num - 1
    elseif vals[2] < 10^(-3) # First peak too small
        println("Second peak too small.")
        return num - 1
    else
        sub_peaks = vals[2:num+1]
        sub_peaks = sub_peaks / sub_peaks[1] # Normalization with respect to first peak
        cost = sum(abs.(sub_peaks .- 1))
        # println("Sucessful. Loss: " * string(cost))
        return cost
    end
end


function augmented_fft(x)
    # Get the length of the original signal

    # Replicate the signal 100 times
    x_extended = repeat(x, replication)

    # Perform FFT on the extended signal
    X = fft(x_extended)

    # Return the full FFT result
    return X

end


function sys!(du, u, p, t)
    # Parameters
    n11, n10, n20, w2 = p

    N1(a1) = n11 * (abs(a1))^2 + n10
    N2(a2) = n20


    a1, a2 = u
    du[1] = im*a1 + N1(a1)*a1 +im*a2
    du[2] = im*w2*a2 + N2(a2)*a2 +im*a1

end


function solve_sys(p)
    range = 0.5
    u0 = [0.1 + 0.1*im, 0.1 + 0.1*im]
    t = 100000.0
    tspan = (0.0, t)

    prob = ODEProblem(sys!, u0, tspan, p)
    abstol = 1e-8
    reltol = 1e-6
    sol = solve(prob, abstol=abstol, reltol=reltol)

    u0 = sol.u[end]
    tspan = (0, 20000)
    prob = ODEProblem(sys!, u0, tspan, p)
    abstol = 1e-10
    reltol = 1e-8
    sol = solve(prob, abstol=abstol, reltol=reltol)
    index = ceil(Int, length(sol) / 2)
    sol = sol[index:end]
    time_points = sol.t
    # println("Solving finished.")

    mean_time_step = 0.005
    t_start, t_end = time_points[1], time_points[end]
    t_interp = t_start:mean_time_step:t_end
    x_sol_1 = [real.(u[1]) for u in sol.u]
    x_sol_2 = [imag.(u[1]) for u in sol.u]
    x_sol_3 = [real.(u[2]) for u in sol.u]
    x_sol_4 = [imag.(u[2]) for u in sol.u]
    if length(x_sol_1) > 100
        interp_func_1 = linear_interpolation(time_points, x_sol_1)
        interp_func_2 = linear_interpolation(time_points, x_sol_2)
        interp_func_3 = linear_interpolation(time_points, x_sol_3)
        interp_func_4 = linear_interpolation(time_points, x_sol_4)
        x_sol_1 = [interp_func_1(t) for t in t_interp]
        x_sol_2 = [interp_func_2(t) for t in t_interp]
        x_sol_3 = [interp_func_3(t) for t in t_interp]
        x_sol_4 = [interp_func_4(t) for t in t_interp]
    end
    x = [x_sol_1, x_sol_2, x_sol_3, x_sol_4]
    return x, t_interp, mean_time_step
end


function repetition_check(x, t_interp, dimensionality=8)
    first_data_point = [x[j][1] for j in 1:dimensionality]
    epsilon = 0.04
    repeating_times = []
    repeating_indices = []

    for i in 1:length(t_interp)
        current_point = [x[j][i] for j in 1:dimensionality]
        if isclose(current_point, first_data_point, epsilon)
            push!(repeating_times, t_interp[i])
            push!(repeating_indices, i)
        end
    end

    repeat_index = -1
    if (repeating_indices[end] > 5000)
        repeat_index = repeating_indices[end]
        println("Repetition found.")
        repetition = true
    end

    # Checking for valid repetition
    if repeat_index == -1
        repeat_index = length(x[1])  # Corrected to use length of the first vector in x
        println("No repeating index.")
        repetition = false
    end

    return repetition, repeat_index
end


function extract_peaks(mean_time_step, x_sol, t_interp, repeat_index, transform, transform_range)
    N = length(x_sol) * replication
    dt = mean_time_step
    freqs = (0:N-1) ./ (N * dt)
    half_N = ceil(Int, N / 2)
    freqs = freqs[1:half_N]
    transform = transform[1:half_N]
    mag_transform = abs.(transform) .^ 2
    pks, vals = findmaxima(mag_transform)
    sorted_indices = sortperm(vals, rev=true)
    sorted_pks = pks[sorted_indices]
    sorted_vals = vals[sorted_indices] # Magnitude of peaks
    peak_frequencies = freqs[sorted_pks] # Frequency of peaks
    #     transform_plot = plot(freqs, mag_transform, title="Fourier Transform", xlabel="Frequency", ylabel="Magnitude", xlims=transform_range, ylims=(1e-9, 1e-1), yaxis=:log)

    # dB scale transform plot
    # mag_transform = 20*log10.(mag_transform ./ sorted_vals[2])
    # yticks!(-60:20:20, string.(-60:20:20))
    # transform_plot = plot(freqs, mag_transform, title="Fourier Transform", xlabel="Frequency", ylabel="Magnitude", xlims=transform_range, ylims=(-60, 20))

    # tseries_plot = plot(t_interp, x_sol, xlims=[t_interp[1], t_interp[end]])
    return peak_frequencies, sorted_vals, freqs, mag_transform
end


function objective(p, plt=false, transform_range=(0, 2.5))
    # try
        # Solve system
        x, t_interp, mean_time_step = solve_sys(p)
        if length(x[1]) < 100
            return 39, [], []
        end
        # Repetition check
        repetition, repeat_index = repetition_check(x, t_interp, 4)

        if repetition == true
            x_sol = x[3][1:repeat_index]
            transform = augmented_fft(x_sol)
            transform = transform / length(x_sol)
        else
            return 39, [], []
        end

        peak_frequencies, sorted_vals, freqs, transform = extract_peaks(mean_time_step, x_sol, t_interp, repeat_index, transform, transform_range)

        return highest_peak_deviation(peak_frequencies, sorted_vals), freqs, transform
    # catch e
    #     println("Error in ode: ", e)
    #     return 10^5
    # end
end

function isclose(point1, point2, eps)
    return all(abs.(point1 - point2) .< eps)
end

function jacobian_eig(p)
    n11, n10, n20, w2 = p

    # Define the time derivatives
    function time_derivatives(vars)
        I1, I2, theta = vars
        dI1dt = 2*n11*(I1^2) + 2*n10*I1 + 2*sqrt(I1*I2)*sin(theta)
        dI2dt = 2*n20*I2 - 2*sqrt(I1*I2)*sin(theta)
        dthetadt = (sqrt(I2/I1) - sqrt(I1/I2)) * cos(theta)
        return [dI1dt, dI2dt, dthetadt]
    end

    # Analytical fixed points
    I1 = I2 = -(n10 + n20) / n11
    theta = asin(n20)

    # Ensure I1 and I2 are positive and real
    if I1 <= 0 || I2 <= 0 || !isreal(theta)
        return []
    end

    fixed_point = [I1, I2, theta]

    # Compute the Jacobian matrix at the fixed point
    J_func = function (vars)
        ForwardDiff.jacobian(time_derivatives, vars)
    end

    J = J_func(fixed_point)

    # Calculate the eigenvalues of the Jacobian matrix
    eigenvalues = eigen(J).values

    return eigenvalues
end

p = [-0.5, 2.88, 0.1, 1]
# loss, f, transform = objective(p)
# print(loss)
# plot(f, transform, xlims=(0.2, 0.4))

# n20_range = LinRange(0.27, 0.5, 200)
# losses = []
# eigs = []

# for i in 1:length(n20_range)
#     println("Iteration: " * string(i))
#     p[3] = n20_range[i]
#     loss, _, _ = objective(p)
#     eigenvlaues = jacobian_eig(p)
#     println("Eigenvalue: " * string(maximum(real.(eigenvlaues))))
#     println("Loss: " * string(loss))
#     push!(losses, loss)
#     push!(eigs, maximum(real.(eigenvlaues)))
# end

# plot(n20_range, losses, label="Losses", xlabel="n20", ylabel="Losses", color=:blue, legend=:topright)
# plot!(twinx(), n20_range, eigs, label="Eigenvalues", ylabel="Eigenvalues", color=:red, legend=:right)



