using DifferentialEquations, FFTW, Statistics, BayesianOptimization, GaussianProcesses, Distributions, Peaks, Interpolations, DSP
using Base: redirect_stdout


num = 40
min = 0


# First replicate the signal by num, then fo Fourier Transform
function augmented_fft(x, replication)
    # Get the length of the original signal
    
    # Replicate the signal 100 times
    x_extended = repeat(x, replication)
    
    # Perform FFT on the extended signal
    X = fft(x_extended) / replication
    
    # Return the full FFT result
    return X

end

# Loss function
function highest_peak_deviation(freqs, vals)
    min = num - 1
    if length(vals) < num + 1
        println("No enough peaks. ")
        return min
    elseif vals[2] < 1e-4
        println("Second peak too small. No oscillations. ")
        return min
    else
        sub_peaks = vals[2:num+1]
        sub_peaks = sub_peaks / sub_peaks[1] # Normalization with respect to second peak
        cost = sum(abs.(sub_peaks .- 1))
        # println("Sucessful. Loss: " * string(cost))
        return cost
    end
end

# Repetition check
function repetition_check(x, t_interp)
    dimensionality = length(x)
    first_data_point = [x[j][1] for j in 1:dimensionality]
    epsilon = 0.05
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
    if (repeating_indices[end] > 50000)
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

function extract_peaks(mean_time_step, x_sol, t_interp, repeat_index, transform, transform_range, replication)
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
    # transform_plot = plot(freqs, mag_transform, title="Fourier Transform", xlabel="Frequency", ylabel="Magnitude", xlims=transform_range, ylims=(1e-9, 1e-1), yaxis=:log)

    # dB scale transform plot
    # mag_transform = 20*log10.(mag_transform ./ sorted_vals[2])
    # yticks!(-60:20:20, string.(-60:20:20))
    # transform_plot = plot(freqs, mag_transform, title="Fourier Transform", xlabel="Frequency", ylabel="Magnitude", xlims=transform_range, ylims=(-60, 20))

    # tseries_plot = plot(t_interp, x_sol, xlims=[t_interp[1], t_interp[end]])
    return peak_frequencies, sorted_vals, mag_transform, freqs, x_sol
end

function isclose(point1, point2, eps)
    return all(abs.(point1 - point2) .< eps)
end