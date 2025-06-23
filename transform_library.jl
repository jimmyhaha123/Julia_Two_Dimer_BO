using DifferentialEquations, FFTW, Statistics, BayesianOptimization, GaussianProcesses, Distributions, Peaks, Interpolations, DSP
using Base: redirect_stdout


num = 40
min = 0


# First replicate the signal by num, then fo Fourier Transform
function augmented_fft(x, replication)
    x_extended = repeat(x, replication)
    
    # Perform FFT on the extended signal
    X = fft(x_extended) / replication
    
    # Return the full FFT result
    return X

end

function mode_gap(freqs, vals)
    tol = 0.05
    sorted_numbers = sort(freqs)
    gaps = diff(sorted_numbers)
    similar(a, b) = (a == 0 && b == 0) || (abs(a - b) ≤ tol * max(abs(a), abs(b)))
    
    # We'll store cluster representatives and their counts.
    clusters = Float64[]
    counts = Int[]
    
    for x in gaps
        found = false
        # Try to add x to an existing cluster.
        for i in 1:length(clusters)
            if similar(x, clusters[i])
                counts[i] += 1
                # Update the cluster's representative as the running average.
                clusters[i] = (clusters[i] * (counts[i]-1) + x) / counts[i]
                found = true
                break
            end
        end
        # If x did not match any existing cluster, start a new one.
        if !found
            push!(clusters, x)
            push!(counts, 1)
        end
    end
    
    # Find the cluster with the maximum count.
    max_count = maximum(counts)
    index = findfirst(isequal(max_count), counts)
    return clusters[index]
end

# Loss function
function highest_peak_deviation(freqs, vals)

    function is_well_separated(numbers::Vector{Float64}, a::Float64, n::Int; tol::Float64=0.05)::Bool
        # Compute the gaps from the sorted numbers
        sorted_numbers = sort(numbers)
        gaps = diff(sorted_numbers)
        if isempty(gaps)
            return false
        end
    
        # Define similarity: two gaps are similar if their relative difference is within tol.
        similar(g1, g2, tol) = abs(g1 - g2) / ((g1 + g2) / 2) <= tol
    
        # Sort the gaps so that similar gap sizes come together
        sorted_gaps = sort(gaps)
    
        clusters = Vector{Vector{Float64}}()  # to store clusters of similar gaps
        current_cluster = [sorted_gaps[1]]
    
        # Cluster contiguous gaps in sorted order that are similar
        for g in sorted_gaps[2:end]
            # Use the mean of the current cluster as the representative value.
            if similar(mean(current_cluster), g, tol)
                push!(current_cluster, g)
            else
                push!(clusters, copy(current_cluster))
                current_cluster = [g]
            end
        end
        push!(clusters, copy(current_cluster))  # add the last cluster
    
        # Check each cluster for the conditions:
        # cluster size > n and mean gap > a.
        for cluster in clusters
            if length(cluster) > n && mean(cluster) < a
                println("Mean gap: ", mean(cluster))
                return false
            end
        end
    
        return true
    end
    
    
    min = num - 1
    if length(vals) < num + 1
        println("No enough peaks. ")
        return min
    elseif vals[2] < 1e-4
        println("Second peak too small. No oscillations. ")
        return min
    elseif ~is_well_separated(freqs[2:num+1], 0.005, 15)  # Some peaks are too close together
        println("Peaks too close together.")
        return min
    else
        sub_peaks = vals[2:num+1]
        sub_peaks = sub_peaks / sub_peaks[1] # Normalization with respect to second peak
        cost = sum(abs.(sub_peaks .- 1))
        # println("Sucessful. Loss: " * string(cost))
        return cost
    end
end

function smoothness_loss(freqs, vals)
    min = 55
    if length(vals) < num + 1
        println("No enough peaks. ")
        return min
    elseif vals[2] < 1e-4
        println("Second peak too small. No oscillations. ")
        return min
    else
        sub_peaks = vals[2:num+1]
        sub_peaks = sub_peaks / sub_peaks[1] # Normalization with respect to second peak
        sub_freqs = freqs[2:num+1]
        deviation_loss = sum(abs.(sub_peaks .- 1))

        mode = mode_gap(sub_freqs, sub_peaks)
        if mode < 0.01
            println("Gaps too small. ")
            return min
        end

        # println("Sucessful. Loss: " * string(cost))
        # Now sort the sub_freqs in increasing order
        indices = sortperm(sub_freqs)
        sub_freqs = sub_freqs[indices]
        sub_peaks = sub_peaks[indices]

        dvals = diff(sub_peaks)
        s = sign.(dvals)
        num_direction_changes = sum(diff(s) .!= 0)
        smoothness_loss = num_direction_changes * 1

        return deviation_loss + smoothness_loss
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
    peak_frequencies = freqs[sorted_pks] # Frequency of peaks (here frequency is sorted)
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