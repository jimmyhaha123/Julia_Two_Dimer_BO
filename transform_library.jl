using DifferentialEquations, FFTW, Statistics, BayesianOptimization, GaussianProcesses, Distributions, Peaks, Interpolations, DSP, Optim, Random
using Base: redirect_stdout


num = 40
min = 0

# Takes in freqs and vals, only keep the freqs and vals within 0.5 to 1.5 of center peak
function filter_freqs_and_vals(freqs, vals)
    center_freq = freqs[1]
    lower_bound = center_freq * 0.5
    upper_bound = center_freq * 1.5

    filtered_freqs = freqs[(freqs .>= lower_bound) .& (freqs .<= upper_bound)]
    filtered_vals = vals[(freqs .>= lower_bound) .& (freqs .<= upper_bound)]

    return filtered_freqs, filtered_vals
end


function exp_f(x_vals, lambda, height, center)
    if x_vals < center
        return lambda*(x_vals - center) + height
    else
        return -lambda*(x_vals - center) + height
    end
end
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
        sub_peaks = to_dB_and_normalize(vals[2:num+1])
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

# The objective function that fits envelopes for loss. 
# Input vals are original magnitudes, not dB.
# Optionally return fitted parameters for visualization.
function fitting_loss(freqs, vals, evenlope="exp_dB", return_fitted_params=false, input_val_is_dB=false)
    # Note that now the subpeaks include the fundamental
    if !input_val_is_dB
        sub_peaks = vals[1:num]
        sub_peaks = sub_peaks ./ sub_peaks[2]  # Normalization with respect to second peak
        sub_peaks = 20*log10.(sub_peaks)
    else
        sub_peaks = vals[1:num]
    end
    sub_freqs = freqs[1:num]
    sub_freqs, sub_peaks = filter_freqs_and_vals(sub_freqs, sub_peaks)

    # The fitting loss funtion, takes in f, freqs and vals, return residue
    function loss(f, freqs, vals)
        fitted_vals = f.(freqs)
        residual = vals .- fitted_vals
        return sum(abs.(residual))  # Sum of squared differences
    end

    function multi_start_optimize(
        loss_func::Function,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real};
        n_restarts::Int = 20,
        inner_algo = LBFGS(),
        options = Optim.Options(g_tol=1e-8, f_tol=1e-8)
    )
        d = length(lb)
        @assert length(ub) == d "lb and ub must have same length"

        best_res = nothing
        best_val = Inf
        best_params = nothing

        for i in 1:n_restarts
            # random init in [lb, ub]
            θ0 = lb .+ rand(d) .* (ub .- lb)

            res = Optim.optimize(
                loss_func,
                lb, ub,
                θ0,
                Fminbox(inner_algo),
                options
            )

            if res.minimum < best_val
                best_val = res.minimum
                best_res = res
                best_params = res.minimizer
            end
        end

        return best_params, best_res, best_val
    end

    if evenlope == "exp_dB"
        # lambda, height, center
        ub = [1000, sub_peaks[1], 1.5*sub_freqs[1]]
        lb = [0, sub_peaks[3], 0.5*sub_freqs[1]]

        # 2 parameters to be fitted: lambda (slope) and height (height of center)
        function exp_dB_loss(params)
            lambda = params[1]
            height = params[2]
            center = params[3]
            return loss(x -> exp_f(x, lambda, height, center), sub_freqs, sub_peaks)
        end
        best_params, best_res, best_val = multi_start_optimize(
            exp_dB_loss, lb, ub; n_restarts=10
        )
        # print("Best parameters: ", best_params, " with loss: ", best_val, "\n")

        best_lambda = best_params[1]

        if return_fitted_params
            return best_lambda, best_params
        else
            return best_lambda
        end
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

# Extracting peaks, but with MA to remove background noise
function extract_peaks_with_MA(mean_time_step, x_sol, t_interp, repeat_index,
                               transform, transform_range, replication)
    # 1) Build the frequency axis
    N = length(x_sol) * replication
    dt = mean_time_step
    freq_axis = (0:N-1) ./ (N * dt)         # broadcasting gives a Vector
    half_N    = ceil(Int, N ÷ 2)
    freq_axis = freq_axis[1:half_N]        # first half (positive freqs)

    # 2) Compute the one-sided power spectrum
    mag_transform = abs.(transform[1:half_N]).^2

    # 3) Locate & sort peaks
    peak_frequencies, sorted_vals = locate_peaks(mag_transform, freq_axis)

    return peak_frequencies, sorted_vals, mag_transform, freq_axis, x_sol
end

"""
    locate_peaks(mag_transform, freq_axis; window_size=150)

Given a power spectrum `mag_transform` and its `freq_axis`, this:

  1. Estimates a smooth baseline via moving average of width `window_size`  
  2. Subtracts that baseline to flatten the spectrum  
  3. Finds local maxima in the residual  
  4. Returns their frequencies & magnitudes sorted by descending magnitude
"""
function extract_peaks_with_MA(mean_time_step, x_sol, t_interp, repeat_index,
                               transform, transform_range, replication)
    # 1) Build freq_axis as a Vector, not a StepRangeLen
    N = length(x_sol) * replication
    dt = mean_time_step
    freq_axis = collect(0:N-1) ./ (N * dt)
    half_N    = ceil(Int, N ÷ 2)
    freq_axis = freq_axis[1:half_N]

    # 2) One‑sided power spectrum
    mag_transform = abs.(transform[1:half_N]).^2

    # 3) Locate & sort peaks
    peak_frequencies, sorted_vals = locate_peaks(mag_transform, freq_axis)

    return peak_frequencies, sorted_vals, mag_transform, freq_axis, x_sol
end

"""
    locate_peaks(mag_transform, freq_axis; window_size=150)

1. Estimate a smooth baseline via moving average  
2. Subtract it (flatten the spectrum)  
3. Find local maxima in the residual  
4. Return their freqs & magnitudes sorted descending
"""
function locate_peaks(mag_transform::AbstractVector{<:Real},
                      freq_axis   ::AbstractVector{<:Real},
                      take_log = true)

    # 1) Estimate & subtract baseline
    window_size = 150
    if take_log
        log_transform = 20 * log10.(mag_transform)
        baseline = moving_average(log_transform, window_size)
        residual = log_transform .- baseline
    else
        residual = mag_transform .- moving_average(mag_transform, window_size)
    end

    # 2) Find all local maxima in the residual
    inds = [ i for i in 2:length(residual)-1
             if residual[i] > residual[i-1] && residual[i] > residual[i+1] ]
    inds = filter(i -> residual[i] > 0, inds)   # keep only positive bumps

    # 3) Pull **raw** magnitudes at those indices
    raw_vals  = mag_transform[inds]
    peak_fqs  = freq_axis[inds]

    # 4) Sort by raw magnitude (largest first)
    ord = sortperm(raw_vals, rev=true)
    return peak_fqs[ord], raw_vals[ord]
end

"""
    moving_average(x, w)

Centered moving average of window `w` over `x`, using Base.max/min
and Statistics.mean so nothing gets shadowed.
"""
function moving_average(x::AbstractVector{T}, w::Int) where T
    n    = length(x)
    half = fld(w, 2)
    y    = similar(x)
    for i in 1:n
        lo = Base.max(1,      i-half)
        hi = Base.min(n,      i+half)
        y[i] = Statistics.mean(x[lo:hi])
    end
    return y
end

# Convert to dB and normalize with respect to the first value
function to_dB_and_normalize(vec)
    vec = vec ./vec[1]
    vec =  20 * log10.(vec)
    return vec
end


