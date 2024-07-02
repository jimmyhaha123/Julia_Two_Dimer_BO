using CSV
using DataFrames
using Plots
include("two_dimer.jl")

command = `ngspice /Users/jimmy/Desktop/Jimmy/WTICS/two_dimer/two_gain_resonators_v2.cir`
run(command)

time_vector = Float64[]
v1_vector = Float64[]
v2_vector = Float64[]

# Read the file line by line
file_path = "/Users/jimmy/Desktop/Jimmy/WTICS/two_dimer/curr_volt_vs_t.dat"
open(file_path, "r") do file
    for line in eachline(file)
        split_line = split(line)
        if length(split_line) >= 3
            push!(time_vector, parse(Float64, split_line[1]))
            push!(v1_vector, parse(Float64, split_line[2]))
            push!(v2_vector, parse(Float64, split_line[3]))
        end
    end
end

# Showing time series
start_index = Int(round(length(time_vector) * 0.6))
time_vector = time_vector[start_index:end]
v1_vector = v1_vector[start_index:end]
v2_vector = v2_vector[start_index:end]
v = [v1_vector, v2_vector]


repetition, repeat_index = repetition_check(v, time_vector, 2)
tseries = plot(time_vector, v1_vector, xlims=(time_vector[1], time_vector[repeat_index]))
display(tseries)

if repetition == true
    x_sol = v1_vector[1:repeat_index]
    transform = fft(x_sol)
    transform = transform / length(x_sol)
else
    # x_sol = x_sol_2[1:repeat_index]
    # hann_window = DSP.Windows.hanning(length(x_sol))
    # x_sol = x_sol .* hann_window
    # transform = fft(x_sol)
    # transform = transform / length(x_sol)
    return 10 ^ 5
end

t_interp = time_vector
mean_time_step = t_interp[2] - t_interp[1]
transform_range = (0, 8e6)
peak_frequencies, sorted_vals, transform_plot, tseries_plot = extract_peaks(mean_time_step, x_sol, t_interp, repeat_index, transform, transform_range)
ylims!(transform_plot, 1e-8, 5)
display(transform_plot)

loss = highest_peak_deviation(peak_frequencies, sorted_vals)


