using DifferentialEquations, Plots, FFTW, Statistics, BayesianOptimization, GaussianProcesses, Distributions, Peaks, Interpolations, DSP
using Base: redirect_stdout

# Systen definition
function sys!(du, u, p, t)
    # Parameters
    w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0, nu1 = p

    aN1(a1) = an11 * (abs(a1))^2 + an10
    aN2(a2) = an20
    bN1(b1) = bn11 * (abs(b1))^2 + bn10
    bN2(b2) = bn20
    nu(a2, b1) = nu0 + nu1*abs(a2 - b1)

    a1, a2, b1, b2 = u
    du[1] = im*a1 + aN1(a1)*a1 +im*a2
    du[2] = im*w2*a2 + aN2(a2)*a2 +im*a1 + im*nu(a2, b1)*b1
    du[3] = im*w3*b1 + bN1(b1)*b1 +im*k*b2 + im*nu(a2, b1)*a2
    du[4] = im*w4*b2 + bN2(b2)*b2 +im*k*b1
end

# Solve system, returns x solution, time solution, and mean time step
function solve_sys(p)
    range = 0.5
    u0 = [range * rand() + im*range * rand(), range * rand() + im*range * rand(), range * rand() + im*range * rand(), range * rand() + im*range * rand()]
    # Fixing initial condition for consistency
    u0 = [0.1 + 0.1*im, 0.1 + 0.1*im, 0.1 + 0.1*im, 0.1 + 0.1*im]
    t = 100000.0
    tspan = (0.0, t)

    prob = ODEProblem(sys!, u0, tspan, p)
    abstol = 1e-8
    reltol = 1e-6
    sol = solve(prob, abstol=abstol, reltol=reltol)

    u0 = sol.u[end]
    tspan = (0, 10000)
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
    x_sol_5 = [real.(u[3]) for u in sol.u]
    x_sol_6 = [imag.(u[3]) for u in sol.u]
    x_sol_7 = [real.(u[4]) for u in sol.u]
    x_sol_8 = [imag.(u[4]) for u in sol.u]
    interp_func_1 = LinearInterpolation(time_points, x_sol_1)
    interp_func_2 = LinearInterpolation(time_points, x_sol_2)
    interp_func_3 = LinearInterpolation(time_points, x_sol_3)
    interp_func_4 = LinearInterpolation(time_points, x_sol_4)
    interp_func_5 = LinearInterpolation(time_points, x_sol_5)
    interp_func_6 = LinearInterpolation(time_points, x_sol_6)
    interp_func_7 = LinearInterpolation(time_points, x_sol_7)
    interp_func_8 = LinearInterpolation(time_points, x_sol_8)
    x_sol_1 = [interp_func_1(t) for t in t_interp]
    x_sol_2 = [interp_func_2(t) for t in t_interp]
    x_sol_3 = [interp_func_3(t) for t in t_interp]
    x_sol_4 = [interp_func_4(t) for t in t_interp]
    x_sol_5 = [interp_func_5(t) for t in t_interp]
    x_sol_6 = [interp_func_6(t) for t in t_interp]
    x_sol_7 = [interp_func_7(t) for t in t_interp]
    x_sol_8 = [interp_func_8(t) for t in t_interp]
    x = [x_sol_1, x_sol_2, x_sol_3, x_sol_4, x_sol_5, x_sol_6, x_sol_7, x_sol_8]
    return x, t_interp, mean_time_step
end

# Do ngspice simulation and returns x vector, time_interp and mean time step
function solve_ngspice_sys(p)
    #=
    parameter list: gainr1, gainr2, resl1, resc1, resl2, resc2, lam, factor, voltdivr
    =#
    write_ngspice_params("two_gain_resonators_v2.cir", p)

    # Define the path to the circuit file
    circuit_file = "/Users/jimmy/Desktop/Jimmy/WTICS/two_dimer/two_gain_resonators_v2.cir"

    # Use mktemp to create a temporary file and redirect stdout to it
    str = mktemp() do path, io
        redirect_stdout(io) do
            run(`ngspice $circuit_file`)
        end
        # Ensure all output is flushed to the file
        Base.Libc.flush_cstdio()
        # Read the content of the file
        readline(path)
    end

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
    start_index = Int(round(length(time_vector) * 0.6)) # Taking last 40% of signal
    time_vector = time_vector[start_index:end]
    v1_vector = v1_vector[start_index:end]
    v2_vector = v2_vector[start_index:end]
    x = [v1_vector, v2_vector]
    return x, time_vector, time_vector[2] - time_vector[1]
end

# Loss function
function highest_peak_deviation(freqs, vals)
    num = 40
    if length(vals) < num + 1
        println("No enough peaks. ")
        return 39
    elseif vals[2] < 10^(-3) # First peak too small
        println("First peak too small. No oscillations. ")
        return 39
    else
        sub_peaks = vals[2:num+1]
        sub_peaks = sub_peaks / sub_peaks[1] # Normalization with respect to first peak
        cost = sum(abs.(sub_peaks .- 1))
        # println("Sucessful. Loss: " * string(cost))
        return cost
    end
end

# Takes in parameters and writes them to local ngspice file
function write_ngspice_params(filename,
    p)

    gainr1 = p[1]
    gainr2 = p[2]
    resl1 = string(p[3]) * "u"
    resc1 = string(p[4]) * "p"
    resl2 = string(p[5]) * "u"
    resc2 = string(p[6]) * "p"
    lam = p[7]
    factor = p[8]
    voltdivr = p[9]

    params = """
    *-------------------------------------------------------------
    * Parameters...
    *--------------------------------------------------------------

    .param pi = 3.1415926535
    .param gainr1 = $gainr1                  \$ gain resonator 1 resistance
    .param gainr2 = $gainr2                  \$ gain resonator 2 resistance
    .param resl1 = $resl1                    \$ resonator inductance of resonator 1
    .param resc1 = $resc1                   \$ resonator capacitance of resonator 1
    .param resl2 = $resl2                    \$ resonator inductance of resonator 2
    .param resc2 = $resc2                   \$ resonator capacitance of resonator 2
    .param lam  = $lam                     \$ scaling of coupling capacitance
    .param factor = $factor                    \$ factor \\in [0:1] modifies the shape of the nonlinear loss

    .param ccoupl = {lam*resc1}            \$ coupling of resonators
    .param natangfreq1 = 1/sqrt(resl1*resc1) \$ natural angular frequency of the circuit
    .param natfreq1 = natangfreq1/(2*pi)    \$ natural frequency of circuit
    .param timestep = 1/(200*natfreq1)     \$ timestep 
    .param tau = 2000/(natfreq1)           \$ total evolution time
    .param voltdivr = $voltdivr

    *-------------------------------------------------------------
    * End of parameters...
    *-------------------------------------------------------------

    *initial voltages
    .ic v(lcnode1)=0.5 v(lcnode2)=0

    *====== circuit definition ======
    *RESONATOR 1 WITH GAIN AND NONLINEAR LOSS
    L1        lcnode1     0           {resl1}
    C1        lcnode1     0           {resc1}

    *Gain 
    * generic op-amp (in+ in- out gnd)
    Xopamp1 lcnode1  vd  vo  0   OpAmp

    * 2:1 voltage divider for 2X gain
    Rd11  vo vd {voltdivr}
    Rd12  vd 0  {voltdivr}

    * positive feedback creating negative resistance
    Rfb1      vo    lcnode1   {gainr1}

    * Nonlinear loss
    *resistance to ground with back-to-back Diodes
    R1 lcnode1 lcnode1_2  {factor*gainr1}
    D11 lcnode1_2 0 1N914
    D12 0 lcnode1_2 1N914


    *RESONATOR 2 WITH LINEAR GAIN
    L2        lcnode2     0           {resl2}
    C2        lcnode2     0           {resc2}

    *Gain
    * generic op-amp (in+ in- out gnd)
    Xopamp2 lcnode2  vd2  vo2  0   OpAmp

    * 2:1 voltage divider for 2X gain
    Rd21  vo2 vd2 {voltdivr}
    Rd22  vd2 0  {voltdivr}

    * positive feedback creating negative resistance
    Rfb2      vo2    lcnode2   {gainr2}




    *COUPLING OF RESONATORS
    *K12     L1     L2    {mu}                  \$ 1 to 2, mutual inductance
    Cc        lcnode1     lcnode2     {ccoupl}  \$ capacitive coupling


    *--------------------------------------------------------------

    * Transient analysis specs

    *--------------------------------------------------------------

    .tran {timestep} {tau} uic
    .option post=none nomod brief
    .control

        run                                \$ auto run

        linearize                          \$ re-sample to only dt step points 

        set wr_singlescale                      \$ only print out one time scale

        wrdata curr_volt_vs_t.dat V(lcnode1),V(lcnode2)  \$ write out voltage and current of the main node
        
        destroy all return [tref_tmp range 0 0] 	\$ cleans the memory

        quit
    .endc



    .END


    *-------------------------------------------------------------
    * Generic op-amp sub-circuit
    *-------------------------------------------------------------

    *  NODE NUMBERS
    *              IN+
    *               | IN-
    *               |  | OUT
    *               |  |  | GND
    *               |  |  |  |
    *               |  |  |  |
    *               |  |  |  |
    .SUBCKT OpAmp   1  2  3  4

    * hi-Z Norton source with voltage diode limit

    Gop    4    3    1    2    1       \$ N+ N- NC+ NC- Transconductance
    *Rsh    3    4    1E5              \$ may be necessary in other circuits?
    Csh    3    4    1.333n            \$ may be necessary in other circuits?
    Vp     vp   4    DC  5            \$ +/- 12 V power supply assumed
    Vm     vm   4    DC -5
    Dp     3    vp   Dop               \$ diode like clipping
    Dm     vm   3    Dop

    .MODEL Dop D(IS=1E-15)            \$ may be necessary in other circuits?

    .ends

    ******

    .MODEL 1N914 D
    + IS=5.0e-09
    + RS=0.8622
    + N=2
    + ISR=9.808e-14
    + NR=2.0
    + CJO=8.68e-13
    + M=0.02504
    + VJ=0.90906
    + FC=0.5
    + TT=6.012e-9
    + BV=100
    + IBV=1e-07
    + EG=0.92

    ******

    .END

    """

    open(filename, "w") do file
        write(file, params)
    end

end

# Check repetition, returns boolean repetition and repeat index.
function repetition_check(x, t_interp, dimensionality=8)
    first_data_point = [x[j][1] for j in 1:dimensionality]
    epsilon = 0.01
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
    if (repeating_indices[end] > 20000)
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

# Extract peaks, return peaj ferquencies, 
# sorted amplitudes, time series and transform
function extract_peaks(mean_time_step, x_sol, t_interp, repeat_index, transform, transform_range)
    N = length(x_sol)
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
    transform_plot = plot(freqs, mag_transform, title="Fourier Transform", xlabel="Frequency", ylabel="Magnitude", xlims=transform_range, ylims=(1e-9, 1e-1), yaxis=:log)

    # dB scale transform plot
    mag_transform = 20*log10.(mag_transform ./ sorted_vals[2])
    yticks!(-60:20:20, string.(-60:20:20))
    transform_plot = plot(freqs, mag_transform, title="Fourier Transform", xlabel="Frequency", ylabel="Magnitude", xlims=transform_range, ylims=(-60, 20))

    tseries_plot = plot(t_interp, x_sol, xlims=[0.003, 0.0035])
    return peak_frequencies, sorted_vals, transform_plot, tseries_plot
end

# Takes in parameters, generate loss, transform plot, and time series plot
function objective(p, plt=false, transform_range=(0, 2.5))
    try
        # Solve system
        x, t_interp, mean_time_step = solve_sys(p)
        # Repetition check        
        repetition, repeat_index = repetition_check(x, t_interp)

        if repetition == true
            x_sol = x[3][1:repeat_index]
            transform = fft(x_sol)
            transform = transform / length(x_sol)
        else
            return 10 ^ 5
        end

        peak_frequencies, sorted_vals, transform_plot, tseries_plot = extract_peaks(mean_time_step, x_sol, t_interp, repeat_index, transform, transform_range)
        if !plt
            return highest_peak_deviation(peak_frequencies, sorted_vals)
        else
            return highest_peak_deviation(peak_frequencies, sorted_vals), transform_plot, tseries_plot
        end
    catch e
        println("Error in ode: ", e)
        return 10^5
    end
end

# Takes in parameters, generate loss, transform plot, time series plot, and min index.
function ngspice_objective(p, plt=false, transform_range=(0, 8e6))
    # try
        # Solve system
        x, t_interp, mean_time_step = solve_ngspice_sys(p)
        # Repetition check        
        repetition, repeat_index = repetition_check(x, t_interp, 2)
        t_interp = t_interp[1:repeat_index]

        if repetition == true
            transforms = []
            for i in 1:length(x)
                x[i] = x[i][1:repeat_index]
                t = fft(x[i])
                push!(transforms, t / length(x[i]))
            end
        else
            return 39
        end

        peak_frequencies = []
        sorted_vals = []
        transform_plots = []
        tseries_plots = []
        losses = Float64[]

        for i in 1:length(transforms)
            result = extract_peaks(mean_time_step, x[i], t_interp, repeat_index, transforms[i], transform_range)
            push!(peak_frequencies, result[1])
            push!(sorted_vals, result[2])
            push!(transform_plots, result[3])
            push!(tseries_plots, result[4])
            push!(losses, float(highest_peak_deviation(result[1], result[2])))
        end
        min_loss = minimum(losses)
        min_idx = argmin(losses)

        if !plt
            return min_loss
        else
            return min_loss, transform_plots[min_idx], tseries_plots[min_idx], min_idx
        end
    # catch e
    #     println("Error in ode: ", e)
    #     return 10^5
    # end
end

# Returns whether the two points are close by element-wise criterion
function isclose(point1, point2, eps)
    return all(abs.(point1 - point2) .< eps)
end

function bayesian_objective(x)
    # w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0, nu1
    p = (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12])
    println("Current: " * string(p))
    return objective(p)
end

function bayesian_ngspice_objective(x)
    #=
    parameter list: gainr1, gainr2, resl1, resc1, resl2, resc2, lam, factor, voltdivr
    =#
    # Optimization variables: gainr1, gainr2, resc1, resc2, lam, factor
    # x[1] = gainr1, x[2] = gainr2, x[3] = resc1, x[4] = resc2, x[5] = lam, x[6] = factor
    p = (x[1], x[2], 200, x[3], 200, x[4], x[5], x[6], 550)
    println("Current: " * string(p))
    flush(stdout)
    println("Loss: " * string(ngspice_objective(p)))
    flush(stdout)
    return ngspice_objective(p)
end

function opt(num_its)
    # [w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0, nu1]
    lower_bound = [0.9, 0.9, 0.8, 1, -0.7, 0.9, 0.4, -0.7, 1.2, 0, 0.4, 0]
    upper_bound = [1.1, 1.1, 1, 1.2, -0.5, 1.1, 0.6, -0.5, 1.4, 0.2, 0.6, 0.2]
    input_dim = length(lower_bound)
    model = ElasticGPE(input_dim, mean=MeanConst(0.0), kernel=SEArd(zeros(input_dim), 5.0), logNoise=0.0, capacity=3000)
    set_priors!(model.mean, [Normal(1, 2)])
    vec1 = vcat(zeros(input_dim) .- 1.0, [0.0])
    vec2 = vcat(zeros(input_dim) .+ 4.0, [10.0])
    modeloptimizer = MAPGPOptimizer(
        every=50,
        noisebounds=[-4, 3],
        kernbounds=[vec1, vec2],
        # GaussianProcesses.get_param_names(model.kernel),
        maxeval=40)

    opt = BOpt(
        bayesian_objective, model,
        ExpectedImprovement(),
        modeloptimizer,
        lower_bound,
        upper_bound,
        repetitions=1,
        maxiterations=num_its,
        sense=Min,
        acquisitionoptions=(method=:LD_LBFGS, restarts=5, maxtime=0.1, maxeval=1000),
        verbosity=Progress)
    result = boptimize!(opt)
    return result
end

function ngspice_opt(num_its)
    #=
    parameter list: gainr1, gainr2, resl1, resc1, resl2, resc2, lam, factor, voltdivr
    =#
    # x[1] = gainr1, x[2] = gainr2, x[3] = resc1, x[4] = resc2, x[5] = lam, x[6] = factor
    lower_bound = [5, 5, 500, 500, 0.05, 0.1]
    upper_bound = [2000, 10000, 1500, 1500, 0.5, 0.5]
    input_dim = length(lower_bound)
    model = ElasticGPE(input_dim, mean=MeanConst(0.0), kernel=SEArd(zeros(input_dim), 5.0), logNoise=0.0, capacity=3000)
    set_priors!(model.mean, [Normal(1, 2)])
    vec1 = vcat(zeros(input_dim) .- 1.0, [0.0])
    vec2 = vcat(zeros(input_dim) .+ 4.0, [10.0])
    modeloptimizer = MAPGPOptimizer(
        every=2,
        noisebounds=[-10, 0],
        kernbounds=[vec1, vec2],
        # GaussianProcesses.get_param_names(model.kernel),
        maxeval=200)

    opt = BOpt(
        bayesian_ngspice_objective, model,
        ExpectedImprovement(),
        modeloptimizer,
        lower_bound,
        upper_bound,
        repetitions=1,
        maxiterations=num_its,
        sense=Min,
        acquisitionoptions=(method=:LD_LBFGS, restarts=5, maxtime=0.1, maxeval=1000),
        verbosity=Progress)
    result = boptimize!(opt)
    return result
end

# Warm-started optimization
function ngspice_opt_warm(num_its)
    #=
    parameter list: gainr1, gainr2, resl1, resc1, resl2, resc2, lam, factor, voltdivr
    =#
    # x[1] = gainr1, x[2] = gainr2, x[3] = resc1, x[4] = resc2, x[5] = lam, x[6] = factor
    lower_bound = [5, 5, 500, 500, 0.05, 0.1]
    upper_bound = [2000, 10000, 1500, 1500, 0.5, 0.5]
    input_dim = length(lower_bound)
    model = ElasticGPE(input_dim, mean=MeanConst(0.0), kernel=SEArd(zeros(input_dim), 5.0), logNoise=0.0, capacity=3000)
    set_priors!(model.mean, [Normal(1, 2)])
    vec1 = vcat(zeros(input_dim) .- 1.0, [0.0])
    vec2 = vcat(zeros(input_dim) .+ 4.0, [10.0])
    modeloptimizer = MAPGPOptimizer(
        every=50,
        noisebounds=[-4, 3],
        kernbounds=[vec1, vec2],
        # GaussianProcesses.get_param_names(model.kernel),
        maxeval=40)

    # Initialization
    x = [73.66503907381308, 6980.047551864795, 543.8022037767349, 1156.0876792090266, 0.4027742682261706, 0.27689905702227396]
    y = -bayesian_ngspice_objective(x)
    append!(model, reshape(x, :, 1), [y]) 

    opt = BOpt(
        bayesian_ngspice_objective, model,
        ExpectedImprovement(),
        modeloptimizer,
        lower_bound,
        upper_bound,
        repetitions=1,
        maxiterations=num_its,
        sense=Min,
        initializer_iterations = 0,
        acquisitionoptions=(method=:LD_LBFGS, restarts=5, maxtime=0.1, maxeval=1000),
        verbosity=Progress)
    result = boptimize!(opt)
    return result
end







