include("transform_library.jl")
include("stability.jl")

using FFTW, Statistics, BayesianOptimization, GaussianProcesses, Distributions, Peaks, Interpolations, DSP
using Base: redirect_stdout

num = 40
min = 0
replication = 50

# Do ngspice simulation and returns x vector, time_interp and mean time step
function solve_ngspice_sys(p)
    #=
    parameter list: 
    gainr1, gainr2, gainr3, gainr4, resc1, resc2, resc3, resc4, factor1, factor2, factor3, factor4, lam1, lam2, lam3
    =#
    write_ngspice_params("four_gain_resonators.cir", p)

    # Define the path to the circuit file
    circuit_file = "four_gain_resonators.cir"

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
    v3_vector = Float64[]
    v4_vector = Float64[]

    # Read the file line by line
    file_path = "./four_gain_resonators.dat"
    open(file_path, "r") do file
        for line in eachline(file)
            split_line = split(line)
            if length(split_line) >= 3
                push!(time_vector, parse(Float64, split_line[1]))
                push!(v1_vector, parse(Float64, split_line[2]))
                push!(v2_vector, parse(Float64, split_line[3]))
                push!(v3_vector, parse(Float64, split_line[4]))
                push!(v4_vector, parse(Float64, split_line[5]))
            end
        end
    end

    # Showing time series
    start_index = Int(round(length(time_vector) * 0.4)) # Taking last 40% of signal
    time_vector = time_vector[start_index:end]
    v1_vector = v1_vector[start_index:end]
    v2_vector = v2_vector[start_index:end]
    v3_vector = v3_vector[start_index:end]
    v4_vector = v4_vector[start_index:end]
    x = [v1_vector, v2_vector, v3_vector, v4_vector]
    return x, time_vector, time_vector[2] - time_vector[1]
end

# Takes in parameters and writes them to local ngspice file
function write_ngspice_params(filename,
    p)
    
    gainr1 = p[1]
    gainr2 = p[2]
    gainr3 = p[3]
    gainr4 = p[4]
    resc1 = string(p[5]) * "p"
    resc2 = string(p[6]) * "p"
    resc3 = string(p[7]) * "p"
    resc4 = string(p[8]) * "p"
    factor1 = p[9]
    factor3 = p[10]
    lam1 = p[11]
    lam2 = p[12]
    lam3 = p[13]

    params = """
    Two coupled RLC Dimers with realistic gain

    *-------------------------------------------------------------
    * Parameters...
    *--------------------------------------------------------------

    .param pi = 3.1415926535

    .param gainr1 = $gainr1                  \$ gain resonator 1 resistance
    .param gainr2 = $gainr2                  \$ gain resonator 2 resistance
    .param gainr3 = $gainr3                  \$ gain resonator 3 resistance
    .param gainr4 = $gainr4                  \$ gain resonator 4 resistance

    .param resc1 = $resc1                   \$ resonator capacitance of resonator 1
    .param resc2 = $resc2                   \$ resonator capacitance of resonator 2
    .param resc3 = $resc3                   \$ resonator capacitance of resonator 3
    .param resc4 = $resc4                   \$ resonator capacitance of resonator 4

    .param resl1 = 200u                    \$ resonator inductance of resonator 1
    .param resl2 = 200u                    \$ resonator inductance of resonator 2
    .param resl3 = 200u                    \$ resonator inductance of resonator 3
    .param resl4 = 200u                    \$ resonator inductance of resonator 4

    .param factor1= $factor1                \$ factor in [0:1] modifies the shape of the nonlinear loss in resonator 1
    .param factor3= $factor3                \$ factor in [0:1] modifies the shape of the nonlinear loss in resonator 3                  


    .param lam1  = $lam1                     \$ scaling of coupling capacitance 1
    .param lam2  = $lam2                     \$ scaling of coupling capacitance 2
    .param lam3  = $lam3                     \$ scaling of coupling capacitance 3

    .param ccoupl1 = {lam1*resc1}            \$ coupling of resonators 1 & 2
    .param ccoupl2 = {lam2*resc1}            \$ coupling of resonators 2 & 3
    .param ccoupl3 = {lam3*resc1}            \$ coupling of resonators 3 & 4

    .param natangfreq = 1/sqrt(resl1*resc1) \$ natural angular frequency of the circuit
    .param natfreq = natangfreq/(2*pi)    \$ natural frequency of circuit
    .param timestep = 1/(200*natfreq)     \$ timestep 
    .param tau = 3 * 2000/(natfreq)           \$ total evolution time
    .param voltdivr = 550


    *initial voltages
    .ic v(lcnode1)=0.5 v(lcnode2)=0 v(lcnode3)=0.25 v(lcnode4)=0

    *====== circuit definition ======
    *RESONATOR 1 WITH GAIN AND NONLINEAR LOSS
    L1        lcnode1     0           {resl1}
    C1        lcnode1     0           {resc1}

    *Gain 
    * generic op-amp (in+ in- out gnd)
    Xopamp1 lcnode1  vd1  vo1  0   OpAmp

    * 2:1 voltage divider for 2X gain
    Rd11  vo1 vd1 {voltdivr}
    Rd12  vd1 0  {voltdivr}

    * positive feedback creating negative resistance
    Rfb1      vo1    lcnode1   {gainr1}

    * Nonlinear loss
    *resistance to ground with back-to-back Diodes
    R1 lcnode1 lcnode1b  {factor1*gainr1}
    D11 lcnode1b 0 1N914
    D12 0 lcnode1b 1N914


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

    * Nonlinear loss
    *resistance to ground with back-to-back Diodes
    * R2 lcnode2 lcnode2b  {factor2*gainr2}
    * D21 lcnode2b 0 1N914
    * D22 0 lcnode2b 1N914


    *RESONATOR 3 WITH GAIN AND NONLINEAR LOSS
    L3        lcnode3     0           {resl3}
    C3        lcnode3     0           {resc3}

    *Gain 
    * generic op-amp (in+ in- out gnd)
    Xopamp3 lcnode3  vd3  vo3  0   OpAmp

    * 2:1 voltage divider for 2X gain
    Rd31  vo3 vd3 {voltdivr}
    Rd32  vd3 0  {voltdivr}

    * positive feedback creating negative resistance
    Rfb3      vo3    lcnode3   {gainr3}

    * Nonlinear loss
    *resistance to ground with back-to-back Diodes
    R3 lcnode3 lcnode3b  {factor3*gainr3}
    D31 lcnode3b 0 1N914
    D32 0 lcnode3b 1N914


    *RESONATOR 4 WITH GAIN AND NONLINEAR LOSS
    L4        lcnode4     0           {resl4}
    C4        lcnode4     0           {resc4}

    *Gain 
    * generic op-amp (in+ in- out gnd)
    Xopamp4 lcnode4  vd4  vo4  0   OpAmp

    * 2:1 voltage divider for 2X gain
    Rd41  vo4 vd4 {voltdivr}
    Rd42  vd4 0  {voltdivr}

    * positive feedback creating negative resistance
    Rfb4      vo4    lcnode4   {gainr4}

    * Nonlinear loss
    *resistance to ground with back-to-back Diodes
    * R4 lcnode4 lcnode4b  {factor4*gainr4}
    * D41 lcnode4b 0 1N914
    * D42 0 lcnode4b 1N914


    *COUPLING OF RESONATORS
    *K12     L1     L2    {mu}                  \$ 1 to 2, mutual inductance
    Cc12        lcnode1     lcnode2     {ccoupl1}  \$ capacitive coupling 1 & 2
    Cc23        lcnode2     lcnode3     {ccoupl2}  \$ capacitive coupling 2 & 3
    Cc34        lcnode3     lcnode4     {ccoupl3}  \$ capacitive coupling 3 & 4


    *--------------------------------------------------------------

    * Transient analysis specs

    *--------------------------------------------------------------

    .tran {timestep} {tau} uic



    .control

        run                                \$ auto run

        linearize                          \$ re-sample to only dt step points 

        

        set wr_singlescale                      \$ only print out one time scale

        wrdata four_gain_resonators.dat V(lcnode1),V(lcnode2),V(lcnode3),V(lcnode4)  \$ write out voltage and current of the main node
        
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

# Takes in parameters, generate loss, transform plot, time series plot, and min index.
function ngspice_objective(p, plt=false, transform_range=(0, 8e6))
    # try
        # Solve system
        x, t_interp, mean_time_step = solve_ngspice_sys(p)
        # Repetition check        
        repetition, repeat_index = repetition_check(x, t_interp)
        t_interp = t_interp[1:repeat_index]

        if repetition == true
            transforms = []
            for i in 1:length(x)
                x[i] = x[i][1:repeat_index]
                t = augmented_fft(x[i], replication)
                push!(transforms, t / (length(x[i])))
            end
        else
            return num - 1, [], [], x[1], t_interp, 0
        end

        peak_frequencies = []
        sorted_vals = []
        mag_transforms = []
        freqs = []
        tseries = []
        losses = Float64[]

        for i in 1:length(transforms)
            result = extract_peaks(mean_time_step, x[i], t_interp, repeat_index, transforms[i], transform_range, replication)
            push!(peak_frequencies, result[1])
            push!(sorted_vals, result[2])
            push!(mag_transforms, result[3])
            push!(freqs, result[4])
            push!(tseries, result[5])
            push!(losses, float(highest_peak_deviation(result[1], result[2])))
        end
        min_loss = minimum(losses)
        min_idx = argmin(losses)

        #db scale
        mag_transforms[min_idx] = mag_transforms[min_idx] / sorted_vals[min_idx][2]
        mag_transforms[min_idx] = 20 * log10.(mag_transforms[min_idx])


        return min_loss, mag_transforms[min_idx], freqs[min_idx], tseries[min_idx], t_interp, min_idx

    # catch e
    #     println("Error in ode: ", e)
    #     return 10^5
    # end
end


function bayesian_ngspice_objective(x)
    #=
    parameter list: 
    gainr1, gainr2, gainr3, gainr4, resc1, resc2, resc3, resc4, factor1, factor3, lam1, lam2, lam3
    =#
    p = x
    println("Current: " * string(p))
    flush(stdout)
    l, _, _, _, _, _ = ngspice_objective(p)
    println("Loss: " * string(l))
    flush(stdout)
    return l
end

function ngspice_opt(num_its)
    #=
    parameter list: 
    gainr1, gainr2, gainr3, gainr4, resc1, resc2, resc3, resc4, factor1, factor2, factor3, factor4, lam1, lam2, lam3
    =#
    lower_bound = [5, 5, 5, 5, 900, 900, 900, 900, 0.1, 0.1, 0.05, 0.05, 0.05]
    upper_bound = [9000, 9000, 9000, 9000, 1100, 1100, 1100, 1100, 0.5, 0.5, 0.5, 0.5, 0.5]
    input_dim = length(lower_bound)
    model = ElasticGPE(input_dim, mean=MeanConst(0.0), kernel=SEArd(zeros(input_dim), 5.0), logNoise=-Inf, capacity=3000)
    set_priors!(model.mean, [Normal(1, 2)])
    vec1 = vcat(zeros(input_dim) .- 1.0, [0.0])
    vec2 = vcat(zeros(input_dim) .+ 4.0, [10.0])
    modeloptimizer = MAPGPOptimizer(
        every=5,
        noisebounds=nothing,
        kernbounds=[vec1, vec2],
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
    lower_bound = [5, 5, 5, 5, 700, 700, 700, 700, 0.1, 0.1, 0.05, 0.05, 0.05]
    upper_bound = [10000, 10000, 10000, 10000, 1400, 1400, 1400, 1400, 0.5, 0.5, 0.5, 0.5, 0.5]
    input_dim = length(lower_bound)
    model = ElasticGPE(input_dim, mean=MeanConst(0.0), kernel=SEArd(zeros(input_dim), 5.0), logNoise=-Inf, capacity=3000)
    set_priors!(model.mean, [Normal(1, 2)])
    vec1 = vcat(zeros(input_dim) .- 1.0, [0.0])
    vec2 = vcat(zeros(input_dim) .+ 4.0, [10.0])
    modeloptimizer = MAPGPOptimizer(
        every=5,
        noisebounds=nothing,
        kernbounds=[vec1, vec2],
        # GaussianProcesses.get_param_names(model.kernel),
        maxeval=40)

    # Initialization
    x = [240.21139646536434, 4867.878914881704, 8384.997250983319, 3108.0357327590455, 962.7116953317021, 1019.2521932376139, 1043.057313471482, 1003.4438974287144, 0.11616358235169735, 0.11402636375999353, 0.4568645360330499, 0.18172062172218917, 0.1946088045238457]
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
        initializer_iterations = 20,
        acquisitionoptions=(method=:LD_LBFGS, restarts=5, maxtime=0.1, maxeval=1000),
        verbosity=Progress)
    result = boptimize!(opt)
    return result
end

ngspice_opt(20)




