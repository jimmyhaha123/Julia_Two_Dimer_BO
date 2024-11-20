include("single_dimer.jl")
using PyPlot, PyCall
PyPlot.matplotlib.use("TkAgg")
pushfirst!(PyVector(pyimport("sys")."path"), "C:/Users/msq3658/Desktop/Julia_Two_Dimer_BO")
@pyimport stability

p = [1.4, 1, -0.5, 2.88, 0.25]
p = [1, 1, -0.5, 2.88, 0.3]
min_loss, mag_transform, freqs, tseries, t_interp, _ = objective(p, true, (0, 1e6))
println(min_loss)
eigenvalue = stability.stability_constraint(p)
println(eigenvalue)



plot(t_interp, tseries)
show()

plot(freqs, mag_transform)
show()


