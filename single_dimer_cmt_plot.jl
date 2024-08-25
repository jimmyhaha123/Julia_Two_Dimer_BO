include("single_dimer.jl")
using PyPlot
PyPlot.matplotlib.use("TkAgg")

p = [1, 1, -0.5, 2.88, 0.3]
min_loss, mag_transform, freqs, tseries, t_interp, _ = objective(p, true, (0, 1e6))

plot(t_interp, tseries)
show()

plot(freqs, mag_transform)
show()
