include("three_dimer.jl")
using PyPlot
PyPlot.matplotlib.use("TkAgg")

params = [1.0966603491055142, 0.9378632655749206, 1.057040343907349, 0.917440458416324, 0.9199003362328986, 1.0723477579553673, 1.025516538639581, 0.9863922249488216, 1.0645605142751668, -0.09835172428429839, -1.530564270477394, -0.46528114211956617, -1.625175837373544, 2.77436605488906, -0.23407390452401033, -1.4547955570899471, 2.706541773105948, -0.8189075954562506]



min_loss, mag_transform, freqs, tseries, t_interp, _ = objective(params, true, (0, 1e6))
println(min_loss)

plot(t_interp, tseries)
show()

# Storing peaks in CSV for loss testing purposes
pks, vals = findmaxima(mag_transform)
sorted_indices = sortperm(vals, rev=true)
sorted_pks = pks[sorted_indices]
sorted_vals = vals[sorted_indices] # Magnitude of peaks
peak_frequencies = freqs[sorted_pks] # Frequency of peaks (here frequency is sorted)
sorted_vals = sorted_vals[2:41]
peak_frequencies = peak_frequencies[2:41]



#db scale
plot(freqs, mag_transform)
scatter(peak_frequencies, sorted_vals, c="red", marker="o", s=20, label="Peaks")
show()



