include("two_dimer.jl")
using PyPlot
PyPlot.matplotlib.use("TkAgg")

an11 = -0.5
an20 = 0
an10 = 2.88
bn11 = -0.5
bn20 = 0
bn10 = 2.88
k = 1
nu0 = 0.34
nu1 = 0
w2 = 1
w3 = 1
w4 = 1

lb = [ 0.3744,  0.6678,  0.9594,  0.7053, -1.2457,  0.2473,  0.0577, -1.1925, 0.6816,  0.1486,  0.5292]
ub = [ 0.6953,  1.2401,  1.7818,  1.3098, -0.6707,  0.4593,  0.1072, -0.6421, 1.2659,  0.2760,  0.9828]

function unnormalize(normalized_coords::Vector{<:Real})
    @assert length(normalized_coords) == length(lb) == length(ub) "All input vectors must have the same length"
    return [norm * (ub[i] - lb[i]) + lb[i] for (i, norm) in enumerate(normalized_coords)]
end

params = [w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0, nu1]
params = [0.9, 0.9, 0.9, 1.1, -0.6, 1.0, 0.5, -0.6, 1.3620790701170864, 0.0, 0.5, 0.0] # Cost: 0.03907570449588289
# params = [0.8796806326187828, 0.9134456252737772, 0.8837446133116293, 1.1156229770371104, -0.5889349731779913, 1.0191034728197605, 0.5082101143775112, -0.5526105932135724, 1.2697904185955133, 0.03751922421656739, 0.5343129591669803, 0.061907611058078675]
# params = [0.8688986063250094, 0.8519311047272581, 0.8783395992215297, 1.1355377800935167, -0.6423207048237176, 1.0235412075877455, 0.4630822944210557, -0.6242294360292484, 1.264516414394163, 0.07310438028640895, 0.45521730256345966, 0.07199144022797095]

# This is the best parameter including nonlinear coupling
params = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 0.9808652766864474, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711, 0.04355262793668439]
# Without nonlinear coupling
params = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 0.9808652766864474, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711]

# params = [ 0.67078778,  1.12303510,  0.68284383,  0.95135297, -0.55687337,
# 0.97508005,  0.38060049, -0.61854603,  1.26118566,  0.05802892,
# 0.60422644]

params = [0.9, 0.9, 0.8063516866447157, 1.2, -0.7, 1.1, 0.4, -0.5, 1.272152040642602, 0.0, 0.6]


params = [1.17, 0.63, 0.564446180651301, 1.56, -0.9099999999999999, 1.0734941117229975, 0.52, -0.65, 1.6502573779699414, 0.0, 0.42]

params = [1.0603125, 0.6384375, 1.0255785514512477, 1.41375, -0.5359375, 0.8834375000000001, 0.41125, -0.5234375, 1.0694028091651873, 0.0, 0.650625]

params = [0.7247037931853786, 0.8630194622503194, 1.0214028372587463, 1.0653901954545089, -0.539670874696093, 0.922719823442953, 0.30268924272958525, -0.3605283562774053, 1.3420227207692053, 0.0, 0.42766061499868274]
min_loss, mag_transform, freqs, tseries, t_interp, _ = objective(params, true, (0, 1e6))
println(min_loss)

# semilogy(freqs, mag_transform)
# show()

plot(t_interp, tseries)
show()


# Finding the peaks used in visualization
# pks, vals = findmaxima(mag_transform)
# sorted_indices = sortperm(vals, rev=true)
# sorted_pks = pks[sorted_indices]
# sorted_vals = vals[sorted_indices] # Magnitude of peaks
# peak_frequencies = freqs[sorted_pks] # Frequency of peaks (here frequency is sorted)
# sorted_vals = sorted_vals[2:41]
# peak_frequencies = peak_frequencies[2:41]

peak_frequencies, sorted_vals = locate_peaks(mag_transform, freqs, false)
best_lambda, best_params = fitting_loss(peak_frequencies, sorted_vals, "exp_dB", true, true)
sorted_vals = sorted_vals[1:40]
peak_frequencies = peak_frequencies[1:40]
peak_frequencies, sorted_vals = filter_freqs_and_vals(peak_frequencies, sorted_vals)



#db scale
center = peak_frequencies[1]
lo, hi = 0.5*center, 1.5*center
plot(freqs, mag_transform)
plot(freqs, exp_f.(freqs, best_lambda, best_params[2], best_params[3]))
xlim(0.5*center, 1.5*center)
ylim(-150, 50)
scatter(peak_frequencies, sorted_vals, c="red", marker="o", s=20, label="Peaks")
show()

