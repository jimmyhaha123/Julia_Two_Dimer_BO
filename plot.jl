include("two_dimer.jl")

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


params = [w2, w3, w4, k, an11, an10, an20, bn11, bn10, bn20, nu0, nu1]
params = [0.9, 0.9, 0.9, 1.1, -0.6, 1.0, 0.5, -0.6, 1.3620790701170864, 0.0, 0.5, 0.0] # Cost: 0.03907570449588289
# params = [0.8796806326187828, 0.9134456252737772, 0.8837446133116293, 1.1156229770371104, -0.5889349731779913, 1.0191034728197605, 0.5082101143775112, -0.5526105932135724, 1.2697904185955133, 0.03751922421656739, 0.5343129591669803, 0.061907611058078675]
# params = [0.8688986063250094, 0.8519311047272581, 0.8783395992215297, 1.1355377800935167, -0.6423207048237176, 1.0235412075877455, 0.4630822944210557, -0.6242294360292484, 1.264516414394163, 0.07310438028640895, 0.45521730256345966, 0.07199144022797095]
params = [0.9259501468342337, 0.9370383303858767, 0.8556021732489235, 1.147449709932502, -0.6150562496186814, 0.9808652766864474, 0.512521117858977, -0.571913546816477, 1.3492302785758965, 0.0494632343878712, 0.533884893558711, 0.04355262793668439]
result, transform_plot, tseries_plot = objective(params, true, (0, 1))
display(transform_plot)
# display(tseries_plot)
println(result)