include("util.jl")
using PyCall
using LinearAlgebra

# Import Python modules
pushfirst!(PyVector(pyimport("sys")."path"), "/Users/jimmy/Documents/GitHub/Julia_Two_Dimer_BO/PIBO")
custom_module = pyimport("example")
println(keys(custom_module))

hi = custom_module.Hi()