###############################################################################
##### Load packages
###############################################################################

using LinearAlgebra, Distributions, StatsBase, Plots , Profile


###############################################################################
##### Source code
###############################################################################

include("../src/treestruct.jl")
include("../src/data.jl")
include("../src/treeutils.jl")
include("../src/preprocess.jl")
include("../src/proposals.jl")
include("../src/sampler.jl")
# include("../src/soft/predict.jl")


###############################################################################
##### Testing implementation
###############################################################################

function g(X::Matrix{Float64})
  10sin.(2*pi * X[:,1] .* X[:,2]) + 20(X[:,3] .- 0.5).^2 + 10X[:,4] + 5X[:,5]
end
n = 500
p = 5
X = rand(n, p)
truesigma = sqrt(0.1)
y = g(X) + rand(Normal(0, truesigma), n)
