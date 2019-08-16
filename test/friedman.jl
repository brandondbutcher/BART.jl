###############################################################################
##### Load packages
###############################################################################

using LinearAlgebra, Distributions, StatsBase


###############################################################################
##### Source code
###############################################################################

include("../src/treestruct.jl")
include("../src/data.jl")
include("../src/treeutils.jl")
include("../src/preprocess.jl")
include("../src/proposals.jl")
include("../src/sampler.jl")
include("../src/predict.jl")


###############################################################################
##### Testing implementation
###############################################################################

function g(X::Matrix{Float64})
  10sin.(pi * X[:,1] .* X[:,2]) + 20(X[:,3] .- 0.5).^2 + 10X[:,4] + 5X[:,5]
end
n = 100
p = 5
X = rand(n, p)
truesigma = sqrt(1)
y = g(X) + rand(Normal(0, truesigma), n)

posterior = softbart(X, y)

describe(softfit.σ2)
plot(posterior.σ2)
