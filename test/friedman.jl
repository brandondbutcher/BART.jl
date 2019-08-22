using BART
using Distributions
using StatsBase

function g(X::Matrix{Float64})
  10sin.(pi * X[:,1] .* X[:,2]) + 20(X[:,3] .- 0.5).^2 + 10X[:,4] + 5X[:,5]
end

n = 250
p = 10
X = rand(n, p)
sigma = sqrt(1)
y = g(X) + rand(Normal(0, sigma), n)

posterior = softbart(X, y)

describe(sqrt.(posterior.σ2))
