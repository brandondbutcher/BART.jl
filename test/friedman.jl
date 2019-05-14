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

softfit = softbart(X, y)

yhatpost = softfit[1]
s2epost = softfit[2]

using RCall
R"""
y <- $y
X <- $X
yhatpost <- $yhatpost
s2epost <- $s2epost

g <- function(X) {
  10*sin(2*pi * X[,1] * X[,2]) + 20*(X[,3] - 0.5)^2 + 10*X[,4] + 5*X[,5]
}

yhatmean <- apply(yhatpost, 1, mean)
yhatup <- apply(yhatpost, 1, quantile, probs = 0.95)
yhatlow <- apply(yhatpost, 1, quantile, probs = 0.05)

dev.new()
plot((g(X) - yhatmean) ~ g(X), pch = 19, ylim = c(-3, 3))
segments(x0 = g(X), x1 = g(X), y0 = (g(X) - yhatlow), y1 = (g(X) - yhatup), col = "gray")
points(g(X), g(X) - yhatmean, pch = 19)

dev.new()
plot(s2epost, pch = 19, col = adjustcolor("gray", 2 / 3))
abline(h = $truesigma^2, lty = 3)
"""
