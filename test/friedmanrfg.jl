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

data = FriedmanRFG(100)
softfit = softbart(data.X, data.yobs)

using RCall
yhatpost = softfit.yhat
s2epost = softfit.σ2
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
