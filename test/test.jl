###############################################################################
##### Load packages
###############################################################################

using LinearAlgebra, Distributions, StatsBase, Plots , Profile


###############################################################################
##### Source code
###############################################################################

include("../src/soft/treestruct.jl")
include("../src/soft/treeutils.jl")
include("../src/soft/preprocess.jl")
include("../src/soft/proposals.jl")
include("../src/soft/sampler.jl")
# include("../src/soft/predict.jl")


###############################################################################
##### Testing implementation
###############################################################################

n = 100
truep = 5
noisep = 25
X = rand(n, truep + noisep)
function g(X::Matrix{Float64})
  10sin.(pi * X[:,1] .* X[:,2]) + 20(X[:,3] .- 5).^2 + 10X[:,4] + 5X[:,5]
end
truesigma = sqrt(50)
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
  10*sin(pi * X[,1] * X[,2]) + 20*(X[,3] - 5)^2 + 10*X[,4] + 5*X[,5]
}

yhatmean <- apply(yhatpost, 1, mean)
yhatup <- apply(yhatpost, 1, quantile, probs = 0.975)
yhatlow <- apply(yhatpost, 1, quantile, probs = 0.025)

dev.new()
plot(yhatmean ~ g(X), pch = 19)
segments(x0 = g(X), x1 = g(X), y0 = yhatlow, y1 = yhatup)
abline(0, 1, lty = 3)

dev.new()
plot(s2epost, pch = 19, col = adjustcolor("gray", 2 / 3))
abline(h = $truesigma^2, lty = 3)
"""

R"""
library(SoftBart)
y <- $y
x <- $X
yscale <- (y - ((max(y) + min(y)) / 2)) / (max(y) - min(y))
sigma_hat <- summary(lm(yscale ~ x))$sigma
system.time(softfit <- softbart(
  X = as.matrix(x), Y = as.matrix(y), X_test = as.matrix(x),
  hypers = Hypers(X = as.matrix(x), Y = as.matrix(y), sigma_hat = sigma_hat, num_tree = 20),
  opts = Opts(
    num_save = 2500, num_burn = 500, num_thin = 1,
    update_s = FALSE, update_alpha = FALSE, update_sigma_mu = FALSE
  )
))

dev.new()
plot(softfit$y_hat_train_mean ~ y, pch = 19)
yhatup <- apply(softfit$y_hat_train, 2, quantile, probs = 0.975)
yhatlow <- apply(softfit$y_hat_train, 2, quantile, probs = 0.025)
segments(x0 = y, x1 = y, y0 = yhatlow, y1 = yhatup)
abline(0, 1, lty = 3)

dev.new()
plot(softfit$sigma^2, pch = 19, col = adjustcolor("gray", 2/3))
abline(h = $truesigma^2, lty = 3)
"""
