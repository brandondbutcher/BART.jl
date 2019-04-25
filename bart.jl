###############################################################################
##### Load packages
###############################################################################

using LinearAlgebra, Distributions, StatsBase, Plots, Profile


###############################################################################
##### Source my BART code
###############################################################################

include("/home/brandon/Dropbox/BART.jl/treestruct.jl")
include("/home/brandon/Dropbox/BART.jl/treeutils.jl")
include("/home/brandon/Dropbox/BART.jl/proposals.jl")
include("/home/brandon/Dropbox/BART.jl/predict.jl")


###############################################################################
##### Simulate data and test code
###############################################################################

n = 100

# f(x) = x^3
# x = rand(Normal(0, 1), n)
# y = f.(x) + rand(Normal(0, 0.3), n)

function g(x)
  sin(pi * x / 2) / (1 + 2 * x^2 * (sign(x) + 1))
end
x = randn(n)
y = g.(x) + rand(Normal(0, 0.3), n)


###############################################################################
##### Set-up parameters
###############################################################################

## BART MCMC sampler
function bartmcmc(x::Vector{Float64}, y::Vector{Float64})

  ## Default number of trees
  m = 200

  ## Prior parameters for probability a node is split
  a = 0.95
  b = 2

  ## Prior variance for node parameters
  k = 2
  ytrain = y .- mean(y)
  s2mu = ((maximum(ytrain) - minimum(ytrain)) / (2*k*sqrt(m)))^2

  ## Set initial residual variance to estimate from linear model &
  ## center outcome by its mean for training
  X = hcat(ones(n), x)
  betahat = X \ ytrain
  rhat = ytrain .- X*betahat
  s2hat = dot(rhat, rhat) / (n - size(X)[2])

  ## Set default Inverse Gamma parameters for residual variance prior
  nu = 3
  q = 0.9
  lambda = 1 / quantile(InverseGamma(nu/2, nu/(2*s2hat)), q)

  ## Set cut points
  ncuts = 100
  increment = (maximum(x) - minimum(x)) / (ncuts + 1)
  lower = minimum(x) + increment
  upper = maximum(x) - increment
  cuts = collect(lower:increment:upper)

  ## Set-up for MCMC sampling
  S = 1000
  trees = initializetrees(ytrain, m)
  s2e = s2hat
  bartpost = BartPosterior(S, m)

  for s in 1:S
    for t in 1:m
      yhat = predict(trees[1:end .!= t], x)
      residual = ytrain .- yhat
      updatetree!(trees[t], residual, x, cuts, a, b, s2e, s2mu)
      updatemu!(trees[t], residual, s2e, s2mu)
      bartpost.trees[s][t] = BartTree(trees[t])
    end
    s2e = updatesigma(ytrain, x, trees, nu, lambda)
    bartpost.s2e[s] = s2e
    if s % 100 == 0
      println("Iteration $s complete.")
    end
  end
  [bartpost, cuts]
end

bartfit = bartmcmc(x, y)

# Profile.print(format = :flat, sortedby = :count)

yhat = predict(bartfit[1], x, bartfit[2])
postsigma = sqrt.(bartfit[1].s2e)

using RCall
R"""
y <- $y
x <- $x
yhat <- $yhat
yhat <- yhat + mean(y)
yhatmean <- colMeans(yhat)
yhatup <- apply(yhat, 2, quantile, 0.975)
yhatlow <- apply(yhat, 2, quantile, 0.025)
postsigma <- $postsigma
S <- 1000
g <- function(x) {
  sin(pi * x / 2) / (1 + 2 * x^2 * (sign(x) + 1))
  # x^3
}

dev.new()
plot(y ~ x, col = "gray", pch = 19)
for (i in 1:S) {
  lines(
    x[order(x)], yhat[i,][order(x)],
    col = adjustcolor("slateblue", 0.01), lwd = 2
  )
}
curve(g(x), add = TRUE)

dev.new()
plot(y ~ x, col = "gray", pch = 19)
lines(x[order(x)], yhatmean[order(x)], col = "slateblue", lwd = 2)
lines(x[order(x)], yhatlow[order(x)], col = "slateblue", lwd = 2, lty = 3)
lines(x[order(x)], yhatup[order(x)], col = "slateblue", lwd = 2, lty = 3)
curve(g(x), add = TRUE)

dev.new()
plot(postsigma, pch = 19, col = adjustcolor("gray", 2/3), ylim = c(0, 2))
abline(h = 0.3, lty = 3, lwd = 1.5, col = "slateblue")
"""

R"""
library(BART)
y <- $y
x <- $x
bartfit <- wbart(x.train = x, y.train = y, nskip = 0)
g <- function(x) {
  sin(pi * x / 2) / (1 + 2 * x^2 * (sign(x) + 1))
  # x^3
}

dev.new()
plot(y ~ x, col = "gray", pch = 19)
for (i in 1:1000) {
  lines(
    x[order(x)], bartfit$yhat.train[i,][order(x)],
    col = adjustcolor("red", 0.01), lwd = 2
  )
}
curve(g(x), add = TRUE)

dev.new()
yhatup <- apply(bartfit$yhat.train, 2, quantile, 0.975)
yhatlow <- apply(bartfit$yhat.train, 2, quantile, 0.025)
plot(y ~ x, col = "gray", pch = 19)
lines(x[order(x)], bartfit$yhat.train.mean[order(x)], col = "red", lwd = 2)
lines(x[order(x)], yhatup[order(x)], col = "red", lwd = 2, lty = 3)
lines(x[order(x)], yhatlow[order(x)], col = "red", lwd = 2, lty = 3)
curve(g(x), add = TRUE)

dev.new()
plot(bartfit$sigma, pch = 19, col = adjustcolor("gray", 2/3), ylim = c(0, 2))
abline(h = 0.3, lty = 3, lwd = 1.5, col = "red")
"""
