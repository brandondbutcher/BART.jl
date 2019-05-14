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

function g(X)
  y = zeros(size(X)[1])
  for i in 1:length(y)
    if (X[i,1] .<= 0.5) .& (X[i,2] .<= 0.5)
      y[i] = 1
    elseif (X[i,1] .<= 0.5) .& (X[i,2] .> 0.5)
      y[i] = 3
    elseif X[i,1] .> 0.5
      y[i] = 5
    end
  end
  y
end

n = 300
truesigma = sqrt(0.25)

x1 = vcat(rand(Uniform(0.1, 0.4), 200), rand(Uniform(0.6, 0.9), 100))
x2 = vcat(rand(Uniform(0.1, 0.4), 100), rand(Uniform(0.6, 0.9), 100), rand(Uniform(0.6, 0.9), 100))
x3 = vcat(rand(Uniform(0.6, 0.9), 200), rand(Uniform(0.1, 0.4), 100))
X = hcat(x1, x2, x3)

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
  ifelse(
    X[,1] <= 0.5 & X[,2] <= 0.5, 1,
    ifelse(
      X[,1] <= 0.5 & X[,2] > 0.5, 3,
      ifelse(
        X[,1] > 0.5, 5, NA
      )
    )
  )
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
