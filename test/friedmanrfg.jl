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

# traindata = TrainData(data.X, data.yobs)
# hypers = Hypers(traindata, m = 200)
softfit = softbart(data.X, data.yobs)

using RCall
ytrue = data.ytrue
truesigma2 = data.σ2
yhatpost = softfit.yhat
s2epost = softfit.σ2
R"""
ytrue <- $ytrue
yhatpost <- $yhatpost
s2epost <- $s2epost

yhatmean <- apply(yhatpost, 1, mean)
ciupper <- apply(yhatpost, 1, quantile, probs = 0.95)
cilower <- apply(yhatpost, 1, quantile, probs = 0.05)

dev.new()
plot(yhatmean ~ ytrue, pch = 19, ylim = c(-3, 3))
abline(0, 1, lty = 3)
segments(
  x0 = ytrue, x1 = ytrue,
  y0 = cilower, y1 = ciupper,
  col = adjustcolor("black", 1/3)
)

covered <- rep(NA, length(ytrue))
for (i in 1:length(ytrue)) {
  covered[i] <- cilower[i] <= ytrue[i] & ytrue[i] <= ciupper[i]
}

dev.new()
plot(s2epost, pch = 19, col = adjustcolor("gray", 2 / 3))
abline(h = $truesigma2, lty = 3)
"""
