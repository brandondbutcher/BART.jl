###############################################################################
##### Data structures for training data, MCMC options, & model hyper-parameters
###############################################################################

struct TrainData
  n::Int64
  p::Int64
  xmin::Matrix{Float64}
  xmax::Matrix{Float64}
  X::Matrix{Float64}
  ybar::Float64
  y::Vector{Float64}
  σhat::Float64
  function TrainData(X::Matrix{Float64}, y::Vector{Float64})
    n = length(y)
    p = size(X)[2]
    xmin = minimum(X, dims = 1)
    xmax = maximum(X, dims = 1)
    ybar = mean(y)
    Q = Matrix(qr(X).Q)
    y = y .- ybar
    yhat = Q * Q' * y
    r = y - yhat
    σhat = p >= n ? std(y) : sqrt(dot(r, r) / (n - p))
    new(n, p, xmin, xmax, X, ybar, y, σhat)
  end
end

struct Opts
  nburn::Int64
  ndraw::Int64
  nthin::Int64
  function Opts(;nburn = 100, ndraw = 1000, nthin = 1)
    new(nburn, ndraw, nthin)
  end
end

struct Hypers
  m::Int64
  k::Int64
  ν::Float64
  δ::Float64
  q::Float64
  α::Float64
  β::Float64
  λmean::Float64
  τ::Float64
  function Hypers(td::TrainData; m = 25, k = 2, ν = 3.0, q = 0.9, α = 0.95, β = 2.0, λmean = 0.1)
    δ = 1 / quantile(InverseGamma(ν / 2, ν / (2 * td.σhat^2)), q)
    τ = ((maximum(td.y) - minimum(td.y)) / (2*k*sqrt(m)))^2
    new(m, k, ν, δ, q, α, β, λmean, τ)
  end
end


###############################################################################
##### BartModel type
###############################################################################

struct BartModel
  hypers::Hypers
  opts::Opts
  td::TrainData
  function BartModel(X::Matrix{Float64}, y::Vector{Float64}, opts::Opts = Opts(); hyperargs...)
    td = TrainData(X, y)
    hypers = Hypers(td; hyperargs...)
    new(hypers, opts, td)
  end
end


###############################################################################
##### Data structure for posterior draws
###############################################################################

struct Posterior
  fdraws::Matrix{Float64}
  σdraws::Vector{Float64}
end
