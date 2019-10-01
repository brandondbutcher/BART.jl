###############################################################################
##### Data types
###############################################################################

struct TrainData
  n::Int64
  p::Int64
  xmin::Matrix{Float64}
  xmax::Matrix{Float64}
  ybar::Float64
  ytrain::Vector{Float64}
  σhat::Float64
  function TrainData(X::Matrix{Float64}, y::Vector{Float64})
    n = length(y)
    p = size(X)[2]
    xmin = minimum(X, dims = 1)
    xmax = maximum(X, dims = 1)
    ybar = mean(y)
    ytrain = y .- ybar
    Q = Matrix(qr(X).Q)
    yhat = Q * Q' * ytrain
    σhat = sqrt(dot(ytrain - yhat, ytrain - yhat) / (n - p))
    new(n, p, xmin, xmax, ybar, ytrain, σhat)
  end
end

struct Opts
  nburn::Int64
  ndraw::Int64
  nthin::Int64
  function Opts(nburn = 500, ndraw = 2500, nthin = 1)
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
    τ = ((maximum(td.ytrain) - minimum(td.ytrain)) / (2*k*sqrt(m)))^2
    new(m, k, ν, δ, q, α, β, λmean, τ)
  end
end

struct Posterior
  fdraws::Matrix{Float64}
  σdraws::Vector{Float64}
end
