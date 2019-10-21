###############################################################################
##### Training Data type and constructors
###############################################################################

struct TrainData
  n::Int64
  p::Int64
  xmin::Matrix{Float64}
  xmax::Matrix{Float64}
  X::Matrix{Float64}
  ybar::Float64
  y::AbstractVector
  σhat::Float64
end

function TrainData(X::Matrix{Float64}, y::Vector{Float64})
  n = length(y)
  p = size(X, 2)
  Q = Matrix(qr(X).Q)
  ybar = mean(y)
  y = y .- ybar
  yhat = Q * Q' * y
  r = y - yhat
  σhat = p >= n ? std(y) : sqrt(dot(r, r) / (n - p))
  dt = fit(ZScoreTransform, transpose(X))
  X = StatsBase.transform(dt, transpose(X))
  X = Matrix(transpose(X))
  xmin = minimum(X, dims = 1)
  xmax = maximum(X, dims = 1)
  TrainData(n, p, xmin, xmax, X, ybar, y, σhat)
end

function TrainData(X::Matrix{Float64}, y::Vector{Int})
  n = length(y)
  p = size(X, 2)
  dt = fit(ZScoreTransform, transpose(X))
  X = StatsBase.transform(dt, transpose(X))
  X = Matrix(transpose(X))
  xmin = minimum(X, dims = 1)
  xmax = maximum(X, dims = 1)
  ybar = mean(y)
  TrainData(n, p, xmin, xmax, X, ybar, y, 1.0)
end


###############################################################################
##### MCMC options type
###############################################################################

struct Opts
  nburn::Int64
  ndraw::Int64
  nthin::Int64
  S::Int64
  function Opts(;nburn = 100, ndraw = 1000, nthin = 1)
    new(nburn, ndraw, nthin, nburn + ndraw)
  end
end


###############################################################################
##### BART hyperparameters type
###############################################################################

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
    if isa(td.y, Vector{Int})
      τ = (3.0 / (k*sqrt(m)))^2
    else
      τ = ((maximum(td.y) - minimum(td.y)) / (2*k*sqrt(m)))^2
    end
    new(m, k, ν, δ, q, α, β, λmean, τ)
  end
end


###############################################################################
##### Bart model type and constructors
###############################################################################

struct BartModel
  hypers::Hypers
  opts::Opts
  td::TrainData
  function BartModel(X::Matrix{Float64}, y::AbstractVector, opts::Opts; hyperargs...)
    td = TrainData(X, y)
    hypers = Hypers(td; hyperargs...)
    new(hypers, opts, td)
  end
end


###############################################################################
##### State of sampler
###############################################################################

abstract type BartState end

mutable struct RegBartState <: BartState
  trees::Vector{Tree}
  fhat::Vector{Float64}
  σ::Float64
end

function RegBartState(bm::BartModel)
  trees = Vector{Tree}(undef, bm.hypers.m)
  μ = mean(bm.td.y) ./ bm.hypers.m
  S = ones(bm.td.n, 1)
  Ω = inv(transpose(S) * S / bm.td.σhat^2 + I / bm.hypers.τ)
  rhat = transpose(S) * bm.td.y / bm.td.σhat^2
  ss = SuffStats(1, Ω, rhat)
  for t in 1:bm.hypers.m
    trees[t] = Tree(Leaf(μ), bm.hypers.λmean, S, ss)
  end
  RegBartState(trees, repeat([μ*bm.hypers.m], bm.td.n), bm.td.σhat)
end

mutable struct ProbitBartState <: BartState
  trees::Vector{Tree}
  fhat::Vector{Float64}
  z::Vector{Float64}
  σ::Float64
end

function ProbitBartState(bm::BartModel)
  trees = Vector{Tree}(undef, bm.hypers.m)
  z = map(y -> y == 1 ? max(randn(), 0) : min(randn(), 0), bm.td.y)
  μ = mean(z) ./ bm.hypers.m
  S = ones(bm.td.n, 1)
  Ω = inv(transpose(S) * S / bm.td.σhat^2 + I / bm.hypers.τ)
  rhat = transpose(S) * (bm.hypers.m - 1) * repeat([μ], bm.td.n) / bm.td.σhat^2
  ss = SuffStats(1, Ω, rhat)
  for t in 1:bm.hypers.m
    trees[t] = Tree(Leaf(μ), bm.hypers.λmean, S, ss)
  end
  ProbitBartState(trees, repeat([μ*bm.hypers.m], bm.td.n), z, 1.0)
end

function suffstats(rt, S, bs, bm)
  Lt = size(S, 2)
  Ω = inv(transpose(S) * S / bs.σ^2 + I / bm.hypers.τ)
  rhat = transpose(S) * rt / bs.σ^2
  ss = SuffStats(Lt, Ω, rhat)
  SuffStats(Lt, Ω, rhat)
end


###############################################################################
##### Posterior draws from BART model
###############################################################################

abstract type BartPosterior end

struct RegBartPosterior <: BartPosterior
  fdraws::Matrix{Float64}
  σdraws::Vector{Float64}
  function RegBartPosterior(bm::BartModel)
    new(
      Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw),
      Vector{Float64}(undef, bm.opts.ndraw)
    )
  end
end

struct ProbitBartPosterior <: BartPosterior
  fdraws::Matrix{Float64}
  zdraws::Matrix{Float64}
  function ProbitBartPosterior(bm::BartModel)
    new(
      Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw),
      Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw)
    )
  end
end
