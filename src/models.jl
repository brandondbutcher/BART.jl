###############################################################################
##### BART response types
###############################################################################

abstract type BartResp end

struct Bart <: BartResp
  y::Vector{Float64}
  ybar::Float64
  function Bart(y::Vector{Float64})
    ybar = mean(y)
    y = y .- ybar
    new(y, ybar)
  end
end

resp(br::Bart) = br.y

mutable struct ProbitBart <: BartResp
  y::Vector{Int}
  z::Vector{Float64}
  function ProbitBart(y::Vector{Int})
    z = map(y -> y == 1 ? max(randn(), 0) : min(randn(), 0), y)
    new(y, z)
  end
end

resp(br::ProbitBart) = br.z

# mutable struct SurvBart <: BartResp
#   time::Vector{Float64}
#   event::Vector{Int}
#   z::Vector{Float64}
# end


###############################################################################
##### Training Data type and constructors
###############################################################################

struct TrainData
  n::Int64
  p::Int64
  xmin::Matrix{Float64}
  xmax::Matrix{Float64}
  X::Matrix{Float64}
  resp::BartResp
  σhat::Float64
end

function TrainData(X::Matrix{Float64}, resp::Bart)
  n = length(resp.y)
  p = size(X, 2)
  Q = Matrix(qr(X).Q)
  y = resp.y .- resp.ybar
  yhat = Q * Q' * y
  r = y - yhat
  σhat = p >= n ? std(y) : sqrt(dot(r, r) / (n - p))
  dt = fit(ZScoreTransform, transpose(X))
  X = StatsBase.transform(dt, transpose(X))
  X = Matrix(transpose(X))
  xmin = minimum(X, dims = 1)
  xmax = maximum(X, dims = 1)
  TrainData(n, p, xmin, xmax, X, resp, σhat)
end

function TrainData(X::Matrix{Float64}, resp::ProbitBart)
  n = length(resp.y)
  p = size(X, 2)
  dt = fit(ZScoreTransform, transpose(X))
  X = StatsBase.transform(dt, transpose(X))
  X = Matrix(transpose(X))
  xmin = minimum(X, dims = 1)
  xmax = maximum(X, dims = 1)
  TrainData(n, p, xmin, xmax, X, resp, 1.0)
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
    if isa(td.resp, ProbitBart)
      τ = (3.0 / (k*sqrt(m)))^2
    else
      τ = ((maximum(td.resp.y) - minimum(td.resp.y)) / (2*k*sqrt(m)))^2
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
  function BartModel(X::Matrix{Float64}, y::AbstractVector, R, opts::Opts; hyperargs...)
    td = TrainData(X, R(y))
    hypers = Hypers(td; hyperargs...)
    new(hypers, opts, td)
  end
end


###############################################################################
##### State of sampler
###############################################################################

mutable struct BartState
  trees::Vector{Tree}
  σ::Float64
  fhat::Vector{Float64}
end

function BartState(bm::BartModel)
  trees = Vector{Tree}(undef, bm.hypers.m)
  μ = mean(resp(bm.td.resp)) ./ bm.hypers.m
  S = ones(bm.td.n, 1)
  Ω = inv(transpose(S) * S / bm.td.σhat^2 + I / bm.hypers.τ)
  rhat = transpose(S) * resp(bm.td.resp) / bm.td.σhat^2
  ss = SuffStats(1, Ω, rhat)
  for t in 1:bm.hypers.m
    trees[t] = Tree(Leaf(μ), bm.hypers.λmean, S, ss)
  end
  BartState(trees, bm.td.σhat, repeat([μ*bm.hypers.m], bm.td.n))
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

struct Posterior
  fdraws::Matrix{Float64}
  σdraws::Vector{Float64}
  function Posterior(bm::BartModel)
    new(
      Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw),
      Vector{Float64}(undef, bm.opts.ndraw)
    )
  end
end
