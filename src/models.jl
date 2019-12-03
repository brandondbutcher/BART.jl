###############################################################################
##### Training Data type and constructors
###############################################################################

struct TrainData
  n::Int
  p::Int
  xmin::Matrix{Float64}
  xmax::Matrix{Float64}
  dt::ZScoreTransform{Float64}
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
  TrainData(n, p, xmin, xmax, dt, X, ybar, y, σhat)
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
  TrainData(n, p, xmin, xmax, dt, X, ybar, y, 1.0)
end


###############################################################################
##### MCMC options type
###############################################################################

struct Opts
  nchains::Int
  nburn::Int
  ndraw::Int
  nthin::Int
  S::Int64
  function Opts(;nchains = 4, nburn = 500, ndraw = 500, nthin = 1)
    new(nchains, nburn, ndraw, nthin, nburn + ndraw)
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
  λfix::Bool
  τ::Float64
  function Hypers(td::TrainData; m = 25, k = 2, ν = 3.0, q = 0.9, α = 0.95, β = 2.0, λmean = 0.1, λfix = false)
    δ = 1 / quantile(InverseGamma(ν / 2, ν / (2 * td.σhat^2)), q)
    if isa(td.y, Vector{Int})
      τ = (3.0 / (k*sqrt(m)))^2
    else
      τ = ((maximum(td.y) - minimum(td.y)) / (2*k*sqrt(m)))^2
    end
    new(m, k, ν, δ, q, α, β, λmean, λfix, τ)
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

mutable struct SuffStats
  Lt::Int
  Ω::Matrix{Float64}
  rhat::Vector{Float64}
end

function suffstats(rt::Vector{Float64}, S::Matrix{Float64}, bs::BartState, bm::BartModel)
  Lt = size(S, 2)
  Ω = inv(transpose(S) * S / bs.σ^2 + I / bm.hypers.τ)
  rhat = transpose(S) * rt / bs.σ^2
  SuffStats(Lt, Ω, rhat)
end

mutable struct BartTree
  tree::Tree
  S::Matrix{Float64}
  ss::SuffStats
end

mutable struct BartEnsemble
  trees::Vector{BartTree}
end

mutable struct RegBartState <: BartState
  ensemble::BartEnsemble
  fhat::Vector{Float64}
  σ::Float64
end

function Base.convert(Node, x)
  isa(x, DecisionTree.Leaf) ? Leaf(x.majority) :
    Branch(x.featid, x.featval, convert(Node, x.left), convert(Node, x.right))
end

function RegBartState(bm::BartModel)
  rf = DecisionTree.fit!(
    DecisionTree.RandomForestRegressor(
      n_trees = bm.hypers.m, max_depth = 5, min_purity_increase = 1, partial_sampling = 0.5),
      bm.td.X, bm.td.y
  )
  trees = Tree.(convert.(Node, rf.ensemble.trees), bm.hypers.λmean)
  S = [leafprob(bm.td.X, tree) for tree in trees]
  fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
  yhat = vec(sum(fhats, dims = 2))
  σhat = sqrt(dot(bm.td.y - yhat, bm.td.y - yhat) / (bm.td.n - bm.td.p))
  bt = Vector{BartTree}(undef, bm.hypers.m)
  for t in eachindex(trees)
    rt = bm.td.y - sum(fhats[:,eachindex(trees) .!= t], dims = 2)
    Ω = inv(transpose(S[t]) * S[t] / σhat^2 + I / bm.hypers.τ)
    rhat = vec(transpose(S[t]) * rt / σhat^2)
    bt[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
  end
  RegBartState(BartEnsemble(bt), yhat, σhat)
end

mutable struct ProbitBartState <: BartState
  ensemble::BartEnsemble
  fhat::Vector{Float64}
  z::Vector{Float64}
  σ::Float64
end

function ProbitBartState(bm::BartModel)
  z = map(y -> y == 1 ?
    rand(Truncated(Normal(), 0, Inf)) :
    rand(Truncated(Normal(), -Inf, 0)),
    bm.td.y
  )
  rf = DecisionTree.fit!(
    DecisionTree.RandomForestRegressor(
      n_trees = bm.hypers.m, max_depth = 5, min_purity_increase = 1, partial_sampling = 0.5),
      bm.td.X, z
  )
  trees = Tree.(convert.(Node, rf.ensemble.trees), bm.hypers.λmean)
  S = [leafprob(bm.td.X, tree) for tree in trees]
  fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
  yhat = vec(sum(fhats, dims = 2))
  bt = BartEnsemble(Vector{BartTree}(undef, bm.hypers.m))
  for t in eachindex(trees)
    rt = z - sum(fhats[:,eachindex(trees) .!= t], dims = 2)
    Ω = inv(transpose(S[t]) * S[t] + I / bm.hypers.τ)
    rhat = vec(transpose(S[t]) * rt)
    bt.trees[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
  end
  ProbitBartState(bt, yhat, z, 1)
end


###############################################################################
##### Posterior draws from BART model
###############################################################################

# abstract type BartPosterior end

struct BartPosterior
  mdraws::Matrix{Float64}
  σdraws::Vector{Float64}
  treedraws::Vector{Vector{Tree}}
  function BartPosterior(bm::BartModel)
    new(
      Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw),
      Vector{Float64}(undef, bm.opts.ndraw),
      Vector{Vector{Tree}}(undef, bm.opts.ndraw)
    )
  end
end

# struct ProbitBartPosterior <: BartPosterior
#   pdraws::Matrix{Float64}
#   lctp::Vector{Float64}
#   treedraws::Vector{Vector{Tree}}
#   function ProbitBartPosterior(bm::BartModel)
#     new(
#       Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw),
#       Vector{Float64}(undef, bm.opts.ndraw),
#       Vector{Vector{Tree}}(undef, bm.opts.ndraw)
#     )
#   end
# end

struct BartInfo
  dt::ZScoreTransform{Float64}
  ybar::Float64
end


struct BartChain
  info::BartInfo
  mdraws::Array{Float64}
  treedraws::Array{Vector{Tree}}
  monitor::Chains
end
