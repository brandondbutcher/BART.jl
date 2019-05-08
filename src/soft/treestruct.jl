###############################################################################
##### Tree data structures for SoftBART
###############################################################################

abstract type SoftNode end

mutable struct SoftLeaf <: SoftNode
  mu::Float64
  parent::Int64
end

mutable struct SoftBranch <: SoftNode
  var::Int64
  cut::Float64
  leftchild::Int64
  rightchild::Int64
  parent::Int64
end

mutable struct SoftTree
  tree::Vector{Union{Nothing, SoftNode, SoftNode}}
  tau::Float64
  Phi::Matrix{Float64}
end

struct TrainData
  n::Int64
  p::Int64
  xmin::Matrix{Float64}
  xmax::Matrix{Float64}
  ymin::Float64
  ymax::Float64
  ymidrange::Float64
  ytrain::Vector{Float64}
  s2ϵhat::Float64
  function TrainData(X::Matrix{Float64}, y::Vector{Float64})
    n = length(y)
    p = size(X)[2]
    xmin = minimum(X, dims = 1)
    xmax = maximum(X, dims = 1)
    ymax = maximum(y)
    ymin = minimum(y)
    ymidrange = (ymax + ymin) / 2
    ytrain = (y .- ymidrange) / (ymax - ymin)
    Q = Matrix(qr(X).Q)
    yhat = Q * Q' * ytrain
    s2ϵhat = dot(ytrain - yhat, ytrain - yhat) / (n - p)
    new(n, p, xmin, xmax, ymin, ymax, ymidrange, ytrain, s2ϵhat)
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
  λ::Float64
  q::Float64
  α::Float64
  β::Float64
  τ_mean::Float64
  s2μ::Float64
  function Hypers(td::TrainData; m = 20, k = 2, ν = 3.0, q = 0.9, α = 0.95, β = 2.0, τ_mean = 0.1)
    λ = 1 / quantile(InverseGamma(ν / 2, ν / (2 * td.s2ϵhat)), q)
    s2μ = (0.5 / (k*sqrt(m)))^2
    new(m, k, ν, λ, q, α, β, τ_mean, s2μ)
  end
end
