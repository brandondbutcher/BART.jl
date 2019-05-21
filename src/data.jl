###############################################################################
##### SoftTree utility functions
###############################################################################

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

struct FriedmanRFG
  X::Matrix{Float64}
  yobs::Vector{Float64}
  ytrue::Vector{Float64}
  σ2::Float64
  function FriedmanRFG(n::Int64; p = 10, stn = 1.0, nfuns = 20)
    X = collect(rand(MvNormal(repeat([0], p), 1), n)')
    a = rand(Uniform(-1, 1), nfuns)
    theta = 2.0
    nvars = Int64.(floor.(1.5 .+ rand(Exponential(theta), nfuns)))
    G = Matrix{Float64}(undef, n, nfuns)
    for l in 1:nfuns
      vars = rand(1:p, nvars[l])
      Z = X[:,vars]
      mu = rand(MvNormal(repeat([0], nvars[l]), 1))
      Q,R = qr(randn(nvars[l], nvars[l]))
      U = Q * Diagonal(sign.(diag(R)))
      lower = minimum(minimum(Z, dims = 1))
      upper = maximum(maximum(Z, dims = 1))
      d = rand(Uniform(lower, upper), nvars[l]).^2
      D = Matrix(Diagonal(d))
      V = U * D * U'
      g = Vector{Float64}(undef, n)
      for i in 1:n
        g[i] = exp(-0.5 * (Z[i,:] - mu)' * V * (Z[i,:] - mu))
      end
      G[:,l] = g
    end
    ytrue = G * a
    σ2 = stn * mean(abs.(ytrue .- median(ytrue)))
    ε = rand(Normal(0, sqrt(σ2)), n)
    yobs = ytrue + ε
    new(X, yobs, ytrue, σ2)
  end
end
