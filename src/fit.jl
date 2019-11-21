###############################################################################
##### Fit method for a continuous response
###############################################################################

function StatsBase.sample(bm::BartModel)
  bs = RegBartState(bm)
  posterior = BartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    if s > bm.opts.nburn
      posterior.lctp[s - bm.opts.nburn] = lctp(bs, bm)
    end
    drawσ!(bs, bm)
    if s > bm.opts.nburn
      posterior.mdraws[:,s - bm.opts.nburn] = predict(bs, bm) .+ bm.td.ybar
      posterior.σdraws[s - bm.opts.nburn] = bs.σ
      posterior.treedraws[s - bm.opts.nburn] = [deepcopy(t.tree) for t in bs.ensemble.trees]
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end

function StatsBase.fit(BartModel, X::Matrix{Float64}, y::Vector{Float64}, opts = Opts(); hyperags...)
  bm = BartModel(X, y, opts; hyperags...)
  post = pmap(sample, [bm for i in 1:bm.opts.nchains])
  monitor = reduce(hcat, [hcat(chain.lctp, chain.σdraws) for chain in post])
  monitor = reshape(monitor, bm.opts.ndraw, 2, bm.opts.nchains)
  BartChain(
    reshape(reduce(hcat, [chain.mdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    Chains(monitor, ["lctp", "sigma"])
  )
end


###############################################################################
##### Fit method for a binary response ---> ProbitBART
###############################################################################

function sample_latent(bm::BartModel)
  bs = ProbitBartState(bm)
  posterior = BartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    if s > bm.opts.nburn
      posterior.mdraws[:,s - bm.opts.nburn] = cdf.(Normal(), predict(bs, bm))
      posterior.lctp[s - bm.opts.nburn] = lctp(bs, bm)
      posterior.treedraws[s - bm.opts.nburn] = [deepcopy(t.tree) for t in bs.ensemble.trees]
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end

function StatsBase.fit(BartModel, X::Matrix{Float64}, y::Vector{Int}, opts = Opts(); hyperags...)
  bm = BartModel(X, y, opts; hyperags...)
  post = pmap(sample_latent, [bm for i in 1:bm.opts.nchains])
  monitor = reduce(hcat, [chain.lctp for chain in post])
  monitor = reshape(monitor, bm.opts.ndraw, 1, bm.opts.nchains)
  BartChain(
    reshape(reduce(hcat, [chain.mdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    Chains(monitor, ["lctp"])
  )
end
