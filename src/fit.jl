###############################################################################
##### Fit method for a continuous response
###############################################################################

function StatsBase.sample(bs::RegBartState, bm::BartModel)
  posterior = RegBartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    drawσ!(bs, bm)
    bm.hypers.sparse ? draws!(bs, bm) : nothing
    if s > bm.opts.nburn
      posterior.mdraws[:,s - bm.opts.nburn] = bs.fhat .+ bm.td.ybar
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
  states = RegBartState(bm)
  post = pmap(bs -> sample(bs, bm), states)
  println("Processing chains...")
  RegBartChain(
    bm,
    reshape(reduce(hcat, [chain.mdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    reshape(reduce(vcat, [chain.σdraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains)
  )
end


###############################################################################
##### Fit method for a binary response ---> ProbitBART
###############################################################################

function StatsBase.sample(bs::ProbitBartState, bm::BartModel)
  posterior = ProbitBartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    bm.hypers.sparse ? draws!(bs, bm) : nothing
    if s > bm.opts.nburn
      posterior.mdraws[:,s - bm.opts.nburn] = cdf.(Normal(), bs.fhat)
      posterior.zdraws[:,s - bm.opts.nburn] = bs.z
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
  states = ProbitBartState(bm)
  post = pmap(bs -> sample(bs, bm), states)
  println("Processing chains...")
  ProbitBartChain(
    bm,
    reshape(reduce(hcat, [chain.mdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(hcat, [chain.zdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains)
  )
end
