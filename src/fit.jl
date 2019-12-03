###############################################################################
##### Fit method for a continuous response
###############################################################################

function StatsBase.sample(bm::BartModel)
  bs = RegBartState(bm)
  posterior = BartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    drawσ!(bs, bm)
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
  post = pmap(sample, [bm for i in 1:bm.opts.nchains])
  treedraws = reduce(vcat, [chain.treedraws for chain in post])
  σdraws = reduce(vcat, [chain.σdraws for chain in post])
  test = [[treedraws[t], σdraws[t]] for t in 1:(bm.opts.nchains*bm.opts.ndraw)]
  println("Processing chains...")
  mpost = pmap(t -> jmp(bm, t), test)
  monitor = reshape(vcat(reshape(σdraws, bm.opts.ndraw, bm.opts.nchains),
    reshape(mpost, bm.opts.ndraw, bm.opts.nchains)), bm.opts.ndraw, 2, bm.opts.nchains)
  println("finished.")
  BartChain(
    BartInfo(bm.td.dt, bm.td.ybar),
    reshape(reduce(hcat, [chain.mdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    Chains(monitor, ["sigma", "jmp"])
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
      posterior.mdraws[:,s - bm.opts.nburn] = cdf.(Normal(), bs.fhat)
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
  treedraws = reduce(vcat, [chain.treedraws for chain in post])
  test = [[treedraws[t], 1.0] for t in 1:(bm.opts.nchains*bm.opts.ndraw)]
  println("Processing chains...")
  mpost = pmap(t -> jmp(bm, t), test)
  monitor = reshape(mpost, bm.opts.ndraw, 1, bm.opts.nchains)
  println("finished.")
  BartChain(
    reshape(reduce(hcat, [chain.mdraws for chain in post]), bm.td.n, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    Chains(monitor, ["jmp"])
  )
end
