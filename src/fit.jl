###############################################################################
##### Fit method for a continuous response
###############################################################################

function StatsBase.sample(bs::RegBartState, bm::BartModel)
  posterior = RegBartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    drawσ!(bs, bm)
    if bm.hypers.sparse
      draws!(bs, bm)
      drawα!(bs, bm)
      # println(bs.shape)
    end
    if s > bm.opts.nburn
      posterior.sdraws[:,s - bm.opts.nburn] = exp.(bs.s)
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
  init_trees = deepcopy(map(state -> Tree[bt.tree for bt in state.ensemble.trees], states))
  post = pmap(bs -> sample(bs, bm), states)
  println("Processing chains...")
  RegBartChain(
    bm,
    init_trees,
    reshape(reduce(hcat, [chain.sdraws for chain in post]), bm.td.p, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    reshape(reduce(vcat, [chain.σdraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains)
  )
end

function update(post::RegBartChain, ndraw::Int)
  s = post.bm.opts.ndraw
  new_opts = Opts(ndraw = ndraw, nburn = 0, nchains = post.bm.opts.nchains)
  states = RegBartState[]
  for c in 1:post.bm.opts.nchains
    trees = post.treedraws[s,1,c]
    σ = post.σdraws[s,1,c]
    S = [leafprob(post.bm.td.X, tree) for tree in trees]
    fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
    yhat = vec(sum(fhats, dims = 2))
    bt = Vector{BartTree}(undef, post.bm.hypers.m)
    for t in eachindex(trees)
      rt = post.bm.td.y - sum(fhats[:,eachindex(trees) .!= t], dims = 2)
      Ω = inv(transpose(S[t]) * S[t] / σ^2 + I / post.bm.hypers.τ)
      rhat = vec(transpose(S[t]) * rt / σ^2)
      bt[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
    end
    bs = RegBartState(BartEnsemble(bt), post.mdraws[:,s,c], σ, ones(post.bm.td.p) ./ post.bm.td.p)
    push!(states, bs)
  end
  bm = BartModel(post.bm.hypers, new_opts, post.bm.td)
  newdraws = pmap(bs -> sample(bs, bm), states)
  RegBartChain(
    BartModel(bm.hypers, Opts(ndraw = ndraw + post.bm.opts.ndraw, nburn = post.bm.opts.nburn), bm.td),
    post.init_trees,
    hcat(post.sdraws,
      reshape(reduce(hcat, [chain.sdraws for chain in newdraws]), bm.td.p, bm.opts.ndraw, bm.opts.nchains)),
    vcat(post.treedraws,
      reshape(reduce(vcat, [chain.treedraws for chain in newdraws]), bm.opts.ndraw, 1, bm.opts.nchains)),
    vcat(post.σdraws,
      reshape(reduce(vcat, [chain.σdraws for chain in newdraws]), bm.opts.ndraw, 1, bm.opts.nchains))
  )
end


###############################################################################
##### Fit method for a binary response ---> ProbitBART
###############################################################################

function StatsBase.sample(bs::ProbitBartState, bm::BartModel)
  posterior = ProbitBartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    if bm.hypers.sparse
      draws!(bs, bm)
      drawα!(bs, bm)
      # println(bs.shape)
    end
    if s > bm.opts.nburn
      posterior.sdraws[:,s - bm.opts.nburn] = exp.(bs.s)
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
  init_trees = deepcopy(map(state -> Tree[bt.tree for bt in state.ensemble.trees], states))
  post = pmap(bs -> sample(bs, bm), states)
  println("Processing chains...")
  ProbitBartChain(
    bm,
    init_trees,
    reshape(reduce(hcat, [chain.sdraws for chain in post]), bm.td.p, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains)
  )
end

function update(post::ProbitBartChain, ndraw::Int)
  s = post.bm.opts.ndraw
  new_opts = Opts(ndraw = ndraw, nburn = 0, nchains = post.bm.opts.nchains)
  states = ProbitBartState[]
  z = map(y -> y == 1 ?
    rand(Truncated(Normal(), 0, Inf)) :
    rand(Truncated(Normal(), -Inf, 0)),
    post.bm.td.y
  )
  for c in 1:post.bm.opts.nchains
    trees = post.treedraws[s,1,c]
    S = [leafprob(post.bm.td.X, tree) for tree in trees]
    fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
    yhat = vec(sum(fhats, dims = 2))
    bt = BartEnsemble(Vector{BartTree}(undef, post.bm.hypers.m))
    for t in eachindex(trees)
      rt = z - sum(fhats[:,eachindex(trees) .!= t], dims = 2)
      Ω = inv(transpose(S[t]) * S[t] + I / post.bm.hypers.τ)
      rhat = vec(transpose(S[t]) * rt)
      bt.trees[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
    end
    push!(states, ProbitBartState(bt, yhat, z, 1, ones(post.bm.td.p) ./ post.bm.td.p))
  end
  bm = BartModel(post.bm.hypers, new_opts, post.bm.td)
  newdraws = pmap(bs -> sample(bs, bm), states)
  ProbitBartChain(
    BartModel(bm.hypers, Opts(ndraw = ndraw + post.bm.opts.ndraw, nburn = post.bm.opts.nburn), bm.td),
    post.init_trees,
    hcat(post.sdraws,
      reshape(reduce(hcat, [chain.s for chain in newdraws]), bm.td.p, bm.opts.ndraw, bm.opts.nchains)),
    vcat(post.treedraws,
      reshape(reduce(vcat, [chain.treedraws for chain in newdraws]), bm.opts.ndraw, 1, bm.opts.nchains))
  )
end


###############################################################################
##### Fit method for a survival response ---> SurvBART
###############################################################################

function preprocess(X::Matrix{Float64}, time::Vector{Float64}, event::Vector{Int})
  n, p = size(X)
  ut = sort(unique(time))
  yij = Int[]
  tj = Float64[]
  obs = Int[]
  for i in 1:n
    for t in filter(t -> t <= time[i], ut)
      t < time[i] ? push!(yij, 0) : push!(yij, event[i])
      push!(tj, t)
      push!(obs, i)
    end
  end
  ns = counts(obs)
  Xrep = [transpose(reshape(repeat(X[i,:],  ns[i]), p, ns[i])) for i in 1:n]
  Xrep = reduce(vcat, Xrep)
  (yij, hcat(tj, Xrep))
end

function StatsBase.fit(BartModel, X::Matrix{Float64}, time::Vector{Float64}, event::Vector{Int},
  opts = Opts(); hyperags...)
  y, X = preprocess(X, time, event)
  bm = BartModel(X, y, opts; hyperags...)
  states = ProbitBartState(bm)
  init_trees = deepcopy(map(state -> Tree[bt.tree for bt in state.ensemble.trees], states))
  post = pmap(bs -> sample(bs, bm), states)
  println("Processing chains...")
  treedraws = reduce(hcat, pmap(p -> p.treedraws, post))
  pdraws = reduce(hcat, pmap(t -> cdf.(Normal(), predict(t, bm.td.X)), treedraws))
  ut = sort(unique(time))
  obs = [i for i in 1:length(time) for t in filter(t -> t <= time[i], ut)]
  ns = counts(obs)
  indices = [findall(obs .== i) for i in 1:bm.td.n]
  sdraws = [cumprod(1 .- pdraws[idx,:], dims = 1) for idx in indices]
  sdraws = reduce(vcat, sdraws)
  SurvBartChain(
    bm,
    time,
    event,
    init_trees,
    reshape(reduce(hcat, [chain.sdraws for chain in post]), bm.td.p, bm.opts.ndraw, bm.opts.nchains),
    reshape(reduce(vcat, [chain.treedraws for chain in post]), bm.opts.ndraw, 1, bm.opts.nchains),
    sdraws
  )
end
