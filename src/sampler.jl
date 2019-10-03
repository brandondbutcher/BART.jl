###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function StatsBase.fit(bartmodel::BartModel)
  trees = initializetrees(bartmodel)
  yhat = treespredict(trees, bartmodel)
  σ = bartmodel.td.σhat
  posterior = Posterior(
    Matrix{Float64}(undef, bartmodel.td.n, bartmodel.opts.ndraw),
    Vector{Float64}(undef, bartmodel.opts.ndraw)
  )
  @time for s in 1:(bartmodel.opts.nburn + bartmodel.opts.ndraw)
    for tree in trees
      yhat_t = yhat .- treepredict(tree)
      rt = bartmodel.td.y .- yhat_t
      updateT!(tree, rt, σ^2, bartmodel)
      updateλ!(rt, tree, σ^2, bartmodel)
      updateμ!(tree, rt, σ^2, bartmodel)
      yhat = yhat_t .+ treepredict(tree)
    end
    yhat = treespredict(trees, bartmodel)
    σ = updateσ(yhat, bartmodel)
    if s > bartmodel.opts.nburn
      posterior.fdraws[:,s - bartmodel.opts.nburn] = yhat .+ bartmodel.td.ybar
      posterior.σdraws[s - bartmodel.opts.nburn] = σ
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end
