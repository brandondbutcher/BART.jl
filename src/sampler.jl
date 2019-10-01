###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function softbart(X::Matrix{Float64}, y::Vector{Float64}, opts::Opts = Opts())
  traindata = TrainData(X, y)
  hypers = Hypers(traindata)
  trees = initializetrees(traindata, hypers)
  yhat = treespredict(trees, traindata)
  σ = traindata.σhat
  posterior = Posterior(
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Vector{Float64}(undef, opts.ndraw)
  )
  @time for s in 1:(opts.nburn + opts.ndraw)
    for tree in trees
      yhat_t = yhat .- treepredict(tree)
      rt = traindata.ytrain .- yhat_t
      updatetree!(tree, rt, X, traindata, σ^2, hypers)
      updateλ!(X, rt, tree, σ^2, traindata, hypers)
      updateμ!(tree, rt, σ^2, hypers)
      yhat = yhat_t .+ treepredict(tree)
    end
    yhat = treespredict(trees, traindata)
    σ = updateσ(yhat, traindata, hypers)
    if s > opts.nburn
      posterior.fdraws[:,s - opts.nburn] = yhat .+ traindata.ybar
      posterior.σdraws[s - opts.nburn] = σ
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end
