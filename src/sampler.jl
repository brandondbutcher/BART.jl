###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function softbart(X::Matrix{Float64}, y::Vector{Float64}, opts::Opts = Opts())
  traindata = TrainData(X, y)
  hypers = Hypers(traindata)
  trees = initializetrees(traindata, hypers)
  yhat = treespredict(trees, traindata)
  s2e = traindata.s2ϵhat
  posterior = Posterior(
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Vector{Float64}(undef, opts.ndraw)
  )
  @time for s in 1:(opts.nburn + opts.ndraw)
    for tree in trees
      yhat_t = yhat .- treepredict(tree)
      rt = traindata.ytrain .- yhat_t
      updatetree!(tree, rt, X, traindata, s2e, hypers)
      updatetau!(X, rt, tree, s2e, traindata, hypers)
      updatemu!(tree, rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(tree)
    end
    yhat = treespredict(trees, traindata)
    s2e = updatesigma(yhat, traindata, hypers)
    if s > opts.nburn
      posterior.yhat[:,s - opts.nburn] = unstandardize(yhat, traindata)
      posterior.σ2[s - opts.nburn] = (traindata.ymax - traindata.ymin)^2 * s2e
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end
