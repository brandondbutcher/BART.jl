###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function softbart(x::Matrix{Float64}, y::Vector{Float64}, opts::Opts = Opts())
  traindata = TrainData(x, y)
  hypers = Hypers(traindata)
  trees = initializetrees(traindata, hypers)
  yhat = treespredict(trees, traindata)
  s2e = traindata.s2ϵhat
  yhatdraws = Matrix{Float64}(undef, n, opts.ndraw)
  s2edraws = Vector{Float64}(undef, opts.ndraw)
  @time for s in 1:(opts.nburn + opts.ndraw)
    for t in 1:hypers.m
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, x, traindata, s2e, hypers)
      updatetau!(x, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    yhat = treespredict(trees, traindata)
    s2e = updatesigma(yhat, traindata, hypers)
    if s > opts.nburn
      s2edraws[s - opts.nburn] = (traindata.ymax - traindata.ymin)^2 * s2e
      yhatdraws[:,s - opts.nburn] = unstandardize(yhat, traindata)
    end
  end
  [yhatdraws, s2edraws]
end
