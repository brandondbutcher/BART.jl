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
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Vector{Float64}(undef, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Int64.(zeros(traindata.p, opts.ndraw))
  )
  @time for s in 1:(opts.nburn + opts.ndraw)
    for t in 1:hypers.m
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, X, traindata, s2e, hypers)
      updatetau!(X, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    yhat = treespredict(trees, traindata)
    s2e = updatesigma(yhat, traindata, hypers)
    if s > opts.nburn
      posterior.yhat[:,s - opts.nburn] = unstandardize(yhat, traindata)
      posterior.σ2[s - opts.nburn] = (traindata.ymax - traindata.ymin)^2 * s2e
      for t in 1:hypers.m
        posterior.treedepth[t, s - opts.nburn] = depth(trees[t])
        posterior.numleaves[t, s - opts.nburn] = length(leafnodes(trees[t]))
        posterior.varcount[:, s - opts.nburn] += varcounts(trees[t], traindata)
      end
    end
  end
  posterior
end

function softbart(X::Matrix{Float64}, y::Vector{Float64}, traindata::TrainData, hypers::Hypers, opts::Opts = Opts())
  # traindata = TrainData(X, y)
  # hypers = Hypers(traindata)
  trees = initializetrees(traindata, hypers)
  yhat = treespredict(trees, traindata)
  s2e = traindata.s2ϵhat
  posterior = Posterior(
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Vector{Float64}(undef, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Int64.(zeros(traindata.p, opts.ndraw))
  )
  @time for s in 1:(opts.nburn + opts.ndraw)
    for t in 1:hypers.m
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, X, traindata, s2e, hypers)
      updatetau!(X, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    yhat = treespredict(trees, traindata)
    s2e = updatesigma(yhat, traindata, hypers)
    if s > opts.nburn
      posterior.yhat[:,s - opts.nburn] = unstandardize(yhat, traindata)
      posterior.σ2[s - opts.nburn] = (traindata.ymax - traindata.ymin)^2 * s2e
      for t in 1:hypers.m
        posterior.treedepth[t, s - opts.nburn] = depth(trees[t])
        posterior.numleaves[t, s - opts.nburn] = length(leafnodes(trees[t]))
        posterior.varcount[:, s - opts.nburn] += varcounts(trees[t], traindata)
      end
    end
  end
  posterior
end

function softbart(x::Vector{Float64}, y::Vector{Float64}, opts::Opts = Opts())
  X = reshape(x, length(x), 1)
  softbart(X, y, opts)
end

function softbart(X::Matrix{Float64}, Xtest::Matrix{Float64}, y::Vector{Float64}, opts::Opts = Opts())
  traindata = TrainData(X, y)
  ntest = size(Xtest)[1]
  hypers = Hypers(traindata)
  trees = initializetrees(traindata, hypers)
  yhat = treespredict(trees, traindata)
  s2e = traindata.s2ϵhat
  posterior = Posterior(
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Matrix{Float64}(undef, size(Xtest)[1], opts.ndraw),
    Vector{Float64}(undef, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Int64.(zeros(traindata.p, opts.ndraw))
  )
  nsplits = 0
  @time for s in 1:(opts.nburn + opts.ndraw)
    for t in 1:hypers.m
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, X, traindata, s2e, hypers)
      updatetau!(X, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    yhat = treespredict(trees, traindata)
    s2e = updatesigma(yhat, traindata, hypers)
    if s > opts.nburn
      posterior.yhat[:,s - opts.nburn] = unstandardize(yhat, traindata)
      posterior.yhat_test[:, s - opts.nburn] = predict(Xtest, trees, traindata)
      posterior.σ2[s - opts.nburn] = (traindata.ymax - traindata.ymin)^2 * s2e
      for t in 1:hypers.m
        posterior.treedepth[t, s - opts.nburn] = depth(trees[t])
        posterior.numleaves[t, s - opts.nburn] = length(leafnodes(trees[t]))
        posterior.varcount[:, s - opts.nburn] += varcounts(trees[t], traindata)
      end
    end
  end
  posterior
end

function mandopt(X::Matrix{Float64}, y::Vector{Float64}, mand::Vector{Int64}, opt::Vector{Int64}, opts::Opts = Opts())
  traindata = TrainData(X, y)
  hypers = Hypers(traindata, m = 50)
  trees = initializetrees(traindata, hypers)
  yhat = treespredict(trees, traindata)
  s2e = traindata.s2ϵhat
  posterior = Posterior(
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Matrix{Float64}(undef, traindata.n, opts.ndraw),
    Vector{Float64}(undef, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Matrix{Int64}(undef, hypers.m, opts.ndraw),
    Int64.(zeros(traindata.p, opts.ndraw))
  )
  m0 = 30
  m1 = 15
  m2 = 5
  @time for s in 1:(opts.nburn + opts.ndraw)
    for t in 1:m0
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, X, mand, traindata, s2e, hypers)
      updatetau!(X, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    for t in (m0+1):m1
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, X, opt, traindata, s2e, hypers)
      updatetau!(X, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    for t in (m1+1):m2
      yhat_t = yhat .- treepredict(trees[t])
      rt = traindata.ytrain .- yhat_t
      trees[t] = updatetree(trees[t], rt, X, vcat(mand, opt), traindata, s2e, hypers)
      updatetau!(X, rt, trees[t], s2e, traindata, hypers)
      updatemu!(trees[t], rt, s2e, hypers)
      yhat = yhat_t .+ treepredict(trees[t])
    end
    yhat = treespredict(trees, traindata)
    s2e = updatesigma(yhat, traindata, hypers)
    if s > opts.nburn
      posterior.yhat[:,s - opts.nburn] = unstandardize(yhat, traindata)
      posterior.σ2[s - opts.nburn] = (traindata.ymax - traindata.ymin)^2 * s2e
      for t in 1:hypers.m
        posterior.treedepth[t, s - opts.nburn] = depth(trees[t])
        posterior.numleaves[t, s - opts.nburn] = length(leafnodes(trees[t]))
        # posterior.varcount[:, s - opts.nburn] += varcounts(trees[t], traindata)
      end
    end
  end
  posterior
end
