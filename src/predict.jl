###############################################################################
##### Predict functions
###############################################################################

function getμ(tree::Tree)
  [leaf.μ for leaf in leafnodes(tree.root)]
end

function StatsBase.predict(bt::BartTree)
  bt.S * getμ(bt.tree)
end

function StatsBase.predict(bs::BartState, bm::BartModel)
  fhat = zeros(bm.td.n)
  for bt in bs.ensemble.trees
    fhat += predict(bt)
  end
  fhat
end

function StatsBase.predict(trees::Vector{Tree}, X::Matrix{Float64})
  yhat = zeros(size(X, 1))
  for tree in trees
    yhat += leafprob(X, tree) * getμ(tree)
  end
  yhat
end

function StatsBase.predict(bc::RegBartChain, X::Matrix{Float64})
  X = StatsBase.transform(bc.bm.td.dt, transpose(X))
  X = Matrix(transpose(X))
  treedraws = reshape(bc.treedraws, size(bc.treedraws, 1)*size(bc.treedraws, 3))
  reduce(hcat, pmap(t -> predict(t, X), treedraws)) .+ bc.info.ybar
end

function StatsBase.predict(bc::ProbitBartChain, X::Matrix{Float64})
  X = StatsBase.transform(bc.bm.td.dt, transpose(X))
  X = Matrix(transpose(X))
  treedraws = reshape(bc.treedraws, size(bc.treedraws, 1)*size(bc.treedraws, 3))
  reduce(hcat, pmap(t -> cdf.(Normal(), predict(t, X)), treedraws))
end
