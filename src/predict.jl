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

# function StatsBase.predict(posterior::Vector{RegBartPosterior}, X::Matrix{Float64})
#   X = StatsBase.transform(posterior[1].dt, transpose(X))
#   X = Matrix(transpose(X))
#   trees = reduce(vcat, [chain.treedraws for chain in posterior])
#   reduce(hcat, pmap(trees -> predict(trees, X), trees)) .+ posterior[1].ybar
# end
