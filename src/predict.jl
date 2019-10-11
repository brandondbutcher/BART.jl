###############################################################################
##### Predict functions
###############################################################################

function getμ(tree::Tree)
  [leaf.μ for leaf in leafnodes(tree)]
end

function StatsBase.predict(tree::Tree)
  tree.S * getμ(tree)
end

function StatsBase.predict(trees::Vector{Tree}, bm::BartModel)
  fhat = zeros(bm.td.n)
  for tree in trees
    fhat += predict(tree)
  end
  fhat
end

function StatsBase.predict(bs::BartState, bm::BartModel)
  fhat = zeros(bm.td.n)
  for tree in bs.trees
    fhat += predict(tree)
  end
  fhat
end

# function StatsBase.predict(X::Matrix{Float64}, trees::Vector{Tree}, td::TrainData)
#   yhat = zeros(size(X)[1])
#   for tree in trees
#     yhat += leafprob(X, tree) * treemu(tree)
#   end
#   yhat .+ td.ybar
# end
