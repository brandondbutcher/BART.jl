###############################################################################
##### Predict functions
###############################################################################

function StatsBase.predict(X::Matrix{Float64}, trees::Vector{Tree}, td::TrainData)
  yhat = zeros(size(X)[1])
  for tree in trees
    yhat += leafprob(X, tree) * treemu(tree)
  end
  yhat .+ td.ybar
end
