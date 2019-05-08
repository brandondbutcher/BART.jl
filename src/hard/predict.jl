function StatsBase.predict(trees::Vector{DecisionTree}, x::Vector{Float64})
  yhat = zeros(n)
  for tree in trees
    leaves = leafnodes(tree)
    for leafnode in leaves
      yhat[leafnode.indices] .+= leafnode.mu
    end
  end
  yhat
end

function StatsBase.predict(tree::BartTree, x::Vector{Float64}, cuts::Vector{Float64})
  n = length(x)
  muhat = zeros(n)
  indices = trues(n)
  predict(tree, tree.tree[1], x, cuts, muhat, indices)
end

function StatsBase.predict(tree::BartTree, node::BranchNode, x::Vector{Float64}, cuts::Vector{Float64}, muhat::Vector{Float64}, indices::BitArray)
  goleft = (x .<= cuts[node.cut]) .& indices
  goright = (x .> cuts[node.cut]) .& indices
  predict(tree, tree.tree[node.leftchild], x, cuts, muhat, goleft)
  predict(tree, tree.tree[node.rightchild], x, cuts, muhat, goright)
end

function StatsBase.predict(tree::BartTree, node::BartLeaf, x::Vector{Float64}, cuts::Vector{Float64}, muhat::Vector{Float64}, indices::BitArray)
  indices = findall(indices)
  muhat[indices] .= node.mu
  muhat
end

function StatsBase.predict(barttrees::Vector{BartTree}, x::Vector{Float64}, cuts::Vector{Float64})
  n = length(x)
  m = length(barttrees)
  yhat = zeros(n)
  for t in 1:m
    yhat .+= predict(barttrees[t], x, cuts)
  end
  yhat
end

function StatsBase.predict(bartpost::BartPosterior, x::Vector{Float64}, cuts::Vector{Float64})
  n = length(x)
  S = length(bartpost.trees)
  yhat = zeros(S, n)
  for s in 1:S
    yhat[s,:] = predict(bartpost.trees[s], x, cuts)
  end
  yhat
end
