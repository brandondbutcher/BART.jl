function addnodes!(leftindex::Int64, leftnode::LeafNode, rightindex::Int64, rightnode::LeafNode, tree::DecisionTree)
  treeindices = 1:length(tree.tree)
  if !(leftindex in treeindices)
    newdepth = depth(leftindex)
    minindex = 2^newdepth
    maxindex = 2^(newdepth + 1)   - 1
    newindices = minindex:maxindex
    for i in newindices
      if i == leftindex
        push!(tree.tree, leftnode)
      elseif i == rightindex
        push!(tree.tree, rightnode)
      else
        push!(tree.tree, nothing)
      end
    end
  else
    tree.tree[leftindex] = leftnode
    tree.tree[rightindex] = rightnode
  end
end

function prunenodes!(node::BranchNode, indices::Vector{Int64}, tree::DecisionTree)
  pruneindex = nodeindex(node, tree)
  tree.tree[node.leftchild] = nothing
  tree.tree[node.rightchild] = nothing
  tree.tree[pruneindex] = LeafNode(0, indices, node.parent)
end

function initializetrees(ytrain::Vector{Float64}, m::Int64)
  trees = Array{DecisionTree}(undef, m)
  mu = mean(ytrain) ./ m
  indices = 1:length(ytrain)
  for t in 1:m
    trees[t] = DecisionTree([LeafNode(mu, indices, 0)])
  end
  trees
end

function grow_tree_ratio(alpha, beta, depth, ncuts)
  log(alpha) + 2*log(1 - (alpha/(2+depth))) - beta*log(1+depth) - log(ncuts)
end

function grow_transition_ratio(b, ncuts, w)
  log(b) + log(ncuts) - log(w)
end

function grow_node_likelihood(leftresid, rightresid, s2e, s2mu)
  nleft = length(leftresid)
  nright = length(rightresid)
  n = nleft + nright
  ssqleft = sum(leftresid)^2 / (s2e + nleft*s2mu)
  ssqright = sum(rightresid)^2 / (s2e + nright*s2mu)
  ssq = (sum(leftresid) + sum(rightresid))^2 / (s2e + n*s2mu)
  a = 0.5*(log(s2e) + log(s2e + n*s2mu) -log(s2e + nleft*s2mu) - log(s2e + nright*s2mu))
  b = (s2mu/(2*s2e))*(ssqleft + ssqright - ssq)
  a + b
end

function birthproposal!(tree::DecisionTree, residual::Vector{Float64}, x::Vector{Float64}, cuts::Vector{Float64}, a, b, s2e, s2mu)
  leaves = goodleaves(tree, cuts, x)
  leaf = rand(leaves)
  leafindex = nodeindex(leaf, tree)
  availcuts = goodcuts(leaf, tree, cuts, x)
  navailcuts = length(availcuts)
  cutindex = rand(availcuts)
  cut = cuts[cutindex]
  leftindices = leaf.indices[x[leaf.indices] .<= cut]
  if length(leftindices) == 0
    println(tree.tree)
    println(cut)
    describe(x)
    throw(ErrorException("Left leaf indices have length 0."))
  end
  rightindices = leaf.indices[x[leaf.indices] .> cut]
  if length(rightindices) == 0
    println(tree.tree)
    println(cut)
    describe(x)
    throw(ErrorException("Right leaf indices have length 0."))
  end
  leftresid = residual[leftindices]
  rightresid = residual[rightindices]
  loglik = grow_node_likelihood(leftresid, rightresid, s2e, s2mu)
  only_parent_nodes = onlyparents(tree)
  w = only_parent_nodes == nothing ? 1 : length(only_parent_nodes) + 1
  transratio = grow_transition_ratio(length(leaves), navailcuts, w)
  treeratio = grow_tree_ratio(a, b, depth(leafindex), navailcuts)
  r = exp(transratio + loglik + treeratio)
  if rand() < r
    leftid = leftindex(leafindex)
    rightid = rightindex(leafindex)
    tree.tree[leafindex] = BranchNode(1, cutindex, leftid, rightid, parentindex(leafindex))
    leftnode = LeafNode(0, leftindices, leafindex)
    rightnode = LeafNode(0, rightindices, leafindex)
    addnodes!(leftid, leftnode, rightid, rightnode, tree)
  end
end

function prune_tree_ratio(alpha, beta, depth, ncuts)
  -1*grow_tree_ratio(alpha, beta, depth, ncuts)
end

function prune_transition_ratio(b, ncuts, w)
  log(w) - log(b-1) - log(ncuts)
end

function prune_node_likelihood(leftresid, rightresid, s2e, s2mu)
  -1*grow_node_likelihood(leftresid, rightresid, s2e, s2mu)
end

function deathproposal!(tree::DecisionTree, residual::Vector{Float64}, x::Vector{Float64}, cuts::Vector{Float64}, a, b, s2e, s2mu)
  availnodes = onlyparents(tree)
  node = rand(availnodes)
  index = nodeindex(node, tree)
  leftleaf = leftchild(node, tree)
  rightleaf = rightchild(node, tree)
  leftresid = residual[leftleaf.indices]
  rightresid = residual[rightleaf.indices]
  indices = vcat(leftleaf.indices, rightleaf.indices)
  navailcuts = length(cuts[indices])
  w = availnodes == nothing ? 1 : length(availnodes) + 1
  leaves = leafnodes(tree)
  transratio = prune_transition_ratio(length(leaves), navailcuts, w)
  treeratio = prune_tree_ratio(a, b, depth(index), navailcuts)
  loglik = prune_node_likelihood(leftresid, rightresid, s2e, s2mu)
  r = exp(transratio + loglik + treeratio)
  if rand() < r
    prunenodes!(node, indices, tree)
  end
end

function updatetree!(tree::DecisionTree, residual::Vector{Float64}, x::Vector{Float64}, cuts::Vector{Float64}, a, b, s2e, s2mu)
  if typeof(tree.tree[1]) == LeafNode
    birthproposal!(tree, residual, x, cuts, a, b, s2e, s2mu)
  elseif rand() < 0.5
    birthproposal!(tree, residual, x, cuts, a, b, s2e, s2mu)
  else
    deathproposal!(tree, residual, x, cuts, a, b, s2e, s2mu)
  end
end

function updatemu!(leaf::LeafNode, residual::Vector{Float64}, s2e, s2mu)
  n = length(leaf.indices)
  residual = residual[leaf.indices]
  mu = s2mu*n*mean(residual) / (s2e + n*s2mu)
  sigma = sqrt(s2e*s2mu / (s2e + n*s2mu))
  leaf.mu = rand(Normal(mu, sigma))
end

function updatemu!(tree::DecisionTree, residual::Vector{Float64}, s2e, s2mu)
  leaves = leafnodes(tree)
  for leaf in leaves
    updatemu!(leaf, residual, s2e, s2mu)
  end
end

function updatesigma(ytrain::Vector{Float64}, x:: Vector{Float64}, trees::Vector{DecisionTree}, nu, lambda)
  yhat = predict(trees, x)
  residual = ytrain .- yhat
  n = length(residual)
  a = (nu + n) / 2
  b = 0.5 * (nu*lambda + sum(residual.^2))
  rand(InverseGamma(a, b))
end
