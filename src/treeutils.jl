###############################################################################
##### SoftTree utility functions
###############################################################################

function root(tree::SoftTree)
  tree.tree[1]
end

function leftchild(node::SoftBranch, tree::SoftTree)
  tree.tree[node.leftchild]
end

function rightchild(node::SoftBranch, tree::SoftTree)
  tree.tree[node.rightchild]
end

function leafnodes(tree::SoftTree)
  leafindices = findall(x -> typeof(x) == SoftLeaf, tree.tree)
  tree.tree[leafindices]
end

function probleft(x::Vector{Float64}, node::SoftBranch, tree::SoftTree)
  1 / (1 + exp((x[node.var] - node.cut) / tree.tau))
end

function leafprob(x::Vector{Float64}, tree::SoftTree)
  prob = Float64[]
  rootnode = root(tree)
  if typeof(rootnode) == SoftLeaf
    return 1.0
  end
  goesleft = probleft(x, rootnode, tree)
  leftnode = leftchild(rootnode, tree)
  goesright = 1 - goesleft
  rightnode = rightchild(rootnode, tree)
  leafprob(x, leftnode, tree, goesleft, prob)
  leafprob(x, rightnode, tree, goesright, prob)
end

function leafprob(x::Vector{Float64}, node::SoftBranch, tree::SoftTree, phi::Float64, prob::Vector{Float64})
  goesleft = phi * probleft(x, node, tree)
  leftnode = leftchild(node, tree)
  goesright = phi * (1 - probleft(x, node, tree))
  rightnode = rightchild(node, tree)
  leafprob(x, leftnode, tree, goesleft, prob)
  leafprob(x, rightnode, tree, goesright, prob)
end

function leafprob(x::Vector{Float64}, node::SoftLeaf, tree::SoftTree, phi::Float64, prob::Vector{Float64})
  push!(prob, phi)
end

function leafprob(X::Matrix{Float64}, tree::SoftTree, td::TrainData)
  Lt = length(leafnodes(tree))
  Phit = zeros(td.n, Lt)
  for i in 1:td.n
    Phit[i,:] .= leafprob(X[i,:], tree)
  end
  Phit
end

function treemu(tree::SoftTree)
  leaves = leafnodes(tree)
  mut = Float64[]
  for leaf in leaves
    push!(mut, leaf.mu)
  end
  mut
end

function isleft(node::SoftNode, tree::SoftTree)
  index = findall(x -> x == node, tree.tree)[1]
  iseven(index)
end

function isright(node::SoftNode, tree::SoftTree)
  index = findall(x -> x == node, tree.tree)[1]
  isodd(index)
end

function isroot(node::SoftNode, tree::SoftTree)
  node == tree.tree[1]
end

function nodeindex(node::SoftNode, tree::SoftTree)
  findall(x -> x == node, tree.tree)[1]
end

function leftindex(index::Int64)
  2*index
end

function rightindex(index::Int64)
  2*index + 1
end

function parentindex(index::Int64)
  Int64(floor(index/2))
end

function Base.parent(node::SoftNode, tree::SoftTree)
  if node.parent == 0
    return node
  else
    tree.tree[node.parent]
  end
end

function depth(tree::SoftTree)
  leafindices = findall(x -> typeof(x) == SoftLeaf, tree.tree)
  maxindex = maximum(leafindices)
  floor(log2(maxindex))
end

function depth(index::Int64)
  if index == 0
    return nothing
  end
  Int64(floor(log2(index)))
end
