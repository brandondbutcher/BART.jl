###############################################################################
##### Tree utility functions
###############################################################################

function root(tree::Tree)
  tree.tree[1]
end

function leftchild(branch::Branch, tree::Tree)
  tree.tree[branch.leftchild]
end

function rightchild(branch::Branch, tree::Tree)
  tree.tree[branch.rightchild]
end

function leafnodes(tree::Tree)
  leafindices = findall(x -> typeof(x) == Leaf, tree.tree)
  tree.tree[leafindices]
end

function probleft(x::Vector{Float64}, branch::Branch, tree::Tree)
  1 / (1 + exp((x[branch.var] - branch.cut) / tree.tau))
end

function leafprob(x::Vector{Float64}, tree::Tree)
  prob = Float64[]
  rootnode = root(tree)
  if typeof(rootnode) == Leaf
    return 1.0
  end
  goesleft = probleft(x, rootnode, tree)
  leftnode = leftchild(rootnode, tree)
  goesright = 1 - goesleft
  rightnode = rightchild(rootnode, tree)
  leafprob(x, leftnode, tree, goesleft, prob)
  leafprob(x, rightnode, tree, goesright, prob)
end

function leafprob(x::Vector{Float64}, branch::Branch, tree::Tree, ψ::Float64, ϕ::Vector{Float64})
  goesleft = ψ * probleft(x, branch, tree)
  leftnode = leftchild(branch, tree)
  goesright = ψ * (1 - probleft(x, branch, tree))
  rightnode = rightchild(branch, tree)
  leafprob(x, leftnode, tree, goesleft, ϕ)
  leafprob(x, rightnode, tree, goesright, ϕ)
end

function leafprob(x::Vector{Float64}, leaf::Leaf, tree::Tree, ψ::Float64, ϕ::Vector{Float64})
  push!(ϕ, ψ)
end

function leafprob(X::Matrix{Float64}, tree::Tree, td::TrainData)
  Lt = length(leafnodes(tree))
  Phit = zeros(td.n, Lt)
  for i in 1:td.n
    Phit[i,:] .= leafprob(X[i,:], tree)
  end
  Phit
end

function treemu(tree::Tree)
  leaves = leafnodes(tree)
  mut = Float64[]
  for leaf in leaves
    push!(mut, leaf.mu)
  end
  mut
end

function isleft(node::Node, tree::Tree)
  # index = findall(x -> x == node, tree.tree)[1]
  iseven(node.index)
end

function isright(node::Node, tree::Tree)
  # index = findall(x -> x == node, tree.tree)[1]
  isodd(node.index)
end

function isroot(node::Node, tree::Tree)
  node == tree.tree[1]
end

# function nodeindex(node::SoftNode, tree::SoftTree)
#   findall(x -> x == node, tree.tree)[1]
# end

function leftindex(index::Int64)
  2*index
end

function rightindex(index::Int64)
  2*index + 1
end

function parentindex(index::Int64)
  Int64(floor(index/2))
end

function Base.parent(node::Node, tree::Tree)
  if isroot(node, tree)
    node
  else
    tree.tree[node.parent]
  end
end

function depth(tree::Tree)
  leafindices = findall(x -> typeof(x) == Leaf, tree.tree)
  maxindex = maximum(leafindices)
  floor(log2(maxindex))
end

function depth(index::Int64)
  if index == 0
    return nothing
  end
  Int64(floor(log2(index)))
end

function branches(tree::Tree)
  branchindices = findall(x -> typeof(x) == Branch, tree.tree)
  tree.tree[branchindices]
end

function onlyparents(tree::Tree)
  if length(tree.tree) == 1
    return nothing
  end
  branchnodes = branches(tree)
  indices = findall(
    x -> (typeof(leftchild(x, tree)) == Leaf) &
      (typeof(rightchild(x, tree)) == Leaf),
    branchnodes
  )
  branchnodes[indices]
end
