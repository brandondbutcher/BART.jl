###############################################################################
##### Tree data structures for SoftBART
###############################################################################

abstract type Node end

mutable struct Branch <: Node
  var::Int64
  cut::Float64
  left::Node
  right::Node
end

mutable struct Leaf <: Node
  μ::Float64
end

mutable struct SuffStats
  Lt::Int
  Ω::Matrix{Float64}
  rhat::Vector{Float64}
end

mutable struct Tree
  root::Node
  λ::Float64
  S::Matrix{Float64}
  ss::SuffStats
end


###############################################################################
##### Tree utility functions
###############################################################################

function leafnodes(tree::Tree)
  leaves = Leaf[]
  if isa(tree.root, Leaf)
    push!(leaves, tree.root)
  else
    leafnodes(tree.root.left, leaves)
    leafnodes(tree.root.right, leaves)
  end
end

function leafnodes(node::Node, leaves::Vector{Leaf})
  if isa(node, Leaf)
    push!(leaves, node)
  else
    leafnodes(node.left, leaves)
    leafnodes(node.right, leaves)
  end
end

function Base.parent(node::Node, tree::Tree)
  parent(node, tree.root)
end

function Base.parent(node::Node, cnode::Branch)
  if (cnode.left == node) || (cnode.right == node)
    return cnode
  else
    parent(node, cnode.left) == nothing ? parent(node, cnode.right) : parent(node, cnode.left)
  end
end

function Base.parent(node::Node, cnode::Leaf)
  nothing
end

function onlyparents(tree::Tree)
  branches = Branch[]
  if isa(tree.root, Leaf)
    return [tree.root]
  else
    onlyparents(tree.root, branches)
  end
end

function onlyparents(branch::Branch, branches::Vector{Branch})
  if isa(branch.left, Leaf) & isa(branch.right, Leaf)
    push!(branches, branch)
  else
    onlyparents(branch.left, branches)
    onlyparents(branch.right, branches)
  end
  return branches
end

function onlyparents(leaf::Leaf, branches::Vector{Branch})
  nothing
end

function depth(node::Node, tree::Tree)
  tree.root == node ? 0 : 1 + depth(parent(node, tree), tree)
end

function isleft(node::Node, tree::Tree)
  parentnode = parent(node, tree)
  parentnode.left == node ? true : false
end
