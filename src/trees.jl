###############################################################################
##### Tree data structures for SoftBART
###############################################################################

abstract type Node end

mutable struct Branch <: Node
  var::Int
  cut::Float64
  left::Node
  right::Node
end

mutable struct Leaf <: Node
  μ::Float64
end

mutable struct Tree
  root::Node
  λ::Float64
end


###############################################################################
##### Tree utility functions
###############################################################################

function leafnodes(node::Node)
  isa(node, Leaf) ? [node] :
    reduce(vcat, [leafnodes(node.left), leafnodes(node.right)])
end

function Base.parent(node::Node, tree::Tree)
  parent(node, tree.root)
end

function Base.parent(node::Node, child_node::Branch)
  if (child_node.left == node) || (child_node.right == node)
    return child_node
  else
    isa(parent(node, child_node.left), Nothing) ?
      parent(node, child_node.right) : parent(node, child_node.left)
  end
end

function Base.parent(node::Node, child_node::Leaf)
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

onlyparents(leaf::Leaf, branches::Vector{Branch}) = nothing

function depth(node::Node, tree::Tree)
  tree.root == node ? 0 : 1 + depth(parent(node, tree), tree)
end

function depth(tree)
  maximum([depth(leaf, tree) for leaf in leafnodes(tree.root)])
end

function isleft(node::Node, tree::Tree)
  parentnode = parent(node, tree)
  parentnode.left == node ? true : false
end
