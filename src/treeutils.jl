###############################################################################
##### Tree utility functions
###############################################################################

## Get leafnodes from tree: Returns Vector{Leaf}
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

## Get the parent node of some node of interest in the tree
## the parent node of root is defined to be of type Nothing
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

## Get the branches in the tree that are only parents
## i.e., branch nodes whose children are both leaf nodes
## returns Vector{Branch}
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

## Determine the depth of a given node
## by convention, the depth of the root node is defined to be zero.
function depth(node::Node, tree::Tree)
  tree.root == node ? 0 : 1 + depth(parent(node, tree), tree)
end

## Determine whether a node is a left node
## returns true if the node is a left node and false if it's not
## (i.e., a right node)
function isleft(node::Node, tree::Tree)
  parentnode = parent(node, tree)
  parentnode.left == node ? true : false
end
