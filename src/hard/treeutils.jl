function Base.parent(node::DecisionNode, tree::DecisionTree)
  if node.parent == 0
    return node
  else
    tree.tree[node.parent]
  end
end

function leftchild(node::BranchNode, tree::DecisionTree)
  tree.tree[node.leftchild]
end

function rightchild(node::BranchNode, tree::DecisionTree)
  tree.tree[node.rightchild]
end

function children(node::BranchNode, tree::DecisionTree)
  tree.tree[node.leftchild:node.rightchild]
end

function branches(tree::DecisionTree)
  branchindices = findall(x -> typeof(x) == BranchNode, tree.tree)
  tree.tree[branchindices]
end

function onlyparents(tree::DecisionTree)
  if length(tree.tree) == 1
    return nothing
  end
  branchnodes = branches(tree)
  indices = findall(
    x -> (typeof(leftchild(x, tree)) == LeafNode) &
      (typeof(rightchild(x, tree)) == LeafNode),
    branchnodes
  )
  branchnodes[indices]
end

function root(tree::DecisionTree)
  tree.tree[1]
end

function depth(tree::DecisionTree)
  leafindices = findall(x -> typeof(x) == LeafNode, tree.tree)
  maxindex = maximum(leafindices)
  floor(log2(maxindex))
end

function depth(index::Int64)
  Int64(floor(log2(index)))
end

function parentindex(index::Int64)
  Int64(floor(index/2))
end

function nodeindex(node::DecisionNode, tree::DecisionTree)
  findall(x -> x == node, tree.tree)[1]
end

function leftindex(index::Int64)
  2*index
end

function rightindex(index::Int64)
  2*index + 1
end

function isleft(node::DecisionNode, tree::DecisionTree)
  index = findall(x -> x == node, tree.tree)[1]
  iseven(index)
end

function isright(node::DecisionNode, tree::DecisionTree)
  index = findall(x -> x == node, tree.tree)[1]
  isodd(index)
end

function leafnodes(tree::DecisionTree)
  leafindices = findall(x -> typeof(x) == LeafNode, tree.tree)
  tree.tree[leafindices]
end

function goodcuts(node::LeafNode, tree::DecisionTree, cuts::Vector{Float64}, x::Vector{Float64})
  ncuts = length(cuts)
  parentnode = parent(node, tree)
  findall((cuts .> minimum(x[node.indices])) .& (cuts .< maximum(x[node.indices])))
end

function goodleaves(tree::DecisionTree, cuts::Vector{Float64}, x::Vector{Float64})
  leaves = leafnodes(tree)
  goodindices = findall(
    y -> (length(y.indices) >= 2)  & (length(goodcuts(y, tree, cuts, x)) >= 2),
    leaves
  )
  leaves[goodindices]
end
