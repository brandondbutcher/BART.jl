abstract type DecisionNode end

mutable struct LeafNode <: DecisionNode
  mu::Float64
  indices::Union{Nothing, Vector{Int64}}
  parent::Int64
end

mutable struct BranchNode <: DecisionNode
  var::Int64
  cut::Int64
  leftchild::Int64
  rightchild::Int64
  parent::Int64
end

mutable struct DecisionTree
  tree::Vector{Union{Nothing, BranchNode, LeafNode}}
end

struct BartLeaf <: DecisionNode
  mu::Float64
  parent::Int64
  function BartLeaf(leafnode::LeafNode)
    new(leafnode.mu, leafnode.parent)
  end
end

struct BartTree
  tree::Vector{Union{Nothing, BranchNode, BartLeaf}}
  function BartTree(tree::DecisionTree)
    barttree = []
    for node in tree.tree
      if typeof(node) == LeafNode
        push!(barttree, BartLeaf(node))
      else
        push!(barttree, node)
      end
    end
    new(barttree)
  end
end

struct BartPosterior
  trees::Vector{Vector{BartTree}}
  s2e::Vector{Float64}
  function BartPosterior(S::Int64, m::Int64)
    new([Vector{BartTree}(undef, m) for s in 1:S], Vector{Float64}(undef, S))
  end
end
