###############################################################################
##### Tree data structures for SoftBART
###############################################################################

abstract type Node end

mutable struct Leaf <: Node
  index::Int64
  mu::Float64
  parent::Int64
end

mutable struct Branch <: Node
  var::Int64
  cut::Float64
  index::Int64
  leftchild::Int64
  rightchild::Int64
  parent::Int64
end

mutable struct Tree
  tree::Vector{Union{Nothing, Node, Node}}
  tau::Float64
  Phi::Matrix{Float64}
end
