###############################################################################
##### Tree data structures for SoftBART
###############################################################################

abstract type SoftNode end

mutable struct SoftLeaf <: SoftNode
  mu::Float64
  parent::Int64
end

mutable struct SoftBranch <: SoftNode
  var::Int64
  cut::Float64
  leftchild::Int64
  rightchild::Int64
  parent::Int64
end

mutable struct SoftTree
  tree::Vector{Union{Nothing, SoftNode, SoftNode}}
  tau::Float64
  Phi::Matrix{Float64}
end
