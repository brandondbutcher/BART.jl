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
  mu::Float64
end

mutable struct Tree
  root::Node
  tau::Float64
  Phi::Matrix{Float64}
end
