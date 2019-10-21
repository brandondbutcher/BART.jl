module BART

  using Distributions
  using LinearAlgebra
  using StatsBase

  include("trees.jl")
  include("models.jl")
  include("proposals.jl")
  include("predict.jl")
  include("fit.jl")

  export
    Node
    Branch
    Leaf
    Tree

  export
    Hypers
    TrainData
    Opts
    BartModel

  export
    fit

end
