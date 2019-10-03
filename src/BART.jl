module BART

  using Distributions
  using LinearAlgebra
  using StatsBase

  include("trees.jl")
  include("treeutils.jl")
  include("models.jl")
  include("proposals.jl")
  include("predict.jl")
  include("preprocess.jl")
  include("sampler.jl")

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
    Posterior

  export
    fit

end
