module BART

  using Distributions
  using LinearAlgebra
  using StatsBase

  include("trees.jl")
  include("treeutils.jl")
  include("data.jl")
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
    Posterior

  export
    softbart

end
