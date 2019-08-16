module BART

  using Distributions
  using LinearAlgebra
  using StatsBase

  include("treestruct.jl")
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
    Opts
    Posterior
    FriedmanRFG

  export
    softbart

end
