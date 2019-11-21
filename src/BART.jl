module BART

  using Distributed
  using Distributions
  using LinearAlgebra
  using StatsBase
  using MCMCChains

  import DecisionTree

  include("trees.jl")
  include("models.jl")
  include("proposals.jl")
  include("predict.jl")
  include("fit.jl")

  export
    BartModel,
    Hypers,
    Opts,
    TrainData

  export
    fit,
    predict

end
