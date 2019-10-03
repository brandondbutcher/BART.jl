###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function StatsBase.fit(bartmodel::BartModel)
  bartstate = BartState(bartmodel)
  posterior = Posterior(bartmodel)
  @time for s in 1:bartmodel.opts.S
    drawtrees!(bartstate, bartmodel)
    drawσ!(bartstate, bartmodel)
    if s > bartmodel.opts.nburn
      posterior.fdraws[:,s - bartmodel.opts.nburn] = bartstate.yhat .+ bartmodel.td.ybar
      posterior.σdraws[s - bartmodel.opts.nburn] = bartstate.σ
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end
