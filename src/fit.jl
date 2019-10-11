###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function StatsBase.fit(BartModel, X, y, resp, opts = Opts(); hyperags...)
  bm = BartModel(X, y, resp, opts; hyperags...)
  bs = BartState(bm)
  posterior = Posterior(bm)
  @time for s in 1:bm.opts.S
    if resp == ProbitBart drawz!(bs, bm) end
    drawtrees!(bs, bm)
    if resp == Bart drawσ!(bs, bm) end
    if s > bm.opts.nburn
      posterior.fdraws[:,s - bm.opts.nburn] = predict(bs, bm)
      posterior.σdraws[s - bm.opts.nburn] = bs.σ
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end
