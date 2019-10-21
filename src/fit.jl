###############################################################################
##### SoftBART MCMC sampler
###############################################################################

function StatsBase.fit(BartModel, X::Matrix{Float64}, y::Vector{Float64}, opts = Opts(); hyperags...)
  bm = BartModel(X, y, opts; hyperags...)
  bs = RegBartState(bm)
  posterior = RegBartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    drawσ!(bs, bm)
    if s > bm.opts.nburn
      posterior.fdraws[:,s - bm.opts.nburn] = predict(bs, bm) .+ bm.td.ybar
      posterior.σdraws[s - bm.opts.nburn] = bs.σ
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end

function StatsBase.fit(BartModel, X::Matrix{Float64}, y::Vector{Int}, opts = Opts(); hyperags...)
  bm = BartModel(X, y, opts; hyperags...)
  bs = ProbitBartState(bm)
  posterior = ProbitBartPosterior(bm)
  @time for s in 1:bm.opts.S
    drawtrees!(bs, bm)
    if s > bm.opts.nburn
      posterior.fdraws[:,s - bm.opts.nburn] = predict(bs, bm)
      posterior.zdraws[:,s - bm.opts.nburn] = bs.z
    end
    if s % 100 == 0
      println("MCMC iteration $s complete.")
    end
  end
  posterior
end
