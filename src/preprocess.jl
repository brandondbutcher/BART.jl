###############################################################################
##### Functions to pre-process training data for fitting SoftBART
###############################################################################

# function standardize(y::Vector{Float64})
#   ymax = maximum(y)
#   ymin = minimum(y)
#   mr = (ymax + ymin) / 2
#   (y .- mr) / (ymax - ymin)
# end

function unstandardize(yhat::Vector{Float64}, td::TrainData)
   yhat .* (td.ymax - td.ymin) .+ td.ymidrange
end
