import BART

function g(X::Matrix{Float64})
  10sin.(pi * X[:,1] .* X[:,2]) + 20(X[:,3] .- 0.5).^2 + 10X[:,4] + 5X[:,5]
end
n = 1000
p = 5
X = rand(n, p)
truesigma = sqrt(1)
y = g(X) + rand(Normal(0, truesigma), n)

posterior = BART.softbart(X, y, BART.Opts(0, 5000, 1))

describe(sqrt.(posterior.σ2))
plot(sqrt.(posterior.σ2))
hline!([truesigma])
