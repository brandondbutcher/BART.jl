###############################################################################
##### Load packages
###############################################################################

using LinearAlgebra, Distributions, StatsBase, Plots


###############################################################################
##### Tree data structures for SoftBART
###############################################################################

abstract type SoftNode end

mutable struct SoftLeaf <: SoftNode
  mu::Float64
  parent::Int64
end

mutable struct SoftBranch <: SoftNode
  cut::Float64
  leftchild::Int64
  rightchild::Int64
  parent::Int64
end

mutable struct SoftTree
  tree::Vector{Union{Nothing, SoftNode, SoftNode}}
  tau::Float64
end


###############################################################################
##### SoftTree utility functions
###############################################################################

function probleft(x::Float64, cut::Float64, tau::Float64)
  1 / (1 + exp((x - cut) / tau))
  # pdf(Normal(0, 1), (x - cut) / tau)
  # cdf(Normal(0, 1), (x - cut) / tau)
end

function root(tree::SoftTree)
  tree.tree[1]
end

function leftchild(node::SoftBranch, tree::SoftTree)
  tree.tree[node.leftchild]
end

function rightchild(node::SoftBranch, tree::SoftTree)
  tree.tree[node.rightchild]
end

function leafnodes(tree::SoftTree)
  leafindices = findall(x -> typeof(x) == SoftLeaf, tree.tree)
  tree.tree[leafindices]
end

function leafprob(x::Float64, tree::SoftTree)
  prob = Float64[]
  rootnode = root(tree)
  if typeof(rootnode) == SoftLeaf
    return 1.0
  end
  goesleft = probleft(x, rootnode.cut, tree.tau)
  leftnode = leftchild(rootnode, tree)
  goesright = 1 - goesleft
  rightnode = rightchild(rootnode, tree)
  leafprob(x, leftnode, tree, goesleft, prob)
  leafprob(x, rightnode, tree, goesright, prob)
end

function leafprob(x::Float64, node::SoftBranch, tree::SoftTree, phi::Float64, prob::Vector{Float64})
  goesleft = phi * probleft(x, node.cut, tree.tau)
  leftnode = leftchild(node, tree)
  goesright = phi * (1 - probleft(x, node.cut, tree.tau))
  rightnode = rightchild(node, tree)
  leafprob(x, leftnode, tree, goesleft, prob)
  leafprob(x, rightnode, tree, goesright, prob)
end

function leafprob(x::Float64, node::SoftLeaf, tree::SoftTree, phi::Float64, prob::Vector{Float64})
  push!(prob, phi)
end

function leafprob(x::Vector{Float64}, tree::SoftTree)
  Lt = length(leafnodes(tree))
  n = length(x)
  Phit = zeros(n, Lt)
  for i in 1:n
    Phit[i,:] .= leafprob(x[i], tree)
  end
  Phit
end

function treemu(tree::SoftTree)
  leaves = leafnodes(tree)
  mut = Float64[]
  for leaf in leaves
    push!(mut, leaf.mu)
  end
  mut
end

function mll(rt, Phit, s2e, s2mu)
  n = length(rt)
  Lt = size(Phit)[2]
  Omega = inv(transpose(Phit) * Phit / s2e + I / s2mu)
  rhat = transpose(Phit) * rt / s2e
  mll = 0.5 * logdet(2 * pi * Omega)
  mll -= 0.5 * n * log(2 * pi * s2e)
  mll -= 0.5 * Lt * log(2 * pi * s2mu)
  mll -= 0.5 * dot(rt, rt) / s2e
  mll += 0.5 * dot(rhat, Omega * rhat)
  mll
end

function initializetrees(ytrain::Vector{Float64}, m::Int64, tau::Float64)
  trees = Array{SoftTree}(undef, m)
  mu = mean(ytrain) ./ m
  indices = 1:length(ytrain)
  for t in 1:m
    trees[t] = SoftTree([SoftLeaf(mu, 0)], tau)
  end
  trees
end

function treepredict(x::Vector{Float64}, trees::Vector{SoftTree})
  n = length(x)
  m = length(trees)
  ytrain_hat = zeros(n)
  for t in 1:m
    Phit = leafprob(x, trees[t])
    mut = treemu(trees[t])
    ytrain_hat += Phit * mut
  end
  ytrain_hat
end

function partresid(ytrain::Vector{Float64}, x::Vector{Float64}, trees::Vector{SoftTree}, t::Int64)
  ytrain .- treepredict(x, trees[1:end .!= t])
end

function StatsBase.predict(x::Vector{Float64}, trees::Vector{SoftTree})
  yhat = treepredict(x, trees)
end

function isleft(node::SoftNode, tree::SoftTree)
  index = findall(x -> x == node, tree.tree)[1]
  iseven(index)
end

function isright(node::SoftNode, tree::SoftTree)
  index = findall(x -> x == node, tree.tree)[1]
  isodd(index)
end

function isroot(node::SoftNode, tree::SoftTree)
  node == tree.tree[1]
end

function nodeindex(node::SoftNode, tree::SoftTree)
  findall(x -> x == node, tree.tree)[1]
end

function leftindex(index::Int64)
  2*index
end

function rightindex(index::Int64)
  2*index + 1
end

function parentindex(index::Int64)
  Int64(floor(index/2))
end

function Base.parent(node::SoftNode, tree::SoftTree)
  if node.parent == 0
    return node
  else
    tree.tree[node.parent]
  end
end

function depth(tree::SoftTree)
  leafindices = findall(x -> typeof(x) == SoftLeaf, tree.tree)
  maxindex = maximum(leafindices)
  floor(log2(maxindex))
end

function depth(index::Int64)
  Int64(floor(log2(index)))
end

function leftparent(node::SoftNode, tree::SoftTree)
  if isroot(node, tree)
    return nothing
  end
  parentnode = parent(node, tree)
  if isleft(node, tree) & (isleft(parentnode, tree) | isroot(parentnode, tree))
    return nothing
  else
    nodeidx = nodeindex(node, tree)
    if iseven(nodeidx)
      return parent(parentnode, tree).cut
    else
      return parentnode.cut
    end
  end
end

function rightparent(node::SoftNode, tree::SoftTree)
  if isroot(node, tree)
    return nothing
  end
  parentnode = parent(node, tree)
  if isright(node, tree) & (isright(parentnode, tree) | isroot(parentnode, tree))
    return nothing
  else
    nodeidx = nodeindex(node, tree)
    if isodd(nodeidx)
      return parent(parentnode, tree).cut
    else
      return parentnode.cut
    end
  end
end

function drawcut(node::SoftLeaf, tree::SoftTree, xmin::Float64, xmax::Float64)
  a = [xmin, leftparent(node, tree)]
  a = filter(x -> x != nothing, a)
  b = [xmax, rightparent(node, tree)]
  b = filter(x -> x != nothing, b)
  lower = maximum(a)
  upper = minimum(b)
  rand(Uniform(lower, upper))
end

function growleaf!(leaf::SoftLeaf, tree::SoftTree, xmin::Float64, xmax::Float64)
  leafindex = nodeindex(leaf, tree)
  leftid = leftindex(leafindex)
  rightid = rightindex(leafindex)
  newcut = drawcut(leaf, tree, xmin::Float64, xmax::Float64)
  tree.tree[leafindex] = SoftBranch(newcut, leftid, rightid, parentindex(leafindex))
  leftnode = SoftLeaf(0, leafindex)
  rightnode = SoftLeaf(0, leafindex)
  treeindices = 1:length(tree.tree)
  if !(leftid in treeindices)
    newdepth = depth(leftid)
    minindex = 2^newdepth
    maxindex = 2^(newdepth + 1)   - 1
    newindices = minindex:maxindex
    for i in newindices
      if i == leftid
        push!(tree.tree, leftnode)
      elseif i == rightid
        push!(tree.tree, rightnode)
      else
        push!(tree.tree, nothing)
      end
    end
  else
    tree.tree[leftid] = leftnode
    tree.tree[rightid] = rightnode
  end
end

function probgrow(depth, alpha, beta)
  alpha * (1.0 + depth) ^ (-beta)
end

function birth_tree_ratio(tree::SoftTree, alpha, beta)
  d = depth(tree)
  numr = probgrow(d, alpha, beta) * (1 - probgrow(d + 1, alpha, beta)) ^ 2
  denomr = 1 - probgrow(d, alpha, beta)
  log(numr) - log(denomr)
end

function birthprob(tree::SoftTree)
  typeof(tree.tree[1]) == SoftLeaf ? 1.0 : 0.5
end

function branches(tree::SoftTree)
  branchindices = findall(x -> typeof(x) == SoftBranch, tree.tree)
  tree.tree[branchindices]
end

function onlyparents(tree::SoftTree)
  if length(tree.tree) == 1
    return nothing
  end
  branchnodes = branches(tree)
  indices = findall(
    x -> (typeof(leftchild(x, tree)) == SoftLeaf) &
      (typeof(rightchild(x, tree)) == SoftLeaf),
    branchnodes
  )
  branchnodes[indices]
end

function birth_trans_ratio(tree::SoftTree, tree_prime::SoftTree)
  pb = birthprob(tree)
  pd = 1 - birthprob(tree_prime)
  Lt = length(leafnodes(tree))
  b = length(onlyparents(tree_prime))
  numr = pd / (b + 1)
  denomr = pb / Lt
  log(numr) - log(denomr)
end

function mll_ratio(rt, Phit, Phit_prime, s2e, s2mu)
  mllt = mll(rt, Phit, s2e, s2mu)
  mllt_prime = mll(rt, Phit_prime, s2e, s2mu)
  mllt_prime - mllt
end

function pruneleaves!(node::SoftBranch, tree::SoftTree)
  pruneindex = nodeindex(node, tree)
  tree.tree[node.leftchild] = nothing
  tree.tree[node.rightchild] = nothing
  tree.tree[pruneindex] = SoftLeaf(0, node.parent)
end

function birthproposal(tree::SoftTree, rt::Vector{Float64}, x::Vector{Float64}, alpha, beta, s2e, s2mu, xmin, xmax)
  tree_prime = SoftTree([], tree.tau)
  tree_prime.tree = copy(tree.tree)
  leaves_prime = leafnodes(tree_prime)
  leaf_prime = rand(leaves_prime)
  growleaf!(leaf_prime, tree_prime, xmin, xmax)
  Phit = leafprob(x, tree)
  Phit_prime = leafprob(x, tree_prime)
  mloglikratio = mll_ratio(rt, Phit, Phit_prime, s2e, s2mu)
  treeratio = birth_tree_ratio(tree, alpha, beta)
  transratio = birth_trans_ratio(tree, tree_prime)
  logr = mloglikratio + treeratio + transratio
  return log(rand()) < logr ? tree_prime : tree
end

function death_tree_ratio(tree::SoftTree, alpha, beta)
  1 / birth_tree_ratio(tree, alpha, beta)
end

function death_trans_ratio(tree::SoftTree, tree_prime::SoftTree)
  numr = birthprob(tree_prime) / (length(leafnodes(tree_prime)) - 1)
  denomr = (1 - birthprob(tree)) / (length(onlyparents(tree_prime)))
  log(numr) - log(denomr)
end

function deathproposal(tree::SoftTree, rt::Vector{Float64}, x::Vector{Float64}, alpha, beta, s2e, s2mu)
  tree_prime = SoftTree([], tree.tau)
  tree_prime.tree = copy(tree.tree)
  branches_prime = onlyparents(tree_prime)
  branch_prime = rand(branches_prime)
  pruneleaves!(branch_prime, tree_prime)
  Phit = leafprob(x, tree)
  Phit_prime = leafprob(x, tree_prime)
  mloglikratio = mll_ratio(rt, Phit, Phit_prime, s2e, s2mu)
  treeratio = death_tree_ratio(tree, alpha, beta)
  transratio = death_trans_ratio(tree, tree_prime)
  logr = mloglikratio + treeratio + transratio
  return log(rand()) < logr ? tree_prime : tree
end

function updatetree(tree::SoftTree, rt::Vector{Float64}, x::Vector{Float64}, alpha, beta, s2e, s2mu, xmin::Float64, xmax::Float64)
  if rand() < birthprob(tree)
    return birthproposal(tree::SoftTree, rt::Vector{Float64}, x::Vector{Float64}, alpha, beta, s2e, s2mu, xmin::Float64, xmax::Float64)
  else
    return deathproposal(tree::SoftTree, rt::Vector{Float64}, x::Vector{Float64}, alpha, beta, s2e, s2mu)
  end
end

function updatetau!(x::Vector{Float64}, rt::Vector{Float64}, tree::SoftTree, s2e, s2mu)
  Phit_tau = leafprob(x, tree)
  mllt_tau = mll(rt, Phit_tau, s2e, s2mu)
  lp_tau = logpdf(Exponential(0.1), tree.tau)
  log_tau = log(tree.tau)
  log_tauprime = log(tree.tau) + rand(Uniform(-1, 1))
  tree.tau = exp(log_tauprime)
  Phit_tauprime = leafprob(x, tree)
  mllt_tauprime = mll(rt, Phit_tauprime, s2e, s2mu)
  lp_tauprime = logpdf(Exponential(0.1), tree.tau)
  logr = mllt_tauprime + lp_tauprime + log_tauprime - (mllt_tau + lp_tau + log_tau)
  if log(rand()) >= logr
    tree.tau = exp(log_tau)
  end
end

function updatemu!(tree::SoftTree, rt::Vector{Float64}, x::Vector{Float64}, s2e, s2mu)
  Phit = leafprob(x, tree)
  Omega = inv(transpose(Phit) * Phit / s2e + I / s2mu)
  rhat = transpose(Phit) * rt / s2e
  newmu = rand(MvNormal(Omega * rhat, Symmetric(Omega)))
  leaves = leafnodes(tree)
  for l in 1:length(leaves)
    leaves[l].mu = newmu[l]
  end
end

function updatesigma(ytrain::Vector{Float64}, x:: Vector{Float64}, trees::Vector{SoftTree}, nu, lambda)
  yhat = predict(x, trees)
  residual = ytrain .- yhat
  n = length(residual)
  a = (nu + n) / 2
  b = 0.5 * (nu*lambda + sum(residual.^2))
  rand(InverseGamma(a, b))
end

function updatesigmamu(trees::Vector{SoftTree}, s2mu)
  mu = []
  for tree in trees
    mut = treemu(tree)
    for l in 1:length(mut)
      push!(mu, mut[l])
    end
  end
  T = length(trees)
  L = length(mu)
  ssmu = dot(mu, mu)
  tauprime = rand(Gamma(0.5*L + 1, 0.5*ssmu))
  smuprime = sqrt(1 / tauprime)
  cauchyplus = Truncated(Cauchy(0, 0.25 / sqrt(T)), 0, Inf)
  llprime = logpdf(cauchyplus, smuprime) * smuprime^3
  llcur = logpdf(cauchyplus, sqrt(s2mu)) * (sqrt(s2mu))^3
  logratio = llprime - llcur
  log(rand()) < logratio ? smuprime^2 : s2mu
end

function standardize(y)
  ymax = maximum(y)
  ymin = minimum(y)
  mr = (ymax + ymin) / 2
  (y .- mr) / (ymax - ymin)
end

function unstandardize(ytrain, y)
  ymax = maximum(y)
  ymin = minimum(y)
  mr = (ymax + ymin) / 2
  ytrain .* (ymax - ymin) .+ mr
end


###############################################################################
##### Testing implementation
###############################################################################

n = 100
x = rand(Normal(0, 1), n)
f(x) = 3 * sin(pi * x / 2) / (1 + 2 * x^2 * ((x > 0) + 1))
truesigma = 0.3
y = f.(x) + rand(Normal(0, truesigma), n)

function softbart(x::Vector{Float64}, y::Vector{Float64})
  ytrain = standardize(y)
  # xtrain = rcopy(R"SoftBart::trank($x)")
  # xmin = minimum(xtrain)
  # xmax = maximum(xtrain)
  xmin = minimum(x)
  xmax = maximum(x)
  m = 20
  tauinit = 0.1
  k = 2
  s2mu = (0.5 / (k*sqrt(m)))^2
  s2e = var(ytrain)
  nu = 3
  q = 0.9
  lambda = 1 / quantile(InverseGamma(nu/2, nu/(2*s2e)), q)
  trees = initializetrees(ytrain, m, tauinit)
  S = 2500
  b = 500
  yhatdraws = Matrix{Float64}(undef, n, S)
  s2edraws = Vector{Float64}(undef, S)
  @time for s in 1:(b + S)
    for t in 1:m
      rt = partresid(ytrain, x, trees, t)
      trees[t] = updatetree(trees[t], rt, x, 0.95, 2.0, s2e, s2mu, xmin, xmax)
      updatetau!(x, rt, trees[t], s2e, s2mu)
      updatemu!(trees[t], rt, x, s2e, s2mu)
    end
    s2e = updatesigma(ytrain, x, trees, nu, lambda)
    # s2mu = updatesigmamu(trees, s2mu)
    yhat = predict(x, trees)
    if s > b
      s2edraws[s-b] = (maximum(y) - minimum(y))^2 * s2e
      yhatdraws[:,s-b] = unstandardize(yhat, y)
    end
    if s % 100 == 0
      println("MCMC Iteration $s complete.")
    end
  end
  [yhatdraws, s2edraws]
end

softfit = softbart(x, y)

yhatpost = softfit[1]
s2epost = softfit[2]

plot(s2epost)
hline!([truesigma^2])

using RCall
R"""
y <- $y
x <- $x
yhatpost <- $yhatpost
f <- function(x) {
  3 * sin(pi * x / 2) / (1 + 2 * x^2 * ((x > 0) + 1))
}

dev.new()
plot(y ~ x, col = adjustcolor("gray", 2/3), pch = 19)
for (s in 1:ncol(yhatpost)) {
  lines(x[order(x)], yhatpost[order(x), s], col = adjustcolor("gray", 0.01), lwd = 3)
}
yhatmean <- apply(yhatpost, 1, mean)
lines(x[order(x)], yhatmean[order(x)], col = "slateblue", lwd = 3.5)
curve(f(x), add = TRUE, lwd = 1.5, lty = 3)

library(SoftBart)
yscale <- (y - ((max(y) + min(y)) / 2)) / (max(y) - min(y))
sigma_hat <- summary(lm(yscale ~ x))$sigma
softfit <- softbart(
  X = as.matrix(x), Y = as.matrix(y), X_test = as.matrix(x),
  hypers = Hypers(X = as.matrix(x), Y = as.matrix(y), sigma_hat = sigma_hat, num_tree = 20),
  opts = Opts(
    num_save = 2500, num_burn = 500, num_thin = 1,
    update_s = FALSE, update_alpha = FALSE, update_sigma_mu = FALSE
  )
)

dev.new()
plot(y ~ x, col = adjustcolor("gray", 2/3), pch = 19)
for (s in 1:2500) {
  lines(x[order(x)], softfit$y_hat_train[s,][order(x)],
        col = adjustcolor("gray", 0.01), lwd = 3)
}
lines(x[order(x)], softfit$y_hat_train_mean[order(x)], col = "slateblue", lwd = 4)
curve(f(x), add = TRUE, lwd = 1.5, lty = 3)

dev.new()
plot(softfit$sigma^2, pch = 19, col = adjustcolor("gray", 2/3))
abline(h = 0.3^2, lty = 3)
"""
