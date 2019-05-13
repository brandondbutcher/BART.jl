###############################################################################
##### SoftBART Gibbs and Metropolis-Hastings Proposals
###############################################################################

function initializetrees(td::TrainData, hypers::Hypers)
  trees = Array{SoftTree}(undef, hypers.m)
  mu = mean(td.ytrain) ./ hypers.m
  for t in 1:hypers.m
    trees[t] = SoftTree([SoftLeaf(mu, 0)], hypers.τ_mean, ones(td.n, 1))
  end
  trees
end

function treepredict(tree::SoftTree)
  tree.Phi * treemu(tree)
end

function treespredict(trees::Vector{SoftTree}, td::TrainData)
  yhat = zeros(td.n)
  for tree in trees
    yhat += treepredict(tree)
  end
  yhat
end

# function drawcut(node::SoftLeaf, tree::SoftTree, td::TrainData)
#   a = [td.xmin[:,1][1], leftparent(node, tree)]
#   a = filter(x -> x != nothing, a)
#   b = [td.xmax[:,1][1], rightparent(node, tree)]
#   b = filter(x -> x != nothing, b)
#   lower = maximum(a)
#   upper = minimum(b)
#   rand(Uniform(lower, upper))
# end

function drawcut(node::SoftBranch, tree::SoftTree, td::TrainData)
  var = node.var
  lower = td.xmin[:,node.var][1]
  upper = td.xmax[:,node.var][1]
  check = isroot(node, tree) ? false : true
  while check
    is_left = isleft(node, tree)
    node = parent(node, tree)
    check = isroot(node, tree) ? false : true
    if (node.var == var)
      check = false
      if (is_left)
        lower = lower
        upper = node.cut
      else
        lower = node.cut
        upper = upper
      end
    end
  end
  rand(Uniform(lower, upper))
end

function growleaf!(leaf::SoftLeaf, tree::SoftTree, td::TrainData, hypers::Hypers)
  leafindex = nodeindex(leaf, tree)
  leftid = leftindex(leafindex)
  rightid = rightindex(leafindex)
  var = rand(1:td.p)
  tree.tree[leafindex] = SoftBranch(var, 0, leftid, rightid, parentindex(leafindex))
  tree.tree[leafindex].cut = drawcut(tree.tree[leafindex], tree, td)
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

function probgrow(depth, hypers::Hypers)
  hypers.α * (1.0 + depth) ^ (-hypers.β)
end

function birth_tree_ratio(tree::SoftTree, hypers::Hypers)
  d = depth(tree)
  numr = probgrow(d, hypers) * (1 - probgrow(d + 1, hypers)) ^ 2
  denomr = 1 - probgrow(d, hypers)
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

function mll(rt::Vector{Float64}, tree::SoftTree, s2e::Float64, hypers::Hypers, td::TrainData)
  Lt = size(tree.Phi)[2]
  Omega = inv(transpose(tree.Phi) * tree.Phi / s2e + I / hypers.s2μ)
  rhat = transpose(tree.Phi) * rt / s2e
  mll = 0.5 * logdet(2 * pi * Omega)
  mll -= 0.5 * td.n * log(2 * pi * s2e)
  mll -= 0.5 * Lt * log(2 * pi * hypers.s2μ)
  mll -= 0.5 * dot(rt, rt) / s2e
  mll += 0.5 * dot(rhat, Omega * rhat)
  mll
end

function mll_ratio(rt, tree, tree_prime, s2e, hypers, td)
  mllt = mll(rt, tree, s2e, hypers, td)
  mllt_prime = mll(rt, tree_prime, s2e, hypers, td)
  mllt_prime - mllt
end

function pruneleaves!(node::SoftBranch, tree::SoftTree)
  pruneindex = nodeindex(node, tree)
  tree.tree[node.leftchild] = nothing
  tree.tree[node.rightchild] = nothing
  tree.tree[pruneindex] = SoftLeaf(0, node.parent)
end

function birthproposal(tree::SoftTree, rt::Vector{Float64}, x::Matrix{Float64}, td::TrainData, s2e, hypers::Hypers)
  tree_prime = SoftTree([], tree.tau, ones(td.n, 1))
  tree_prime.tree = copy(tree.tree)
  leaves_prime = leafnodes(tree_prime)
  leaf_prime = rand(leaves_prime)
  growleaf!(leaf_prime, tree_prime, td, hypers)
  tree_prime.Phi = leafprob(x, tree_prime, td)
  mloglikratio = mll_ratio(rt, tree, tree_prime, s2e, hypers, td)
  treeratio = birth_tree_ratio(tree, hypers)
  transratio = birth_trans_ratio(tree, tree_prime)
  logr = mloglikratio + treeratio + transratio
  return log(rand()) < logr ? tree_prime : tree
end

function death_tree_ratio(tree::SoftTree, hypers::Hypers)
  1 / birth_tree_ratio(tree, hypers)
end

function death_trans_ratio(tree::SoftTree, tree_prime::SoftTree)
  numr = birthprob(tree_prime) / (length(leafnodes(tree_prime)) - 1)
  denomr = (1 - birthprob(tree)) / (length(onlyparents(tree_prime)))
  log(numr) - log(denomr)
end

function deathproposal(tree::SoftTree, rt::Vector{Float64}, x::Matrix{Float64}, td::TrainData, s2e, hypers::Hypers)
  tree_prime = SoftTree([], tree.tau, ones(td.n, 1))
  tree_prime.tree = copy(tree.tree)
  branches_prime = onlyparents(tree_prime)
  branch_prime = rand(branches_prime)
  pruneleaves!(branch_prime, tree_prime)
  tree_prime.Phi = leafprob(x, tree_prime, td)
  mloglikratio = mll_ratio(rt, tree, tree_prime, s2e, hypers, td)
  treeratio = death_tree_ratio(tree, hypers)
  transratio = death_trans_ratio(tree, tree_prime)
  logr = mloglikratio + treeratio + transratio
  return log(rand()) < logr ? tree_prime : tree
end

function updatetree(tree::SoftTree, rt::Vector{Float64}, x::Matrix{Float64}, td::TrainData, s2e, hypers::Hypers)
  if rand() < birthprob(tree)
    return birthproposal(tree, rt, x, td, s2e, hypers)
  else
    return deathproposal(tree, rt, x, td, s2e, hypers)
  end
end

function updatetau!(x::Matrix{Float64}, rt::Vector{Float64}, tree::SoftTree, s2e, td::TrainData, hypers::Hypers)
  Phit_tau = tree.Phi
  mllt_tau = mll(rt, tree, s2e, hypers, td)
  lp_tau = logpdf(Exponential(0.1), tree.tau)
  log_tau = log(tree.tau)
  log_tauprime = log(tree.tau) + rand(Uniform(-1, 1))
  tree.tau = exp(log_tauprime)
  tree.Phi = leafprob(x, tree, td)
  mllt_tauprime = mll(rt, tree, s2e, hypers, td)
  lp_tauprime = logpdf(Exponential(0.1), tree.tau)
  logr = mllt_tauprime + lp_tauprime + log_tauprime - (mllt_tau + lp_tau + log_tau)
  if log(rand()) >= logr
    tree.tau = exp(log_tau)
    tree.Phi = Phit_tau
  end
end

function updatemu!(tree::SoftTree, rt::Vector{Float64}, s2e, hypers::Hypers)
  Omega = inv(transpose(tree.Phi) * tree.Phi / s2e + I / hypers.s2μ)
  rhat = transpose(tree.Phi) * rt / s2e
  newmu = rand(MvNormal(Omega * rhat, Symmetric(Omega)))
  leaves = leafnodes(tree)
  for l in 1:length(leaves)
    leaves[l].mu = newmu[l]
  end
end

function updatesigma(yhat::Vector{Float64}, td::TrainData, hypers::Hypers)
  residual = td.ytrain - yhat
  a = 0.5 * (hypers.ν + td.n)
  b = 0.5 * (hypers.ν * hypers.λ + sum(residual.^2))
  rand(InverseGamma(a, b))
end

# function updatesigmamu(trees::Vector{SoftTree}, hypers::Hypers)
#   mu = []
#   for tree in trees
#     mut = treemu(tree)
#     for l in 1:length(mut)
#       push!(mu, mut[l])
#     end
#   end
#   T = length(trees)
#   L = length(mu)
#   ssmu = dot(mu, mu)
#   tauprime = rand(Gamma(0.5*L + 1, 0.5*ssmu))
#   smuprime = sqrt(1 / tauprime)
#   cauchyplus = Truncated(Cauchy(0, 0.25 / sqrt(T)), 0, Inf)
#   llprime = logpdf(cauchyplus, smuprime) * smuprime^3
#   llcur = logpdf(cauchyplus, sqrt(params.s2μ)) * (sqrt(hypers.s2μ))^3
#   logratio = llprime - llcur
#   log(rand()) < logratio ? smuprime^2 : hypers.s2μ
# end
