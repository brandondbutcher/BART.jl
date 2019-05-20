###############################################################################
##### SoftBART Gibbs and Metropolis-Hastings Proposals
###############################################################################

function initializetrees(td::TrainData, hypers::Hypers)
  trees = Vector{Tree}(undef, hypers.m)
  mu = mean(td.ytrain) ./ hypers.m
  for t in 1:hypers.m
    trees[t] = Tree([Leaf(1, mu, 0)], hypers.τ_mean, ones(td.n, 1))
  end
  trees
end

function treepredict(tree::Tree)
  tree.Phi * treemu(tree)
end

function treespredict(trees::Vector{Tree}, td::TrainData)
  yhat = zeros(td.n)
  for tree in trees
    yhat += treepredict(tree)
  end
  yhat
end

function drawcut(branch::Branch, tree::Tree, td::TrainData)
  var = branch.var
  lower = td.xmin[:,branch.var][1]
  upper = td.xmax[:,branch.var][1]
  check = isroot(branch, tree) ? false : true
  while check
    left = isleft(branch, tree)
    branch = parent(branch, tree)
    check = isroot(branch, tree) ? false : true
    if (branch.var == var)
      check = false
      if (left)
        lower = lower
        upper = branch.cut
      else
        lower = branch.cut
        upper = upper
      end
    end
  end
  rand(Uniform(lower, upper))
end

function growleaf!(leaf::Leaf, tree::Tree, td::TrainData, hypers::Hypers)
  leftid = leftindex(leaf.index)
  rightid = rightindex(leaf.index)
  var = rand(1:td.p)
  tree.tree[leaf.index] = Branch(var, 0, leaf.index, leftid, rightid, parentindex(leaf.index))
  tree.tree[leaf.index].cut = drawcut(tree.tree[leaf.index], tree, td)
  leftnode = Leaf(leftid, 0, leaf.index)
  rightnode = Leaf(rightid, 0, leaf.index)
  treeindices = 1:length(tree.tree)
  if !(leftid in treeindices)
    newdepth = depth(leftid)
    minindex = 2^newdepth
    maxindex = 2^(newdepth + 1) - 1
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

function growleaf!(leaf::Leaf, tree::Tree, vars::Vector{Int64}, td::TrainData, hypers::Hypers)
  leftid = leftindex(leaf.index)
  rightid = rightindex(leaf.index)
  # var = rand(1:td.p)
  var = rand(vars)
  tree.tree[leaf.index] = Branch(var, 0, leaf.index, leftid, rightid, parentindex(leaf.index))
  tree.tree[leaf.index].cut = drawcut(tree.tree[leaf.index], tree, td)
  leftnode = Leaf(leftid, 0, leaf.index)
  rightnode = Leaf(rightid, 0, leaf.index)
  treeindices = 1:length(tree.tree)
  if !(leftid in treeindices)
    newdepth = depth(leftid)
    minindex = 2^newdepth
    maxindex = 2^(newdepth + 1) - 1
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

function birth_tree_ratio(tree::Tree, hypers::Hypers)
  d = depth(tree)
  numr = probgrow(d, hypers) * (1 - probgrow(d + 1, hypers)) ^ 2
  denomr = 1 - probgrow(d, hypers)
  log(numr) - log(denomr)
end

function birthprob(tree::Tree)
  typeof(tree.tree[1]) == Leaf ? 1.0 : 0.5
end

function birth_trans_ratio(tree::Tree, tree_prime::Tree)
  pb = birthprob(tree)
  pd = 1 - birthprob(tree_prime)
  Lt = length(leafnodes(tree))
  b = length(onlyparents(tree_prime))
  numr = pd / (b + 1)
  denomr = pb / Lt
  log(numr) - log(denomr)
end

function mll(residt::Vector{Float64}, tree::Tree, s2e::Float64, hypers::Hypers, td::TrainData)
  Lt = size(tree.Phi)[2]
  Omega = inv(transpose(tree.Phi) * tree.Phi / s2e + I / hypers.s2μ)
  rhat = transpose(tree.Phi) * residt / s2e
  mll = 0.5 * logdet(2 * pi * Omega)
  mll -= 0.5 * td.n * log(2 * pi * s2e)
  mll -= 0.5 * Lt * log(2 * pi * hypers.s2μ)
  mll -= 0.5 * dot(residt, residt) / s2e
  mll += 0.5 * dot(rhat, Omega * rhat)
  mll
end

function mll_ratio(residt::Vector{Float64}, tree::Tree, tree_prime::Tree, s2e::Float64, hypers::Hypers, td::TrainData)
  mllt = mll(residt, tree, s2e, hypers, td)
  mllt_prime = mll(residt, tree_prime, s2e, hypers, td)
  mllt_prime - mllt
end

function pruneleaves!(branch::Branch, tree::Tree)
  tree.tree[branch.leftchild] = nothing
  tree.tree[branch.rightchild] = nothing
  tree.tree[branch.index] = Leaf(branch.index, 0, branch.parent)
end

function birthproposal(tree::Tree, residt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  tree_prime = Tree([], tree.tau, ones(td.n, 1))
  tree_prime.tree = copy(tree.tree)
  leaves_prime = leafnodes(tree_prime)
  leaf_prime = rand(leaves_prime)
  growleaf!(leaf_prime, tree_prime, td, hypers)
  tree_prime.Phi = leafprob(X, tree_prime, td)
  mloglikratio = mll_ratio(residt, tree, tree_prime, s2e, hypers, td)
  treeratio = birth_tree_ratio(tree, hypers)
  transratio = birth_trans_ratio(tree, tree_prime)
  logr = mloglikratio + treeratio + transratio
  return log(rand()) < logr ? tree_prime : tree
end

function death_tree_ratio(tree::Tree, hypers::Hypers)
  1 / birth_tree_ratio(tree, hypers)
end

function death_trans_ratio(tree::Tree, tree_prime::Tree)
  numr = birthprob(tree_prime) / (length(leafnodes(tree_prime)) - 1)
  denomr = (1 - birthprob(tree)) / (length(onlyparents(tree_prime)))
  log(numr) - log(denomr)
end

function deathproposal(tree::Tree, residt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  tree_prime = Tree([], tree.tau, ones(td.n, 1))
  tree_prime.tree = copy(tree.tree)
  branches_prime = onlyparents(tree_prime)
  branch_prime = rand(branches_prime)
  pruneleaves!(branch_prime, tree_prime)
  tree_prime.Phi = leafprob(X, tree_prime, td)
  mloglikratio = mll_ratio(residt, tree, tree_prime, s2e, hypers, td)
  treeratio = death_tree_ratio(tree, hypers)
  transratio = death_trans_ratio(tree, tree_prime)
  logr = mloglikratio + treeratio + transratio
  return log(rand()) < logr ? tree_prime : tree
end

function updatetree(tree::Tree, residt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  if rand() < birthprob(tree)
    return birthproposal(tree, residt, X, td, s2e, hypers)
  else
    return deathproposal(tree, residt, X, td, s2e, hypers)
  end
end

function updatetau!(X::Matrix{Float64}, residt::Vector{Float64}, tree::Tree, s2e::Float64, td::TrainData, hypers::Hypers)
  Phit_tau = tree.Phi
  mllt_tau = mll(residt, tree, s2e, hypers, td)
  lp_tau = logpdf(Exponential(hypers.τ_mean), tree.tau)
  log_tau = log(tree.tau)
  log_tauprime = log(tree.tau) + rand(Uniform(-1, 1))
  tree.tau = exp(log_tauprime)
  tree.Phi = leafprob(X, tree, td)
  mllt_tauprime = mll(residt, tree, s2e, hypers, td)
  lp_tauprime = logpdf(Exponential(0.1), tree.tau)
  logr = mllt_tauprime + lp_tauprime + log_tauprime - (mllt_tau + lp_tau + log_tau)
  if log(rand()) >= logr
    tree.tau = exp(log_tau)
    tree.Phi = Phit_tau
  end
end

function updatemu!(tree::Tree, residt::Vector{Float64}, s2e::Float64, hypers::Hypers)
  Omega = inv(transpose(tree.Phi) * tree.Phi / s2e + I / hypers.s2μ)
  rhat = transpose(tree.Phi) * residt / s2e
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
