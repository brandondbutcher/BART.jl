###############################################################################
##### Gibbs and Metropolis-Hastings Proposals
###############################################################################

## Initialize trees at stumps
## The initial leaf parameter at the root Node is the
## mean of scaled y divided by the number of Trees
function initializetrees(td::TrainData, hypers::Hypers)
  trees = Vector{Tree}(undef, hypers.m)
  μ = mean(td.ytrain) ./ hypers.m
  for t in 1:hypers.m
    trees[t] = Tree(Leaf(μ), hypers.λmean, ones(td.n, 1))
  end
  trees
end

## Probability a Node is a Branch
function probgrow(d::Int64, hypers::Hypers)
  hypers.α * (1.0 + d) ^ (-hypers.β)
end

## Prior probability of a Tree
function log_tree_prior(tree::Tree, hypers::Hypers)
  if isa(tree.root, Leaf)
    d = depth(tree.root, tree)
    return log(1 - probgrow(d, hypers))
  end
  lp = log(probgrow(depth(tree.root, tree), hypers))
  log_tree_prior(tree.root.left, tree, hypers, lp) +
    log_tree_prior(tree.root.right, tree, hypers, lp)
end

function log_tree_prior(trees::Vector{T} where T <: Tree, hypers::Hypers)
  ltp = 0
  for tree in trees
    ltp += log_tree_prior(tree, hypers)
  end
  ltp
end

function log_tree_prior(branch::Branch, tree::Tree, hypers::Hypers, lp::Float64)
  d = depth(branch, tree)
  lp = lp + log(probgrow(d, hypers))
  log_tree_prior(branch.left, tree, hypers, lp) +
    log_tree_prior(branch.right, tree, hypers, lp)
end

function log_tree_prior(leaf::Leaf, tree::Tree, hypers::Hypers, lp::Float64)
  d = depth(leaf, tree)
  lp + log(1 - probgrow(d, hypers))
end

## Probability that an observtion goes left at a given Branch
function probleft(x::Vector{Float64}, branch::Branch, tree::Tree)
  1 / (1 + exp((x[branch.var] - branch.cut) / tree.λ))
end

function probleft(X::Matrix{Float64}, branch::Branch, tree::Tree)
  1 ./ (1 .+ exp.((X[:,branch.var] .- branch.cut) ./ tree.λ))
end

## Probability that observations end up in each Leaf
function leafprob(x::Vector{Float64}, tree::Tree)
  if isa(tree.root, Leaf)
    return 1.0
  end
  S = Float64[]
  goesleft = probleft(x, tree.root, tree)
  goesright = 1 - goesleft
  leafprob(x, tree.root.left, tree, goesleft, S)
  leafprob(x, tree.root.right, tree, goesright, S)
end

function leafprob(x::Vector{Float64}, branch::Branch, tree::Tree, psi::Float64, S::Vector{Float64})
  goesleft = psi * probleft(x, branch, tree)
  goesright = psi - goesleft
  leafprob(x, branch.left, tree, goesleft, S)
  leafprob(x, branch.right, tree, goesright, S)
end

function leafprob(x::Vector{Float64}, leaf::Leaf, tree::Tree, psi::Float64, S::Vector{Float64})
  push!(S, psi)
end

function leafprob(X::Matrix{Float64}, tree::Tree, td::TrainData)
  St = zeros(td.n, length(leafnodes(tree)))
  for i in 1:td.n
    St[i,:] .= leafprob(X[i,:], tree)
  end
  St
end

function leafprob(X::Matrix{Float64}, tree::Tree)
  n = size(X)[1]
  St = zeros(n, length(leafnodes(tree)))
  for i in 1:n
    St[i,:] .= leafprob(X[i,:], tree)
  end
  St
end

## Draw a new cut value for the proposed split
function drawcut(leaf::Leaf, var::Int64, tree::Tree, td::TrainData)
  branch = leaf
  lower = [td.xmin[:,var][1]]
  upper = [td.xmax[:,var][1]]
  check = branch == tree.root ? false : true
  while check
    left = isleft(branch, tree)
    branch = parent(branch, tree)
    check = branch == tree.root ? false : true
    if (branch.var == var)
      check
      if (left)
        upper = push!(upper, branch.cut)
      else
        lower = push!(lower, branch.cut)
      end
    end
  end
  lower = maximum(lower)
  upper = minimum(upper)
  rand(Uniform(lower, upper))
end

## If the birthproposal is accepted
## grow the Leaf into a Branch with two new Leafs
function birthleaf!(leaf::Leaf, tree::Tree, branch::Branch)
  if isa(tree.root, Leaf)
    tree.root = branch
  else
    parentnode = parent(leaf, tree)
    if parentnode.left == leaf
      parentnode.left = branch
    else
      parentnode.right = branch
    end
  end
end

## Probability of making a birth proposal
## If the tree only has a root node, a birth proposal has to be made
## Otherwise randomly choose a Birth or Death proposal with probability 0.5
function birthprob(tree::Tree)
  isa(tree.root, Leaf) ? 1.0 : 0.5
end

function birthprob(S::Matrix{Float64})
  size(S)[2] == 1 ? 1.0 : 0.5
end

## The log ratio of the transition probabilities for a birth proposal
function log_birth_trans(tree::Tree, S_prime::Matrix{Float64})
  # Probability of transitioning from proposed Tree back to the current Tree
  pd = 1 - birthprob(S_prime)
  b = length(onlyparents(tree))
  numr = pd / b
  # Probability of transitioning from the current Tree to the proposed Tree
  pb = birthprob(tree)
  Lt = length(leafnodes(tree))
  denomr = pb / Lt
  log(numr) - log(denomr)
end

## The log ratio of the tree probabilities
function log_birth_tree(node::Node, tree::Tree, hypers::Hypers)
  d = depth(node, tree)
  numr = probgrow(d, hypers) * (1 - probgrow(d + 1, hypers)) ^ 2
  denomr = 1 - probgrow(d, hypers)
  log(numr) - log(denomr)
end

## Compute the mariginal log likelihood of the current Tree residuals
function mll(rt::Vector{Float64}, S::Matrix{Float64}, s2e::Float64, hypers::Hypers, td::TrainData)
  Lt = size(S)[2]
  Omega = inv(transpose(S) * S / s2e + I / hypers.τ)
  rhat = transpose(S) * rt / s2e
  mll = 0.5 * logdet(2 * pi * Omega)
  mll -= 0.5 * td.n * log(2 * pi * s2e)
  mll -= 0.5 * Lt * log(2 * pi * hypers.τ)
  mll -= 0.5 * dot(rt, rt) / s2e
  mll += 0.5 * dot(rhat, Omega * rhat)
  mll
end

## Computed the ratio of the marginal log likelihood
## of the proposed tree to the current Tree
function mllratio(rt::Vector{Float64}, S::Matrix{Float64}, S_prime::Matrix{Float64}, s2e::Float64, hypers::Hypers, td::TrainData)
  mllt = mll(rt, S, s2e, hypers, td)
  mllt_prime = mll(rt, S_prime, s2e, hypers, td)
  mllt_prime - mllt
end

## Function to get the Leaf parameters
## Returns a Vector that will get post
## multiplied by S to make predictions
## at that given Tree
function treeμ(tree::Tree)
  [leaf.μ for leaf in leafnodes(tree)]
end

## Function to get the predicted values at a single tree
function treepredict(tree::Tree)
  tree.S * treeμ(tree)
end

## Function to get yhat on the transformed scale of y
function treespredict(trees::Vector{Tree}, td::TrainData)
  yhat = zeros(td.n)
  for tree in trees
    yhat += treepredict(tree)
  end
  yhat
end

## Conduct a Birth proposal
function birthproposal!(tree::Tree, rt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  leaves = leafnodes(tree)
  leaf = rand(leaves)
  index = findall(x -> x == leaf, leaves)[1]
  newvar = rand(1:td.p)
  newcut = drawcut(leaf, newvar, tree, td)
  branch = Branch(newvar, newcut, Leaf(0.0), Leaf(0.0))
  goesleft = tree.S[:,index] .* probleft(X, branch, tree)
  goesright = tree.S[:,index] .- goesleft
  if size(tree.S)[2] == 1
    S_prime = hcat(goesleft, goesright)
  else
    indices = [index, index + 1]
    S_prime = zeros(td.n, length(leaves) + 1)
    S_prime[:,indices] = hcat(goesleft, goesright)
    S_prime[:,setdiff(1:end, indices)] = tree.S[:,setdiff(1:end, index)]
  end
  mloglikratio = mllratio(rt, tree.S, S_prime, s2e, hypers, td)
  treeratio = log_birth_tree(leaf, tree, hypers)
  transratio = log_birth_trans(tree, S_prime)
  logr = mloglikratio + treeratio + transratio
  if log(rand()) < logr
    birthleaf!(leaf, tree, branch)
    tree.S = S_prime
  end
end

## The log ratio of the transition probabilities for a Death proposal
function log_death_trans(tree::Tree, S_prime::Matrix{Float64})
  pb = birthprob(S_prime)
  pd = 1 - birthprob(tree)
  Lt = length(leafnodes(tree))
  b = length(onlyparents(tree))
  numr = pb / (Lt - 1)
  denomr = pd / b
  log(numr) - log(denomr)
end

## The log ratio of the Tree probabilities
function log_death_tree(node::Node, tree::Tree, hypers::Hypers)
  -1.0*log_birth_tree(node, tree, hypers)
end

## If doing a Death proposal, kill two Leaf Nodes of
## a randomly selected Branch and turn the Branch into a Leaf
function deathbranch!(branch::Branch, tree::Tree)
  if tree.root == branch
    tree.root = Leaf(0.0)
  else
    parentnode = parent(branch, tree)
    if parentnode.left == branch
      parentnode.left = Leaf(0.0)
    else
      parentnode.right = Leaf(0.0)
    end
  end
end

## Conduct a Death proposal
function deathproposal!(tree::Tree, rt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  branch = rand(onlyparents(tree))
  indexes = findall(x -> (x == branch.left) || (x == branch.right), leafnodes(tree))
  lp = sum(tree.S[:,indexes], dims = 2)
  S_prime = copy(tree.S)
  S_prime[:,indexes[1]] = lp
  S_prime = S_prime[:,1:end .!= indexes[2]]
  mloglikratio = mllratio(rt, tree.S, S_prime, s2e, hypers, td)
  treeratio = log_death_tree(branch, tree, hypers)
  transratio = log_death_trans(tree, S_prime)
  logr = mloglikratio + treeratio + transratio
  if log(rand()) < logr
    deathbranch!(branch, tree)
    tree.S = S_prime
  end
end

## Function to perform Metropolis-Hastings step for a single Tree update
function updatetree!(tree::Tree, rt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  if rand() < birthprob(tree)
    birthproposal!(tree, rt, X, td, s2e, hypers)
  else
    deathproposal!(tree, rt, X, td, s2e, hypers)
  end
end

## Function to perform Metropolis-Hastings step to update λ on a single Tree
function updateλ!(X::Matrix{Float64}, rt::Vector{Float64}, tree::Tree, s2e::Float64, td::TrainData, hypers::Hypers)
  St_λ = tree.S
  mllt_λ = mll(rt, tree.S, s2e, hypers, td)
  lp_λ = logpdf(Exponential(hypers.λmean), tree.λ)
  log_λ = log(tree.λ)
  log_λprime = log(tree.λ) + rand(Uniform(-1, 1))
  tree.λ = exp(log_λprime)
  tree.S = leafprob(X, tree, td)
  mllt_λprime = mll(rt, tree.S, s2e, hypers, td)
  lp_λprime = logpdf(Exponential(hypers.λmean), tree.λ)
  logr = mllt_λprime + lp_λprime + log_λprime - (mllt_λ + lp_λ + log_λ)
  if log(rand()) >= logr
    tree.λ = exp(log_λ)
    tree.S = St_λ
  end
end

## Gibbs step to update leaf parameters for a single Tree
function updateμ!(tree::Tree, rt::Vector{Float64}, s2e::Float64, hypers::Hypers)
  Omega = inv(transpose(tree.S) * tree.S / s2e + I / hypers.τ)
  rhat = transpose(tree.S) * rt / s2e
  newμ = rand(MvNormal(Omega * rhat, Symmetric(Omega)))
  leaves = leafnodes(tree)
  for l in 1:length(leaves)
    leaves[l].μ = newμ[l]
  end
end

## Gibbs step to update error variance
function updateσ(yhat::Vector{Float64}, td::TrainData, hypers::Hypers)
  residual = td.ytrain - yhat
  a = 0.5 * (hypers.ν + td.n)
  b = 0.5 * (hypers.ν * hypers.δ + sum(residual.^2))
  sqrt(rand(InverseGamma(a, b)))
end
