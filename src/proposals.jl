###############################################################################
##### Gibbs and Metropolis-Hastings Proposals
###############################################################################

## Initialize trees at stumps leaf parameter at the mean of scaled y
## divided by the number of trees
function initializetrees(td::TrainData, hypers::Hypers)
  trees = Vector{Tree}(undef, hypers.m)
  mu = mean(td.ytrain) ./ hypers.m
  for t in 1:hypers.m
    trees[t] = Tree(Leaf(mu), hypers.τ_mean, ones(td.n, 1))
  end
  trees
end

## Probability that an observtion goes left at a given node
function probleft(x::Vector{Float64}, branch::Branch, tree::Tree)
  1 / (1 + exp((x[branch.var] - branch.cut) / tree.tau))
end

function probleft(X::Matrix{Float64}, branch::Branch, tree::Tree)
  1 ./ (1 .+ exp.((X[:,branch.var] .- branch.cut) ./ tree.tau))
end

## Probability that observations end up in the leaf nodes
function leafprob(x::Vector{Float64}, tree::Tree)
  if isa(tree.root, Leaf)
    return 1.0
  end
  phi = Float64[]
  goesleft = probleft(x, tree.root, tree)
  goesright = 1 - goesleft
  leafprob(x, tree.root.left, tree, goesleft, phi)
  leafprob(x, tree.root.right, tree, goesright, phi)
end

function leafprob(x::Vector{Float64}, branch::Branch, tree::Tree, psi::Float64, phi::Vector{Float64})
  goesleft = psi * probleft(x, branch, tree)
  goesright = psi - goesleft
  leafprob(x, branch.left, tree, goesleft, phi)
  leafprob(x, branch.right, tree, goesright, phi)
end

function leafprob(x::Vector{Float64}, leaf::Leaf, tree::Tree, psi::Float64, phi::Vector{Float64})
  push!(phi, psi)
end

function leafprob(X::Matrix{Float64}, tree::Tree, td::TrainData)
  Phit = zeros(td.n, length(leafnodes(tree)))
  for i in 1:td.n
    Phit[i,:] .= leafprob(X[i,:], tree)
  end
  Phit
end

function leafprob(X::Matrix{Float64}, tree::Tree)
  n = size(X)[1]
  Phit = zeros(n, length(leafnodes(tree)))
  for i in 1:n
    Phit[i,:] .= leafprob(X[i,:], tree)
  end
  Phit
end

## Draw a new cut value for new proposed split
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

## If the birthproposal is accepted birth the leaf node into a branch node
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

## Probability a node is a Branch node
function probgrow(d::Int64, hypers::Hypers)
  hypers.α * (1.0 + d) ^ (-hypers.β)
end

## Probability of making a birth proposal
## If the tree only has a root node, a birth proposal has to be made
## Otherwise choose between a Birth or Death proposal with probability 1/2
function birthprob(tree::Tree)
  isa(tree.root, Leaf) ? 1.0 : 0.5
end

function birthprob(Phi::Matrix{Float64})
  size(Phi)[2] == 1 ? 1.0 : 0.5
end

## The log ratio of the transition probabilities for a birth proposal
function log_birth_trans(tree::Tree, Phi_prime::Matrix{Float64})
  # Probability of transitioning from proposed tree back to the current tree
  pd = 1 - birthprob(Phi_prime)
  b = length(onlyparents(tree))
  numr = pd / b
  # Probability of transitioning from the current tree to the proposed tree
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

## Compute the mariginal log likelihood of the current tree residuals
function mll(rt::Vector{Float64}, Phi::Matrix{Float64}, s2e::Float64, hypers::Hypers, td::TrainData)
  Lt = size(Phi)[2]
  Omega = inv(transpose(Phi) * Phi / s2e + I / hypers.s2μ)
  rhat = transpose(Phi) * rt / s2e
  mll = 0.5 * logdet(2 * pi * Omega)
  mll -= 0.5 * td.n * log(2 * pi * s2e)
  mll -= 0.5 * Lt * log(2 * pi * hypers.s2μ)
  mll -= 0.5 * dot(rt, rt) / s2e
  mll += 0.5 * dot(rhat, Omega * rhat)
  mll
end

## Computed the marginal log likelihood of the proposed tree to the current tree
function mllratio(rt::Vector{Float64}, Phi::Matrix{Float64}, Phi_prime::Matrix{Float64}, s2e::Float64, hypers::Hypers, td::TrainData)
  mllt = mll(rt, Phi, s2e, hypers, td)
  mllt_prime = mll(rt, Phi_prime, s2e, hypers, td)
  mllt_prime - mllt
end

## Function to get the leaf parameters
## Returns a vector that will get post
## multiplied by Phi to make predictions
## at that given tree
function treemu(tree::Tree)
  [leaf.mu for leaf in leafnodes(tree)]
end

## Function to get the predicted values at a single tree
function treepredict(tree::Tree)
  tree.Phi * treemu(tree)
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
  goesleft = tree.Phi[:,index] .* probleft(X, branch, tree)
  goesright = tree.Phi[:,index] .- goesleft
  if size(tree.Phi)[2] == 1
    Phi_prime = hcat(goesleft, goesright)
  else
    indices = [index, index + 1]
    Phi_prime = zeros(td.n, length(leaves) + 1)
    Phi_prime[:,indices] = hcat(goesleft, goesright)
    Phi_prime[:,setdiff(1:end, indices)] = tree.Phi[:,setdiff(1:end, index)]
  end
  mloglikratio = mllratio(rt, tree.Phi, Phi_prime, s2e, hypers, td)
  treeratio = log_birth_tree(leaf, tree, hypers)
  transratio = log_birth_trans(tree, Phi_prime)
  logr = mloglikratio + treeratio + transratio
  if log(rand()) < logr
    birthleaf!(leaf, tree, branch)
    tree.Phi = Phi_prime
  end
end

## The log ratio of the transition probabilities for a death proposal
function log_death_trans(tree::Tree, Phi_prime::Matrix{Float64})
  pb = birthprob(Phi_prime)
  pd = 1 - birthprob(tree)
  Lt = length(leafnodes(tree))
  b = length(onlyparents(tree))
  numr = pb / (Lt - 1)
  denomr = pd / b
  log(numr) - log(denomr)
end

## The log ratio of the tree probabilities
function log_death_tree(node::Node, tree::Tree, hypers::Hypers)
  -1.0*log_birth_tree(node, tree, hypers)
end

## If doing a death proposal, kill two leaf nodes of a randomly
## selected branch
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

## Conduct a death proposal
function deathproposal!(tree::Tree, rt::Vector{Float64}, X::Matrix{Float64}, td::TrainData, s2e::Float64, hypers::Hypers)
  branch = rand(onlyparents(tree))
  indexes = findall(x -> (x == branch.left) || (x == branch.right), leafnodes(tree))
  lp = sum(tree.Phi[:,indexes], dims = 2)
  Phi_prime = copy(tree.Phi)
  Phi_prime[:,indexes[1]] = lp
  Phi_prime = Phi_prime[:,1:end .!= indexes[2]]
  mloglikratio = mllratio(rt, tree.Phi, Phi_prime, s2e, hypers, td)
  treeratio = log_death_tree(branch, tree, hypers)
  transratio = log_death_trans(tree, Phi_prime)
  logr = mloglikratio + treeratio + transratio
  if log(rand()) < logr
    deathbranch!(branch, tree)
    tree.Phi = Phi_prime
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

## Function to perform Metropolis-Hastings step to update tau on a single Tree
function updatetau!(X::Matrix{Float64}, rt::Vector{Float64}, tree::Tree, s2e::Float64, td::TrainData, hypers::Hypers)
  Phit_tau = tree.Phi
  mllt_tau = mll(rt, tree.Phi, s2e, hypers, td)
  lp_tau = logpdf(Exponential(hypers.τ_mean), tree.tau)
  log_tau = log(tree.tau)
  log_tauprime = log(tree.tau) + rand(Uniform(-1, 1))
  tree.tau = exp(log_tauprime)
  tree.Phi = leafprob(X, tree, td)
  mllt_tauprime = mll(rt, tree.Phi, s2e, hypers, td)
  lp_tauprime = logpdf(Exponential(hypers.τ_mean), tree.tau)
  logr = mllt_tauprime + lp_tauprime + log_tauprime - (mllt_tau + lp_tau + log_tau)
  if log(rand()) >= logr
    tree.tau = exp(log_tau)
    tree.Phi = Phit_tau
  end
end

## Gibbs step to update leaf parameters for a single Tree
function updatemu!(tree::Tree, rt::Vector{Float64}, s2e::Float64, hypers::Hypers)
  Omega = inv(transpose(tree.Phi) * tree.Phi / s2e + I / hypers.s2μ)
  rhat = transpose(tree.Phi) * rt / s2e
  newmu = rand(MvNormal(Omega * rhat, Symmetric(Omega)))
  leaves = leafnodes(tree)
  for l in 1:length(leaves)
    leaves[l].mu = newmu[l]
  end
end

## Gibbs step to update error variance
function updatesigma(yhat::Vector{Float64}, td::TrainData, hypers::Hypers)
  residual = td.ytrain - yhat
  a = 0.5 * (hypers.ν + td.n)
  b = 0.5 * (hypers.ν * hypers.λ + sum(residual.^2))
  rand(InverseGamma(a, b))
end
