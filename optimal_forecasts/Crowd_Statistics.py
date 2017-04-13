from numpy.random import multivariate_normal as multi_norm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform as uniform
import math
from scipy.stats import norm
from scipy.special import comb
import pprint

def transform_probability(bid, mean, stddev):
	transformed_bid = float(bid - mean)/float(stddev)
	return norm.cdf(transformed_bid)

def print_cov(cov_mat):
    print("COVARIANCE MATRIX")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    for row in cov_mat:
        total_str = ""
        for entry in row:
            total_str = total_str + "      " + str(round(entry,2))
        total_str = total_str + "\n"
        print(total_str)

def print_bids(bids):
    print("BIDS")
    print("////////////////////////////////////")
    total_str = ""
    for bid in bids:
        total_str = total_str + " " + str(round(bid, 3))
    print(total_str)

def expected_error(num_A, num_B, var_A, var_B, cov_A, cov_B, cov_AB):
    if num_A + num_B == 0:
        print("ERROR NO ONE IN CROWD")
    else:
        denominator = num_A + num_B
        numerator = num_A*var_A + num_B*var_B + 2*comb(num_A, 2, exact=False)*cov_A + 2*comb(num_B, 2, exact=False)*cov_B + 2*num_A*num_B*cov_AB
        return float(numerator)/float(denominator)

def utility(bid, activation_threshold):
    # Plug in different scoring rules
    payoff_1 = 2*bid - bid**2 - (1-bid)**2
    payoff_2 = 2*(1-bid) - bid**2 - (1-bid)**2
    return bid*payoff_1 + (1-bid)*payoff_2 - activation_threshold

def calc_optimal(group_size, var_A, var_B, cov_A, cov_B, cov_AB):
    tc = float(cov_A + cov_B - 2*cov_AB)
    print(tc)
    term1 = float((var_B - var_A) - (cov_B - cov_A)) / float(2*group_size*tc)
    term2 = float(cov_B - cov_AB) / float(tc)
    return term1 + term2


# Define types and their populations
type_populations = {
    "A":50,
    "B":50
}

activation_threshold = 0

var_bounds = [0, 10]
intra_cov_lb = []
inter_cov_lb = 0
V = 0

# Define variances, within-type covariance, across-type covariance
variances = {}
for key in type_populations.keys():
    variances[key] = uniform(var_bounds[0], var_bounds[1])

intra_covs = {}
for key in type_populations.keys():
    possible_value = uniform(intra_cov_lb, variances[key])
    intra_covs[key + key] = possible_value

inter_covs = {}
for key1 in type_populations.keys():
    for key2 in type_populations.keys():
        if key1 == key2:
            continue
        else:
            key = ''.join(sorted(key1 + key2))
            if key in inter_covs.keys():
                continue
            upper_bound = float(intra_covs[key1 + key1] + intra_covs[key2 + key2]) / float(2)
            possible_value = uniform(inter_cov_lb, upper_bound)
            inter_covs[key] = possible_value

# Define the covariance matrix
num_agents = sum(type_populations.values())
cov_mat = [[0 for x in range(num_agents)] for y in range(num_agents)]

agents = []
for agent_type in type_populations.keys():
    for i in range(type_populations[agent_type]):
        agents.append(agent_type)

for i in range(num_agents):
    for j in range(num_agents):
        if j < i:
            continue
        agent1 = agents[i]
        agent2 = agents[j]
        if i == j:
            cov_mat[i][j] = variances[agents[i]]
        elif (i != j) and (agent1 == agent2):
            cov_mat[i][j] = intra_covs[agent1 + agent2]
            cov_mat[j][i] = intra_covs[agent1 + agent2]
        else:
            cov_mat[i][j] = inter_covs[''.join(sorted(agent1 + agent2))]
            cov_mat[j][i] = inter_covs[''.join(sorted(agent1 + agent2))]

means = [V for i in range(num_agents)]

draw = multi_norm(means, cov_mat)

# Convert these to probabilities
bids = []
for i, bid in enumerate(draw):
    b = (i, transform_probability(bid, means[i], math.sqrt(variances[agents[i]])))
    bids.append(b)

realized_bids = []
realized_payoffs = []

expected_payoffs = []
for bid in bids:
    ep = (bid[0], utility(bid[1], activation_threshold))
    expected_payoffs.append(ep)

pos_expected_payoffs = [payoff for payoff in expected_payoffs if payoff[1] > 0]


count_A = 0
count_B = 0
proportion_A_random = [0]
while len(pos_expected_payoffs) > 0:
    rand_pick = int(round(uniform(0, len(pos_expected_payoffs) - 1)))

    picked_tuple = pos_expected_payoffs[rand_pick]
    picked_agent = picked_tuple[0]
    picked_payoff = picked_tuple[1]
    picked_bids_list = [bid for bid in bids if bid[0] == picked_agent]
    picked_bid = picked_bids_list[0][1]

    if agents[picked_agent] == 'A':
        count_A += 1
    else:
        count_B += 1
    proportion_A_random.append(float(count_A)/float(count_A + count_B))

    realized_bids.append(picked_bids_list[0])
    realized_payoffs.append((picked_agent, picked_payoff))
    bids.remove(picked_bids_list[0])

    expected_payoffs = []
    for bid in bids:
        expected_payoffs.append((bid[0], utility(bid[1], activation_threshold)))

    pos_expected_payoffs = [payoff for payoff in expected_payoffs if payoff[1] > 0]

proportion_A_optimal = [0]
for i in range(len(agents)):
    group_size = 1 + i
    proportion_A_optimal.append(calc_optimal(group_size, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB']))

x = range(len(agents) + 1)
plt.plot(x, proportion_A_random, 'r--', x, proportion_A_optimal, 'g^')
plt.show()












