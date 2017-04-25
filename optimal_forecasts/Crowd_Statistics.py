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
    	print(num_A, num_B, var_A, var_B, cov_A, cov_B, cov_AB)
        denominator = num_A + num_B
        numerator = num_A*var_A + num_B*var_B + 2*comb(num_A, 2, exact=False)*cov_A + 2*comb(num_B, 2, exact=False)*cov_B + 2*num_A*num_B*cov_AB
        return float(numerator)/float(denominator)

def utility(bid, activation_threshold, incentive, C):
    # Plug in different scoring rules
    payoff_1 = 2*bid - bid**2 - (1-bid)**2
    payoff_2 = 2*(1-bid) - bid**2 - (1-bid)**2
    # print("Payout w/o Incentive", bid*payoff_1 + (1-bid)*payoff_2 - activation_threshold)
    # print("Payout w/ Incentive", (bid*payoff_1 + (1-bid)*payoff_2 - activation_threshold + C*incentive))
    return (bid*payoff_1 + (1-bid)*payoff_2 - activation_threshold + C*incentive)

def calc_optimal(group_size, var_A, var_B, cov_A, cov_B, cov_AB):
    tc = float(cov_A + cov_B - 2*cov_AB)
    term1 = float((var_B - var_A) - (cov_B - cov_A)) / float(2*group_size*tc)
    term2 = float(cov_B - cov_AB) / float(tc)
    return term1 + term2


# Define types and their populations
type_populations = {
    "A":20,
    "B":10
}

activation_threshold = 0
V = 0

# Define variances, within-type covariance, across-type covariance
variances = {
    "A":0.1,
    "B":0.2
}

intra_covs = {
    "AA": 0.6,
    "BB": 1
}

inter_covs = {
    "AB": -0.5
}

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

cov_mat = np.dot(np.matrix(cov_mat), np.matrix(cov_mat).T)
# print(cov_mat)
means = [V for i in range(num_agents)]

draw = multi_norm(means, cov_mat)

cov_mat = cov_mat.tolist()

variances = {
    "A":cov_mat[0][0],
    "B":cov_mat[len(cov_mat) - 1][len(cov_mat) - 1]
}

intra_covs = {
    "AA": cov_mat[0][1],
    "BB": cov_mat[len(cov_mat) - 1][len(cov_mat) - 2]
}

inter_covs = {
    "AB": cov_mat[type_populations["A"] - 1][type_populations["A"] + 1]
}

# Convert these to probabilities
bids = []

for i, bid in enumerate(draw):
    b = (i, transform_probability(bid, means[i], math.sqrt(variances[agents[i]])))
    bids.append(b)

# print([bid[1] for bid in bids])
# print("MEAN", sum([bid[1] for bid in bids])/len(bids))
realized_bids = []
realized_payoffs = []
realized_draws = []


incentives_for_A = [0]
incentives_for_B = [0]

group_error = [0]
expected_group_error = [0]



# First do it with no incentive

expected_payoffs = []
for bid in bids:
    ep = (bid[0], utility(bid[1], activation_threshold, 0, 0))
    expected_payoffs.append(ep)

pos_expected_payoffs = [payoff for payoff in expected_payoffs if payoff[1] > 0]

count_A = 0
count_B = 0
proportion_A = [0]

count = 0
while len(pos_expected_payoffs) > 0:
    rand_pick = int(round(uniform(0, len(pos_expected_payoffs) - 1)))
    picked_tuple = pos_expected_payoffs[rand_pick]

    picked_agent        = picked_tuple[0]
    picked_payoff       = picked_tuple[1]
    picked_bids_list    = [bid for bid in bids if bid[0] == picked_agent]
    picked_bid          = picked_bids_list[0][1]

    if agents[picked_agent] == 'A':
        count_A += 1
    else:
        count_B += 1
    count = count + 1

    proportion_A.append(float(count_A)/float(count_A + count_B))

    err_diff_for_A = expected_error(count_A, count_B, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB']) - expected_error(count_A + 1, count_B, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB'])
    err_diff_for_B = expected_error(count_A, count_B, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB']) - expected_error(count_A, count_B + 1, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB'])

    C = 0
    if err_diff_for_A < 0 and err_diff_for_B < 0:
    	# print("BOTH LOW")
    	C = min(float(1)/float(abs(err_diff_for_A)), float(1)/float(abs(err_diff_for_B)))
    else:
    	# print("One Positive")
    	smaller_diff = min(err_diff_for_A, err_diff_for_B)
    	C = float(1)/float(abs(smaller_diff))
    
    incentives_for_A.append(float(C)*float(err_diff_for_A))
    incentives_for_B.append(float(C)*float(err_diff_for_B)) 

    bids.remove(picked_bids_list[0])

    realized_bids.append(picked_bids_list[0][1])
    realized_payoffs.append((picked_agent, picked_payoff))
    realized_draws.append(draw[picked_bids_list[0][0]])

    # print(np.mean(realized_bids))
    group_error.append((sum(realized_draws)/len(realized_draws))**2)
    expected_group_error.append(expected_error(count_A, count_B, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB']))


    favored = ''
    if err_diff_for_A > err_diff_for_B:
    	favored = 'A'
    else:
    	favored = 'B'
    # print("Favored", favored, "Incentive A", C*err_diff_for_A, "Incentive B", C*err_diff_for_B)
    expected_payoffs = []
    for bid in bids:
    	incentive_A = float(err_diff_for_A)
        incentive_B = float(err_diff_for_B)
        incentive = 0
        if agents[bid[0]] == 'A':
            incentive = incentive_A
            # if err_diff_for_A > err_diff_for_B:
            # 	C = float(1)/float(err_diff_for_A)
            # else:
            # 	C = abs(float(1)/float(err_diff_for_A))
        else:
            incentive = incentive_B
            # if err_diff_for_B > err_diff_for_A:
            # 	C = float(1)/float(err_diff_for_B)
            # else:
            # 	C = abs(float(1)/float(err_diff_for_B))
        ep = (bid[0], utility(bid[1], activation_threshold, incentive, C))
        # ep = (bid[0], utility(bid[1], activation_threshold, 0, 0))
        expected_payoffs.append(ep)

    pos_expected_payoffs = [payoff for payoff in expected_payoffs if payoff[1] > 0]

proportion_A_optimal = [0]
for i in range(len(realized_bids)):
    group_size = 1 + i
    if group_size == 1:
        if variances['A'] < variances['B']:
            proportion_A_optimal.append(1)
        else:
            proportion_A_optimal.append(0)
    else:
        proportion_A_optimal.append(calc_optimal(group_size, variances['A'], variances['B'], intra_covs['AA'], intra_covs['BB'], inter_covs['AB']))

plt.figure(1)
x = range(len(realized_bids) + 1)
line1, = plt.plot(x, proportion_A, 'r--', label="Actual Proportion of A")
line2, = plt.plot(x, proportion_A_optimal, 'g^', label="Optimal Proportion of A")
# line3, = plt.plot(x, incentives_for_A, 'r.', label="Improvement in Error with Extra A")
# line4, = plt.plot(x, incentives_for_B, 'b.', label="Improvement in Error with Extra B")
line6, = plt.plot(x, expected_group_error, 'r.', label="Expected Group Error")
line5, = plt.plot(x, group_error, 'g.', label='Group Error')
plt.legend(handles=[line1, line2, line6, line5])
plt.show()

# plt.figure(2)
# plt.hist(realized_draws, bins='auto')
# plt.show()

# print("MEAN", sum(realized_draws)/len(realized_draws))












