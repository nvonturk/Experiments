import trial_net_sqrt
import numpy as np
import matplotlib.pyplot as plt

# Parameters
#
# size = 100
# iterations = 100
# breadth = 100
# num_nets = 11
#
# result = trial_net_sqrt.run(iterations, size, breadth, num_nets)
# multi_errors = result[1]
# statistic_sets = result[2]
# # ss = (overlap_mean, overlap_variance)
# overlap_data = []
# for ss in statistic_sets:
# 	overlap_data.append(ss[1])
# overlap_mean = [overlap_data[i][0] for i in range(len(overlap_data))]
# overlap_var = [overlap_data[i][1] for i in range(len(overlap_data))]
#
# combined = zip(multi_errors, overlap_mean, overlap_var)
# combined.sort(key = lambda t: t[0])
# combined = [list(t) for t in zip(*combined)]
#
# fig, ax = plt.subplots()
# line1, = ax.plot(range(len(combined[0])), combined[0], linewidth=2, label="Error")
# line2, = ax.plot(range(len(combined[1])), combined[1], linewidth=2, label="Mean Overlap")
#
# ax.legend(loc='middle right')
# plt.show()

# def numberNetsComparison():

size = 100
iterations = 100
breadth = 10000
num_nets = [1,3,5,11,15]

single_errors = []
multi_errors = []

for num in num_nets:
	result = trial_net_sqrt.run(iterations, size, breadth, num)
	single_errors.append(np.average(result[0]))
	multi_errors.append(np.average(result[1]))

fig, ax = plt.subplots()
line1, = ax.plot(num_nets, multi_errors, linewidth=2, label="Multiple Nets, Split Info")
line2, = ax.plot(num_nets, single_errors, linewidth=2, label="Single Nets")

ax.legend(loc='middle right')
plt.show()
