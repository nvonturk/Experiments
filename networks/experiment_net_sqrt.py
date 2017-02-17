import trial_net_sqrt
import numpy as np
import matplotlib.pyplot as plt

size = 100
breadth = 100000
num_nets = [1,3,5,7,9,11,21,25]

singlenet_errors = []

multinet_errors_splitinfo = []
diffs_splitinfo = []

multinet_errors_fullinfo = []
diffs_fullinfo = []

for num in num_nets:
	result = trial_net_sqrt.run(size, breadth, num)
	singlenet_errors.append(result[0])
	multinet_errors_splitinfo.append(result[1])
	multinet_errors_fullinfo.append(result[2])
	diffs_splitinfo.append(result[0] - result[1])
	diffs_fullinfo.append(result[0] - result[2])

fig, ax = plt.subplots()
line1, = ax.plot(num_nets, multinet_errors_splitinfo, linewidth=2, label="Multiple Nets, Split Info")
line2, = ax.plot(num_nets, multinet_errors_fullinfo, linewidth=2, label="Multiple Nets, Full Info")
line3, = ax.plot(num_nets, singlenet_errors, linewidth=2, label="Single Nets")

ax.legend(loc='middle right')
plt.show()