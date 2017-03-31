import net_sqrt

# To facilitate running multiple experiments to average out run variability
def run(iterations, size, breadth, num_nets):

	# Error info
	single_net_errors = []
	multi_nets_errors = []
	statistic_sets = []

	for i in range(iterations):
		ret = net_sqrt.run(size, breadth, num_nets)
		single_net_errors.append(ret[0])
		multi_nets_errors.append(ret[1])
		statistic_sets.append(ret[2])

	return (single_net_errors, multi_nets_errors, statistic_sets)
