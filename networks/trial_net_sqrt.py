import net_sqrt
import fullinfo_net_sqrt

# Input Parameters
def run(size, breadth, num_nets):
	split_factor = size/num_nets

	iterations = 100

	gross_singlenet_error = 0
	gross_multiplenet_error = 0
	gross_multiplenet_error_fullinfo = 0
	for i in range(iterations):
		errors = net_sqrt.singleVsMultiple(size, breadth, num_nets, split_factor)
		gross_singlenet_error += errors[0]
		gross_multiplenet_error += errors[1]
		gross_multiplenet_error_fullinfo += errors[2]


	average_singlenet_error = gross_singlenet_error/iterations
	average_multiplenet_error = gross_multiplenet_error/iterations
	average_multiplenet_error_fullinfo = gross_multiplenet_error_fullinfo/iterations

	print("2")
	return (average_singlenet_error, average_multiplenet_error, average_multiplenet_error_fullinfo)

