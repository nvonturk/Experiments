from numpy.random import multivariate_normal as multi_norm

cov = [[0.9,0.7,0.4 ],
       [0.7,0.8 ,0.3 ],
       [0.4,0.3 ,0.7 ]]
means = [0,0,0]
print(multi_norm(means, cov))
