import numpy as np

AGENT = 'MPC'
metrics = np.loadtxt(f'experiment_results/{AGENT}.csv', delimiter=',')

print(metrics)

# number of collisions
# number of deadlocks
# slower ttg
# avg delta V
# avg path deviation