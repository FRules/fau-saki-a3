import mdptoolbox
import numpy as np

# Definition of the three different transition probability matrices
tpmLazy = np.array([[0, 0.95, 0.04, 0.01],[0, 1, 0, 0],[0, 0.2, 0.8, 0],[0, 0, 0.8, 0.2]])
tpmDiligent = np.array([[0, 0.95, 0.04, 0.01],[0, 0.3, 0.7, 0],[0, 0.2, 0.8, 0], [0, 0, 0.8, 0.2]])
tpmEDiligent = np.array([[0, 0.95, 0.04, 0.01],[0, 0.1, 0.2, 0.7], [0, 0, 0.7, 0.3],[0, 0, 0.8, 0.2]])

# This is an example how the construction of the reward matrix is not working as input for the mdp
rewardLazy = np.array([0, 1, 2, 4])
rewardDiligent = np.array([0, -5, 2, 4])
rewardEDiligent = np.array([0, -20, -10, 4])

# This is an example how to build the reward matrix
rewardfull = np.array([[0, 0, 0],[1, -1, -10], [2, 2, -1], [10, 10, 10]])

# Bring all the tpm together
tmpFull = np.array([tpmLazy, tpmDiligent, tpmEDiligent])

# Definition of the mdp with discount factor, maximal iterations, the tranisition probability matrix and the reward matrix
mdpresultPolicy = mdptoolbox.mdp.PolicyIteration(tmpFull,rewardfull,0.99, max_iter=100)
mdpresultValue = mdptoolbox.mdp.ValueIteration(tmpFull,rewardfull,0.99, max_iter=100)

# Run the MDP
mdpresultPolicy.run()
mdpresultValue.run()

"""-------- HERE ARE THE SOLUTIONS ----------------"""

print('PolicyIteration:')
print(mdpresultPolicy.policy)
print(mdpresultPolicy.V)
print(mdpresultPolicy.iter)

print('ValueIteration:')
print(mdpresultValue.policy)
print(mdpresultValue.V)
print(mdpresultValue.iter)
