import numpy as np
from numpy.lib import utils
from mdp import GridMDP
from utils import calculate_kl_divergence, find_unaccrued_features, trajectory_likelihood

nS = 81; nA = 9; nF = 4;
allow_diagonal_transitions = True
feature_map = np.zeros((nS, nA, nF))
feature_map[:, [0,1,2,3], 0] += 1 # distance
feature_map[:, [4,5,6,7], 0] += np.sqrt(2) # distance
feature_map[42:45, :, 1] += 1 # green
feature_map[36:39, :, 2] += 1 # blue
feature_map[80, :, 3] += 1 # goal
feature_wts = np.array([-4, 0, 0, 0])

feature_constraints = np.array([1,2]) # green, blue
state_constraints = nF+np.array([21,31,41,49,57])
action_constraints = nF+nS+np.array([4,6])
augmented_feature_constraints = True
constraints = np.hstack([state_constraints,action_constraints,feature_constraints])

goal = 80; start = 8; state_dim = 9; discount = 1.0; infinite_horizon = False;
print("nominal mdp")
nominal_mdp = GridMDP(nS, goal, start, state_dim, [], [], infinite_horizon, discount, feature_map, feature_wts, allow_diagonal_transitions)
print("true mdp")
true_mdp = GridMDP(nominal_mdp, constraints, augmented_feature_constraints)

T = 25; demos = 100;
print("generate demos")
true_mdp.backward_pass(T)
true_mdp.forward_pass(T)
demonstrations = true_mdp.produce_demonstrations(T, demos)
state_seqs, action_seqs = demonstrations

n_constraints = 20; d_kl_eps = 0.01
nominal_mdp.backward_pass(T)
nominal_mdp.forward_pass(T)
unaccrued_features = find_unaccrued_features(demonstrations, nominal_mdp)
estimated_mdp = GridMDP(nominal_mdp)
likely_constraints = np.zeros((n_constraints))
eliminated_trajectories = np.zeros((n_constraints))
d_kl_history = np.zeros((n_constraints+1))
d_kl_history[0] = calculate_kl_divergence(state_seqs, estimated_mdp)

for i in range(n_constraints):
    fah = estimated_mdp.feature_accrual_history[unaccrued_features, -1]
    max_index = np.argmax(fah)
    eliminated_trajectories[i] = fah[max_index]
    
    if eliminated_trajectories[i] == 0:
        d_kl_history[i+1] = d_kl_history[i]
        continue

    likely_constraints[i] = unaccrued_features[max_index]
    estimated_mdp = GridMDP(estimated_mdp, likely_constraints[i:i+1], True)
    estimated_mdp.backward_pass(T)
    estimated_mdp.forward_pass(T)

    d_kl_history[i+1] = calculate_kl_divergence(state_seqs, estimated_mdp)

    if np.abs(d_kl_history[i+1]-d_kl_history[i]) < d_kl_eps:
        print("stopping")
        estimated_mdp = GridMDP(nominal_mdp, likely_constraints[:i+1], True)
        estimated_mdp.backward_pass(T)
        estimated_mdp.forward_pass(T)
        break

d_kl_true = calculate_kl_divergence(state_seqs, true_mdp)
