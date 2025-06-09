import numpy as np

def find_unaccrued_features(demonstrations, mdp):
    num_demos = len(demonstrations[0])
    accrued = np.zeros((mdp.num_augmented_features()))

    for i in range(num_demos):
        for t in range(len(demonstrations[1][i])):
            features = mdp.augmented_indicator_feature_map(demonstrations[0][i][t], demonstrations[1][i][t])
            accrued += features.reshape(-1)
    
    unaccrued_features = []
    for i in range(mdp.num_augmented_features()):
        if accrued[i] == 0.:
            unaccrued_features.append(i)

    return unaccrued_features

def infer_actions_from_states(state_seq, mdp):
    action_seq = np.zeros((len(state_seq)))
    for i in range(len(state_seq)-1):
        most_likely_action = np.argmax(mdp.actions[state_seq[i], :, state_seq[i+1]])
        action_seq[i] = most_likely_action
    action_seq[-1] = mdp.terminal_action
    return action_seq.reshape(-1, 1)

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def trajectory_likelihood(state_seq, mdp):
    assert(mdp.backward)
    action_seq = infer_actions_from_states(state_seq, mdp)
    members = [item for item in state_seq.reshape(-1) if item in mdp.obstacles]
    if len(members) > 0: return 0
    trajectory_reward_indices = sub2ind(mdp.rewards.shape, state_seq, action_seq)
    path_cost = np.sum(mdp.rewards.T.reshape(-1)[trajectory_reward_indices.astype(int).reshape(-1)])
    unnormalized_path_prob = np.exp(path_cost)
    path_prob = unnormalized_path_prob / mdp.Zs.T.reshape(-1)[state_seq[0][0]] * mdp.D0[state_seq[0][0]]
    return path_prob

def calculate_kl_divergence(demonstrations, mdp):
    empirical = {}
    estimated = {}
    for i in range(len(demonstrations)):
        if demonstrations[i].tostring() not in empirical.keys():
            empirical[demonstrations[i].tostring()] = 1 / len(demonstrations)
            estimated[demonstrations[i].tostring()] = trajectory_likelihood(demonstrations[i], mdp)
        else:
            empirical[demonstrations[i].tostring()] += 1 / len(demonstrations)
    d_kl = 0
    for key in empirical.keys():
        p = empirical[key]
        q = estimated[key]
        assert(q > 0)
        d_kl += p * np.log(p / q)
    return d_kl