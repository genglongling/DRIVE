import numpy as np
from numpy.lib.shape_base import expand_dims
from copy import deepcopy as dc

class MDP:
    def __init__(self):
        self.nS, self.nA, self.goal, self.actions, \
        self.terminal_action, self.rewards, self.D0, \
        self.obstacles, self.state_constraints, \
        self.action_constraints, self.feature_constraints, \
        self.infinite_horizon, self.discount_factor, \
        self.nF, self.feature_map, self.feature_wts, \
        self.backward, self.forward, self.Zs, \
        self.local_action_probs, \
        self.total_state_visitation_freqs, \
        self.state_visitation_history, \
        self.unique_visitations, \
        self.feature_accrual_history = [None, ]*24
        self.obstacles, self.state_constraints, \
        self.action_constraints, self.feature_constraints = [], [], [], []
        self.average_reward = 0  # Initialize the average reward attribute
        self.success_rate = 0
        self.violation_rate = 0

    def backward_forward(self):
        return self.backward and self.forward
    
    def reset_backward_forward(self):
        self.backward = False
        self.forward = False
        self.Zs, self.local_action_probs, \
        self.total_state_visitation_freqs, \
        self.state_visitation_history, \
        self.unique_visitations, \
        self.feature_accrual_history = [[],]*6
        self.eyeS, self.eyeA = np.eye(self.nS), np.eye(self.nA)
    
    def set_rewards(self, rewards):
        self.rewards = rewards
        self.reset_backward_forward()
    
    def update_rewards_from_features(self):
        feature_map_by_action = self.feature_map.transpose((0,2,1))
        feature_based_rewards = np.zeros((self.nS, self.nA))
        for i in range(self.nA):
            feature_based_rewards[:, i] = \
                feature_map_by_action[:, :, i].dot(self.feature_wts)
        self.set_rewards(feature_based_rewards)
    
    def augmented_indicator_feature_map(self, s, a):
        feature_component = (self.feature_map[s, a, :] > 0).astype(int)
        state_component = self.eyeS[s]
        action_component = self.eyeA[a]
        return np.hstack([feature_component, state_component, action_component])
    
    def num_augmented_features(self):
        return self.nF+self.nS+self.nA
    
    def augmented_feature_feature_indices(self):
        return np.array(range(self.nF))
    
    def augmented_feature_state_indices(self):
        return self.nF+np.array(range(self.nS))
    
    def augmented_feature_action_indices(self):
        return self.nF+self.nS+np.array(range(self.nA))



    def get_reward(self):
        return self.average_reward
    def get_success_rate(self):
        return self.success_rate
    def get_violation_rate(self):
        return self.violation_rate

    def get_next_state(self, state, action):
        """
        Given a state and an action, return the next state.
        This function assumes that `self.actions` contains the transition dynamics.
        """
        # This should return the resulting state based on the action taken from the state.
        # You might need to adjust this based on your implementation of transitions.
        return np.argwhere(self.actions[state, action, :] != 0)

import numpy as np
import csv


class GridMDP(MDP):
    def __init__(self, *args):
        # super().__init__()
        super().__init__()

        if len(args) > 0 and type(args[0]) == GridMDP:
            self.nS, self.nA, self.goal, self.actions, \
            self.terminal_action, self.rewards, self.D0, \
            self.obstacles, self.state_constraints, \
            self.action_constraints, self.feature_constraints, \
            self.infinite_horizon, self.discount_factor, \
            self.nF, self.feature_map, self.feature_wts, \
            self.backward, self.forward, self.Zs, \
            self.local_action_probs, \
            self.total_state_visitation_freqs, \
            self.state_visitation_history, \
            self.unique_visitations, \
            self.feature_accrual_history = \
                dc(args[0].nS), dc(args[0].nA), dc(args[0].goal), dc(args[0].actions), \
                dc(args[0].terminal_action), dc(args[0].rewards), dc(args[0].D0), \
                dc(args[0].obstacles), dc(args[0].state_constraints), \
                dc(args[0].action_constraints), dc(args[0].feature_constraints), \
                dc(args[0].infinite_horizon), dc(args[0].discount_factor), \
                dc(args[0].nF), dc(args[0].feature_map), dc(args[0].feature_wts), \
                dc(args[0].backward), dc(args[0].forward), dc(args[0].Zs), \
                dc(args[0].local_action_probs), \
                dc(args[0].total_state_visitation_freqs), \
                dc(args[0].state_visitation_history), \
                dc(args[0].unique_visitations), \
                dc(args[0].feature_accrual_history)
            
            if len(args) >= 2:
                
                if len(args) >= 3 and args[2]:
                    constraints = args[1]
                    state_constraints = [item-self.nF for item in self.augmented_feature_state_indices() if item in constraints]
                    action_constraints = [item-self.nF-self.nS for item in self.augmented_feature_action_indices() if item in constraints]
                    feature_constraints = [item for item in self.augmented_feature_feature_indices() if item in constraints]
                    print("New MDP:")
                    self.enforce_state_constraints(state_constraints)
                    self.obstacles += state_constraints
                    self.state_constraints += state_constraints
                    self.enforce_action_constraints(action_constraints)
                    self.action_constraints += action_constraints
                    self.enforce_feature_constraints(feature_constraints)
                    self.feature_constraints += feature_constraints
                    print("")
                else:
                    obstacles = args[2]
                    print("New MDP:")
                    self.enforce_state_constraints(obstacles)
                    self.obstacles += obstacles
                    self.state_constraints += obstacles
                    print("")

                self.reset_backward_forward()
            
        elif len(args) >= 5:
            
            self.nS, self.goal, init_state, self.grid_height = args[:4]
            double_cost_states = args[4] if len(args) >= 5 else []
            self.obstacles = args[5] if len(args) >= 6 else []
            self.infinite_horizon = args[6] if len(args) >= 7 else False
            self.discount_factor = args[7] if len(args) >= 8 else 1
            feature_map = args[8] if len(args) >= 9 else []
            feature_wts = args[9] if len(args) >= 10 else []
            allow_diagonal_transitions = args[10] if len(args) >= 11 and args[10] != [] else False

            initial_state_distribution = np.zeros((self.nS, 1))
            if type(init_state) == dict:
                for s in init_state.keys():
                    initial_state_distribution[s] = init_state[s]
                initial_state_distribution /= 1.0 * sum(list(init_state.values()))
            else:
                initial_state_distribution[init_state] = 1
            self.D0 = initial_state_distribution

            nA = 5
            if allow_diagonal_transitions: nA += 4
            terminal_action = nA-1
            actions = np.zeros((self.nS, nA, self.nS))
            for s in range(self.nS):
                if type(self.goal) == dict and s in self.goal.keys():
                    continue
                elif s == self.goal: continue
                if (s % self.grid_height) != 0: actions[s, 0, s-1] = 1;
                if (s % self.grid_height) != self.grid_height-1: actions[s, 1, s+1] = 1;
                if (s >= self.grid_height): actions[s, 2, s-self.grid_height] = 1;
                if (self.nS-1 - s >= self.grid_height): actions[s, 3, s+self.grid_height] = 1;
                if allow_diagonal_transitions:
                    if (s % self.grid_height) != 0 and (s >= self.grid_height): actions[s, 4, s-self.grid_height-1] = 1;
                    if (s % self.grid_height) != self.grid_height-1 and (s >= self.grid_height): actions[s, 5, s-self.grid_height+1] = 1;
                    if (s % self.grid_height) != 0 and (self.nS-1 - s >= self.grid_height): actions[s, 6, s+self.grid_height-1] = 1;
                    if (s % self.grid_height) != self.grid_height-1 and (self.nS-1 - s >= self.grid_height): actions[s, 7, s+self.grid_height+1] = 1;
            self.actions = actions
            print("New MDP:")
            self.enforce_state_constraints(self.obstacles)
            print("")
            self.terminal_action = terminal_action
            self.nA = nA

            if feature_map == [] or feature_wts == []:
                feature_map_by_state = np.eye(self.nS)
                feature_wts = -2 * np.ones((self.nS, 1))
                feature_map = np.zeros((self.nS, nA, self.nS))
                for i in range(self.nA):
                    feature_map[:, i, :] = feature_map_by_state
                feature_wts[double_cost_states] *= 2
                feature_wts[self.goal] = 0
            
            self.nF = feature_map.shape[2]
            self.feature_map = feature_map
            self.feature_wts = feature_wts
            self.update_rewards_from_features()
            self.reset_backward_forward()
    
    def enforce_state_constraints(self, state_constraints):
        print("Obstacles: %s" % self.obstacles)
        print("State constraints: %s" % state_constraints)
        for i in range(len(state_constraints)):
            constraint = state_constraints[i]
            self.actions[constraint, :, :] *= 0
    
    def enforce_action_constraints(self, action_constraints):
        print("Action constraints: %s" % action_constraints)
        for i in range(len(action_constraints)):
            constraint = action_constraints[i]
            self.actions[:, constraint, :] *= 0
    
    def enforce_feature_constraints(self, feature_constraints):
        print("Feature constraints: %s" % feature_constraints)
        for i in range(len(feature_constraints)):
            constraint = feature_constraints[i]
            locs = np.argwhere(self.feature_map[:, :, constraint] == 1)
            for j in range(len(locs)):
                self.actions[locs[j, 0], locs[j, 1], :] *= 0
    
    def backward_pass(self, T):
        Zs = np.zeros((self.nS, T))
        if type(self.goal) == dict:
            for possible_goal in list(self.goal.keys()):
                Zs[possible_goal, T-1] = np.exp(self.rewards[possible_goal, self.terminal_action])
        else:
            Zs[self.goal, T-1] = np.exp(self.rewards[self.goal, self.terminal_action])
        goal_constraint = True
        if not goal_constraint:
            Zs[:, T-1] = np.exp(self.rewards[:, self.terminal_action])
            Zs[self.obstacles, T-1] *= 0
        
        local_action_probs = np.zeros((self.nS, self.nA, T))
        local_action_probs[:, self.terminal_action, T-1] = np.ones((self.nS))

        for i in range(T-1)[::-1]:
            for s in range(self.nS):

                if type(self.goal) == dict and s in list(self.goal.keys()):
                    Zs[s, i] = np.exp(self.rewards[s, self.terminal_action])
                    local_action_probs[s, self.terminal_action, i] = 1
                    continue
                elif s == self.goal:
                    Zs[self.goal, i] = np.exp(self.rewards[self.goal, self.terminal_action])
                    local_action_probs[self.goal, self.terminal_action, i] = 1
                    continue

                Za = np.zeros((self.nA))
                for a in range(self.nA):
                    future_states = np.argwhere(self.actions[s, a, :] != 0)
                    for s_prime in future_states:
                        weighted_value = np.log(Zs[s_prime, i+1] * self.actions[s, a, s_prime])
                        Za[a] += np.exp(self.rewards[s, a] + weighted_value)
                Zs[s, i] = np.sum(Za)

                if Zs[s, i] > 0:
                    for a in range(self.nA):
                        local_action_probs[s, a, i] = Za[a] / Zs[s, i]
                else:
                    local_action_probs[s, :, i] = np.ones((1, self.nA)) / self.nA 

        self.local_action_probs = local_action_probs
        self.Zs = Zs
        self.backward = True

    # calculate the average reward also
    def forward_pass(self, T):
        """
        Perform the forward pass, compute the average reward,
        and store it in reward_history for each epoch.
        """
        assert (np.abs(np.sum(self.D0) - 1) < 1e-5)

        next_svf = np.zeros((self.nS))
        for s in range(self.nS):
            next_svf[s] = self.D0[s]
        svf = np.copy(next_svf)
        total_svf = np.copy(svf)

        svh = np.zeros((self.nS, T))
        svh[:, 0] = np.copy(svf)

        next_fa = np.zeros((self.nS, self.num_augmented_features()))
        fa = np.copy(next_fa)
        fah = np.zeros((self.num_augmented_features(), T))
        unique_visitations = fah[self.num_augmented_features() - 1, 0]

        # Initialize total reward variable for averaging
        total_reward = 0
        success_count = 0
        total_count = 0
        violation_count = 0
        total_count = self.nS  # Total number of states in the state space

        for t in range(T - 1):
            next_svf = np.zeros((self.nS))
            next_fa = np.zeros((self.nS, self.num_augmented_features()))
            if t > 0:
                fah[:, t] = fah[:, t - 1]

            for s in range(self.nS):
                if s in self.state_constraints:  # Check if the state is in the state_constraints
                    violation_count += 1  # Count this as a violation

                for a in range(self.nA):
                    new_fa = np.multiply(self.augmented_indicator_feature_map(s, a).T, (svf[s] - fa[s, :]))
                    if self.infinite_horizon:
                        local_action_prob = self.local_action_probs[s, a, 0]
                    else:
                        local_action_prob = self.local_action_probs[s, a, t]
                    fah[:, t] += (new_fa * local_action_prob)

                    future_states = np.argwhere(self.actions[s, a, :] != 0)
                    for s_prime in future_states:
                        next_svf[s_prime] += svf[s] * local_action_prob * self.actions[s, a, s_prime]
                        next_fa[s_prime, :] += (fa[s, :] + new_fa) * local_action_prob * self.actions[s, a, s_prime]

                    # Add reward for the current state-action pair to the total reward
                    total_reward += self.rewards[s, a] * local_action_prob

                    # Get the next state for this state-action pair
                    next_state = self.get_next_state(s, a)
                    # Check if the next state is not an obstacle
                    if next_state not in self.obstacles:
                        success_count += 1
                    total_count += 1

            svf = np.copy(next_svf)
            total_svf += svf
            svh[:, t + 1] = svf
            fa = np.copy(next_fa)
            last_unique_visitations = unique_visitations
            unique_visitations = fah[self.augmented_feature_state_indices(), t]


        if T > 1:
            fah[:, T - 1] = fah[:, T - 2]
        for s in range(self.nS):
            terminal_new_fa = np.multiply(self.augmented_indicator_feature_map(s, self.terminal_action),
                                          (svf[s] - fa[s, :]))
            fah[:, T - 1] += terminal_new_fa
        unique_visitations = fah[self.augmented_feature_state_indices(), T - 1]

        self.total_state_visitation_freqs = total_svf
        self.state_visitation_history = svh
        self.unique_visitations = unique_visitations
        self.feature_accrual_history = fah
        self.forward = True

        # Calculate and store the average reward in the MDP
        self.average_reward = total_reward / T  # Store average reward as a scalar
        # Append the average reward to the history list
        # print(self.average_reward)
        # self.reward_history.append(self.average_reward)
        # print(self.reward_history)
        # Return the success rate as a percentage
        success_rate = success_count / total_count
        self.success_rate = success_rate
        violation_rate = violation_count / total_count
        self.violation_rate = violation_rate

    def produce_demonstrations(self, N, num_demos):
        demonstrations = [[None,]*num_demos for i in range(2)]
        assert(self.backward_forward())

        import tqdm
        for i in tqdm.trange(num_demos):
            state_seq, action_seq = [], []
            state_seq.append(np.random.choice(self.nS, 1, True, self.D0.reshape(-1)))
            for j in range(N-1):
                if self.infinite_horizon:
                    local_action_probs = self.local_action_probs[state_seq[j], :, 0]
                else:
                    local_action_probs = self.local_action_probs[state_seq[j], :, j]
                lap = np.copy(local_action_probs)
                action_seq.append(np.random.choice(self.nA, 1, True, lap.reshape(-1)))
                tr = np.copy(self.actions[state_seq[-1], action_seq[-1], :])
                try:
                    state_seq.append(np.random.choice(self.nS, 1, True, tr.reshape(-1)))
                except:
                    break
                if state_seq[j+1] == self.goal or (j+1 == N-1):
                    action_seq.append(np.array([self.terminal_action]))
                    break
            demonstrations[0][i] = np.array(state_seq)
            demonstrations[1][i] = np.array(action_seq)
        
        return demonstrations

import numpy as np

class BeamMDP(GridMDP):
    def __init__(self, *args):
        super().__init__()
        self.beam_width = 5  # Width of the beam search
        if len(args) > 0 and type(args[0]) == GridMDP:
            self.nS, self.nA, self.goal, self.actions, \
                self.terminal_action, self.rewards, self.D0, \
                self.obstacles, self.state_constraints, \
                self.action_constraints, self.feature_constraints, \
                self.infinite_horizon, self.discount_factor, \
                self.nF, self.feature_map, self.feature_wts, \
                self.backward, self.forward, self.Zs, \
                self.local_action_probs, \
                self.total_state_visitation_freqs, \
                self.state_visitation_history, \
                self.unique_visitations, \
                self.feature_accrual_history = \
                dc(args[0].nS), dc(args[0].nA), dc(args[0].goal), dc(args[0].actions), \
                    dc(args[0].terminal_action), dc(args[0].rewards), dc(args[0].D0), \
                    dc(args[0].obstacles), dc(args[0].state_constraints), \
                    dc(args[0].action_constraints), dc(args[0].feature_constraints), \
                    dc(args[0].infinite_horizon), dc(args[0].discount_factor), \
                    dc(args[0].nF), dc(args[0].feature_map), dc(args[0].feature_wts), \
                    dc(args[0].backward), dc(args[0].forward), dc(args[0].Zs), \
                    dc(args[0].local_action_probs), \
                    dc(args[0].total_state_visitation_freqs), \
                    dc(args[0].state_visitation_history), \
                    dc(args[0].unique_visitations), \
                    dc(args[0].feature_accrual_history)

            if len(args) >= 2:

                if len(args) >= 3 and args[2]:
                    constraints = args[1]
                    state_constraints = [item - self.nF for item in self.augmented_feature_state_indices() if
                                         item in constraints]
                    action_constraints = [item - self.nF - self.nS for item in self.augmented_feature_action_indices()
                                          if item in constraints]
                    feature_constraints = [item for item in self.augmented_feature_feature_indices() if
                                           item in constraints]
                    print("New MDP:")
                    self.enforce_state_constraints(state_constraints)
                    self.obstacles += state_constraints
                    self.state_constraints += state_constraints
                    self.enforce_action_constraints(action_constraints)
                    self.action_constraints += action_constraints
                    self.enforce_feature_constraints(feature_constraints)
                    self.feature_constraints += feature_constraints
                    print("")
                else:
                    obstacles = args[2]
                    print("New MDP:")
                    self.enforce_state_constraints(obstacles)
                    self.obstacles += obstacles
                    self.state_constraints += obstacles
                    print("")

                self.reset_backward_forward()

        elif len(args) >= 5:

            self.nS, self.goal, init_state, self.grid_height = args[:4]
            double_cost_states = args[4] if len(args) >= 5 else []
            self.obstacles = args[5] if len(args) >= 6 else []
            self.infinite_horizon = args[6] if len(args) >= 7 else False
            self.discount_factor = args[7] if len(args) >= 8 else 1
            feature_map = args[8] if len(args) >= 9 else []
            feature_wts = args[9] if len(args) >= 10 else []
            allow_diagonal_transitions = args[10] if len(args) >= 11 and args[10] != [] else False

            initial_state_distribution = np.zeros((self.nS, 1))
            if type(init_state) == dict:
                for s in init_state.keys():
                    initial_state_distribution[s] = init_state[s]
                initial_state_distribution /= 1.0 * sum(list(init_state.values()))
            else:
                initial_state_distribution[init_state] = 1
            self.D0 = initial_state_distribution

            nA = 5
            if allow_diagonal_transitions: nA += 4
            terminal_action = nA - 1
            actions = np.zeros((self.nS, nA, self.nS))
            for s in range(self.nS):
                if type(self.goal) == dict and s in self.goal.keys():
                    continue
                elif s == self.goal:
                    continue
                if (s % self.grid_height) != 0: actions[s, 0, s - 1] = 1;
                if (s % self.grid_height) != self.grid_height - 1: actions[s, 1, s + 1] = 1;
                if (s >= self.grid_height): actions[s, 2, s - self.grid_height] = 1;
                if (self.nS - 1 - s >= self.grid_height): actions[s, 3, s + self.grid_height] = 1;
                if allow_diagonal_transitions:
                    if (s % self.grid_height) != 0 and (s >= self.grid_height): actions[
                        s, 4, s - self.grid_height - 1] = 1;
                    if (s % self.grid_height) != self.grid_height - 1 and (s >= self.grid_height): actions[
                        s, 5, s - self.grid_height + 1] = 1;
                    if (s % self.grid_height) != 0 and (self.nS - 1 - s >= self.grid_height): actions[
                        s, 6, s + self.grid_height - 1] = 1;
                    if (s % self.grid_height) != self.grid_height - 1 and (self.nS - 1 - s >= self.grid_height):
                        actions[s, 7, s + self.grid_height + 1] = 1;

            self.actions = actions
            print("New MDP:")
            self.enforce_state_constraints(self.obstacles)
            print("")
            self.terminal_action = terminal_action
            self.nA = nA

            if feature_map == [] or feature_wts == []:
                feature_map_by_state = np.eye(self.nS)
                feature_wts = -2 * np.ones((self.nS, 1))
                feature_map = np.zeros((self.nS, nA, self.nS))
                for i in range(self.nA):
                    feature_map[:, i, :] = feature_map_by_state
                feature_wts[double_cost_states] *= 2
                feature_wts[self.goal] = 0

            self.nF = feature_map.shape[2]
            self.feature_map = feature_map
            self.feature_wts = feature_wts
            self.update_rewards_from_features()
            self.reset_backward_forward()

    def enforce_state_constraints(self, state_constraints):
        print("Obstacles: %s" % self.obstacles)
        print("State constraints: %s" % state_constraints)
        for i in range(len(state_constraints)):
            constraint = state_constraints[i]
            self.actions[constraint, :, :] *= 0

    def enforce_action_constraints(self, action_constraints):
        print("Action constraints: %s" % action_constraints)
        for i in range(len(action_constraints)):
            constraint = action_constraints[i]
            self.actions[:, constraint, :] *= 0

    def enforce_feature_constraints(self, feature_constraints):
        print("Feature constraints: %s" % feature_constraints)
        for i in range(len(feature_constraints)):
            constraint = feature_constraints[i]
            locs = np.argwhere(self.feature_map[:, :, constraint] == 1)
            for j in range(len(locs)):
                self.actions[locs[j, 0], locs[j, 1], :] *= 0

    def forward_pass(self, T):
        """
        Perform the forward pass, compute the average reward,
        and store it in reward_history for each epoch.
        Uses a beam search approach to limit the number of states considered at each step.
        """
        assert (np.abs(np.sum(self.D0) - 1) < 1e-5)

        # Initialize state visitation frequencies (SVF) and feature accrual (FA)
        next_svf = np.zeros((self.nS))
        for s in range(self.nS):
            next_svf[s] = self.D0[s]
        svf = np.copy(next_svf)
        total_svf = np.copy(svf)

        svh = np.zeros((self.nS, T))
        svh[:, 0] = np.copy(svf)

        next_fa = np.zeros((self.nS, self.num_augmented_features()))
        fa = np.copy(next_fa)
        fah = np.zeros((self.num_augmented_features(), T))
        unique_visitations = fah[self.num_augmented_features() - 1, 0]

        # Initialize reward variables
        total_reward = 0
        success_count = 0
        total_count = self.nS
        violation_count = 0

        # Beam Search Initialization: For each time step, keep the best `beam_width` states
        beam_width = self.beam_width
        beam = [(0, s) for s in range(self.nS)]  # (priority, state)

        for t in range(T - 1):
            # Initialize temporary containers for next step
            next_svf = np.zeros((self.nS))
            next_fa = np.zeros((self.nS, self.num_augmented_features()))

            # Keep track of state-action pairs in the beam
            beam = sorted(beam, reverse=True)[:beam_width]  # Sort by priority and keep top `beam_width`

            if t > 0:
                fah[:, t] = fah[:, t - 1]

            # Iterate over states in the beam
            for priority, s in beam:
                if s in self.state_constraints:
                    violation_count += 1

                # Iterate over actions for each state
                for a in range(self.nA):
                    # Compute feature accrual
                    new_fa = np.multiply(self.augmented_indicator_feature_map(s, a).T, (svf[s] - fa[s, :]))
                    local_action_prob = self.local_action_probs[s, a, t] if not self.infinite_horizon else \
                    self.local_action_probs[s, a, 0]
                    fah[:, t] += (new_fa * local_action_prob)

                    # Transition to future states based on the current action
                    future_states = np.argwhere(self.actions[s, a, :] != 0)
                    for s_prime in future_states:
                        s_prime = s_prime[0]  # Extract the state index from the result
                        next_svf[s_prime] += svf[s] * local_action_prob * self.actions[s, a, s_prime]
                        next_fa[s_prime, :] += (fa[s, :] + new_fa) * local_action_prob * self.actions[s, a, s_prime]

                    # Accumulate reward for the current state-action pair
                    total_reward += self.rewards[s, a] * local_action_prob

                    # Determine the next state (if not an obstacle)
                    next_state = self.get_next_state(s, a)
                    if next_state not in self.obstacles:
                        success_count += 1
                    total_count += 1

            # Update state visitation frequencies and feature accrual
            svf = np.copy(next_svf)
            total_svf += svf
            svh[:, t + 1] = svf
            fa = np.copy(next_fa)

            # Update unique visitations
            last_unique_visitations = unique_visitations
            unique_visitations = fah[self.augmented_feature_state_indices(), t]

            # Update beam for the next iteration
            beam = []
            for s in range(self.nS):
                priority = np.sum(svf[s])  # A simple heuristic for beam priority (could be changed)
                beam.append((priority, s))

        # Final step adjustments for terminal state and feature accrual
        if T > 1:
            fah[:, T - 1] = fah[:, T - 2]

        for s in range(self.nS):
            terminal_new_fa = np.multiply(self.augmented_indicator_feature_map(s, self.terminal_action),
                                          (svf[s] - fa[s, :]))
            fah[:, T - 1] += terminal_new_fa

        unique_visitations = fah[self.augmented_feature_state_indices(), T - 1]

        # Store results
        self.total_state_visitation_freqs = total_svf
        self.state_visitation_history = svh
        self.unique_visitations = unique_visitations
        self.feature_accrual_history = fah
        self.forward = True

        # Calculate and store the average reward in the MDP
        self.average_reward = total_reward / T
        self.success_rate = success_count / total_count
        self.violation_rate = violation_count / total_count

        # Optionally print average reward
        # print(self.average_reward)

        # return self.average_reward, self.success_rate, self.violation_rate

    def backward_pass(self, T):
        # Initialize Zs and local_action_probs for all states and time steps
        Zs = np.zeros((self.nS, T))
        if type(self.goal) == dict:
            # Goal is a dictionary of goal states
            for possible_goal in list(self.goal.keys()):
                Zs[possible_goal, T - 1] = np.exp(self.rewards[possible_goal, self.terminal_action])
        else:
            Zs[self.goal, T - 1] = np.exp(self.rewards[self.goal, self.terminal_action])

        goal_constraint = True
        if not goal_constraint:
            Zs[:, T - 1] = np.exp(self.rewards[:, self.terminal_action])
            Zs[self.obstacles, T - 1] *= 0

        local_action_probs = np.zeros((self.nS, self.nA, T))
        local_action_probs[:, self.terminal_action, T - 1] = 1  # Terminal action probabilities are 1 at T-1

        # Backward iteration over time steps
        for i in range(T - 1)[::-1]:
            for s in range(self.nS):
                if type(self.goal) == dict and s in list(self.goal.keys()):
                    Zs[s, i] = np.exp(self.rewards[s, self.terminal_action])
                    local_action_probs[s, self.terminal_action, i] = 1
                    continue
                elif s == self.goal:
                    Zs[self.goal, i] = np.exp(self.rewards[self.goal, self.terminal_action])
                    local_action_probs[self.goal, self.terminal_action, i] = 1
                    continue

                Za = np.zeros(self.nA)
                for a in range(self.nA):
                    future_states = np.argwhere(self.actions[s, a, :] != 0)
                    for s_prime in future_states:
                        weighted_value = np.log(Zs[s_prime, i + 1] * self.actions[s, a, s_prime])
                        Za[a] += np.exp(self.rewards[s, a] + weighted_value)
                Zs[s, i] = np.sum(Za)

                if Zs[s, i] > 0:
                    for a in range(self.nA):
                        local_action_probs[s, a, i] = Za[a] / Zs[s, i]
                else:
                    local_action_probs[s, :, i] = np.ones(self.nA) / self.nA  # Uniform distribution for non-positive Zs

        self.local_action_probs = local_action_probs
        self.Zs = Zs
        self.backward = True

    def select_best_states(self, candidate_states, next_svf):
        """
        Selects the best states from the candidate set based on some criteria (e.g., value).
        Here, we simply sort by the value of the next state visitation frequency.
        """
        # Calculate values based on next_svf or a reward-based value function
        state_values = [(s, next_svf[s]) for s in candidate_states]
        # Sort states by value (descending order)
        state_values_sorted = sorted(state_values, key=lambda x: x[1], reverse=True)
        # Select the top `beam_width` states
        selected_states = [state for state, value in state_values_sorted[:self.beam_width]]
        return selected_states

