from collections import defaultdict

from drl.agents.learning_rate_schedules import scaled_learning_rate


def exploration_function(frequency_cutoff, exploration_reward=1.0):
    def fn(utility, frequency):
        if frequency < frequency_cutoff:
            return exploration_reward
        else:
            return utility

    return fn


class QLearningAgent:
    def __init__(self, env, epochs, frequency_cutoff=3, gamma=1.0, learning_rate_fn=None):
        self.env = env
        self.gamma = gamma
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0))
        self.frequency_table = defaultdict(lambda: 0)  # N[s,a]
        self.exploration_fn = exploration_function(frequency_cutoff)
        self.previous_state = None  # s
        self.previous_action = None  # a
        self.previous_reward = 0.0  # r

        if learning_rate_fn is None:
            self.learning_rate_fn = scaled_learning_rate(epochs)
        else:
            self.learning_rate_fn = learning_rate_fn

    def predict(self, observation):
        current_state = str(observation)
        max_action_utility = None
        max_action = None
        # TODO: Move this to sample entire action space
        for action, utility in self.q_table[current_state]:
            if max_action_utility is None or utility > max_action_utility:
                max_action_utility = utility
                max_action = action

        return max_action

    def update(self, observation, reward, done, info):
        if done:
            self.q_table[(self.previous_state, self.previous_action)] = reward

        current_state = str(observation)
        self.frequency_table[(self.previous_state, self.previous_action)] += 1
        max_action_delta = None
        # TODO: Move this to sample entire action space
        for action, utility in self.q_table[current_state]:
            action_delta = utility - self.q_table[self.previous_state][self.previous_action]
            if max_action_delta is None:
                max_action_delta = action_delta
            max_action_delta = max(max_action_delta, action_delta)
        self.q_table[(self.previous_state, self.previous_action)] += self.learning_rate_fn(info['epoch']) * (self.previous_reward + self.gamma * max_action_delta)
