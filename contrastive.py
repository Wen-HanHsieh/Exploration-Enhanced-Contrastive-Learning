import numpy as np
from sklearn.neighbors import KDTree

class ContrastiveModule:
    def __init__(self, state_dim, threshold=0.1, max_exploration_reward=0.75, reward_decay=0.997, max_states=1000):
        self.state_dim = state_dim
        self.threshold = threshold
        self.max_exploration_reward = max_exploration_reward
        self.current_exploration_reward = max_exploration_reward
        self.reward_decay = reward_decay
        self.max_states = max_states
        self.explored_states = np.empty((0, state_dim))
        self.kd_tree = None

    def add_state(self, state):
        if len(self.explored_states) >= self.max_states:
            self.explored_states = self.explored_states[1:]  # Remove the oldest state
        self.explored_states = np.vstack([self.explored_states, state])
        self.kd_tree = KDTree(self.explored_states)  # Rebuild the k-d tree

    def is_new_state(self, state):
        if self.explored_states.shape[0] == 0:
            return True
        distances, _ = self.kd_tree.query([state], k=1)
        return distances[0][0] >= self.threshold

    def get_exploration_reward(self, state):
        reward = 0.0
        if self.is_new_state(state):
            self.add_state(state)
            reward = self.current_exploration_reward
            self.current_exploration_reward *= self.reward_decay
        return reward
