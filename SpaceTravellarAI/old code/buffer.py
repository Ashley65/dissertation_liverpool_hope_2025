from collections import deque

import numpy as np


class PrioritisedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, max_memory_size=10000, alpha=0.6):
        self.memory = deque(maxlen=max_memory_size)
        self.priorities = deque(maxlen=max_memory_size)
        self.alpha = alpha

    def add_experience(self, state, action, reward, next_state, done):
        """Store a new experience in the memory."""
        max_priority = max(self.priorities, default=1.0)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample_batch(self, batch_size=64, beta=0.4):
        """Sample a batch of experiences with prioritized sampling."""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority