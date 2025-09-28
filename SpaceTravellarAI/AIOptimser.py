import numpy as np
from typing import Dict, List, Optional, Any
import logging


class AIOptimiser:
    """
    This optimises the AI captain's decision-making process by tuning the hyperparameters
    and improving action selection strategies based on the mission performance data.
    """


    def __init__(self, captain=None, learning_rate=0.01, momentum=0.9):

        self.captain = captain
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}  # For momentum-based updates

        # Performance tracking
        self.performance_history = []
        self.rewards_window = []
        self.window_size = 100

        # Default values when captain is None
        default_actions = {'maintain_course', 'increase_velocity', 'emergency_protocol', 'investigate_anomaly',
                           'refuel'}

        # Hyperparameters to tune with safe defaults
        self.tunable_parameters = {
            "learning_rate": (0.0001, 0.1),
            'epsilon': 0.1 if captain is None else captain.epsilon,
            'gamma': 0.99 if captain is None else captain.gamma,
            'anomaly_threshold': 1.0,
            'action_biases': {action: 0.0 for action in (captain.actions.values() if captain else default_actions)}
        }

        logging.info(f"AIOptimiser initialised with learning rate: {learning_rate}, momentum: {momentum}")



    def update_from_experience(self, reward: float, action_name: str, mission_phase: str):
        """Update the optimiser parameters based on the mission performance."""

        # Track rewards for performance evaluation
        self.rewards_window.append(reward)
        if len(self.rewards_window) > self.window_size:
            self.rewards_window = self.rewards_window[-self.window_size:]

        #Adjust action biases based on the performance
        if action_name in self.tunable_parameters['action_biases']:
            # Calculate gradient (simplified) - increase bias for actions with high rewards
            gradient = reward - np.mean(self.rewards_window)



            # Update velocity
            if action_name not in self.velocity:
                self.velocity[action_name] = 0.0

            self.velocity[action_name] = (self.momentum * self.velocity[action_name] +
                                         self.learning_rate * gradient)

            # Update bias
            self.tunable_parameters['action_biases'][action_name] += self.velocity[action_name]



    def optimise_training(self, batch_size=None, learning_rate=None):
        """ Optimise the training parameters based on recent performance data."""

        avg_reward = np.mean(self.rewards_window) if self.rewards_window else 0.0

        # Record performance history
        self.performance_history.append({
            'step': self.captain.mission_step,
            'avg_reward': avg_reward,
            'epsilon' : self.captain.epsilon,
        })

        # Dynamic batch size adjustment based on performance trend
        if batch_size is None and len(self.performance_history) >= 3:
            recent_trend = self.performance_history[-1]['avg_reward'] - self.performance_history[-3]['avg_reward']

            # If performance is declining, increase batch size for more stable learning
            if recent_trend < -0.1:
                return min(256, self.captain.model.batch_size * 1.5)
            # If performance is improving, decrease batch size for faster adaptation
            elif recent_trend > 0.1:
                return max(32, self.captain.model.batch_size * 0.8)

        return batch_size

    def get_action_biases(self, phase_name: str) -> Dict[str, float]:
        """Get optimised action biases for the current mission phase."""

        base_biases = self.tunable_parameters['action_biases']

        # Phase-specific adjustments
        phase_adjustments = {
            'TRANSIT': {'maintain_course': 0.1, 'increase_velocity': 0.05},
            'EXPLORATION': {'investigate_anomaly': 0.15},
            'CRITICAL': {'emergency_protocol': 0.2},
            'EMERGENCY': {'emergency_protocol': 0.3, 'refuel': 0.1},
            'LAUNCH': {'increase_velocity': 0.15}
        }

        # Combine base biases with phase-specific adjustments
        result = base_biases.copy()
        if phase_name in phase_adjustments:
            for action, adjustment in phase_adjustments[phase_name].items():
                if action in result:
                    result[action] += adjustment

        return result

    def get_anomaly_threshold_factor(self, phase_name: str) -> float:
        """"""
        base_factor = self.tunable_parameters['anomaly_threshold_factor']

        # Adjust the threshold based on phase (lower = more sensitive)
        phase_factors = {
            'CRITICAL': 0.8,  # More sensitive in critical phases
            'EMERGENCY': 0.7,  # Most sensitive in emergency
            'LAUNCH': 0.9,  # Less sensitive during launch
            'EXPLORATION': 1.1  # Less sensitive during exploration
        }

        return base_factor * phase_factors.get(phase_name, 1.0)


    def optimise_epsilon(self) -> float:
        """
        """

        if len(self.performance_history) < 5:
            return self.captain.epsilon

            # Calculate performance improvement over last 5 steps
            recent_perf = [p['avg_reward'] for p in self.performance_history[-5:]]
            perf_improvement = recent_perf[-1] - recent_perf[0]

            # If performance is improving, reduce exploration slightly faster
            if perf_improvement > 0.05:
                return max(self.captain.epsilon_min, self.captain.epsilon * 0.95)
            # If performance is declining, slow down the epsilon decay
            elif perf_improvement < -0.05:
                return min(0.5, self.captain.epsilon * 1.05)

            # Otherwise use standard decay
            return max(self.captain.epsilon_min, self.captain.epsilon * self.captain.epsilon_decay)