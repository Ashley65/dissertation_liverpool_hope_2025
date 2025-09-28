
import numpy as np
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger("MissionEvaluator")


class MarsDeepSpaceEvaluator:
    """
    Evaluation framework for Mars deep space missions.
    Assesses AI performance based on mission success rate, decision latency,
    policy convergence, and memory system effectiveness.
    """

    def __init__(self, max_steps=1000, evaluation_frequency=50):
        self.metrics = {
            "mission_success": [],
            "decision_latency": [],
            "policy_convergence": [],
            "memory_effectiveness": [],
            "fuel_efficiency": [],
            "anomaly_response": []
        }
        self.action_distribution = defaultdict(int)
        self.phase_transitions = []
        self.decision_times = []
        self.max_steps = max_steps
        self.evaluation_frequency = evaluation_frequency
        self.waypoint_completion_times = {}
        self.mission_results = []

    def measure_decision_latency(self, captain, state):
        """Measure time taken to make a decision."""
        start_time = time.time()
        action = captain.get_action(state)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.decision_times.append(latency)
        return action, latency

    def assess_policy_convergence(self, captain, training_episodes=10):
        """Evaluate policy convergence by measuring action consistency."""
        test_states = self._generate_test_states()
        action_consistency = []

        for _ in range(training_episodes):
            actions_before = []
            for state in test_states:
                actions_before.append(captain.get_action(state))

            # Run some training
            captain.train(batch_size=64)

            actions_after = []
            for state in test_states:
                actions_after.append(captain.get_action(state))

            # Calculate consistency (% of actions that remained the same)
            consistency = sum(a1 == a2 for a1, a2 in zip(actions_before, actions_after)) / len(test_states)
            action_consistency.append(consistency)

        # Later episodes should have higher consistency as policy converges
        convergence_score = np.mean(action_consistency[-3:]) if len(action_consistency) >= 3 else 0
        return convergence_score

    def evaluate_memory_effectiveness(self, captain):
        """Evaluate how effectively the memory system influences decisions."""
        if not hasattr(captain, 'mission_memory') or not captain.mission_memory.memory:
            return 0.0

        # Get recent experiences from memory
        recent_experiences = captain.mission_memory.get_recent_history(10)
        if not recent_experiences:
            return 0.0

        # Calculate memory influence score
        memory_scores = []
        for i in range(1, len(recent_experiences)):
            prev_state, prev_action, _, _, _ = recent_experiences[i - 1]
            curr_state, curr_action, _, _, _ = recent_experiences[i]

            # Measure state similarity
            state_similarity = 1.0 - np.linalg.norm(np.array(prev_state) - np.array(curr_state)) / (
                        np.linalg.norm(np.array(prev_state)) + 1e-6)

            # Check if similar states led to similar actions in sequence
            action_consistency = 1.0 if prev_action == curr_action else 0.0

            # If states are similar but actions differ, the memory might be influencing decisions
            memory_influence = abs(state_similarity - action_consistency)
            memory_scores.append(memory_influence)

        return np.mean(memory_scores) if memory_scores else 0.0

    def run_periodic_evaluation(self, captain):
        """
        Runs evaluation periodically during the mission.

        Args:
            captain: The mission captain object
        """
        if not hasattr(self, 'trajectory') or not self.trajectory:
            return

        # Create temporary arrays from collected data
        trajectory = np.array(self.trajectory)
        fuel_levels = self.fuel_history if hasattr(self, 'fuel_history') else []
        actions_taken = self.action_history if hasattr(self, 'action_history') else []
        phases = self.phase_history if hasattr(self, 'phase_history') else []

        # Get waypoints if available
        waypoints = []
        if hasattr(captain, 'target_waypoints'):
            waypoints = captain.target_waypoints

        # Calculate intermediate metrics
        if waypoints:
            waypoints_reached = self._calculate_waypoints_reached(trajectory, waypoints)
            mission_success_rate = waypoints_reached / len(waypoints) if waypoints else 0
            self.metrics["mission_success"].append(mission_success_rate)

        if fuel_levels:
            fuel_efficiency = self._calculate_fuel_efficiency(trajectory, fuel_levels, waypoints)
            self.metrics["fuel_efficiency"].append(fuel_efficiency)

        # Update action distribution
        for action in actions_taken:
            self.action_distribution[action] += 1

        logger.debug(f"Periodic evaluation at step {self.step_count}: " +
                     f"Success rate: {self.metrics['mission_success'][-1] if self.metrics['mission_success'] else 'N/A'}, " +
                     f"Fuel efficiency: {self.metrics['fuel_efficiency'][-1] if self.metrics['fuel_efficiency'] else 'N/A'}")

    def update_metrics(self, captain, position, fuel_level, action, mission_phase):
        """
        Update mission metrics with current data.

        Args:
            captain: The mission captain object
            position: Current spacecraft position
            fuel_level: Current fuel level
            action: Action taken this step
            mission_phase: Current mission phase
        """
        # Store data for trajectory analysis
        if not hasattr(self, 'trajectory'):
            self.trajectory = []
        self.trajectory.append(position.copy())

        # Store fuel consumption data
        if not hasattr(self, 'fuel_history'):
            self.fuel_history = []
        self.fuel_history.append(fuel_level)

        # Store action history
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(action)

        # Store phase history
        if not hasattr(self, 'phase_history'):
            self.phase_history = []
        self.phase_history.append(mission_phase)

        # Calculate distance to target
        if hasattr(captain, 'target_position'):
            distance_to_target = np.linalg.norm(position - captain.target_position)
            if not hasattr(self, 'distance_history'):
                self.distance_history = []
            self.distance_history.append(distance_to_target)

        # Track mission steps
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1

        # Periodically run evaluation if needed
        if hasattr(self, 'evaluation_frequency') and self.step_count % self.evaluation_frequency == 0:
            self.run_periodic_evaluation(captain)

    def evaluate_mission(self, captain, trajectory, fuel_levels, actions_taken, phases, waypoints):
        """
        Perform comprehensive mission evaluation after completion.

        Args:
            captain: The AI captain instance
            trajectory: List of position vectors during mission
            fuel_levels: List of fuel levels at each step
            actions_taken: List of actions taken during mission
            phases: List of mission phases during each step
            waypoints: List of waypoints that defined the mission

        Returns:
            evaluation_results: Dictionary with evaluation metrics
        """
        trajectory = np.array(trajectory)

        # Calculate mission success rate
        waypoints_reached = self._calculate_waypoints_reached(trajectory, waypoints)
        mission_success_rate = waypoints_reached / len(waypoints) if waypoints else 0

        # Calculate average decision latency
        avg_decision_latency = np.mean(self.decision_times) if self.decision_times else 0

        # Calculate fuel efficiency
        fuel_efficiency = self._calculate_fuel_efficiency(trajectory, fuel_levels, waypoints)

        # Analyze action distribution
        for action in actions_taken:
            self.action_distribution[action] += 1

        # Calculate phase transition efficiency
        phase_changes = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i - 1])
        phase_efficiency = len(waypoints) / (phase_changes + 1) if phase_changes > 0 else 0

        # Calculate policy convergence if possible
        if hasattr(captain, 'train'):
            policy_convergence = self.assess_policy_convergence(captain, training_episodes=5)
        else:
            policy_convergence = 0

        # Calculate memory effectiveness
        memory_effectiveness = self.evaluate_memory_effectiveness(captain)

        # Calculate anomaly response efficiency
        anomaly_response = self._calculate_anomaly_response(captain, actions_taken)

        # Compile results
        results = {
            "mission_success_rate": mission_success_rate,
            "waypoints_reached": waypoints_reached,
            "total_waypoints": len(waypoints),
            "avg_decision_latency_ms": avg_decision_latency,
            "policy_convergence": policy_convergence,
            "memory_effectiveness": memory_effectiveness,
            "fuel_efficiency": fuel_efficiency,
            "phase_transition_efficiency": phase_efficiency,
            "anomaly_response": anomaly_response,
            "action_distribution": dict(self.action_distribution)
        }

        self.mission_results.append(results)
        return results

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        if not self.mission_results:
            return "No mission data available for evaluation."

        latest_result = self.mission_results[-1]
        avg_results = self._average_mission_results()

        report = "=== MARS MISSION EVALUATION REPORT ===\n\n"

        # Mission Success
        report += "MISSION SUCCESS METRICS:\n"
        report += f"- Latest mission success rate: {latest_result['mission_success_rate']:.2%}\n"
        report += f"- Average mission success rate: {avg_results['mission_success_rate']:.2%}\n"
        report += f"- Waypoints reached: {latest_result['waypoints_reached']}/{latest_result['total_waypoints']}\n\n"

        # Decision Latency
        report += "DECISION PERFORMANCE:\n"
        report += f"- Average decision latency: {latest_result['avg_decision_latency_ms']:.2f} ms\n"
        report += f"- Policy convergence score: {latest_result['policy_convergence']:.2f}/1.0\n\n"

        # Memory Effectiveness
        report += "MEMORY SYSTEM EFFECTIVENESS:\n"
        report += f"- Memory utilization score: {latest_result['memory_effectiveness']:.2f}/1.0\n\n"

        # Efficiency Metrics
        report += "EFFICIENCY METRICS:\n"
        report += f"- Fuel efficiency: {latest_result['fuel_efficiency']:.2f}/1.0\n"
        report += f"- Phase transition efficiency: {latest_result['phase_transition_efficiency']:.2f}\n"
        report += f"- Anomaly response efficiency: {latest_result['anomaly_response']:.2f}/1.0\n\n"

        # Action Distribution
        report += "ACTION DISTRIBUTION:\n"
        for action, count in latest_result['action_distribution'].items():
            percentage = count / sum(latest_result['action_distribution'].values())
            report += f"- {action}: {count} ({percentage:.1%})\n"

        return report

    def plot_evaluation_metrics(self):
        """Generate visualizations for key evaluation metrics."""
        if not self.mission_results:
            logger.warning("No mission data available for plotting.")
            return

        metrics_over_time = {
            "mission_success_rate": [],
            "avg_decision_latency_ms": [],
            "policy_convergence": [],
            "memory_effectiveness": [],
            "fuel_efficiency": []
        }

        for result in self.mission_results:
            for key in metrics_over_time:
                metrics_over_time[key].append(result[key])

        # Plot metrics over time
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Mars Mission Performance Metrics', fontsize=16)

        # Mission success rate
        axes[0, 0].plot(metrics_over_time["mission_success_rate"], 'b-', marker='o')
        axes[0, 0].set_title('Mission Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1.1)

        # Decision latency
        axes[0, 1].plot(metrics_over_time["avg_decision_latency_ms"], 'r-', marker='o')
        axes[0, 1].set_title('Decision Latency')
        axes[0, 1].set_ylabel('Latency (ms)')

        # Policy convergence
        axes[1, 0].plot(metrics_over_time["policy_convergence"], 'g-', marker='o')
        axes[1, 0].set_title('Policy Convergence')
        axes[1, 0].set_ylabel('Convergence Score')
        axes[1, 0].set_ylim(0, 1.1)

        # Memory effectiveness
        axes[1, 1].plot(metrics_over_time["memory_effectiveness"], 'c-', marker='o')
        axes[1, 1].set_title('Memory System Effectiveness')
        axes[1, 1].set_ylabel('Memory Utilization')
        axes[1, 1].set_ylim(0, 1.1)

        # Fuel efficiency
        axes[2, 0].plot(metrics_over_time["fuel_efficiency"], 'm-', marker='o')
        axes[2, 0].set_title('Fuel Efficiency')
        axes[2, 0].set_ylabel('Efficiency Score')
        axes[2, 0].set_ylim(0, 1.1)

        # Action distribution (latest mission)
        latest_actions = self.mission_results[-1]['action_distribution']
        actions = list(latest_actions.keys())
        counts = list(latest_actions.values())
        axes[2, 1].bar(actions, counts)
        axes[2, 1].set_title('Action Distribution')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_xticklabels(actions, rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        return fig

    def _calculate_waypoints_reached(self, trajectory, waypoints):
        """Calculate how many waypoints were successfully reached."""
        reached_count = 0

        # Handle different possible waypoint formats
        waypoint_positions = []
        for wp in waypoints:
            if isinstance(wp, dict) and "position" in wp:
                waypoint_positions.append(wp["position"])
            elif isinstance(wp, np.ndarray):
                waypoint_positions.append(wp)
            elif hasattr(wp, "__iter__") and len(wp) >= 3:
                waypoint_positions.append(np.array(wp))

        waypoint_positions = np.array(waypoint_positions)

        if len(waypoint_positions) == 0:
            return 0

        for waypoint_pos in waypoint_positions:
            # Check if any position in trajectory comes within threshold of waypoint
            distances = np.linalg.norm(trajectory - waypoint_pos, axis=1)
            if np.min(distances) < 1e9:  # 1 million km threshold
                reached_count += 1

        return reached_count

    def _calculate_fuel_efficiency(self, trajectory, fuel_levels, waypoints):
        """Calculate fuel efficiency score based on distance traveled and fuel consumed."""
        if not fuel_levels or len(fuel_levels) < 2:
            return 0.0

        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(trajectory)):
            total_distance += np.linalg.norm(trajectory[i] - trajectory[i - 1])

        # Calculate fuel consumed
        fuel_consumed = fuel_levels[0] - fuel_levels[-1]
        if fuel_consumed <= 0:
            return 1.0  # Perfect efficiency if no fuel consumed (unlikely)

        # Calculate theoretical minimum distance (sum of straight lines between waypoints)
        waypoint_positions = []
        for wp in waypoints:
            if isinstance(wp, dict) and "position" in wp:
                waypoint_positions.append(wp["position"])
            elif isinstance(wp, np.ndarray):
                waypoint_positions.append(wp)
            elif hasattr(wp, "__iter__") and len(wp) >= 3:
                waypoint_positions.append(np.array(wp))

        waypoint_positions = np.array(waypoint_positions)

        min_distance = 0
        if len(waypoint_positions) >= 2:
            for i in range(1, len(waypoint_positions)):
                min_distance += np.linalg.norm(waypoint_positions[i] - waypoint_positions[i - 1])

        # Calculate efficiency ratio (closer to 1.0 is better)
        if min_distance == 0:
            return 0.5  # Default if we can't calculate

        path_efficiency = min_distance / total_distance if total_distance > 0 else 0
        fuel_per_distance = fuel_consumed / total_distance if total_distance > 0 else float('inf')

        # Normalize to 0-1 scale (lower fuel per distance is better)
        norm_fuel_efficiency = np.exp(-fuel_per_distance * 1e-10) if fuel_per_distance < float('inf') else 0

        # Combined score (weighted average)
        return 0.6 * path_efficiency + 0.4 * norm_fuel_efficiency

    def _calculate_anomaly_response(self, captain, actions_taken):
        """Calculate how efficiently anomalies were handled."""
        # We need sensor data history to properly evaluate this
        if not hasattr(captain, 'sensor_data') or not hasattr(captain.sensor_data, 'history'):
            return 0.5  # Default if we can't calculate

        anomaly_responses = []
        anomaly_steps = []

        # Find steps where anomalies were detected
        for step, readings in enumerate(captain.sensor_data.history):
            if readings.get('anomaly_detector', False):
                anomaly_steps.append(step)

        # Check if appropriate actions were taken within 3 steps of anomaly detection
        for anomaly_step in anomaly_steps:
            response_window = range(anomaly_step, min(anomaly_step + 4, len(actions_taken)))
            appropriate_response = False

            for step in response_window:
                if step < len(actions_taken) and actions_taken[step] in [
                    'investigate_anomaly', 'emergency_protocol', 'adjust_trajectory'
                ]:
                    appropriate_response = True
                    break

            anomaly_responses.append(appropriate_response)

        # Calculate response rate
        if anomaly_responses:
            return sum(anomaly_responses) / len(anomaly_responses)
        else:
            return 1.0  # No anomalies to respond to

    def _generate_test_states(self):
        """Generate a set of test states for evaluating policy consistency."""
        test_states = []

        # Generate a diverse set of test states
        for i in range(20):
            # Create states with different positions, velocities, fuel levels
            position = np.array([1e9 * (i % 5), 1e8 * (i % 3), 0])
            velocity = np.array([1e4 * ((i % 4) - 1.5), 1e3 * ((i % 3) - 1), 0])
            fuel_level = 0.1 + 0.8 * (i / 20)

            # Create a simulated preprocessed state vector
            state = np.zeros(12)  # Assuming state dimension is 12
            state[0:3] = position / 1e10  # Normalized position
            state[3:6] = velocity / 1e4  # Normalized velocity
            state[6] = fuel_level
            state[7] = i / 20  # Simulated mission progress

            # Add some anomaly indicators for a few states
            if i % 5 == 0:
                state[8] = 1.0  # Anomaly indicator
                state[9] = 0.7  # Anomaly score

            test_states.append(state)

        return test_states

    def _average_mission_results(self):
        """Calculate average results across all missions."""
        if not self.mission_results:
            return {}

        avg_results = {}
        for key in self.mission_results[0]:
            if key != 'action_distribution':
                avg_results[key] = np.mean([r[key] for r in self.mission_results])

        return avg_results

    def plot_evaluation_metrics(self):
        """
        Generate and display plots for mission evaluation metrics.
        """
        if not hasattr(self, 'trajectory') or not self.trajectory:
            raise ValueError("No trajectory data available for plotting.")

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle("Mission Evaluation Metrics", fontsize=16)

        # Plot 1: Trajectory
        trajectory = np.array(self.trajectory)
        axs[0].plot(trajectory[:, 0], trajectory[:, 1], marker='o', label="Trajectory")
        axs[0].set_title("Spacecraft Trajectory")
        axs[0].set_xlabel("X Position (km)")
        axs[0].set_ylabel("Y Position (km)")
        axs[0].legend()
        axs[0].grid()

        # Plot 2: Fuel Efficiency
        if hasattr(self, 'fuel_history') and self.fuel_history:
            axs[1].plot(self.fuel_history, marker='o', label="Fuel Level")
            axs[1].set_title("Fuel Level Over Time")
            axs[1].set_xlabel("Mission Step")
            axs[1].set_ylabel("Fuel Level")
            axs[1].legend()
            axs[1].grid()
        else:
            axs[1].text(0.5, 0.5, "No fuel data available", ha='center', va='center', fontsize=12)
            axs[1].set_title("Fuel Level Over Time")

        # Plot 3: Mission Success Rate
        if hasattr(self, 'metrics') and "mission_success" in self.metrics:
            axs[2].plot(self.metrics["mission_success"], marker='o', label="Success Rate")
            axs[2].set_title("Mission Success Rate Over Time")
            axs[2].set_xlabel("Evaluation Step")
            axs[2].set_ylabel("Success Rate")
            axs[2].legend()
            axs[2].grid()
        else:
            axs[2].text(0.5, 0.5, "No success rate data available", ha='center', va='center', fontsize=12)
            axs[2].set_title("Mission Success Rate Over Time")

        # Adjust layout and show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        return fig


def evaluate_mission_run(captain, trajectory, fuel_levels, actions_taken, phases, waypoints):
    """
    Convenience function to evaluate a single mission run.

    Args:
        captain: The AI captain instance
        trajectory: List of position vectors during mission
        fuel_levels: List of fuel levels at each step
        actions_taken: List of actions taken during mission
        phases: List of mission phases during each step
        waypoints: List of waypoints that defined the mission

    Returns:
        evaluation_report: String containing evaluation results
    """
    evaluator = MarsDeepSpaceEvaluator()
    results = evaluator.evaluate_mission(captain, trajectory, fuel_levels, actions_taken, phases, waypoints)
    return evaluator.generate_evaluation_report()