"""
Reinforcement Learning Agent for Weather Temperature Prediction.
Uses Q-learning to predict whether temperature will go up or down.
"""
import numpy as np
import pickle


class WeatherRLAgent:
    """
    Q-learning agent that predicts temperature trends.
    
    Actions:
        0: Predict temperature will go DOWN
        1: Predict temperature will go UP
    
    States:
        Discretized temperature difference from the last observation
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the RL agent.
        
        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps (state, action) to Q-value
        self.q_table = {}
        
        # Actions: 0 = DOWN, 1 = UP
        self.actions = [0, 1]
        
    def _discretize_state(self, temp_diff):
        """
        Convert continuous temperature difference to discrete state.
        
        Args:
            temp_diff: Temperature difference from previous day
            
        Returns:
            Discretized state as integer
        """
        # Bin the temperature difference into discrete states
        if temp_diff < -2:
            return 0  # Large decrease
        elif temp_diff < -0.5:
            return 1  # Small decrease
        elif temp_diff < 0.5:
            return 2  # No change
        elif temp_diff < 2:
            return 3  # Small increase
        else:
            return 4  # Large increase
    
    def get_state(self, temperatures, current_idx):
        """
        Get the current state based on recent temperature history.
        
        Args:
            temperatures: Array of temperatures
            current_idx: Current index in the temperature array
            
        Returns:
            State representation
        """
        if current_idx == 0:
            return 2  # Neutral state for first observation
        
        temp_diff = temperatures[current_idx] - temperatures[current_idx - 1]
        return self._discretize_state(temp_diff)
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action (0 or 1)
        """
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        # Exploitation: choose best action based on Q-values
        q_values = [self.q_table.get((state, action), 0.0) for action in self.actions]
        max_q = max(q_values)
        
        # If multiple actions have same Q-value, choose randomly among them
        best_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.q_table.get((state, action), 0.0)
        
        # Get max Q-value for next state
        next_q_values = [self.q_table.get((next_state, a), 0.0) for a in self.actions]
        max_next_q = max(next_q_values)
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save the agent's Q-table to a file.
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
    
    def load(self, filepath):
        """
        Load the agent's Q-table from a file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
