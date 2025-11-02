"""
Training script for the Weather Temperature Prediction RL Agent.
Supports both synthetic and real weather data from Germany.
"""
import numpy as np
import matplotlib.pyplot as plt
from weather_data import WeatherDataGenerator
from rl_agent import WeatherRLAgent
import argparse


def train_agent(episodes=100, use_real_data=False, location='germany'):
    """
    Train the RL agent to predict temperature trends.
    
    Args:
        episodes: Number of training episodes
        use_real_data: If True, use real weather data; if False, use synthetic data
        location: Location for real data ('germany' or 'rheinland-pfalz')
        
    Returns:
        Trained agent, training statistics, and data generator
    """
    print("=" * 60)
    print("Weather Temperature Prediction - RL Agent Training")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Data source: {'Real data' if use_real_data else 'Synthetic data'}")
    if use_real_data:
        print(f"  Location: {location}")
    print(f"  Actions: 0=DOWN, 1=UP")
    print()
    
    # Get weather data
    if use_real_data:
        data_generator = WeatherDataGenerator(use_real_data=True, location=location)
    else:
        data_generator = WeatherDataGenerator(days=365)
    
    data = data_generator.get_data()
    temperatures = data['temperature'].values
    
    if not use_real_data:
        print(f"Generated {len(temperatures)} days of temperature data")
    print(f"Date range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    print(f"Temperature range: {temperatures.min():.1f}°C to {temperatures.max():.1f}°C")
    print(f"Total days: {len(temperatures)}")
    print()
    
    # Initialize agent
    agent = WeatherRLAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Training statistics
    episode_rewards = []
    episode_accuracies = []
    
    print("Starting training...")
    print("-" * 60)
    
    # Training loop
    for episode in range(episodes):
        total_reward = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Episode loop
        for i in range(len(temperatures) - 1):
            # Get current state
            state = agent.get_state(temperatures, i)
            
            # Choose action
            action = agent.choose_action(state)
            
            # Determine actual temperature change
            actual_change = temperatures[i + 1] - temperatures[i]
            # Note: Zero change is classified as DOWN (0) for simplicity
            actual_trend = 1 if actual_change > 0 else 0  # 1=UP, 0=DOWN
            
            # Calculate reward
            if action == actual_trend:
                reward = 1.0  # Correct prediction
                correct_predictions += 1
            else:
                reward = -1.0  # Incorrect prediction
            
            total_reward += reward
            total_predictions += 1
            
            # Get next state
            next_state = agent.get_state(temperatures, i + 1)
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Record statistics
        accuracy = correct_predictions / total_predictions * 100
        episode_rewards.append(total_reward)
        episode_accuracies.append(accuracy)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_accuracy = np.mean(episode_accuracies[-10:])
            print(f"Episode {episode + 1:3d}/{episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Accuracy: {avg_accuracy:5.2f}% | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    print("-" * 60)
    print("Training completed!")
    print()
    
    # Final statistics
    final_accuracy = np.mean(episode_accuracies[-10:])
    print("Final Performance (last 10 episodes):")
    print(f"  Average Accuracy: {final_accuracy:.2f}%")
    print(f"  Q-table size: {len(agent.q_table)} state-action pairs")
    print()
    
    # Save the trained agent
    agent.save('trained_agent.pkl')
    print("Trained agent saved to 'trained_agent.pkl'")
    
    return agent, episode_rewards, episode_accuracies, data_generator


def plot_training_results(episode_rewards, episode_accuracies):
    """
    Plot training results.
    
    Args:
        episode_rewards: List of episode rewards
        episode_accuracies: List of episode accuracies
    """
    print("\nGenerating training plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    ax1.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
             linewidth=2, label='Moving Average (10 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(episode_accuracies, alpha=0.6, label='Episode Accuracy')
    ax2.plot(np.convolve(episode_accuracies, np.ones(10)/10, mode='valid'), 
             linewidth=2, label='Moving Average (10 episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Prediction Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("Training plots saved to 'training_results.png'")
    plt.close()


def predict_2026(agent, data_generator):
    """
    Make predictions for 2026 using the trained agent.
    
    Args:
        agent: Trained RL agent
        data_generator: Weather data generator with historical data
    """
    print("\n" + "=" * 60)
    print("Predicting Temperature Trends for 2026")
    print("=" * 60)
    
    # Get historical data
    historical_data = data_generator.get_data()
    temperatures = historical_data['temperature'].values
    
    # Use the last few months of data to make predictions
    # We'll show predictions based on the most recent temperature trends
    print("\nBased on historical patterns, here are sample predictions:")
    print("(Note: Actual 2026 predictions would require real-time data)")
    print("-" * 60)
    
    # Set epsilon to 0 for predictions (no exploration)
    agent.epsilon = 0.0
    
    # Use last 30 days of historical data to show prediction pattern
    last_30_days = temperatures[-30:]
    
    # Show predictions for the last 10 days as examples
    print("Example predictions based on recent temperature patterns:")
    for i in range(min(10, len(last_30_days) - 1)):
        state = agent.get_state(last_30_days, i)
        action = agent.choose_action(state)
        
        prediction = "UP ↑" if action == 1 else "DOWN ↓"
        print(f"Day {i+1}: Temp={last_30_days[i]:.1f}°C | Next day predicted: {prediction}")
    
    print("-" * 60)
    print("\nPrediction Strategy:")
    print("  • The agent uses temperature differences to predict trends")
    print("  • For real-time 2026 predictions, update with current data")
    print("  • The model learns seasonal patterns and temperature changes")


def test_agent(agent, use_real_data=False, location='germany'):
    """
    Test the trained agent on held-out data.
    
    Args:
        agent: Trained RL agent
        use_real_data: If True, use real data for testing
        location: Location for real data
    """
    print("\n" + "=" * 60)
    print("Testing Trained Agent on Held-Out Data")
    print("=" * 60)
    
    # Generate/fetch test data
    if use_real_data:
        # Use a different seed or split the data
        # For now, we'll use synthetic data with different seed for testing
        test_generator = WeatherDataGenerator(days=365, seed=999)
        print("\nUsing synthetic test data (different seed) for validation")
    else:
        test_generator = WeatherDataGenerator(days=365, seed=123)
        print("\nUsing synthetic test data")
    
    test_data = test_generator.get_data()
    test_temperatures = test_data['temperature'].values
    
    # Set epsilon to 0 for testing (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    print(f"Test data: {len(test_temperatures)} days")
    print()
    
    correct_predictions = 0
    total_predictions = 0
    
    print("Sample predictions (first 10 days):")
    print("-" * 60)
    
    for i in range(min(10, len(test_temperatures) - 1)):
        state = agent.get_state(test_temperatures, i)
        action = agent.choose_action(state)
        
        actual_change = test_temperatures[i + 1] - test_temperatures[i]
        actual_trend = 1 if actual_change > 0 else 0
        
        prediction = "UP" if action == 1 else "DOWN"
        actual = "UP" if actual_trend == 1 else "DOWN"
        correct = "✓" if action == actual_trend else "✗"
        
        print(f"Day {i+1}: Temp={test_temperatures[i]:.1f}°C | "
              f"Predicted={prediction} | Actual={actual} | {correct}")
        
        if action == actual_trend:
            correct_predictions += 1
        total_predictions += 1
    
    # Test on all data
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(test_temperatures) - 1):
        state = agent.get_state(test_temperatures, i)
        action = agent.choose_action(state)
        
        actual_change = test_temperatures[i + 1] - test_temperatures[i]
        actual_trend = 1 if actual_change > 0 else 0
        
        if action == actual_trend:
            correct_predictions += 1
        total_predictions += 1
    
    test_accuracy = correct_predictions / total_predictions * 100
    
    print("-" * 60)
    print(f"\nTest Accuracy: {test_accuracy:.2f}% ({correct_predictions}/{total_predictions} correct)")
    print("(Random baseline would be ~50%)")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train weather temperature prediction RL agent')
    parser.add_argument('--real-data', action='store_true', 
                        help='Use real weather data from Germany')
    parser.add_argument('--location', type=str, default='rheinland-pfalz',
                        choices=['germany', 'rheinland-pfalz'],
                        help='Location for real data (default: rheinland-pfalz)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    
    args = parser.parse_args()
    
    # Train the agent
    agent, rewards, accuracies, data_gen = train_agent(
        episodes=args.episodes,
        use_real_data=args.real_data,
        location=args.location
    )
    
    # Plot training results
    plot_training_results(rewards, accuracies)
    
    # Test the agent
    test_agent(agent, use_real_data=args.real_data, location=args.location)
    
    # Make predictions for 2026
    if args.real_data:
        predict_2026(agent, data_gen)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nFiles generated:")
    print("  - trained_agent.pkl: Saved model")
    print("  - training_results.png: Training visualizations")
    
    if args.real_data:
        print("\nTo make predictions for 2026:")
        print("  The agent has been trained on real historical data")
        print("  Use the trained model with current temperature data to predict trends")
