# Weather Forecasting RL Agent

A reinforcement learning agent that learns to predict temperature trends using Q-learning. The agent trains on **realistic German weather data from 2016-2025** (Rheinland-Pfalz region) and predicts whether the temperature will go up or down based on historical patterns.

## Overview

This project implements a simple Q-learning agent that:
- Takes historical temperature data from Germany (2016-2025, ~10 years)
- Predicts whether temperature will increase or decrease
- Learns from rewards based on prediction accuracy
- Improves over time through reinforcement learning
- Can make predictions for 2026 based on learned patterns

## Features

- **Q-Learning Agent**: Implements a Q-learning algorithm for temperature trend prediction
- **Real German Weather Data**: Uses realistic weather patterns from Rheinland-Pfalz/Germany (2016-2025)
- **Synthetic Data Generation**: Also supports simple synthetic data for testing
- **Training Visualization**: Plots training progress and accuracy metrics
- **Model Persistence**: Save and load trained models
- **Testing Framework**: Evaluate agent performance on new data
- **2026 Predictions**: Make predictions for the next year based on historical patterns

## Project Structure

```
weather-forecasting-rl-agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── weather_data.py          # Weather data generator
├── rl_agent.py              # Q-learning RL agent
└── train.py                 # Training script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Eliana-Di-Lodovico/weather-forecasting-rl-agent.git
cd weather-forecasting-rl-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training with Real German Weather Data

To train the agent using realistic German weather patterns (simulated historical data from 2016-2025):

```bash
python train.py --real-data --location rheinland-pfalz --episodes 100
```

Options:
- `--real-data`: Use realistic German weather patterns (10 years of data)
- `--location`: Choose `rheinland-pfalz` or `germany` (default: rheinland-pfalz)
- `--episodes`: Number of training episodes (default: 100)

This will:
- Generate realistic German weather data based on historical patterns (2016-2025)
- Train the agent on ~3,650 days of temperature data
- Make predictions for 2026 based on learned patterns
- Save the trained model to `trained_agent.pkl`
- Generate training visualizations in `training_results.png`

### Training with Synthetic Data

Run the training script with synthetic data:

```bash
python train.py
```

This will:
- Generate synthetic weather data (365 days)
- Train the agent for 100 episodes
- Save the trained model to `trained_agent.pkl`
- Generate training visualizations in `training_results.png`
- Test the agent on new data

### Expected Output (Real Data)

```
============================================================
Weather Temperature Prediction - RL Agent Training
============================================================

Training Configuration:
  Episodes: 100
  Data source: Real data
  Location: rheinland-pfalz
  Actions: 0=DOWN, 1=UP

Fetching real weather data for rheinland-pfalz...
  Location: Rheinland-Pfalz (Mainz)
  Coordinates: 49.9929°N, 8.2473°E
  Date range: 2016-01-01 to 2025-12-31
  ✓ Generated 3653 days of realistic temperature data
  Temperature range: -13.1°C to 41.6°C
  Average temperature: 11.4°C

...

============================================================
Predicting Temperature Trends for 2026
============================================================

Based on historical patterns, here are sample predictions:
Example predictions based on recent temperature patterns:
Day 1: Temp=6.1°C | Next day predicted: DOWN ↓
Day 2: Temp=-1.6°C | Next day predicted: UP ↑
...
```

### Expected Output (Synthetic Data)

```
============================================================
Weather Temperature Prediction - RL Agent Training
============================================================

Training Configuration:
  Episodes: 100
  Days of data: 365
  Actions: 0=DOWN, 1=UP

Generating weather data...
Generated 365 days of temperature data
Temperature range: 3.2°C to 27.8°C

Starting training...
------------------------------------------------------------
Episode  10/100 | Avg Reward:  -45.60 | Avg Accuracy: 43.70% | Epsilon: 0.9044
Episode  20/100 | Avg Reward:  -12.20 | Avg Accuracy: 48.32% | Epsilon: 0.8179
...
Episode 100/100 | Avg Reward:   78.40 | Avg Accuracy: 60.77% | Epsilon: 0.0100
------------------------------------------------------------
Training completed!

Final Performance (last 10 episodes):
  Average Accuracy: 60.77%
  Q-table size: 10 state-action pairs

Trained agent saved to 'trained_agent.pkl'
```

## How It Works

### 1. State Representation

The agent discretizes temperature differences into 5 states:
- State 0: Large decrease (< -2°C)
- State 1: Small decrease (-2°C to -0.5°C)
- State 2: No change (-0.5°C to 0.5°C)
- State 3: Small increase (0.5°C to 2°C)
- State 4: Large increase (> 2°C)

### 2. Actions

The agent can take two actions:
- Action 0: Predict temperature will go DOWN
- Action 1: Predict temperature will go UP

### 3. Rewards

- **+1**: Correct prediction
- **-1**: Incorrect prediction

### 4. Q-Learning Algorithm

The agent uses the Q-learning update rule:

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

Where:
- α = learning rate (0.1)
- γ = discount factor (0.95)
- r = reward
- s = current state
- a = action taken
- s' = next state

## Customization

You can customize the training parameters by modifying `train.py`:

```python
agent = WeatherRLAgent(
    learning_rate=0.1,      # How fast the agent learns
    discount_factor=0.95,   # Importance of future rewards
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995,    # Exploration decay rate
    epsilon_min=0.01        # Minimum exploration rate
)

# Training configuration
train_agent(episodes=100, days=365)
```

## Requirements

- Python 3.7+
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- requests >= 2.31.0 (for API access to real weather data)

## License

See LICENSE file for details.

## Future Improvements

Potential enhancements:
- Add more features (humidity, pressure, wind speed)
- Implement deep Q-learning (DQN)
- Use real weather data from APIs
- Add multi-day forecasting
- Implement different RL algorithms (SARSA, Actor-Critic)
