import numpy as np
import matplotlib.pyplot as plt

data_path = "./experiments/HCCB_BASE/results.npy"

data = np.load(data_path)

rewards = data[:, 0]
steps = data[:, 1]

def get_rolling_avg(data, window_size=10):
    """
    Calculate the rolling average of the data.
    
    Parameters:
    data: The data to calculate the rolling average for.
    window_size: The size of the rolling window.
    
    Returns:
    rolling_avg: The rolling average of the data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

avg_rewards = get_rolling_avg(rewards, window_size=10)
avg_steps = get_rolling_avg(steps, window_size=10)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(avg_rewards, label="Average Reward")
plt.xlabel("Episode")

plt.ylabel("Reward")
plt.title("Rolling Average Reward per Episode")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(avg_steps, label="Average Steps", color="orange")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Rolling Average Steps per Episode")
plt.legend()
plt.grid()

plt.show()
