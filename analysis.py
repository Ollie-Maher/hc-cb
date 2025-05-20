import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

rewards = np.empty((3, 2000), dtype=float)
steps = np.empty((3, 2000), dtype=float)
for i in range(3):
    data_path = f"./experiments/base_test_start_weights_{i}/results.npy"
    data = np.load(data_path)
    rewards[i] = data[:, 0]
    steps[i] = data[:, 1]
    

def get_rolling_vals(data, window_size=100):
    """
    Calculate the rolling average of the data.
    
    Parameters:
    data: The data to calculate the rolling average for.
    window_size: The size of the rolling window.
    
    Returns:
    rolling_avg: The rolling average of the data.
    """
    mean_avg = np.mean(data, axis=0)
    std_avg = np.std(data, axis=0)
    max_vals = np.max(data, axis=0)
    min_vals = np.min(data, axis=0)
    iqr = stats.iqr(data, axis=0)

    rolling_mean = np.convolve(mean_avg, np.ones(window_size)/window_size, mode='valid')
    rolling_std = np.convolve(std_avg, np.ones(window_size)/window_size, mode='valid')
    rolling_max = np.convolve(max_vals, np.ones(window_size)/window_size, mode='valid')
    rolling_min = np.convolve(min_vals, np.ones(window_size)/window_size, mode='valid')
    rolling_iqr = np.convolve(iqr, np.ones(window_size)/window_size, mode='valid')

    return rolling_mean, rolling_std, rolling_max, rolling_min, rolling_iqr

ones = 0
for i in range(rewards.shape[1]):
    ones += 1 if rewards[0, i] > 0 else 0
print(f"Number of ones in rewards: {ones}")


# Plot raw data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards[0], label="Replicate 1", marker = '.', ls = 'None')
plt.plot(rewards[1], label="Replicate 2", marker = '.', ls = 'None')
plt.plot(rewards[2], label="Replicate 3", marker = '.', ls = 'None')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards per Episode")
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(steps[0], label="Replicate 1", marker = '.', ls = 'None')
plt.plot(steps[1], label="Replicate 2", marker = '.', ls = 'None')
plt.plot(steps[2], label="Replicate 3", marker = '.', ls = 'None')
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per Episode")
plt.legend()
plt.grid()


avg_rewards, std_rewards, max_rewards, min_rewards, iqr_rewards = get_rolling_vals(rewards)
avg_steps, std_steps, max_steps, min_steps, iqr_steps = get_rolling_vals(steps)

# Plotting the analysed data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(avg_rewards, label="Average Reward")
plt.fill_between(range(len(avg_rewards)), avg_rewards - iqr_rewards/2, avg_rewards + iqr_rewards/2, color='yellow', alpha=0.5)
plt.fill_between(range(len(avg_rewards)), min_rewards, max_rewards, color='grey', alpha=0.2)
plt.xlabel("Episode")

plt.ylabel("Reward")
plt.title("Rolling Average Reward per Episode")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(avg_steps, label="Average Steps", color="purple")
plt.fill_between(range(len(avg_steps)), avg_steps - iqr_steps/2, avg_steps + iqr_steps/2, color='orange', alpha=0.5)
plt.fill_between(range(len(avg_steps)), min_steps, max_steps, color='grey', alpha=0.2)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Rolling Average Steps per Episode")
plt.legend()
plt.grid()

plt.show()
