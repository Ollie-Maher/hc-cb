import numpy as np
import matplotlib.pyplot as plt

data_path = "./experiments/test1/results.npy"

data = np.load(data_path)

rewards = data[:, 0]
steps = data[:, 1]

# Plotting the data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards, label="Total Reward")
plt.xlabel("Episode")

plt.ylabel("Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(steps, label="Total Steps", color="orange")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Total Steps per Episode")
plt.legend()
plt.grid()

plt.show()
