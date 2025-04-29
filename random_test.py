# This script is used to test the random agent.

import torch
import numpy as np
import json
import os
from environments import T_Maze, water_maze
from collections import deque
import numpy as np
import minigrid


# Parameters:
# PATHS:
EXPERIMENT_ID = "random_water_test3"
PATH = f"./HCCB/experiments/{EXPERIMENT_ID}"

# Environment parameters
ENV_NAME = "water-maze"
ENV_MAX_STEPS = 100
ENV_TASK_SWITCH = 100
SEED = 873 #665, 873, 323

EPISODES = 2000

# Config dictionaries
object_cfg = {
    "env": {
        "name": ENV_NAME,
        "max_steps": ENV_MAX_STEPS,
        "task_switch": ENV_TASK_SWITCH
    },
    "storage": {
        "episodes": EPISODES,
        "path": "./storage"
    }
}

train_cfg = {
    "episodes": EPISODES,
    "ep_max_steps": ENV_MAX_STEPS,
    "seed": SEED
}

def main():
    # Make objects
    env, storage = make_Objects(object_cfg)
    print("Objects created successfully.")

    # Train agent on environment
    train(env, storage, train_cfg)
    print("Training completed.")

    # Save agent and data
    save(storage, object_cfg, train_cfg)
    print("Agent and data saved successfully.")

def make_Objects(cfg):
    
    # Unpack object configuration parameters
    new_env = get_env(object_cfg["env"])

    new_store = storage(object_cfg["storage"]) 

    return new_env, new_store

# Get the environment based on the type
def get_env(env_cfg) -> object:
    """
    Get the environment based on the type.

    Parameters:
    env_type: The type of environment.

    Returns:
    env: The environment object.
    """
    if env_cfg["name"] == "t-maze":
        env = T_Maze(max_steps=env_cfg["max_steps"], task_switch=env_cfg["task_switch"])
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    elif env_cfg["name"] == "water-maze":
        env = water_maze(max_steps=env_cfg["max_steps"], task_switch=env_cfg["task_switch"])
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env)

    else:
        raise ValueError("Unknown environment type")

    return env

# Storage class for storing performance data
class storage():
    def __init__(self, config):
        episodes = config["episodes"]
        self.data = np.empty((episodes, 2))

    def store(self, episode, total_reward, total_steps):
        print("Storing data...")
        self.data[episode] = [total_reward, total_steps]

    def new_path(self):
        print("Creating new path...")

    def end_path(self):
        print("Ending path...")

    def save_path(self, action, done):
        pass

def train(env, storage, train_cfg):
    """
    Train the agent on the environment.

    Parameters:
    env: The environment to train the agent on.
    storage: The storage for saving the agent and data.
    train_cfg: The training configuration parameters.

    Returns:
    None
    """
    
    # Unpack training configuration parameters
    episodes = train_cfg["episodes"]
    max_steps = train_cfg["ep_max_steps"]
    seed = train_cfg["seed"]
    rng = np.random.default_rng(seed)
    
    # Training loop
    for episode in range(episodes):
        # Run episode
        total_reward, total_steps = run_Episode(env, rng, max_steps, storage)
        # Save the data
        storage.store(episode, total_reward, total_steps)
        # Print progress
        print(f"Episode {episode + 1}/{episodes} completed.")

def run_Episode(env, rng, max_steps, storage):
    """
    Run a single episode of training.

    Parameters:
    env: The environment to train the agent on.
    agent: The agent to train.
    target: The target network for the agent.
    buffer: The replay buffer for storing experiences.
    storage: The storage for saving the agent and data.

    Returns:
    None
    """
    print("Running episode...")

    
    # Reset environment and agent
    state = env.reset()[0]["image"] # Get the image from the state IMPLEMENT WRAPPER TO FIX THIS
    state = torch.tensor(state, dtype=torch.float32)

    # Initialise path storage
    storage.new_path()

    # Run steps in the episode
    for step in range(max_steps):
        # Get random action
        action = rng.integers(0, 3)
        # Take action in environment
        next_state, reward, done, trunc, *_ = env.step(action)
        next_state = torch.tensor(next_state["image"], dtype=torch.float32) # Get the image from the state IMPLEMENT WRAPPER TO FIX THIS

        done = done or trunc # Check if episode is done

        # Update state
        state = next_state
        # Save path to storage
        storage.save_path(action, done)

        # Update the agent

        # End if episode is done
        if done:
            break
        
    # End path storage
    storage.end_path()
    
    return reward, step + 1 # Reward, total steps


def save(storage, object_cfg, train_cfg):
    """
    Save the agent and data to the specified storage.

    Parameters:
    agent: The agent to save.
    storage: The data from training.
    object_cfg: The configuration of the objects.
    train_cfg: The training configuration parameters.
    """
    # Save agent and data to storage
    print("Saving agent and data...")

    # Make directory
    os.makedirs(PATH, exist_ok=True)
    print(f"Directory {PATH} created.")

    # Set paths for saving
    config_path = f"{PATH}/config.json"
    results_path = f"{PATH}/results.npy"

    # Save dictionary using torch
    config_dict = {
        "experiment_id": EXPERIMENT_ID,
        "object_cfg": object_cfg,
        "train_cfg": train_cfg
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f)
    print(f"Dictionary saved to {config_path}")

    # Save NumPy array
    np.save(results_path, storage.data)
    print(f"NumPy array saved to {results_path}")

if __name__ == "__main__":
    main()