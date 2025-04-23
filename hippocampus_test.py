# This script is used to test the hippocampus agent.

from training import train
from objects import make_Objects

import torch
import numpy as np
import json
import os


# Parameters:
# PATHS:
EXPERIMENT_ID = "test"
PATH = f"./HCCB/experiments/{EXPERIMENT_ID}"

# Environment parameters
ENV_NAME = "t-maze"
ENV_MAX_STEPS = 100
ENV_TASK_SWITCH = 100


# Agent parameters
AGENT_CLASS = "hippocampus"
AGENT_NAME = "HC-CB" # Options: "HC-CB", "no HC-CB", "HC-no CB", "no HC-no CB"
AGENT_NOISE = "" # "encoder", "cb_input", "cb_output"
AGENT_HC_GRU_SIZE = 512
AGENT_HC_CA1_SIZE = 512
AGENT_CB_SIZES = [512, 256] # Need biological data to set these sizes
AGENT_OUTPUT_SIZE = 3
AGENT_LR = 0.01
AGENT_GAMMA = 0.99
AGENT_EPSILON = 0.1
AGENT_UPDATE_FREQ = 10

# Other parameters
EPISODES = 2000
BUFFER_SIZE = 10000
BATCH_SIZE = 32
SEQUENCE_LENGTH = 10 # Number of steps to unroll the GRU for training

# Config dictionaries
object_cfg = {
    "env": {
        "name": ENV_NAME,
        "max_steps": ENV_MAX_STEPS,
        "task_switch": ENV_TASK_SWITCH
    },
    "agent": {
        "class": AGENT_CLASS,
        "name": AGENT_NAME,
        "noise": AGENT_NOISE,
        "hc_gru_size": AGENT_HC_GRU_SIZE,
        "hc_ca1_size": AGENT_HC_CA1_SIZE,
        "cb_sizes": AGENT_CB_SIZES,
        "output_size": AGENT_OUTPUT_SIZE,
        "lr": AGENT_LR,
        "gamma": AGENT_GAMMA,
        "epsilon": AGENT_EPSILON,
        "target_update_freq": AGENT_UPDATE_FREQ
    },
    "buffer": {
        "size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH
    },
    "storage": {
        "episodes": EPISODES,
        "path": "./storage"
    }
}

train_cfg = {
    "episodes": EPISODES,
    "ep_max_steps": ENV_MAX_STEPS,
    "batch_size": BATCH_SIZE
}


def main():
    # Make objects
    env, agent, target, buffer, storage = make_Objects(object_cfg)
    print("Objects created successfully.")

    # Train agent on environment
    train(env, agent, target, buffer, storage, train_cfg)
    print("Training completed.")

    # Save agent and data
    save(agent, storage, object_cfg, train_cfg)
    print("Agent and data saved successfully.")

def save(agent, storage, object_cfg, train_cfg):
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
    weights_path = f"{PATH}/agent_weights.pth"
    config_path = f"{PATH}/config.json"
    results_path = f"{PATH}/results.npy"

    # Save agent weights
    torch.save(agent.state_dict(), weights_path)
    print(f"Agent weights saved to {weights_path}")

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
    # Run the main function