# This script is used to test the hippocampus agent.
import argparse

from training import train
from objects import make_Objects

import torch
import numpy as np
import json
import os

parser = argparse.ArgumentParser(prog = 'HPC-CB test',
                                 description="Trains agent with defined parameters.")

parser.add_argument('--experiment_id', type=str, default="test",
                    help="Experiment ID for saving the results.")

parser.add_argument('--env_name', type=str, default="t-maze",
                    help="Environment to be trained on.")

parser.add_argument('-hp', '--hippocampus', help='Use plastic HPC', action='store_true')
parser.add_argument('-c', '--cerebellum', help='Use plastic CB', action='store_true')
parser.add_argument('--encoder_noise', help='Add noise to encoder', action='store_true')
parser.add_argument('--cb_input_noise', help='Add noise to CB input', action='store_true')
parser.add_argument('--cb_output_noise', help='Add noise to CB output', action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Set learning rate for the agent')

parser.add_argument('--replicate', type=int,
                    help="Replicate number for the experiment [0,1,2].")
parser.add_argument('--episodes', type=int, default=2000,
                    help="Set number of episodes.")

args = parser.parse_args()

# Parameters:
# PATHS:
EXPERIMENT_ID = args.experiment_id
PATH = f"../experiments/{EXPERIMENT_ID}"

# Environment parameters
ENV_NAME = args.env_name
ENV_MAX_STEPS = 100
ENV_TASK_SWITCH = 100


# Agent parameters
AGENT_CLASS = "hippocampus"

# Set agent learning parameters
hc_str = 'HC' if args.hippocampus else 'no HC'
cb_str = 'CB' if args.cerebellum else 'no CB'
AGENT_NAME = f"{hc_str}-{cb_str}" # Options: "HC-CB", "no HC-CB", "HC-no CB", "no HC-no CB"

# Set noise parameters
enc_noise = "encoder" if args.encoder_noise else ""
cb_input_noise = "cb_input" if args.cb_input_noise else ""
cb_output_noise = "cb_output" if args.cb_output_noise else ""
AGENT_NOISE = f"{enc_noise}-{cb_input_noise}-{cb_output_noise}" # Options: "encoder", "cb_input", "cb_output"

# Set agent sizes
AGENT_HC_GRU_SIZE = 512
AGENT_HC_CA1_SIZE = 512
AGENT_CB_SIZES = [1024, 256] # Need biological data to set these sizes
AGENT_OUTPUT_SIZE = 3

# Set agent learning parameters
AGENT_LR = args.learning_rate
AGENT_GAMMA = 0.99
AGENT_EPSILON = 0.1
TAU = 0.01

# Other parameters
EPISODES = args.episodes
BUFFER_SIZE = 10000
BATCH_SIZE = 32
SEQUENCE_LENGTH = 10 # Number of steps to unroll the GRU for training
SEEDS = [665, 873, 323]
if args.replicate is not None:
    SEEDS = [SEEDS[args.replicate]]


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
        "tau": TAU
    },
    "buffer": {
        "size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH
    },
    "storage": {
        "episodes": EPISODES,
        "path": PATH
    }
}

train_cfg = {
    "episodes": EPISODES,
    "ep_max_steps": ENV_MAX_STEPS,
    "batch_size": BATCH_SIZE
}


def main():

    for i in range(len(SEEDS)):
        seed = SEEDS[i]
        replicate=[665, 873, 323].index(seed)

        # Make directory
        os.makedirs(f"{PATH}_{replicate}", exist_ok=True)
        print(f"Directory {PATH}_{replicate} created.")

        print(f"Running replicate {i+1} with seed {seed}")
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        object_cfg["agent"]["seed"] = seed
        object_cfg["storage"]["replicate"] = replicate
        # Make objects
        env, agent, target, buffer, storage = make_Objects(object_cfg)
        print("Objects created successfully.")

        # Train agent on environment
        train(env, agent, target, buffer, storage, train_cfg)
        print("Training completed.")

        # Save agent and data
        save(agent, storage, object_cfg, train_cfg, replicate)
        print("Agent and data saved successfully.")

def save(agent, storage, object_cfg, train_cfg, replicate=0):
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

    # Set experiment id and path        
    path = f"{PATH}_{replicate}"
    id = f"{EXPERIMENT_ID}_{replicate}"
    
    # Make directory
    os.makedirs(path, exist_ok=True)
    print(f"Directory {path} created.")

    # Set paths for saving
    weights_path = f"{path}/agent_weights.pth"
    config_path = f"{path}/config.json"
    results_path = f"{path}/results.npy"

    # Save agent weights
    torch.save(agent.state_dict(), weights_path)
    print(f"Agent weights saved to {weights_path}")

    # Save dictionary using torch
    config_dict = {
        "experiment_id": id,
        "object_cfg": object_cfg,
        "train_cfg": train_cfg
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f)
    print(f"Dictionary saved to {config_path}")



if __name__ == "__main__":
    main()
    # Run the main function