# This script is used to create the objects for the HCCB project.
import torch.nn as nn
import torch

from agents import HC_CB_agent, HC_agent
from environments import T_Maze, water_maze
from collections import deque
import numpy as np
from itertools import islice

import minigrid

# Makes all objects needed for the training loop
def make_Objects(object_cfg):
    """
    Create the objects defined in config.

    Parameters:
    object_cfg: The configuration of the objects.

    Returns:
    env: The environment object.
    agent: The agent object.
    target: The target network for the agent.
    buffer: The replay buffer for storing experiences.
    storage: The storage for saving the agent and data.
    """
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Unpack object configuration parameters
    new_env = get_env(object_cfg["env"])
    image_shape = new_env.observation_space.spaces["image"].shape

    new_agent, new_target = get_agent(object_cfg["agent"], image_shape, device)
    new_buffer = replay_buffer(object_cfg["buffer"], device)
    new_store = storage(object_cfg["storage"]) 

    
    return new_env, new_agent, new_target, new_buffer, new_store

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

# Get the agent based on the type
def get_agent(agent_cfg, image_shape, device) -> object:
    """
    Get the agent based on the type.

    Parameters:
    agent_type: The type of agent.

    Returns:
    agent: The agent object.
    """
    if agent_cfg["class"] == "hippocampus-cerebellum":
        agent = HC_CB_agent(agent_cfg, image_shape, device=device)
        target = HC_CB_agent(agent_cfg, image_shape, device=device, is_target=True)
        target.load_state_dict(agent.state_dict())
    elif agent_cfg["class"] == "hippocampus":
        agent = HC_agent(agent_cfg, image_shape, device=device)
        target = HC_agent(agent_cfg, image_shape, device=device, is_target=True)
        target.load_state_dict(agent.state_dict())
    else:
        raise ValueError("Unknown agent type")

    return agent, target

# Class definitions
# Storage class for storing performance data
class storage():
    def __init__(self, config):
        self.root = f"{config['path']}_{config['replicate']}"
        self.path = f"{config['path']}_{config['replicate']}/results.npy"
        episodes = config["episodes"]
        self.data = np.empty((episodes, 2))

    def store(self, episode, total_reward, total_steps):
        print("Storing data...")
        self.data[episode] = [total_reward, total_steps]

    def new_path(self):
        pass
        #print("Creating new path...")

    def end_path(self):
        pass
        #print("Ending path...")

    def save_path(self, action, done):
        pass

    def save(self):
        np.save(self.path, self.data)

# Replay buffer class for storing experiences
class replay_buffer():
    '''
    A simple replay buffer for storing experiences.
    
    Stores:
    - state: The current state of the environment.
    - action: The action taken by the agent.
    - reward: The reward received from the environment.
    - next_state: The next state of the environment.
    - done: Whether the episode is done.
    - hidden_state: The hidden state of the agent.
    - cb_input: The CA3 input to the cerebellum at t-1.

    '''
    def __init__(self, buffer_cfg, device):
        self.buffer_size = buffer_cfg["size"]
        self.batch_size = buffer_cfg["batch_size"]
        self.sequence_length = buffer_cfg["sequence_length"]
        self.buffer = deque(maxlen=self.buffer_size)

        self.device = device # Device for sample tensors

    def store(self, state, hidden, action, reward, next_state, done):
        # Store the experience in the buffer
        if len(self.buffer) >= self.buffer_size:
            x = self.buffer.popleft() # Remove the oldest
            del x # Free memory
        self.buffer.append((state, hidden, action, reward, next_state, done))
    
    def sample(self):
        # Sample a batch of experience sequences from the buffer
        
        # Randomly sample indices upto length of buffer - sequence length
        # Prevents sampling out-of-bounds
        # Buffer is a deque so samples will scroll through
        # Therefore removing newest samples does not prevent sampling them
        indices = np.random.choice(len(self.buffer) - self.sequence_length, self.batch_size, replace=True)
        
        batch = [] # List of sequences
        done = 0 # Done flag for the sequence

        for i in indices:
            # Create empty sequences
            state_sequence = []
            hidden_sequence = [] # Hidden state sequence for the agent
            action_sequence = torch.zeros(self.sequence_length, dtype=int, device=self.device)
            reward_sequence = torch.zeros(self.sequence_length, dtype=float, device=self.device)
            next_state_sequence = []
            done_sequence = torch.zeros(self.sequence_length, dtype=int, device=self.device)

            buffer_sequence = list(islice(self.buffer, i, i + self.sequence_length)) # Get the sequence from the buffer

            for j in range(self.sequence_length): # Loop through the sequence length
                if done == 1: # Leave rest of sequence as zeros; done from t-1
                    for k in range(j, self.sequence_length):
                        state_sequence.append(buffer_sequence[j][0])
                        hidden_sequence.append(buffer_sequence[j][1])
                        action_sequence[k] = buffer_sequence[j][2]
                        reward_sequence[k] = buffer_sequence[j][3]
                        next_state_sequence.append(buffer_sequence[j][0])
                        done_sequence[k] = 1
                    break

                done = buffer_sequence[j][5]

                state_sequence.append(buffer_sequence[j][0])
                hidden_sequence.append(buffer_sequence[j][1])
                action_sequence[j] = buffer_sequence[j][2]
                reward_sequence[j] = buffer_sequence[j][3]
                next_state_sequence.append(buffer_sequence[j][4])
                done_sequence[j] = done
            
            # Convert sequences to tensors
            state_sequence = torch.stack(state_sequence)
            next_state_sequence = torch.stack(next_state_sequence)
            hidden_sequence = torch.stack(hidden_sequence)

            batch.append((state_sequence, hidden_sequence, action_sequence, reward_sequence, next_state_sequence, done_sequence))
        
        return batch # Return the batch of sequences
       
    def reward_check(self):
        '''Checks if there is a non-zero reward in the buffer'''
        for i in range(len(self.buffer)):
            if self.buffer[i][3] != 0:
                return True
        return False

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)

# Abstract base class (ABC) for agents and targets
class agent(nn.Module):
    def reset():
        # Reset the agent
        pass

    def forward():
        # Run forward pass through the agent
        pass

# ABC for environments
class env():
    def reset():
        # Reset the environment
        pass

    def step():
        # Take a step in the environment
        pass

def main():
    # Parameters:
    # Environment parameters
    ENV_NAME = "t-maze"
    ENV_MAX_STEPS = 100
    ENV_TASK_SWITCH = 1


    # Agent parameters
    AGENT_CLASS = "hippocampus"
    AGENT_NAME = "HC-CB" # Options: "HC-CB", "no HC-CB", "HC-no CB", "no HC-no CB"
    AGENT_HC_GRU_SIZE = 512
    AGENT_HC_CA1_SIZE = 512
    AGENT_CB_SIZES = [512, 256] # Need biological data to set these sizes
    AGENT_OUTPUT_SIZE = 3
    AGENT_LR = 0.001
    AGENT_GAMMA = 0.99

    # Other parameters
    EPISODES = 10
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    object_cfg = {
        "env": {
            "name": ENV_NAME,
            "max_steps": ENV_MAX_STEPS,
            "task_switch": ENV_TASK_SWITCH
        },
        "agent": {
            "class": AGENT_CLASS,
            "name": AGENT_NAME,
            "hc_gru_size": AGENT_HC_GRU_SIZE,
            "hc_ca1_size": AGENT_HC_CA1_SIZE,
            "cb_sizes": AGENT_CB_SIZES,
            "output_size": AGENT_OUTPUT_SIZE,
            "lr": AGENT_LR,
            "gamma": AGENT_GAMMA,
            "target_update_freq": 10
        },
        "buffer": {
            "size": BUFFER_SIZE
        },
        "storage": {
            "episodes": EPISODES,
            "path": "./storage"
        }
    }

    make_Objects(object_cfg)    
    print("Objects created successfully.")

if __name__ == "__main__":
    main()