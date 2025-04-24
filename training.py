# This file contains the training loop for the agent in the HCCB project.
import torch
import numpy as np

def train(env, agent, target, buffer, storage, train_cfg):
    """
    Train the agent on the environment.

    Parameters:
    env: The environment to train the agent on.
    agent: The agent to train.
    target: The target network for the agent.
    buffer: The replay buffer for storing experiences.
    storage: The storage for saving the agent and data.
    train_cfg: The training configuration parameters.

    Returns:
    None
    """
    
    # Unpack training configuration parameters
    episodes = train_cfg["episodes"]
    max_steps = train_cfg["ep_max_steps"]
    batch_size = train_cfg["batch_size"]

    
    # Training loop
    for episode in range(episodes):
        # Run episode
        total_reward, total_steps = run_Episode(env, max_steps, agent, target, buffer, storage, batch_size)
        # Save the agent and data
        storage.store(episode, total_reward, total_steps)
        # Print progress
        print(f"Episode {episode + 1}/{episodes} completed.")

def run_Episode(env, max_steps, agent, target, buffer, storage, batch_size):
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
    agent.reset()

    # Initialise path storage
    storage.new_path()

    # Run steps in the episode
    for step in range(max_steps):
        # Get action from agent
        q_values, hidden_state, next_cb_input = agent(state)
        action = int((torch.argmax(q_values).item() if np.random.rand() > agent.epsilon else np.random.randint(0, q_values.shape[1])))
        # Take action in environment
        next_state, reward, done, trunc, *_ = env.step(action)
        next_state = next_state["image"] # Get the image from the state IMPLEMENT WRAPPER TO FIX THIS

        done = done or trunc # Check if episode is done

        # Store experience in buffer
        buffer.store(state, action, reward, next_state, done)
        # Update state
        state = next_state
        # Save path to storage
        storage.save_path(action, done)

        # Update the agent
        update_Agent(agent, target, buffer, batch_size)

        # End if episode is done
        if done:
            break
        
    # End path storage
    storage.end_path()
    
    return reward, step + 1 # Reward, total steps


def update_Agent(agent, target, buffer, batch_size):
    """
    Update the agent using the experiences stored in the buffer.

    Parameters:
    agent: The agent to train.
    target: The target network for the agent.
    buffer: The replay buffer for storing experiences.
    batch_size: The size of the batch to sample from the buffer.

    Returns:
    None
    """
    
    # Sample a batch of experiences from the buffer
    if len(buffer) < batch_size or len(buffer) < buffer.sequence_length:
        return  # Not enough samples to update
    if not buffer.reward_check():
        return # No rewards in the buffer
    batch = buffer.sample()
    
    # Update the agent using the batch
    for sequence in batch:
        agent.train(target, *sequence)
    # Update the target network
    if agent.update_count % agent.target_update_freq == 0:
        target.load_state_dict(agent.state_dict())
        agent.update_count += 1
    
