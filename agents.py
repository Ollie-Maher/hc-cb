# Script to define the agent class and its components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np


class noise_layer(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class res_encoder(nn.Module):
    def __init__(self, obs_shape, device):
        super().__init__()
        self.device = device

        # Use pretrained ResNet18 encoder
        self.model = resnet18(pretrained=True)
        self.model.to(self.device)
    

        # Transform the input to the correct size
        self.transform = transforms.Compose([transforms.Resize(224)])

        # Freeze the parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.repr_dim = 1024

        # Get the output shape of the convolutional layers
        x = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)

        # Initialization
        nn.init.orthogonal_(self.fc.weight.data)
        self.fc.bias.data.fill_(0.0)

        self.to(self.device)

    @torch.no_grad()
    def forward_conv(self, obs: np.ndarray):
        # Convert the input to a tensor
        obs = torch.tensor(obs, device = self.device).float()
        if len(obs.shape) == 3:
            obs = obs.permute(2,0,1).unsqueeze(0)  # Add batch dimension & change to (C, H, W) format from (H, W, C)
        elif len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2) # Change to (B, C, H, W) format from (B, H, W, C)
        else:
            raise ValueError("Input shape must be (H, W, C) or (B, H, W, C)")
        # Normalize the input
        obs = obs / 255.0 - 0.5
        obs = self.transform(obs)

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == "layer2":
                break

        # Flatten the output to (batch_size, -1)
        conv = obs.reshape(obs.size(0), -1)

        return conv

    def forward(self, obs):

        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)

        return out


class HC_CB_agent(nn.Module):
    def __init__(self, config: dict, image_shape, device, is_target=False):
        super().__init__()
        self.config = config

        # Type of agent
        self.name = config["name"]

        # Parameters
        self.hc_gru_size = config["hc_gru_size"]
        self.hc_ca1_size = config["hc_ca1_size"]
        self.cb_sizes = config["cb_sizes"]
        self.output_size = config["output_size"]
        self.lr = config["lr"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.target_update_freq = config["target_update_freq"]
        self.update_count = 0

        # Noise params
        self.noise = config["noise"] # Noise location
        self.noise_params = {
            "mean": 0,
            "std": 1,
        }

        # Device
        self.device = device

        # Encoder
        features_extractor = res_encoder(image_shape, device)
        if self.noise >= "encoder":
            self.encoder = nn.Sequential(
                features_extractor,
                noise_layer(mean=self.noise_params["mean"], std=self.noise_params["std"])
            )
        else:
            self.encoder = features_extractor
        self.encoder.to(self.device)
        self.features_size = features_extractor.repr_dim # Output size

        # Hippocampus using GRU
        self.gru = nn.GRU(self.features_size + self.cb_sizes[1], self.hc_gru_size, device=self.device)
        self.ca1 = nn.Linear(self.hc_gru_size, self.hc_ca1_size, device=self.device)
        self.action = nn.Linear(self.hc_ca1_size, self.output_size, device=self.device)

        # Cerebellum using linear layers w/ noise
        if self.noise >= "cb_input":
            self.cb = nn.Sequential(                
                noise_layer(mean=self.noise_params["mean"], std=self.noise_params["std"]),
                nn.Linear(self.hc_gru_size, self.cb_sizes[0], device=self.device),
                nn.ReLU(),
                nn.Linear(self.cb_sizes[0], self.cb_sizes[1], device=self.device)
            )
        elif self.noise >= "cb_output":
            self.cb = nn.Sequential(                
                nn.Linear(self.hc_gru_size, self.cb_sizes[0], device=self.device),
                nn.ReLU(),
                nn.Linear(self.cb_sizes[0], self.cb_sizes[1], device=self.device),
                noise_layer(mean=self.noise_params["mean"], std=self.noise_params["std"])
            )
        else:
            self.cb = nn.Sequential(
                nn.Linear(self.hc_gru_size, self.cb_sizes[0], device=self.device),
                nn.ReLU(),
                nn.Linear(self.cb_sizes[0], self.cb_sizes[1], device=self.device)
            )


        # Set learning according to type of agent
        if self.name >= "no HC" or is_target:
            self.gru.requires_grad_ = False
            self.ca1.requires_grad_ = False
            self.action.requires_grad_ = False
        
        if self.name >= "no CB" or is_target:
            self.cb.requires_grad_ = False

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Loss function
        self.criterion = nn.MSELoss()

        self.reset() # Initialize the prediction and hidden state
    
    def reset(self):
        # Reset the agent
        self.prediction = torch.zeros(self.cb_sizes[1], device=self.device)
        self.gru_hidden = torch.zeros(1, self.hc_gru_size, device=self.device)

    def forward(self, x):
        # Run the input through the encoder
        features = self.encoder(x).squeeze(0)  # Remove batch dimension

        # Store the initial prediction and hidden state
        hidden_state = self.gru_hidden
        init_prediction = self.prediction
        
        # Concatenate the conv and prediction
        hc_input = torch.cat((features, init_prediction.squeeze()), dim=0).unsqueeze(0)
        # No batch dimension
        # hc_input must have shape (sequence length, features + predictions), sequence length = 1
        

        # Pass the input through the GRU
        out, self.gru_hidden = self.gru(hc_input, hidden_state)
        out = F.relu(self.ca1(F.relu(out)))
        out = self.action(out)
        out = F.softmax(out, dim=1)

        # Pass the hidden activity through the cerebellum
        self.prediction = self.cb(self.gru_hidden.detach())
        
        return out, hidden_state.detach(), self.gru_hidden.detach() # Return q values, gru (CA3) hidden state at t-1, and hidden state at t (used for CB input at t+1)
    
    def _reset_batch(self):
        # Reset the prediction and hidden state for batch processing
        self.batch_prediction = torch.zeros(self.cb_sizes[1], device=self.device)
        self.batch_gru_hidden = torch.zeros(1, self.hc_gru_size, device=self.device)
    
    def batch_forward(self, x): # THIS NEEDS TO BE CHANGED SO IT DOESN'T INTERFERE WITH THE HIDDEN STATE

        # Get the batch size
        batch_size = x.shape[0]

        # Stack state sequences
        # This is necessary to run the encoder on the entire batch
        x = np.stack(x, axis=0) # Shape (batch_size, H, W, C)

        self._reset_batch() # Reset the prediction and hidden state 

        # Run the input through the encoder
        features_sequence = self.encoder(x) # Shape (batch_size, features_size)

        out_sequence = torch.zeros(batch_size, self.output_size, device=self.device) # Shape (batch_size, output_size)
        prediction_sequence = torch.zeros(batch_size, self.cb_sizes[1], device=self.device) # Shape (batch_size, cb_sizes[1])

        # Store the initial prediction and hidden state
        for i in range(batch_size):
            # Get the features for the current step
            features = features_sequence[i].squeeze(0)

            # Concatenate the conv and prediction
            hc_input = torch.cat((features, self.batch_prediction.squeeze()), dim=0).unsqueeze(0)
            # No batch dimension
            # hc_input must have shape (sequence length, features + predictions), sequence length = 1
        
            # Pass the input through the GRU
            out, self.batch_gru_hidden = self.gru(hc_input, self.batch_gru_hidden)
            out = F.relu(self.ca1(F.relu(out)))
            out = self.action(out)
            out = F.softmax(out, dim=1)

            out_sequence[i] = out.squeeze(0) # Store the output for the current step

            # Pass the hidden activity through the cerebellum
            self.batch_prediction = self.cb(self.batch_gru_hidden.detach())

            prediction_sequence[i] = self.batch_prediction.squeeze() # Store the prediction for the current step
        
        return out_sequence, prediction_sequence # Return the output sequence and prediction sequence

    def train(self, target_net, state_sequence, action_sequence, reward_sequence, next_state_sequence, done_sequence):
        """
        Train the agent with sample sequences from the replay buffer.
        """
        # Get the action Q-values for the current state sequence
        q_values, predictions = self.batch_forward(state_sequence)
        
        action_qs = q_values.gather(1, action_sequence.unsqueeze(0)).squeeze(1) # Get the Q-values for the actions taken

        # Get the next Q-values for the next state sequence
        next_q_values, _ = target_net.batch_forward(next_state_sequence)
        # Get max Q-values for the next state sequence
        next_max_values, _ = next_q_values.max(1)

        # Remove size dimension
        reward_sequence.squeeze_()
        done_sequence.squeeze_()

        # Compute target Q-values
        target_q_values = reward_sequence + self.gamma * next_max_values * (1 - done_sequence)
        
        # Compute loss (using MSE)
        # target_q_values needs to be explicitly cast to float
        # because it is cast as double during calculation
        loss = self.criterion(action_qs, target_q_values.float().squeeze())
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    '''
    #test res_encoder
    obs_shape = (56, 56, 3) # Example observation shape (H, W, C)
    encoder = res_encoder(obs_shape, torch.device("cpu"))
    print("Encoder created successfully.")
    obs = np.random.rand(10, 56, 56, 3) # Example observation
    features = encoder(obs)
    print("Features shape:", features.shape) # Should be (1, 1024)
    '''

    
    # test HC_CB_agent

    # Parameters:
    # PATHS:
    EXPERIMENT_ID = "test"
    PATH = f"./HCCB/experiments/{EXPERIMENT_ID}"

    # Environment parameters
    ENV_NAME = "t-maze"
    ENV_MAX_STEPS = 100
    ENV_TASK_SWITCH = 1


    # Agent parameters
    AGENT_CLASS = "hippocampus"
    AGENT_NAME = "HC-CB" # Options: "HC-CB", "no HC-CB", "HC-no CB", "no HC-no CB"
    AGENT_NOISE = "" # "encoder", "cb_input", "cb_output"
    AGENT_HC_GRU_SIZE = 512
    AGENT_HC_CA1_SIZE = 512
    AGENT_CB_SIZES = [512, 256] # Need biological data to set these sizes
    AGENT_OUTPUT_SIZE = 3
    AGENT_LR = 0.001
    AGENT_GAMMA = 0.99
    AGENT_EPSILON = 0.1
    AGENT_UPDATE_FREQ = 10

    # Other parameters
    EPISODES = 1
    BUFFER_SIZE = 10000
    BATCH_SIZE = 1
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

    obs_shape = (56, 56, 3) # Example observation shape (H, W, C)
    
    test_agent = HC_CB_agent(object_cfg["agent"], obs_shape, torch.device("cpu"))
    print("Agent created successfully.")
    obs = np.random.rand(56, 56, 3) # Example observation
    out = test_agent(obs)
    print("Output shape:", out[0].shape) # Should be (1, 3)
    seq_obs = np.random.rand(10, 56, 56, 3) # Example observation sequence
    out_seq = test_agent.batch_forward(seq_obs)
    print("Output sequence shape:", out_seq[0].shape) # Should be (10, 3)
    


if __name__ == "__main__":
    main()