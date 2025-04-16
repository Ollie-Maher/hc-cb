# Script to define the agent class and its components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

class res_encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        # Use pretrained ResNet18 encoder
        self.model = resnet18(pretrained=True)

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

    @torch.no_grad()
    def forward_conv(self, obs: np.ndarray):
        # Convert the input to a tensor
        obs = torch.from_numpy(obs).float()
        obs = obs.permute(2, 0, 1).unsqueeze(0)  # Change to (C, H, W) format
        # Normalize the input
        obs = obs / 255.0 - 0.5
        obs = self.transform(obs)

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == "layer2":
                break

        # Flatten the output to (batch_size, -1)
        conv = obs.view(obs.size(0), -1)

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
        self.target_update_freq = config["target_update_freq"]
        self.update_count = 0

        # Device
        self.device = device

        # Encoder 
        self.encoder = res_encoder(image_shape)
        self.encoder.to(self.device)
        self.features_size = self.encoder.repr_dim # Output size

        # Hippocampus using GRU
        self.gru = nn.GRU(self.features_size + self.cb_sizes[1], self.hc_gru_size, device=self.device)
        self.ca1 = nn.Linear(self.hc_gru_size, self.hc_ca1_size, device=self.device)
        self.action = nn.Linear(self.hc_ca1_size, self.output_size, device=self.device)

        # Cerebellum using linear layer
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
        out = self.ca1(out)
        out = self.action(out)
        out = F.softmax(out, dim=1)

        # Pass the hidden activity through the cerebellum
        self.prediction = self.cb(self.gru_hidden.detach())
        
        return out, hidden_state.detach(), self.gru_hidden.detach() # Return q values, gru (CA3) hidden state at t-1, and hidden state at t (used for CB input at t+1)
    
    def train(self, target_net, state, action, reward, next_state, done, hidden_state, cb_input):
        """
        Train the agent with samples from the replay buffer.
        """
        # Set hidden states
        self.gru_hidden = hidden_state
        # CB prediction from previous step
        self.prediction = self.cb(cb_input)

        # Get action Q-values
        q_values, *_ = self.forward(state)

        action_q = q_values.squeeze()[action]
        
        # Get next state Q-values
        with torch.no_grad():
            target_net.gru_hidden = self.gru_hidden # Set hidden state for target network
            target_net.prediction = self.prediction # Set prediction for target network
            next_q_values, *_ = target_net(next_state)
        
        # Compute target Q-values
        target_q_value = reward + self.gamma * torch.max(next_q_values) * (1 - done)
        
        # Compute loss (using MSE)
        loss = self.criterion(action_q, target_q_value)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    '''
    #test res_encoder
    obs_shape = (56, 56, 3) # Example observation shape (H, W, C)
    encoder = res_encoder(obs_shape)
    print("Encoder created successfully.")
    obs = np.random.rand(56, 56, 3) # Example observation
    features = encoder(obs)
    print("Features shape:", features.shape) # Should be (1, 1024)
    '''
    # test HC_CB_agent

    # Agent parameters FROM HIPPOCAMPUS_TEST.PY
    AGENT_CLASS = "hippocampus"
    AGENT_NAME = "HC-CB" # Options: "HC-CB", "no HC-CB", "HC-no CB", "no HC-no CB"
    AGENT_HC_GRU_SIZE = 512
    AGENT_HC_CA1_SIZE = 512
    AGENT_CB_SIZES = [512, 256] # Need biological data to set these sizes
    AGENT_OUTPUT_SIZE = 3
    AGENT_LR = 0.001
    AGENT_BATCH_SIZE = 32

    obs_shape = (56, 56, 3) # Example observation shape (H, W, C)
    
    test_agent = HC_CB_agent({"class": AGENT_CLASS,
                            "name": AGENT_NAME,
                            "hc_gru_size": AGENT_HC_GRU_SIZE,
                            "hc_ca1_size": AGENT_HC_CA1_SIZE,
                            "cb_sizes": AGENT_CB_SIZES,
                            "output_size": AGENT_OUTPUT_SIZE,
                            "lr": AGENT_LR,
                            "gamma": 0.99},
                            obs_shape, torch.device("cpu"))
    print("Agent created successfully.")
    obs = np.random.rand(56, 56, 3) # Example observation
    out = test_agent(obs)
    print("Output shape:", out[0].shape) # Should be (1, 3)


if __name__ == "__main__":
    main()