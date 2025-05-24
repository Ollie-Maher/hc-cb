import torch
from agents import HC_CB_agent
from objects import get_env

path = "./experiments/base_test_2/"

# Load the agent weights from the specified path
agent1 = torch.load(f"{path}/agent_weights_0.pth", map_location=torch.device('cpu'))
agent2 = torch.load(f"{path}/agent_weights_final.pth", map_location=torch.device('cpu'))

def compare_agents(agent1, agent2):
    """
    Compare two agents by checking if their weights are the same.
    
    Parameters:
    agent1: The first agent to compare.
    agent2: The second agent to compare.
    
    Returns:
    True if the agents are the same, False otherwise.
    """
    for key in agent1.keys():
        if not torch.equal(agent1[key], agent2[key]):
            print(f"Difference found in {key}")
            return False
    return True

def test_encoder(agent, obs_shape, env):
    """
    Test the encoder of the agent by checking if it produces the same output for the same input.
    
    Parameters:
    agent: The agent to test.
    
    Returns:
    True if the encoder is working correctly, False otherwise.
    """
    state = env.reset()[0]["image"]
    input_tensor1 = torch.tensor(state, dtype=torch.float32, device=agent.device)


    state = env.reset()[0]["image"]
    input_tensor2 = torch.tensor(state, dtype=torch.float32, device=agent.device)



    output1 = agent.encoder(agent.forward_conv(input_tensor1))
    output2 = agent.encoder(agent.forward_conv(input_tensor2))

    print("Output 1:", output1)
    print("Output 2:", output2)
    
    if torch.equal(output1, output2):
        print("Encoder is working correctly.")
        return True
    else:
        print("Encoder is not working correctly.")
        return False
    
    

if __name__ == "__main__":
    are_same = compare_agents(agent1, agent2)
    if are_same:
        print("The agents are the same.")
    else:
        print("The agents are different.")
        # test HC_CB_agent

    # Parameters:
    # Agent parameters
    AGENT_CLASS = "hippocampus-cerebellum"
    AGENT_NAME = "HC-CB" # Options: "HC-CB", "no HC-CB", "HC-no CB", "no HC-no CB"
    AGENT_NOISE = "--" # "encoder", "cb_input", "cb_output"
    AGENT_HC_GRU_SIZE = 512
    AGENT_HC_CA1_SIZE = 512
    AGENT_CB_SIZES = [1024, 256] # Need biological data to set these sizes
    AGENT_OUTPUT_SIZE = 3
    AGENT_LR = 0.0001
    AGENT_GAMMA = 0.99
    AGENT_EPSILON = 0.1
    TAU = 0.01
    AGENT_SEED = 323 

    # Env Parameters
    env_cfg = {"name": "t-maze", "max_steps": 100, "task_switch": 1}


    # Config dictionaries
    agent_cfg= {
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
    }

    env = get_env(env_cfg)
    obs_shape = env.observation_space.spaces["image"].shape
    print("OBS SHAPE:", obs_shape)

    real_agent = HC_CB_agent(agent_cfg, obs_shape, device=torch.device('cpu'))
    real_agent.load_state_dict(agent2)

    encoder_test = test_encoder(real_agent, obs_shape, env)
    if encoder_test:
        print("Encoder test passed.")
    else:
        print("Encoder test failed.")