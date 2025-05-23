import torch

path = "./experiments/base_test_2/"

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

if __name__ == "__main__":
    are_same = compare_agents(agent1, agent2)
    if are_same:
        print("The agents are the same.")
    else:
        print("The agents are different.")