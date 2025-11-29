# Add the controller Webots Python library path
import sys
import os
import numpy as np
import torch

webots_path = r'C:\Program Files\Webots\lib\controller\python'
sys.path.append(webots_path)

from agent import Agent_PPO

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # Parameters
    save_path = './results'   
    load_path = './results/final_weights.pt'
    train_mode = True
    
    max_episodes = 1000 if train_mode else 10
    max_timesteps = 500 # max timesteps in one episode
    
    # PPO Hyperparameters
    update_timestep = 2000      # update policy every n timesteps
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    gamma = 0.99            # discount factor
    K_epochs = 40           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    hidden_size = 64        # hidden layer size (increased for PPO)

    # Agent Instance
    agent = Agent_PPO(save_path, load_path, max_episodes, max_timesteps, 
                      lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_size)
    
    if train_mode:
        # Initialize Training
        agent.train()
    else:
        # Test
        agent.test()