import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import timedelta

from model import ActorCritic
from environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Agent_PPO:
    def __init__(self, save_path, load_path, max_episodes, max_timesteps, 
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_size):
        
        self.save_path = save_path
        self.load_path = load_path
        os.makedirs(self.save_path, exist_ok=True)

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(input_size=3, hidden_size=hidden_size, output_size=3).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(input_size=3, hidden_size=hidden_size, output_size=3).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.env = Environment()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_probs = self.policy_old.act(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(dist.log_prob(action))

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, path):
        torch.save(self.policy_old.state_dict(), self.save_path + path)
        
    def load(self):
        self.policy_old.load_state_dict(torch.load(self.load_path, map_location=device))
        self.policy.load_state_dict(torch.load(self.load_path, map_location=device))
        
    def train(self):
        print("Starting training with PPO...")
        start_time = time.time()
        
        update_timestep = 2000      # update policy every n timesteps
        time_step = 0
        reward_history = []
        steps_history = []
        best_score = -np.inf
        
        for i_episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            current_ep_reward = 0
            
            for t in range(self.max_timesteps):
                time_step += 1
                
                # Select action with policy_old
                action = self.select_action(state)
                state, reward, done = self.env.step_env(action, self.max_timesteps)
                
                # Saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)
                
                current_ep_reward += reward
                
                # update PPO agent
                if time_step % update_timestep == 0:
                    self.update()
                
                if done:
                    break
            
            reward_history.append(current_ep_reward)
            steps_history.append(t + 1)
            avg_reward = np.mean(reward_history[-100:])
            
            if current_ep_reward > best_score:
                best_score = current_ep_reward
                self.save('/best_weights.pt')
            
            print(f'Episode: {i_episode} \t Reward: {current_ep_reward:.2f} \t Avg Reward: {avg_reward:.2f}')
            
        self.save('/final_weights.pt')
        self.plot_rewards(reward_history)
        self.plot_steps(steps_history)
        
        elapsed_time = time.time() - start_time
        print(f'Total Training Time: {str(timedelta(seconds=elapsed_time)).split(".")[0]}')

    def test(self):
        print("Testing PPO agent...")
        self.load()
        self.policy_old.eval()
        
        for i_episode in range(1, 11):
            state = self.env.reset()
            ep_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                state, reward, done = self.env.step_env(action, self.max_timesteps)
                ep_reward += reward
                
            print(f'Episode: {i_episode} \t Reward: {ep_reward:.2f}')

    def plot_rewards(self, rewards):
        sma = np.convolve(rewards, np.ones(25)/25, mode='valid')
        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.savefig(self.save_path + '/reward_plot.png', format='png', dpi=1000, bbox_inches='tight')
        plt.close()

    def plot_steps(self, steps):
        sma = np.convolve(steps, np.ones(25)/25, mode='valid')
        plt.figure()
        plt.title("Episode Steps")
        plt.plot(steps, label='Raw Steps', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.savefig(self.save_path + '/steps_plot.png', format='png', dpi=1000, bbox_inches='tight')
        plt.close()
