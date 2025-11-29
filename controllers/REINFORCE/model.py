import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """Neural network model representing the Actor-Critic architecture for PPO."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCritic, self).__init__()
        
        # Shared layers (optional, but good for feature extraction)
        # Here we keep them separate for simplicity and stability as per common PPO implementations for simple envs
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        """
        Query the actor network.
        Returns action probabilities.
        """
        return self.actor(state)
    
    def evaluate(self, state, action):
        """
        Evaluate the state and action.
        Returns action log probabilities, state values, and entropy.
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
