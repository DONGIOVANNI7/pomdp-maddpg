import torch
import numpy as np # Added import
from maddpg_networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, 
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.agent_name = f'agent_{agent_idx}'

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions)
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float)
        
        with torch.no_grad():
            actions = self.actor.forward(state)
            
        return actions.detach().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        for target, local in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(tau*local.data + (1.0-tau)*target.data)
        for target, local in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(tau*local.data + (1.0-tau)*target.data)