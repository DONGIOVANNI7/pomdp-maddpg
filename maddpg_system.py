import torch
import torch.nn.functional as F
from maddpg_agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, 
                 gamma=0.99, tau=0.01):
        self.agents = []
        self.n_agents = n_agents
        self.gamma = gamma
        
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, 
                                     agent_idx, alpha=alpha, beta=beta, fc1=fc1, fc2=fc2, 
                                     gamma=gamma, tau=tau))

    def choose_action(self, raw_obs):
        return [agent.choose_action(raw_obs[i]) for i, agent in enumerate(self.agents)]

    def learn(self, memory):
        if not memory.ready(64): return

        actor_states, states, rewards, actor_new_states, dones = memory.sample_buffer(64)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to tensors
        actor_states = [torch.tensor(actor_states[i], dtype=torch.float).to(device) for i in range(self.n_agents)]
        actor_new_states = [torch.tensor(actor_new_states[i], dtype=torch.float).to(device) for i in range(self.n_agents)]
        actions = [torch.tensor(states[i], dtype=torch.float).to(device) for i in range(self.n_agents)]
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        all_states = torch.cat(actor_states, dim=1)
        all_new_states = torch.cat(actor_new_states, dim=1)
        all_actions = torch.cat(actions, dim=1)

        for agent_idx, agent in enumerate(self.agents):
            new_actions = []
            with torch.no_grad():
                for i, other in enumerate(self.agents):
                    new_actions.append(other.target_actor.forward(actor_new_states[i]))
                all_new_actions = torch.cat(new_actions, dim=1)
                
                critic_val = agent.target_critic.forward(all_new_states, all_new_actions).flatten()
                critic_val[dones[:, agent_idx]] = 0.0
                target = rewards[:, agent_idx] + agent.gamma * critic_val

            q_expected = agent.critic.forward(all_states, all_actions).flatten()
            critic_loss = F.mse_loss(q_expected, target)
            
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
            agent.critic.optimizer.step()

            mu_states = []
            for i, other in enumerate(self.agents):
                if i == agent_idx:
                    mu_states.append(other.actor.forward(actor_states[i]))
                else:
                    mu_states.append(other.actor.forward(actor_states[i]).detach())

            actor_loss = -agent.critic.forward(all_states, torch.cat(mu_states, dim=1)).mean()
            
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor.optimizer.step()

            agent.update_network_parameters()