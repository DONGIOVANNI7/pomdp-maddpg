import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, n_agents, obs_dims, act_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents

        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, obs_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, obs_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, act_dims[i])))

        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

    def store_transition(self, raw_obs, state, actions, rewards, raw_obs_, state_, dones):
        index = self.mem_cntr % self.mem_size
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = actions[agent_idx]

        self.reward_memory[index] = rewards
        self.terminal_memory[index] = dones
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states, states_, actions = [], [], []
        for i in range(self.n_agents):
            states.append(self.actor_state_memory[i][batch])
            states_.append(self.actor_new_state_memory[i][batch])
            actions.append(self.actor_action_memory[i][batch])

        return states, actions, self.reward_memory[batch], states_, self.terminal_memory[batch]

    def ready(self, batch_size):
        return self.mem_cntr >= batch_size