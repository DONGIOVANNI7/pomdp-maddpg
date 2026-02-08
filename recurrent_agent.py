import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class RecurrentAgent:
    def __init__(self, model, lr=1e-3, gamma=0.99, device='cpu'):
        self.device = device
        self.model = model.to(self.device) # Move model to GPU/CPU
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.memory = []

    def get_action(self, state, hidden):
        with torch.no_grad():
            # Convert state to tensor and move to DEVICE
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # If hidden state exists, move it to device too
            if hidden is not None:
                if isinstance(hidden, tuple): # LSTM (h, c)
                    hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
                else: # GRU (h)
                    hidden = hidden.to(self.device)

            logits, _, new_hidden = self.model(state_t, hidden)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
        return action.item(), new_hidden

    def store_transition(self, state, action, reward):
        self.memory.append((state, action, reward))

    def update(self):
        if len(self.memory) == 0: return 0

        states, actions, rewards = zip(*self.memory)
        
        # Move Batch to DEVICE
        state_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor(actions).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Forward pass
        logits, values, _ = self.model(state_tensor, hidden=None)

        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Move Returns to DEVICE
        returns = torch.FloatTensor(returns).unsqueeze(0).unsqueeze(-1).to(self.device)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(action_tensor.squeeze(-1)).unsqueeze(-1)

        advantage = returns - values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.memory = []
        return total_loss.item()