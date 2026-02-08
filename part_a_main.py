import gymnasium as gym
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from wrapper import LunarLanderPOMDPWrapper
from rnn_networks import RecurrentACModel, FeatureExtractorACModel
from recurrent_agent import RecurrentAgent

def run_experiment(config):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running config {config['id']} on device: {device}")

    env = gym.make("LunarLander-v3")
    env = LunarLanderPOMDPWrapper(env)
    
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ModelClass = RecurrentACModel if config['arch'] == 'simple' else FeatureExtractorACModel
        
    model = ModelClass(input_dim, action_dim, 
                       hidden_dim=config['hidden_size'],
                       rnn_type=config['rnn_type'],
                       activation=config['activation'])
    
    # Pass device to Agent
    agent = RecurrentAgent(model, lr=config['lr'], device=device)
    scores = []
    
    for ep in range(config['episodes']):
        state, _ = env.reset()
        done = False
        score = 0
        hidden = None 
        
        while not done:
            action, hidden = agent.get_action(state, hidden)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store_transition(state, action, reward)
            state = next_state
            score += reward
            
        agent.update()
        scores.append(score)
        
        if (ep+1) % 50 == 0:
            print(f"Config {config['id']} | Ep {ep+1} | Avg Score: {np.mean(scores[-50:]):.2f}")
            
    return scores

def plot_results(results):
    plt.figure(figsize=(12, 7))
    for exp_name, scores in results.items():
        smoothed = [np.mean(scores[max(0, i-50):(i+1)]) for i in range(len(scores))]
        plt.plot(smoothed, label=exp_name)
    
    plt.title("Part A: LSTM vs GRU vs Architectures (Smoothed)")
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("part_a_comparison.png")
    print("Saved plot to part_a_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Define Experiments
    experiments = [
        {'id': 'lstm_baseline', 'arch': 'simple', 'rnn_type': 'LSTM', 'lr': 0.002, 'hidden_size': 64, 'activation': 'relu', 'episodes': 500},
        {'id': 'gru_baseline',  'arch': 'simple', 'rnn_type': 'GRU',  'lr': 0.002, 'hidden_size': 64, 'activation': 'relu', 'episodes': 500},
        {'id': 'lstm_arch2',    'arch': 'feature', 'rnn_type': 'LSTM', 'lr': 0.002, 'hidden_size': 64, 'activation': 'relu', 'episodes': 500},
        {'id': 'lstm_high_lr',  'arch': 'simple', 'rnn_type': 'LSTM', 'lr': 0.01,  'hidden_size': 64, 'activation': 'relu', 'episodes': 500},
        {'id': 'lstm_big_net',  'arch': 'simple', 'rnn_type': 'LSTM', 'lr': 0.002, 'hidden_size': 128, 'activation': 'relu', 'episodes': 500},
    ]
    
    results = {}
    for exp in experiments:
        print(f"--- Running Experiment: {exp['id']} ---")
        results[exp['id']] = run_experiment(exp)
        
    with open("part_a_results.json", "w") as f:
        json.dump(results, f)

    plot_results(results)