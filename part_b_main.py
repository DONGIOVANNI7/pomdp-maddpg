import numpy as np
import torch
import os
import sys

# Suppress the "future release" warning from PettingZoo
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pettingzoo.mpe import simple_tag_v3
from maddpg_system import MADDPG
from buffer import MultiAgentReplayBuffer

# Fix relative imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_state_vector(obs_dict, agent_names, actor_dims):
    """
    Robustly converts the observation dictionary to a fixed-order list.
    If an agent is missing (dead), we return a zero-vector so the 
    Neural Network doesn't crash due to shape mismatch.
    """
    state = []
    for i, agent in enumerate(agent_names):
        if agent in obs_dict:
            state.append(obs_dict[agent])
        else:
            # Return zero padding for dead agents
            state.append(np.zeros(actor_dims[i]))
    return state

def plot_learning_curve(scores):
    import matplotlib.pyplot as plt
    running_avg = [np.mean(scores[max(0, i-100):(i+1)]) for i in range(len(scores))]
    plt.figure(figsize=(10,6))
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Score')
    plt.plot(running_avg, color='red', label='Moving Avg (100)')
    plt.title("Part B: MADDPG Learning Curve (Simple Tag)")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("maddpg_learning_curve.png")
    print("Saved plot to maddpg_learning_curve.png")

def main():
    # Setup Environment
    env = simple_tag_v3.parallel_env(render_mode=None, max_cycles=25, continuous_actions=False)
    env.reset()
    
    # Extract fixed properties
    agent_names = env.agents
    n_agents = len(agent_names)
    
    # Get observation dimensions for each agent
    actor_dims = []
    for agent in agent_names:
        actor_dims.append(env.observation_space(agent).shape[0])
    
    critic_dims = sum(actor_dims)
    n_actions = env.action_space(agent_names[0]).n

    # Initialize System
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01)

    memory = MultiAgentReplayBuffer(1000000, n_agents, actor_dims, [n_actions]*n_agents)

    MAX_GAMES = 2000
    score_history = []
    
    print(f"Starting training on Simple Tag with {n_agents} agents...")

    for i in range(MAX_GAMES):
        obs_dict, _ = env.reset()
        done = False
        score = 0
        total_steps = 0
        
        # In PettingZoo parallel, we loop until the returned 'terminations' dict says everyone is done
        # OR we just check if the env.agents list is empty (which happens when all are done)
        while env.agents:
            # 1. Get Actions
            # Use helper to guarantee list order matches agent_names
            obs_list = get_state_vector(obs_dict, agent_names, actor_dims)
            
            actions_probs = maddpg_agents.choose_action(obs_list)
            
            # Create action dictionary for ONLY the currently active agents
            actions_dict = {}
            for idx, agent in enumerate(agent_names):
                if agent in env.agents: # Only step active agents
                    action_index = np.argmax(actions_probs[idx]) 
                    actions_dict[agent] = action_index

            # 2. Step Environment
            obs_dict_, rewards_dict, terminations, truncations, _ = env.step(actions_dict)

            # 3. Process Transitions safely
            obs_list_ = get_state_vector(obs_dict_, agent_names, actor_dims)
            
            # Use .get(agent, 0/False) to handle dead agents safely
            rewards = [rewards_dict.get(a, 0) for a in agent_names]
            dones_list = [terminations.get(a, False) or truncations.get(a, False) for a in agent_names]
            
            # Store in buffer
            memory.store_transition(obs_list, obs_list, actions_probs, rewards, 
                                    obs_list_, obs_list_, dones_list)

            # 4. Learn
            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            obs_dict = obs_dict_
            score += sum(rewards)
            total_steps += 1
            
            # Break if everyone is done
            if all(dones_list):
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if i % 10 == 0:
            print(f"Episode {i} | Avg Score: {avg_score:.1f} | Steps: {total_steps}")

    np.save("maddpg_scores.npy", np.array(score_history))
    plot_learning_curve(score_history)

if __name__ == '__main__':
    main()