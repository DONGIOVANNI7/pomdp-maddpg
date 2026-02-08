import gymnasium as gym
import numpy as np

class LunarLanderPOMDPWrapper(gym.ObservationWrapper):
    """
    Wraps LunarLander-v2 to make it Partially Observable (POMDP).
    We mask (remove) the velocity vectors:
    - x_vel (index 2), y_vel (index 3), ang_vel (index 5)
    New Shape: 5
    """
    def __init__(self, env):
        super().__init__(env)
        mask_indices = [2, 3, 5]
        low = np.delete(self.observation_space.low, mask_indices)
        high = np.delete(self.observation_space.high, mask_indices)
        
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return np.delete(observation, [2, 3, 5])