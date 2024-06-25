import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NQueensEnv(gym.Env):
    def __init__(self, n=8):
        super(NQueensEnv, self).__init__()
        self.n = n  # Size of the board
        self.action_space = spaces.Discrete(n * n)  # Action space
        self.observation_space = spaces.MultiDiscrete([n] * n)  # Observation space
        self.current_step = 0
        self.max_steps = 500
        self.reset()

    def reset(self, **kwargs):
        self.state = np.random.choice(self.n, size=self.n, replace=False)  # Random initial state
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step= self.current_step +1
        row = action % self.n
        col = action // self.n
        self.state[row] = col
        reward = self.calculate_reward()
        if reward == 0:
          done = True
          truncated = False
        elif  self.current_step > self.max_steps:
          truncated = True
          done = False
        else:
          done = False
          truncated = False
        return self.state, reward, done, truncated, {}

    def calculate_reward(self):
        # Initialize violations count
        violations = 0
        # Create sets to keep track of conflicts
        row_set = set()
        diag1_set = set()
        diag2_set = set()

        # Iterate through each queen
        for i in range(self.n):
            # Calculate positions on diagonals
            diag1 = self.state[i] + i
            diag2 = self.state[i] - i

            # Check for conflicts
            if self.state[i] in row_set or diag1 in diag1_set or diag2 in diag2_set:
                violations += 1

            # Update sets
            row_set.add(self.state[i])
            diag1_set.add(diag1)
            diag2_set.add(diag2)

        # Reward is negative number of violations
        return -violations

    def is_terminal_state(self):
        return self.calculate_reward() == 0  # Terminal state reached when there are no violations

    def render(self, mode='human'):
        board = [['_' for _ in range(self.n)] for _ in range(self.n)]
        for i, row in enumerate(board):
            row[self.state[i]] = 'Q'
            print(' '.join(row))

# # Example of creating and using the environment
# if __name__ == "__main__":
#     env = NQueensEnv()
#     state, _ = env.reset()
#     print("Initial state:")
#     env.render()
#     # Example of taking a step
#     action = env.action_space.sample()
#     #print(action)
#     state, reward, done, truncated, info = env.step(action)
#    # print("\nState after one step:")
#     print(state)
#     env.render()
#     print(f"Reward: {reward}, Done: {done}")
