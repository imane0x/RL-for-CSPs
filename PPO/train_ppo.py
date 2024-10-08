import argparse
from stable_baselines3 import PPO
import gym
from custom_callback import CustomCallback
from N_queens_env import NQueensEnv
def train_ppo(board_size, total_timesteps, learning_rate, use_custom_callback):
    # Create the environment
    #env = gym.make(env_name)
    env = NQueensEnv(n = board_size)
    
    # Initialize the PPO model
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, verbose=1)
    
    # Initialize the callback
    callback = CustomCallback() if use_custom_callback else None
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the model
    model.save("ppo_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Model")
    parser.add_argument("--board_size", type=int, required=True, help="Size of the board")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--use_callback", default = False, help="Use custom callback for storing trajectories")

    args = parser.parse_args()

    train_ppo(args.board_size, args.timesteps, args.lr, args.use_callback)
