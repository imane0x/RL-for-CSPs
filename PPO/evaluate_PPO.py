import numpy as np
import argparse
import time
from stable_baselines3 import PPO  
import gym

def evaluate_ppo(model_path, env_name, episodes):
    # Load the environment
    env = gym.make(env_name)

    # Load the trained PPO model
    model = PPO.load(model_path)

    rewards_list = []
    episode_lengths = []
    successes = 0

    for ep in range(episodes):
        obs, info = env.reset()
        env.render()
        print(f"Episode {ep}")
        print(f"Initial observation: {obs}")

        total_reward = 0
        steps = 0
        done = False
        truncated = False

        while not done and not truncated:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, info = env.step(action)
            print(f"Action: {action}, New state: {obs}, Reward: {rewards}, Done: {done}, Truncated: {truncated}")

            total_reward += rewards
            steps += 1

            if done or truncated:
                break

        rewards_list.append(total_reward)
        episode_lengths.append(steps)
        if done:
            successes += 1

    average_reward = np.mean(rewards_list)
    average_episode_length = np.mean(episode_lengths)
    success_rate = successes / episodes

    print(f"Average Reward per Episode: {average_reward}")
    print(f"Average Episode Length: {average_episode_length}")
    print(f"Success Rate: {success_rate * 100}%")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run for evaluation")

    args = parser.parse_args()

    evaluate_ppo(args.model_path, args.env_name, args.episodes)
