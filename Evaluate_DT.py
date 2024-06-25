import torch
import numpy as np
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from custom_data_collator import CustomDataCollator  # Assuming custom_data_collator.py is in the same directory
from trainable_dt import TrainableDT  # Assuming trainable_dt.py is in the same directory
from decision_transformer import DecisionTransformerConfig  # Assuming this is where the config is defined
from decision_transformer_collator import DecisionTransformerGymDataCollator  # Assuming this is where the collator is defined
from colabgymrender.recorder import Recorder  # Assuming you have this module for recording videos

def get_action(model, states, actions, rewards, target_return, timesteps):
    # Define the logic to get action from the model
    # You need to implement this function based on your model's requirements
    pass

def evaluate_dts(model_name, dataset_name, output_dir, episodes, max_ep_len, target_return, scale,my_env):
    # Load the dataset and initialize collator
    dataset = load_dataset(dataset_name)
    collator = DecisionTransformerGymDataCollator(dataset['train'])

    # Load the trained model
    model = TrainableDT.from_pretrained(model_name)
    model = model.to("cpu")

    # Environment setup
    env = gym.make(my_env)

    # Evaluation settings
    device = "cpu"
    state_mean = collator.state_mean.astype(np.float32)
    state_std = collator.state_std.astype(np.float32)
    state_dim = 10
    act_dim = 1

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    episode_lengths = []
    final_rewards = []
    successes = 0

    for _ in range(episodes):
        episode_return, episode_length = 0, 0
        state, _ = env.reset()
        print(state)
        target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
        states = torch.from_numpy(np.array(state)).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        total_reward = 0
        steps = 0

        for t in range(max_ep_len):
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            action = get_action(
                model,
                (states - state_mean) / state_std,
                actions,
                rewards,
                target_return_tensor,
                timesteps,
            )
            steps += 1
            action = torch.argmax(action)
            actions[-1] = action
            action = action.detach().cpu().numpy()
            action = np.reshape(action, (1,))
            state, reward, done, truncated, _ = env.step(action)

            total_reward += reward
            cur_state = torch.from_numpy(np.array(state)).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            rewards[-1] = torch.tensor(reward / scale)

            pred_return = target_return_tensor[0, -1] - (reward / scale)
            target_return_tensor = torch.cat([target_return_tensor, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                final_rewards.append(total_reward)
                episode_lengths.append(steps)
                successes += 1
                break

    average_reward = np.mean(final_rewards)
    average_episode_length = np.mean(episode_lengths)
    success_rate = successes / episodes
    print(f"Average Reward per Episode: {average_reward}")
    print(f"Average Episode Length: {average_episode_length}")
    print(f"Success Rate: {success_rate * 100}%")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Decision Transformer Model")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--env", type=str, default "", help="env_name")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the evaluation video")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run for evaluation")
    parser.add_argument("--max_ep_len", type=int, default=20, help="Maximum episode length")
    parser.add_argument("--target_return", type=float, default=-15, help="Target return for evaluation")
    parser.add_argument("--scale", type=float, default 1.0, help="Normalization scale for rewards/returns")
 


    args = parser.parse_args()

    evaluate_dts(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        episodes=args.episodes,
        max_ep_len=args.max_ep_len,
        target_return=args.target_return,
        scale=args.scale
    )