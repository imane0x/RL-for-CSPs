import torch
import numpy as np
import argparse
from datasets import load_dataset
from custom_data_collator import CustomDataCollator  
from trainable_dt import TrainableDT 
from decision_transformer import DecisionTransformerConfig  
from decision_transformer_collator import DecisionTransformerGymDataCollator  

def get_action(model, states, actions, rewards, returns_to_go, timesteps):

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # The prediction is conditioned on up to 20 previous time-steps
    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]

    # pad all tokens to sequence length, this is required if we process batches
    padding = model.config.max_length - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, 1)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    # perform the prediction
    state_preds, action_preds, return_preds = model.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,)
    return action_preds[0, -1]

def evaluate_dts(model_name, dataset_name, episodes, max_ep_len, target_return, scale,my_env):
    # Load the dataset and initialize collator
    dataset = load_dataset(dataset_name)
    collator = DecisionTransformerGymDataCollator(dataset['train'])
    config = DecisionTransformerConfig(state_dim=10, act_dim=1)
    model = TrainableDT(config)
    model.load_state_dict(torch.load('model_10_DT.pth'))
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
    parser.add_argument("--env", type=str, default "nqueens", help="env_name")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face")
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
