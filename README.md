# RL-for-CSPs

## Project Overview
This project aims to solve the N-Queens problem using two approaches in reinforcement learning and sequence modeling: Proximal Policy Optimization (PPO) and Decision Transformers. The N-Queens problem involves placing N chess queens on an N×N chessboard so that no two queens threaten each other.

## Installation
Dependencies can be installed using the following command:

```
conda env create -f conda_env.yml
```

## Training
### PPO

```
python PPO/train_ppo.py  --board_size <boardsize> --timesteps <total_timesteps> --lr <learning_rate> 
```
### DT

```
python Decision_Transformers/train_dts.py --state_dim <state_dim> --dataset_name <dataset_name> --output_dir <output_dir> --epochs <num_train_epochs> --batch_size <per_device_train_batch_size> --learning_rate <learning_rate> --weight_decay <weight_decay>  --optim <optimizer> 
```

## Evaluation
### PPO

```
python PPO/evaluate_ppo.py --model_path <model_path> --board_size <board_size> --episodes <number_of_episodes>

```
### DT

```
python Decision_Transformers/evaluate_dts.py --model_path <model_path> --states_dim <number_of_queens> --dataset_name <dataset_name> --episodes <number_of_episodes> --max_ep_len <maximum_episode_length> --target_return <target_return_value> --scale <reward_scale>

```
