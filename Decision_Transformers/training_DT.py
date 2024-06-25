import argparse
import torch
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
from custom_data_collator import CustomDataCollator  
from DT_model import TrainableDT  
from decision_transformer import DecisionTransformerConfig  
from decision_transformer_collator import DecisionTransformerGymDataCollator  

def train_dts(model_name, dataset_name, output_dir, num_train_epochs, per_device_train_batch_size, evaluation_strategy, learning_rate, weight_decay, warmup_ratio, optim, max_grad_norm):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the Decision Transformer Gym Data Collator
    collator = DecisionTransformerGymDataCollator(dataset['train'])
    
    # Initialize the Decision Transformer configuration and custom model
    config = DecisionTransformerConfig(state_dim=10, act_dim=1)  # Adjust these dimensions as necessary
    model = TrainableDT(config)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        optim=optim,
        max_grad_norm=max_grad_norm,
        evaluation_strategy=evaluation_strategy,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    torch.save(model.state_dict(), 'model_DT.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Transformer Model with Custom Model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name from Hugging Face to load the configuration")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=0.25, help="Max gradient norm")

    args = parser.parse_args()

    train_dts(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy=args.eval_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm
    )