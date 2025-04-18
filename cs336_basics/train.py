import argparse
import torch
import os
import wandb
import time
from cs336_basics.torch_modules.custom_modules import TransformerLM, cross_entropy_loss 
from cs336_basics.torch_modules.adamw_opt import AdamW, learning_rate_scheduling, gradient_clipping
from cs336_basics.torch_modules.dataloader import DataLoader
from cs336_basics.torch_modules.check_pointing import save_checkpoint, load_checkpoint  

def main(args):
    # WandB setup
    wandb.login()
    run = wandb.init(
                    project="cs336_a1",  # Specify your project
                    name=args.checkpoint_tag,  # Specify your run name
                    config=args, 
                )

    # If debugging
    if (args.debug):
        args['checkpoint_tag'] = 'debug'

    # First things first
    data_dir = '/data/c-worledge/t_results/' 
    save_dir = f'/data/c-worledge/a1_checkpoints/{args.checkpoint_tag}/'
      
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set device and dtype
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    dtype = getattr(torch, args.dtype) 

    # Initialize the dataloader
    if args.train_data_str == 'ts':
        train_data_fp = data_dir + "ts_train_tokens.npy" 
        val_data_fp = data_dir + "ts_val_tokens.npy"
    elif args.train_data_str == 'owt':
        train_data_fp = data_dir + "" # TODO merge the two files and its name here
        val_data_fp = data_dir + "owt_val_tokens.npy"
    else:
        raise ValueError("Invalid train data string. Must be 'ts' or 'owt'.")
    
    train_dataset = DataLoader(
        batch_size=args.batch_size,
        context_length=args.context_length,
        data_file_path=train_data_fp,
        dtype=dtype,
    )

    val_dataset = DataLoader(
        batch_size=args.batch_size,
        context_length=args.context_length,
        data_file_path=val_data_fp,
        dtype=dtype,
    )

    # Initialize the model
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        theta=args.theta,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        eps=args.model_eps,
        device=device,
        dtype=dtype,
    )
    model.to(device)

    # Initialize the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps
    )

    # Train
    step_count = args.total_tokens_processed // (args.batch_size * args.context_length)
    start_time = time.time()

    # step_count = 10 
    for t in range(step_count):
        
        # get the data batch
        train_batch, target_batch = train_dataset.get_random_batch()
        train_batch = train_batch.to(device)
        target_batch = target_batch.to(device)

        # zero gradients for new batch
        optimizer.zero_grad()

        # compute a forward pass
        logits = model(train_batch)

        # compute the loss
        train_loss = cross_entropy_loss(logits, target_batch)
        train_loss.backward()
        print(train_loss.item())

        # Gradient clipping
        gradient_clipping(model.parameters(), args.max_norm)
        
        # Take an optimizer step with loss scheduling
        lr_t = learning_rate_scheduling(t, args.lr, args.a_min, int(step_count * args.T_w_fraction), int(step_count * args.T_c_fraction)) 
        optimizer.step(scheduled_lr=lr_t)

        # get val loss
        if (t % 100 == 0):
            avg_val_loss = run_validation(model, val_dataset, device, 10)
            if (not args.debug):
                wandb.log({"val_loss": avg_val_loss
                })

        # Implement loss tracking (WandB)
        if (not args.debug):
            wandb.log({"train_loss": train_loss.item(),
                       "wallclock_time": time.time() - start_time
            })

        # Implement checkpointing
        if (t % 100 == 0):
            save_checkpoint(model, optimizer, t, f'{save_dir}/checkpoint.pt')
    
    # Save the final model 
    save_checkpoint(model, optimizer, t, f'{save_dir}/final.pt')

    # Get the final val loss
    avg_val_loss = run_validation(model, val_dataset, device, 1000) 

    print(f"Final validation loss for a_max={args.lr}: {avg_val_loss}")

    if (not args.debug):
        wandb.log({
            "final_val_loss": avg_val_loss,
            })

    # finish wandb run
    wandb.finish()
       

def run_validation(model:torch.nn.Module, 
                    val_dataset:DataLoader, 
                    device:str,
                    num_samples:int,
                    ):
    val_loss_sum = 0
    for t in range(num_samples): 
        x_batch, y_batch = val_dataset.get_random_batch()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        val_loss = cross_entropy_loss(logits, y_batch)
        val_loss_sum += val_loss.item()

    avg_val_loss = val_loss_sum / num_samples
    return avg_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences in a batch')
    parser.add_argument('--context_length', type=int, default=256, help='Number of tokens in a sequence')
    parser.add_argument('--train_data_str', type=str, default='ts', help='Train data string') # 'ts', or 'owt'
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--d_ff', type=int, default=1344, help='Dimension of the feedforward layer')
    parser.add_argument('--theta', type=float, default=10000, help='Theta value for the model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--model_eps', type=float, default=1e-5, help='Epsilon value for the model')
    parser.add_argument('--max_norm', type=float, default=1e-2, help='Max norm for gradient clipping') 
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for the model') 
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 value for the optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 value for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer') 
    parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Epsilon value for the optimizer')
    parser.add_argument('--total_tokens_processed', type=int, default=327680000, help='batch_size*step_count*context_length')
    parser.add_argument('--checkpoint_tag', type=str, default='debug', help='name of the directory for the checkpoint files')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--a_min', type=float, default=0, help='Minimum value for parameter a')
    parser.add_argument('--T_w_fraction', type=float, default=0.1, help='Fraction of step_count for T_w')
    parser.add_argument('--T_c_fraction', type=float, default=0.8, help='Fraction of step_count for T_c')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')

    args = parser.parse_args()
    main(args)
