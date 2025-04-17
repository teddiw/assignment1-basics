import argparse
import torch
from cs336_basics.torch_modules.custom_modules import TransformerLM, cross_entropy_loss 
from cs336_basics.torch_modules.adamw_opt import AdamW, learning_rate_scheduling, gradient_clipping
from cs336_basics.torch_modules.dataloader import DataLoader
from cs336_basics.torch_modules.check_pointing import save_checkpoint, load_checkpoint  


def main(args):
    # First things first
    data_dir = '/data/c-worledge/t_results/' 
    save_dir = '/data/c-worledge/a1_checkpoints/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype) 

    # Initialize the dataloader
    if args.train_data_str == 'tst':
        train_data_fp = data_dir + "ts_train_tokens.npy" 
    elif args.train_data_str == 'tsv':
        train_data_fp = data_dir + "ts_val_tokens.npy"
    elif args.train_data_str == 'owtt':
        train_data_fp = data_dir + "" # TODO merge the two files and its name here
    elif args.train_data_str == 'owtv':
        train_data_fp = data_dir + "owt_val_tokens.npy"
    else:
        raise ValueError("Invalid train data string. Must be 'tst', 'tsv', 'owtt', or 'owtv'.")
    
    train_dataset = DataLoader(
        batch_size=args.batch_size,
        context_length=args.context_length,
        data_file_path=train_data_fp,
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
    # TODO where to set max_norm for gradient clipping (and where does it happen in AdamW?)
    # TODO how can I use learning_rate_scheduling with AdamW?
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps
    )

    # Train
    step_count = args.total_tokens_processed // (args.batch_size * args.context_length)
    for t in range(10): # TODO revert
        # get the data batch
        train_batch, target_batch = train_dataset.get_random_batch()
        train_batch = train_batch.to(device)
        target_batch = target_batch.to(device)

        # zero gradients for new batch
        optimizer.zero_grad()

        # compute a forward pass
        logits = model(train_batch)

        # compute the loss
        # (logits: Float[Tensor, '... batch vocab_size'],
        #                targets: Int[Tensor, '... batch']
        loss = cross_entropy_loss(logits, target_batch)
        loss.backward()
        print(loss.item())

        optimizer.step()

        # take an optimizer step



    # cross_entropy_loss(logits: Float[Tensor, '... batch vocab_size'],
    #                    targets: Int[Tensor, '... batch']
    #                    ) -> Float[Tensor, '...']

    # Implement checkpointing
    # Implement loss tracking (WandB)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=str, default=4, help='Number of sequences in a batch')
    parser.add_argument('--context_length', type=int, default=256, help='Number of tokens in a sequence')
    parser.add_argument('--train_data_str', type=str, default='tsv', help='Train data string') # 'tst', 'tsv','owtt', or 'owtv'
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--d_ff', type=int, default=1344, help='Dimension of the feedforward layer')
    parser.add_argument('--theta', type=float, default=10000, help='Theta value for the model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--model_eps', type=float, default=1e-5, help='Epsilon value for the model')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping') # TODO get default value
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for the model') 
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer') # TODO get default value
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 value for the optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 value for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer') # TODO get default value
    parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Epsilon value for the optimizer')
    parser.add_argument('--wandb_project', type=str, default='cs336_a1', help='WandB project name')
    parser.add_argument('--total_tokens_processed', type=int, default=327680000, help='batch_size*step_count*context_length')
    
    # TODO May want to set these based off of args.lr and step_count
    parser.add_argument('--a_max', type=float, default=1.0, help='Maximum value for parameter a')
    parser.add_argument('--a_min', type=float, default=0.0, help='Minimum value for parameter a')
    parser.add_argument('--T_w', type=int, default=100, help='Warmup steps for learning rate scheduling')
    parser.add_argument('--T_c', type=int, default=1000, help='Cooldown steps for learning rate scheduling')

    args = parser.parse_args()
    main(args)
