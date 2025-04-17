import argparse
import torch
from cs336_basics.torch_modules import TransformerLM, cross_entropy_loss, learning_rate_scheduling, AdamW, DataLoader

def main(args):
    # First things first
    data_dir = '/data/c-worledge/t_results/' 
    save_dir = '/data/c-worledge/a1_checkpoints/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

     # Initialize the dataloader
    if args.train_data_str == 'tst':
        train_data_fp = data_dir + "ts_train_tokens.npy" 
    elif args.train_data_str == 'tsv':
        train_data_fp = data_dir + "ts_val_tokens.npy"
    elif args.train_data_str == 'owtt':
        train_data_fp = data_dir + "" # TODO merge the two files and its name here
    elif args.train_data_str == 'owtv':
        train_data_fp = data_dir + "owt_val_tokens.npy"
    
    train_dataset = DataLoader(
        batch_size=args.batch_size,
        context_length=args.context_length,
        data_file_path=train_data_fp
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
        dtype=args.dtype,
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
    for t in range(args.num_batches):
        # get the data batch
        train_batch, target_batch = train_dataset.get_random_batch()
        train_batch = train_batch.to(device)
        target_batch = target_batch.to(device)

        # compute a forward pass
        logits = model(train_batch)

        # compute the loss
        loss = cross_entropy_loss(logits, target_batch)
        gradients = loss.backward()
        breakpoint()

        # take an optimizer step



    # cross_entropy_loss(logits: Float[Tensor, '... batch vocab_size'],
    #                    targets: Int[Tensor, '... batch']
    #                    ) -> Float[Tensor, '...']

    # Implement checkpointing
    # Implement loss tracking (WandB)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=str, required=True, help='Number of sequences in a batch')
    parser.add_argument('--context_length', type=int, default=42, help='Number of tokens in a sequence')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to train')
    parser.add_argument('--train_data_str', type=str, default='ts', help='Train data string') # 'tst', 'tsv','owtt', or 'owtv'
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--vocab_size', type=int, default=256, help='Vocabulary size')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of the feedforward layer')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta value for the model')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--model_eps', type=float, default=1e-5, help='Epsilon value for the model')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type for the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 value for the optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 value for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer')
    parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Epsilon value for the optimizer')
    args = parser.parse_args()
    main(args)
