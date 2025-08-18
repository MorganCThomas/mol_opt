import torch
import argparse
import os
from tqdm import tqdm 
from random import shuffle 
from module import GCN 
from chemutils import smiles2feature, load_vocabulary
from utils import Molecule_Dataset


def collate_fn(batch_lst):
    return batch_lst


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for molecular property prediction')
    parser.add_argument('--data-file', '-d', required=True,
                        help='Input file containing cleaned SMILES data')
    parser.add_argument('--vocabulary-file', '-v', required=True,
						help='Vocabulary file containing valid substructures')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for training (default: cpu)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num-workers', '-w', type=int, default=1,
                        help='Number of workers for data loading (default: 1)')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--valid-every', type=int, default=5000,
                        help='Validate every N iterations (default: 5000)')
    parser.add_argument('--save-dir', '-s', default='pretrained_model',
                        help='Directory to save model checkpoints (default: save_model)')
    parser.add_argument('--nfeat', type=int, default=50,
                        help='Number of input features (default: 50)')
    parser.add_argument('--nhid', type=int, default=100,
                        help='Number of hidden units (default: 100)')
    parser.add_argument('--num-layers', '-l', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--train-split', type=float, default=0.9,
                        help='Fraction of data to use for training (default: 0.9)')
    parser.add_argument('--shuffle-data', action='store_true',
                        help='Shuffle the data before splitting')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load and prepare data
    data_file = args.data_file
    with open(data_file, 'r') as fin:
        lines = fin.readlines()

    if args.shuffle_data:
        shuffle(lines)
    lines = [line.strip() for line in lines]
    N = int(len(lines) * args.train_split)
    train_data = lines[:N]
    valid_data = lines[N:]
    
    vocabulary = load_vocabulary(args.vocabulary_file)
    
    print(f"Training samples: {len(train_data)}, Validation samples: {len(valid_data)}")

    training_set = Molecule_Dataset(train_data)
    valid_set = Molecule_Dataset(valid_data)
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers}

    train_generator = torch.utils.data.DataLoader(training_set, collate_fn=collate_fn, **params)
    valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn=collate_fn, **params)

    gnn = GCN(
        nfeat=args.nfeat,
        nhid=args.nhid,
        num_layer=args.num_layers,
        vocabulary_size=len(vocabulary),
        device=device,
        ).to(device)
    print('GNN is built!')

    cost_lst = []
    valid_loss_lst = []
    epoch = args.epochs
    every_k_iters = args.valid_every
    save_folder = os.path.join(args.save_dir, "GNN_epoch_")
    
    for ep in tqdm(range(epoch), desc="Epochs"):
        for i, smiles in tqdm(enumerate(train_generator), desc=f"Epoch {ep+1}", total=len(training_set)):
            ### 1. training
            smiles = smiles[0]
            try:
            	node_mat, adjacency_matrix, idx, label = smiles2feature(smiles, vocabulary=vocabulary)
            except Exception as e:
                continue
            node_mat = torch.FloatTensor(node_mat).to(device)
            adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
            label = torch.LongTensor([label]).view(-1).to(device)
            cost = gnn.learn(node_mat, adjacency_matrix, idx, label)
            cost_lst.append(cost)

            #### 2. validation 
            if (i % every_k_iters == 0) and (i != 0):
                gnn.eval()
                valid_loss, valid_num = 0, 0
                for smiles in valid_generator:
                    smiles = smiles[0]
                    try:
                        node_mat, adjacency_matrix, idx, label = smiles2feature(smiles, vocabulary=vocabulary)
                    except Exception as e:
                        continue  # Skip invalid SMILES
                    node_mat = torch.FloatTensor(node_mat).to(device)
                    adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
                    label = torch.LongTensor([label]).view(-1).to(device)
                    cost, _ = gnn.infer(node_mat, adjacency_matrix, idx, label)
                    valid_loss += cost
                    valid_num += 1
                valid_loss = valid_loss / valid_num
                valid_loss_lst.append(valid_loss)
                file_name = f"{save_folder}{ep}_validloss_{str(valid_loss)[:7]}.ckpt"
                torch.save(gnn, file_name)
                print(f"Epoch {ep}, Iteration {i}, Validation Loss: {valid_loss:.4f}")
                print(f"Model saved to {file_name}")
                gnn.train()


if __name__ == "__main__":
    main()




