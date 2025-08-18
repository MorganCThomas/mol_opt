#!/usr/bin/env python

import argparse
import torch
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm


from data_structs import MolData, Vocabulary,MolData_v
from model_trans import Transformer_
from utils import decrease_learning_rate
rdBase.DisableLog('rdApp.error')
from tdc.generation import MolGen
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# import setproctitle
# setproctitle.setproctitle("reposition@ft")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = MolGen(name = 'ZINC')
# data = MolGen(name = 'ChEMBL')
def pretrain(voc_file, smiles_file, output_dir, output_name, restore_from=None):
    """Trains the Prior transformer"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)

    # Read SMILES from file instead of using MolGen
    print(f"Reading SMILES from: {smiles_file}")
    with open(smiles_file, 'r') as f:
        data_all = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(data_all)} SMILES from file")
    
    # Filter out unwanted SMILES (same filtering logic as original)
    shape_arr = []
    out_list=['i','I','.','P','[Se]','[2H]','B','9','[N@+]','[3H]','[C-]','[O]','[18F]']
    for inter in data_all:
        pass_=False
        if type(inter)==float:
            aaaaa=1
        else:
            for inter1 in out_list:
                if inter1 in inter:
                    pass_=True
            if pass_:
                aaa=1
            else:
                a1='i' in inter
                a2='I' in inter
                a3='.' in inter
                if a1 or a2:
                    aaa = 1
                elif a3:
                    aaaa=1
                else:
                    shape_arr.append(inter)


    print('# Create a Dataset from a SMILES file')
    # moldata = MolData("data/mols_filtered.smi", voc)
    moldata=MolData_v(shape_arr, voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)
    print('build DataLoader')

    Prior = Transformer_(voc,device)
    toLoad=False
    
    # Can restore from a saved RNN
    if restore_from:
        Prior.transformer.load_state_dict(torch.load(restore_from))
    if toLoad:
        # Loading the checkpoint
        checkpoint_path = "data/Prior_transformer-epoch67-valid-86.71875.ckpt"#"data/Prior_transformer-epoch5-valid-54.6875.ckpt"
        Prior.transformer.load_state_dict(torch.load(checkpoint_path))
        print("loaded",checkpoint_path)
    print("build Transformer")
    optimizer = torch.optim.Adam(Prior.transformer.parameters(), lr = 0.001)
    print("begin to learn")
    for epoch in range(1, 6):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate los
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()
            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            # print("loss:",loss)
            optimizer.step()



            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                with torch.no_grad():
                # tqdm.write("*" * 50)
                # tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                    seqs, likelihood, _ = Prior.sample(128)
                    valid = 0
                    for i, seq in enumerate(seqs.cpu().numpy()):
                        smile = voc.decode(seq)
                        if Chem.MolFromSmiles(smile):
                            valid += 1
                        if i < 5:
                            tqdm.write(smile)
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("*" * 50 + "\n")
                # torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")
            #torch.cuda.empty_cache()

        # Save the Prior
        output_path = os.path.join(output_dir, f"{output_name}_transformer-epoch{epoch}-valid-{100 * valid / len(seqs)}.ckpt")
        torch.save(Prior.transformer.state_dict(), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain Transformer for molecular generation')
    parser.add_argument('--voc_file', type=str, default='data/Voc',
                        help='Path to vocabulary file (default: data/Voc)')
    parser.add_argument('--smiles_file', type=str, default='data/mols_filtered.smi',
                        help='Path to SMILES file for training (default: data/mols_filtered.smi)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for saving model (default: data)')
    parser.add_argument('--output_name', type=str, default='Prior',
                        help='Output model name (default: Prior)')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Path to checkpoint to restore from (optional)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    pretrain(args.voc_file, args.smiles_file, args.output_dir, args.output_name, args.restore_from)
