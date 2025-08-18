#!/usr/bin/env python

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')
from tdc.chem_utils import MolConvert
converter = MolConvert(src = 'SELFIES', dst = 'SMILES',)
# selfies_lst = converter(smiles_lst)

def pretrain(voc_file, selfies_file, output_dir, output_name, restore_from=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)
    print('# Create a Dataset from a SELFIES file')
    moldata = MolData(selfies_file, voc)


    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)
    print('build DataLoader')

    Prior = RNN(voc)
    print("build RNN")
    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    print("begin to learn")
    for epoch in range(1, 6):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                # tqdm.write("*" * 50)
                # tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    # smile = converter(selfies)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SELFIES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                output_path = os.path.join(output_dir, f"{output_name}.ckpt")
                torch.save(Prior.rnn.state_dict(), output_path)

        # Save the Prior
        output_path = os.path.join(output_dir, f"{output_name}.ckpt")
        torch.save(Prior.rnn.state_dict(), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain RNN for molecular generation using SELFIES')
    parser.add_argument('--voc_file', type=str, default='data/Voc',
                        help='Path to vocabulary file (default: data/Voc)')
    parser.add_argument('--selfies_file', type=str, default='data/zinc.selfies',
                        help='Path to SELFIES file for training (default: data/zinc.selfies)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for saving model (default: data)')
    parser.add_argument('--output_name', type=str, default='Prior',
                        help='Output model name (default: Prior)')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Path to checkpoint to restore from (optional)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    pretrain(args.voc_file, args.selfies_file, args.output_dir, args.output_name, args.restore_from)
