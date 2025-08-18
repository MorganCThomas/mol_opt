#!/usr/bin/env python
import argparse
import json
import multiprocessing
import os
import sys
import time
from itertools import chain
from os import path

import torch
from rdkit import rdBase
from tqdm.auto import tqdm

from model import Model
from dataset import Dataset, calculate_nlls_from_model
from vocabulary import create_vocabulary, SMILESTokenizer
import utils


rdBase.DisableLog("rdApp.error")


# ---- Main ----
def main(args):
    # Make absolute output directory
    args.output_directory = path.abspath(args.output_directory)
    if not path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Save all args out
    param_file = f"Prior_{args.suffix}.json"
    with open(os.path.join(args.output_directory, param_file), "wt") as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = utils.set_device(args.device)
    print(f"Device set to {device.type}")

    # Setup Tensorboard
    try:
        import wandb
        writer = wandb.init(
            project="mol_opt",  # Specify your project
            name=f"aug_mem_{args.suffix}",  # Name of the run
            dir=args.output_directory,  # Directory to save logs
        )
    except ImportError:
        writer = None

    # Load smiles
    print("Loading smiles")
    train_smiles = utils.read_smiles(args.train_smiles)
    # Load other smiles
    all_smiles = train_smiles
    if args.valid_smiles is not None:
        valid_smiles = utils.read_smiles(args.valid_smiles)
        all_smiles += valid_smiles
    if args.test_smiles is not None:
        test_smiles = utils.read_smiles(args.test_smiles)
        all_smiles += test_smiles

    # Set tokenizer
    tokenizer = SMILESTokenizer()

    # Create vocabulary
    print("Creating vocabulary")
    smiles_vocab = create_vocabulary(smiles_list=all_smiles, tokenizer=tokenizer)

    # Create dataset
    dataset = Dataset(
        smiles_list=train_smiles, vocabulary=smiles_vocab, tokenizer=tokenizer
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Dataset.collate_fn,
        generator=torch.Generator(device=device)
    )

    # Set network params
    network_params = {
        "layer_size": args.layer_size,
        "num_layers": args.num_layers,
        "cell_type": args.cell_type,
        "embedding_layer_size": args.embedding_layer_size,
        "dropout": args.dropout,
        "layer_normalization": args.layer_normalization,
    }
    # Create model
    print("Loading model")
    prior = Model(
        vocabulary=smiles_vocab,
        tokenizer=tokenizer,
        network_params=network_params,
        max_sequence_length=256,
        #device=device,
    )

    # Setup optimizer update to adaptive learning
    optimizer = torch.optim.Adam(prior.network.parameters(), lr=args.learning_rate)

    # Train model
    print("Beginning training")
    global_step = 0
    start_time = time.time()
    for e in range(1, args.n_epochs + 1):
        print(f"Epoch {e}")
        for step, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            # Update total step
            global_step += 1

            # Sample from DataLoader
            input_vectors = batch.long()

            # Calculate loss
            log_p = prior.likelihood(input_vectors)
            loss = log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Decrease learning rate
            if (step % 500 == 0) & (step != 0):
                # Decrease learning rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 1 - 0.03  # Decrease by

            # Validate
            if step % args.validate_frequency == 0:
                # Validate
                prior.network.eval()
                with torch.no_grad():
                    # Sample new molecules
                    sampled_smiles, sampled_likelihood = prior.sample_smiles()
                    validity, mols = utils.fraction_valid_smiles(sampled_smiles)
                    img = utils.draw_mols(sampled_smiles)

                    # Check likelihood on other datasets
                    train_dataloader, _ = calculate_nlls_from_model(prior, train_smiles)
                    train_likelihood = next(train_dataloader)
                    
                    # Check validation smiles
                    if args.valid_smiles is not None:
                        valid_dataloader, _ = calculate_nlls_from_model(
                            prior, valid_smiles
                        )
                        valid_likelihood = next(valid_dataloader)
                    else:
                        valid_likelihood = None
                    
                    # Check test smiles
                    if args.test_smiles is not None:
                        test_dataloader, _ = calculate_nlls_from_model(
                            prior, test_smiles
                        )
                        test_likelihood = next(test_dataloader)
                    else:
                        test_likelihood = None
                    
                    if writer is not None:
                        writer.log({
                            "epoch": e,
                            "validity": validity,
                            "train_ll": train_likelihood.mean(),
                            "sample_ll": sampled_likelihood.mean(),
                            "valid_ll": valid_likelihood.mean() if valid_likelihood is not None else None,
                            "test_ll": test_likelihood.mean() if test_likelihood is not None else None,
                            "mols": wandb.Image(img) if img is not None else None,
                            })
                    else:
                        print(f"Epoch {e}\t|\tValidity: {validity:.2f}%\t|\tTrain NLL: {train_likelihood.mean():.2f}\t|\tSample NLL: {sampled_likelihood.mean():.2f}\t|\t")
                prior.network.train()

        # Save every epoch
        prior.save(
            file=path.join(args.output_directory, f"Prior_{args.suffix}_Epoch-{e}.ckpt")
        )

    # Add training time
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # in minutes
    # Add this to params
    with open(os.path.join(args.output_directory, param_file), "r") as f:
        params = json.load(f)
    params.update({"total_time_mins": training_time})
    params.update({"epoch_time_mins": training_time / args.n_epochs})
    with open(os.path.join(args.output_directory, param_file), "w") as f:
        json.dump(params, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train an initial prior model based on smiles data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--train_smiles", type=str, help="Path to smiles file", required=True
    )
    required.add_argument(
        "-o",
        "--output_directory",
        type=str,
        help="Output directory to save model",
        required=True,
    )
    required.add_argument(
        "-s", "--suffix", type=str, help="Suffix to name files", required=True
    )

    # optional = parser.add_argument_group('Optional arguments')
    #parser.add_argument(
    #    "--randomize",
    #    action="store_true",
    #    help="Training smiles will be randomized using default arguments (10 restricted)",
    #)
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="If randomizing use multiple cores"
    )
    parser.add_argument("--valid_smiles", help="Validation smiles")
    parser.add_argument("--test_smiles", help="Test smiles")
    parser.add_argument("--validate_frequency", default=500, help=" ")
    parser.add_argument("--n_epochs", type=int, default=5, help=" ")
    parser.add_argument("--batch_size", type=int, default=128, help=" ")
    parser.add_argument(
        "-d", "--device", default="gpu", help="cpu/gpu or device number"
    )
    parser.add_argument("--layer_size", type=int, default=512, help=" ")
    parser.add_argument("--num_layers", type=int, default=3, help=" ")
    parser.add_argument("--cell_type", choices=["lstm", "gru"], default="gru", help=" ")
    parser.add_argument("--embedding_layer_size", type=int, default=256, help=" ")
    parser.add_argument("--dropout", type=float, default=0.0, help=" ")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help=" ")
    parser.add_argument("--layer_normalization", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)