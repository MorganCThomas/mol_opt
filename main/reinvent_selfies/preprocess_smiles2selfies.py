#!/usr/bin/env python

import argparse
import os
import sys
from tqdm import tqdm
from moleval.utils import read_smiles
from tdc.chem_utils import MolConvert
from data_structs import Vocabulary
import re

# Initialize converter
converter = MolConvert(src='SMILES', dst='SELFIES')

def construct_selfies_vocabulary(selfies_list):
    """Constructs a vocabulary from a list of SELFIES strings"""
    add_chars = set()
    for selfies in tqdm(selfies_list, desc="Building vocabulary"):
        if selfies and isinstance(selfies, str):
            # Parse SELFIES tokens
            words = selfies.strip().strip('[]').split('][')
            char_list = ['[' + word + ']' for word in words if word]
            for char in char_list:
                add_chars.add(char)
    
    print(f"Number of unique SELFIES tokens: {len(add_chars)}")
    return add_chars

def save_vocabulary(vocab_chars, output_file):
    """Save vocabulary characters to a file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for char in sorted(vocab_chars):
            f.write(char + "\n")
    print(f"Vocabulary saved to: {output_file}")

def save_selfies(selfies_list, output_file):
    """Save SELFIES strings to a file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for selfies in selfies_list:
            if selfies:  # Only write non-empty SELFIES
                f.write(selfies + "\n")
    print(f"SELFIES saved to: {output_file}")

def convert_smiles_to_selfies(smiles_list):
    """Convert a list of SMILES to SELFIES, filtering out failed conversions"""
    selfies_list = []
    failed_count = 0
    
    for smiles in tqdm(smiles_list, desc="Converting SMILES to SELFIES"):
        try:
            selfies = converter(smiles)
            if selfies and selfies != '' and selfies != 'None':
                selfies_list.append(selfies)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Successfully converted {len(selfies_list)} SMILES to SELFIES")
    print(f"Failed conversions: {failed_count}")
    
    return selfies_list

def main():
    parser = argparse.ArgumentParser(description='Convert SMILES to SELFIES and create vocabulary')
    parser.add_argument('--input_smiles', type=str, required=True,
                        help='Path to input SMILES file')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for SELFIES and vocabulary files (default: data)')
    parser.add_argument('--output_selfies', type=str, default='molecules.selfies',
                        help='Output SELFIES filename (default: molecules.selfies)')
    parser.add_argument('--output_vocab', type=str, default='Voc',
                        help='Output vocabulary filename (default: Voc)')
    
    args = parser.parse_args()
    
    # Create full paths
    output_selfies_path = os.path.join(args.output_dir, args.output_selfies)
    output_vocab_path = os.path.join(args.output_dir, args.output_vocab)
    
    # Read SMILES from file
    print(f"Reading SMILES from: {args.input_smiles}")
    try:
        smiles_list = read_smiles(args.input_smiles)
        print(f"Read {len(smiles_list)} SMILES")
            
    except Exception as e:
        print(f"Error reading SMILES file: {e}")
        sys.exit(1)
    
    # Convert SMILES to SELFIES
    print("Converting SMILES to SELFIES...")
    selfies_list = convert_smiles_to_selfies(smiles_list)
    
    if not selfies_list:
        print("No valid SELFIES generated. Exiting.")
        sys.exit(1)
    
    # Save SELFIES to file
    save_selfies(selfies_list, output_selfies_path)
    
    # Construct vocabulary from SELFIES
    print("Constructing vocabulary...")
    vocab_chars = construct_selfies_vocabulary(selfies_list)
    
    # Save vocabulary to file
    save_vocabulary(vocab_chars, output_vocab_path)
    
    print(f"\nProcessing complete!")
    print(f"- Input SMILES: {len(smiles_list)}")
    print(f"- Valid SELFIES: {len(selfies_list)}")
    print(f"- Vocabulary size: {len(vocab_chars)}")
    print(f"- SELFIES saved to: {output_selfies_path}")
    print(f"- Vocabulary saved to: {output_vocab_path}")

if __name__ == "__main__":
    main()