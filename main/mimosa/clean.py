from tqdm import tqdm 
import os
import argparse
# from chemutils import vocabulary, smiles2word 
from chemutils import is_valid, logp_modifier, load_vocabulary

def main():
    parser = argparse.ArgumentParser(description='Clean SMILES database by removing invalid molecules')
    parser.add_argument('--input', '-i', required=True,
                        help='Input file containing SMILES data (e.g., zinc.smi)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output file for cleaned SMILES data')
    parser.add_argument('--vocabulary', '-v', required=True,
						help='Vocabulary file containing valid substructures')
    parser.add_argument('--skip-header', action='store_true',
                        help='Skip the first line of the input file (header)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    smiles_database = args.input
    clean_smiles_database = args.output
    vocabulary	 = load_vocabulary(args.vocabulary)

    with open(smiles_database, 'r') as fin:
        lines = fin.readlines()
        if args.skip_header:
            lines = lines
    
    smiles_lst = [i.strip().strip('"') for i in lines]
    print(f"Loaded {len(smiles_lst)} SMILES from {smiles_database}")

    clean_smiles_lst = []
    for smiles in tqdm(smiles_lst, desc="Validating SMILES"):
        if is_valid(smiles, vocabulary):
            clean_smiles_lst.append(smiles)
    
    clean_smiles_set = set(clean_smiles_lst)
    print(f"Found {len(clean_smiles_lst)} valid SMILES, {len(clean_smiles_set)} unique")
    
    with open(clean_smiles_database, 'w') as fout:
        for smiles in clean_smiles_set:
            fout.write(smiles + '\n')
    
    print(f"Cleaned SMILES saved to {clean_smiles_database}")


if __name__ == "__main__":
    main()


