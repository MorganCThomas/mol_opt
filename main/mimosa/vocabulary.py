# from chemutils import smiles2word

import os
import argparse
from collections import defaultdict 
from tqdm import tqdm 
from rdkit import Chem


def smiles2mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol 

## input: smiles, output: word lst;  
def smiles2word(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None 
    word_lst = []

    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques_smiles = []
    for clique in cliques:
        try:
            clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=True)
            cliques_smiles.append(clique_smiles)
        except Exception as e:
            print(e) #print(f"Error fragmenting molecule: {e}")
    atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
    return cliques_smiles + atom_not_in_rings_list 


def main():
    parser = argparse.ArgumentParser(description='Generate vocabulary from SMILES data')
    parser.add_argument('--input', '-i', required=True, 
                        help='Input file containing SMILES data (e.g., zinc.tab)')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Output directory where substructure.txt and vocabulary.txt will be saved')
    parser.add_argument('--min-count', '-m', type=int, default=5000,
                        help='Minimum count threshold for vocabulary selection (default: 5000)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    rawdata_file = args.input
    all_vocabulary_file = os.path.join(args.output_dir, "substructure.txt")
    select_vocabulary_file = os.path.join(args.output_dir, "vocabulary.txt")

    if not os.path.exists(all_vocabulary_file):
        with open(rawdata_file) as fin:
            lines = fin.readlines()
            smiles_lst = [line.strip().strip('"') for line in lines]
        word2cnt = defaultdict(int)
        for smiles in tqdm(smiles_lst):
            word_lst = smiles2word(smiles)
            if word_lst is not None:
                for word in word_lst:
                    word2cnt[word] += 1
        word_cnt_lst = [(word,cnt) for word,cnt in word2cnt.items()]
        word_cnt_lst = sorted(word_cnt_lst, key=lambda x:x[1], reverse = True)

        with open(all_vocabulary_file, 'w') as fout:
            for word, cnt in word_cnt_lst:
                fout.write(word + '\t' + str(cnt) + '\n')
    else:
        with open(all_vocabulary_file, 'r') as fin:
            lines = fin.readlines()
            word_cnt_lst = [(line.split('\t')[0], int(line.split('\t')[1])) for line in lines]

    word_cnt_lst = list(filter(lambda x:x[1]>args.min_count, word_cnt_lst))
    print(f"Selected {len(word_cnt_lst)} vocabulary items with count > {args.min_count}")

    with open(select_vocabulary_file, 'w') as fout:
        for word, cnt in word_cnt_lst:
            fout.write(word + '\t' + str(cnt) + '\n')


if __name__ == "__main__":
    main()



