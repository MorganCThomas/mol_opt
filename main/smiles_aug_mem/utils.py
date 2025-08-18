import torch
import os
import io
import gzip
import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import RenumberAtoms

from vocabulary import SMILESTokenizer

_ST = SMILESTokenizer()


def get_randomized_smiles(smiles_list, prior) -> list:
    """takes a list of SMILES and returns a list of randomized SMILES"""
    randomized_smiles_list = []
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        if mol:
            try:
                randomized_smiles = randomize_smiles(mol)
                # there may be tokens in the randomized SMILES that are not in the Vocabulary
                # check if the randomized SMILES can be encoded
                tokens = _ST.tokenize(randomized_smiles)
                encoded = prior.vocabulary.encode(tokens)
                randomized_smiles_list.append(randomized_smiles)
            except KeyError:
                randomized_smiles_list.append(smiles)
        else:
            randomized_smiles_list.append(smiles)

    return randomized_smiles_list


def randomize_smiles(mol) -> str:
    """
    Returns a randomized SMILES given an RDKit Mol object.
    :param mol: An RDKit Mol object
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    from reinvent-chemistry
    """
    new_atom_order = list(range(mol.GetNumHeavyAtoms()))
    # reinvent-chemistry uses random.shuffle
    # use np.random.shuffle for reproducibility since PMO fixes the np seed
    np.random.shuffle(new_atom_order)
    random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
    return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def set_device(device="gpu"):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if not torch.cuda.is_available() or (device == "cpu"):
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        device = torch.device("cpu")
        return device
    elif (
        device in ["gpu", "cuda"]
    ) and torch.cuda.is_available():  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        device = torch.device("cuda")
        return device
    elif torch.cuda.is_available():  # Assume an index
        raise NotImplementedError


def read_smiles(file_path):
    """Read a smiles file separated by \n"""
    if any(["gz" in ext for ext in os.path.basename(file_path).split(".")[1:]]):
        with gzip.open(file_path) as f:
            smiles = f.read().splitlines()
            smiles = [smi.decode("utf-8") for smi in smiles]
    else:
        with open(file_path, "rt") as f:
            smiles = f.read().splitlines()
    return smiles


def save_smiles(smiles, file_path):
    """Save smiles to a file path seperated by \n"""
    if (not os.path.exists(os.path.dirname(file_path))) and (
        os.path.dirname(file_path) != ""
    ):
        os.makedirs(os.path.dirname(file_path))
    if any(["gz" in ext for ext in os.path.basename(file_path).split(".")[1:]]):
        with gzip.open(file_path, "wb") as f:
            _ = [
                f.write((smi + "\n").encode("utf-8"))
                for smi in smiles
                if smi is not None
            ]
    else:
        with open(file_path, "wt") as f:
            _ = [f.write(smi + "\n") for smi in smiles if smi is not None]
    return

def fraction_valid_smiles(smiles):
    i = 0
    mols = []
    for smile in smiles:
        try:
            mol = MolFromSmiles(smile)
            if mol:
                i += 1
                mols.append(mol)
        except TypeError:  # None passed as smile
            pass
    fraction = 100 * i / len(smiles)
    return round(fraction, 2), mols


def draw_mols(
    smis,
    mols_per_row=5,
    legends=None,
    size_per_mol=(300, 300),
):
    """
    Adds molecules in a grid.
    """
    mols = np.random.choice(smis, 10)
    mols = [MolFromSmiles(smi) for smi in mols if smi is not None]
    if any(mols):
        image = Draw.MolsToGridImage(
            mols, molsPerRow=mols_per_row, subImgSize=size_per_mol, legends=legends
        )
    else:
        image = None
    return image