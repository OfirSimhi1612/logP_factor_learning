"""
Data loading and preprocessing for the Contextual Atom Scalar MLP.
"""

import os
import io
import pickle
import requests
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

from src.mp_graph.featurizer import Featurizer
from src.mp_graph.mp_graph import MessagePassingGraph
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.mlp_regressor.training import MoleculeDataset, collate_molecules

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
CACHE_FILE = DATA_DIR / "processed_molecules_cache.pkl"


def get_dataset():
    """Load the lipophilicity dataset, downloading if necessary."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / 'lipophilicity.csv'

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print("Downloading dataset...")
        response = requests.get(DATA_URL)
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    return df


def get_features_and_targets(df, use_cache=True):
    if use_cache and CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['X_atom_fps'], cache_data['atom_rdkit_score'], cache_data['mol_indexs'], cache_data['mol_data']

    X_atom_fps = []
    y_atom_target = []
    atom_rdkit_score = []
    mol_indexs = []
    mol_data = []

    mol_counter = 0
    cmpnn = MessagePassingGraph(depth=3)
    featurizer = Featurizer()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None: continue

        atom_contribs = np.array([x[0] for x in Crippen._GetAtomContribs(mol)])
        rdkit_logp = atom_contribs.sum()

        exp_logp = row['exp']

        num_atoms = mol.GetNumAtoms()

        mol_info = {
            'mol_index': mol_counter,
            'smiles': row['smiles'],
            'exp_logp': exp_logp,
            'rdkit_logp': rdkit_logp,
            'mw': Descriptors.MolWt(mol),
            'num_atoms': num_atoms,
            'atom_contribs': atom_contribs.copy()
        }
        mol_data.append(mol_info)
        
        atom_feats, bond_feats, adj = featurizer.featurize_molecule(mol)
        
        atom_feats_t = torch.from_numpy(atom_feats)
        bond_feats_t = torch.from_numpy(bond_feats)
        adj_t = torch.from_numpy(adj)
        
        atom_embeddings = cmpnn(adj_t, atom_feats_t, bond_feats_t, return_atoms=True)
        atom_embeddings = atom_embeddings.detach().numpy()

        for atom_idx in range(num_atoms):
            env_fp = atom_embeddings[atom_idx]
            X_atom_fps.append(env_fp)
            atom_rdkit_score.append(atom_contribs[atom_idx])
            mol_indexs.append(mol_counter)
            y_atom_target.append(exp_logp)

        mol_counter += 1

    X_atom_fps = np.array(X_atom_fps)
    y_atom_target = np.array(y_atom_target)
    atom_rdkit_score = np.array(atom_rdkit_score)
    mol_indexs = np.array(mol_indexs)

    if use_cache:
        cache_data = {
            'X_atom_fps': X_atom_fps,
            'y_atom_target': y_atom_target,
            'atom_rdkit_score': atom_rdkit_score,
            'mol_indexs': mol_indexs,
            'mol_data': mol_data,
            'num_molecules': len(mol_data),
            'num_atoms': len(X_atom_fps)
        }
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)

    return X_atom_fps, atom_rdkit_score, mol_indexs, mol_data


def create_and_save_splits(mol_indexs, mol_data):
    """Create and save train/validation/test splits."""
    SPLITS_DIR.mkdir(exist_ok=True, parents=True)

    unique_mol_indexs = np.unique(mol_indexs)
    train_mol_indexs, test_mol_indexs = train_test_split(
        unique_mol_indexs, test_size=0.2, random_state=42
    )
    train_mol_indexs, val_mol_indexs = train_test_split(
        train_mol_indexs, test_size=0.1, random_state=42
    )

    # Save splits to CSV
    def save_split_csv(indices, filename):
        split_data = []
        for idx in indices:
            mol_info = mol_data[idx]
            split_data.append(mol_info)

        pd.DataFrame(split_data).to_csv(filename, index=False)

    save_split_csv(train_mol_indexs, SPLITS_DIR / 'train.csv')
    save_split_csv(val_mol_indexs, SPLITS_DIR / 'val.csv')
    save_split_csv(test_mol_indexs, SPLITS_DIR / 'test.csv')

    return train_mol_indexs, val_mol_indexs, test_mol_indexs


def get_dataloaders(batch_size):
    df = get_dataset()
    X_atoms, atom_contribs, mol_indexs, mol_data = get_features_and_targets(df)

    train_mol_indexs, val_mol_indexs, test_mol_indexs = create_and_save_splits(mol_indexs, mol_data)

    train_dataset = MoleculeDataset(X_atoms, atom_contribs, mol_indexs, mol_data, train_mol_indexs)
    val_dataset = MoleculeDataset(X_atoms, atom_contribs, mol_indexs, mol_data, val_mol_indexs)
    test_dataset = MoleculeDataset(X_atoms, atom_contribs, mol_indexs, mol_data, test_mol_indexs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_molecules)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)
    
    return train_loader, val_loader, test_loader

