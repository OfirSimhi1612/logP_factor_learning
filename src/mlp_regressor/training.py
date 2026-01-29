"""
Training and evaluation utilities for the Contextual Atom Scalar MLP.

This module contains dataset classes, training loops, and evaluation functions
for per-atom scalar prediction with molecular context awareness.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MoleculeDataset(Dataset):
    def __init__(self, X_atoms, atom_contribs, mol_indexs, mol_data, mol_index_list):
        self.X_atoms = X_atoms
        self.atom_contribs = atom_contribs
        self.mol_indexs = mol_indexs
        self.mol_data = mol_data
        self.mol_index_list = mol_index_list

    def __len__(self):
        return len(self.mol_index_list)

    def __getitem__(self, idx):
        mol_index = self.mol_index_list[idx]

        # Get all atoms for this molecule
        atom_mask = self.mol_indexs == mol_index

        # Atom features and RDKit contributions
        atom_features = torch.FloatTensor(self.X_atoms[atom_mask])
        atom_rdkit_score = torch.FloatTensor(self.atom_contribs[atom_mask])

        # Molecule-level target
        exp_logp = self.mol_data[mol_index]["exp_logp"]
        rdkit_logp = self.mol_data[mol_index]["rdkit_logp"]
        mw = self.mol_data[mol_index]["mw"]

        return {
            "atom_features": atom_features,
            "atom_rdkit_score": atom_rdkit_score,
            "exp_logp": torch.FloatTensor([exp_logp]),
            "rdkit_logp": torch.FloatTensor([rdkit_logp]),
            "mw": mw,
            "mol_index": mol_index,
        }


def collate_molecules(batch):
    """Custom collate function since molecules have different numbers of atoms"""
    return batch  # Return list of molecules


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to run training on (CPU or GPU)

    Returns:
        Average loss over all molecules in the epoch
    """
    model.train()
    total_loss = 0
    num_molecules = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch_loss = 0

        for mol_data in batch:
            # Get molecule data
            atom_features = mol_data["atom_features"].to(device)
            atom_rdkit_score = mol_data["atom_rdkit_score"].to(device)
            exp_logp = mol_data["exp_logp"].to(device)

            # Forward pass: get context-aware scalar for each atom
            # The network sees ALL atoms at once and outputs a vector of scalars
            atom_scalars = model(atom_features)  # (num_atoms,)

            # Aggregate: sum(rdkit_contrib * scalar)
            predicted_logp = torch.sum(atom_rdkit_score * atom_scalars)

            # Compute loss
            loss = criterion(predicted_logp.unsqueeze(0), exp_logp)
            batch_loss += loss

            num_molecules += 1

        # Backward pass
        optimizer.zero_grad()
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * len(batch)

    return total_loss / num_molecules


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: The neural network model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on (CPU or GPU)

    Returns:
        Tuple of numpy arrays:
        - predictions: Predicted LogP values
        - targets: True experimental LogP values
        - baselines: RDKit baseline LogP values
        - mol_weights: Molecular weights
        - mol_indexs: Molecule IDs
    """
    model.eval()
    predictions = []
    targets = []
    baselines = []
    mol_weights = []
    mol_indexs_eval = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            for mol_data in batch:
                # Get molecule data
                atom_features = mol_data["atom_features"].to(device)
                atom_rdkit_score = mol_data["atom_rdkit_score"].to(device)
                exp_logp = mol_data["exp_logp"].item()
                rdkit_logp = mol_data["rdkit_logp"].item()
                mw = mol_data["mw"]
                mol_index = mol_data["mol_index"]

                # Forward pass
                atom_scalars = model(atom_features)
                predicted_logp = torch.sum(atom_rdkit_score * atom_scalars).item()

                predictions.append(predicted_logp)
                targets.append(exp_logp)
                baselines.append(rdkit_logp)
                mol_weights.append(mw)
                mol_indexs_eval.append(mol_index)

    return (
        np.array(predictions),
        np.array(targets),
        np.array(baselines),
        np.array(mol_weights),
        np.array(mol_indexs_eval),
    )


def get_atom_scalars(model, mol_data, device):
    """
    Get per-atom scalars for a single molecule.

    Args:
        model: The neural network model
        mol_data: Dictionary containing molecule data from dataset
        device: Device to run inference on

    Returns:
        numpy array of per-atom scalars
    """
    model.eval()
    with torch.no_grad():
        atom_features = mol_data["atom_features"].to(device)
        atom_scalars = model(atom_features)
        return atom_scalars.cpu().numpy()
