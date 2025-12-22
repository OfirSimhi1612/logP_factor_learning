"""
Visualize per-atom scalar weights from the trained contextual MLP model.
Shows which atoms are scaled up/down to correct RDKit's LogP predictions.

Note: This script now uses the refactored visualization utilities from src.utils.visualization
for improved code reusability and maintainability.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from rdkit import Chem
from src.mlp_regressor.mlp import ContextualAtomScalarMLP

# Import refactored visualization utilities
from src.utils.visualization import (
    get_atom_features_from_mol,
    predict_atom_scalars,
    visualize_molecule_with_weights,
    visualize_molecule_3d
)

# Configuration (must match training)
RADIUS = 2
HIDDEN_LAYERS = [40, 40, 32]
CACHE_FILE = "processed_molecules_cache.pkl"
MODEL_FILE = "best_model_contextual.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_input_dim():
    """Dynamically determine the input dimension for the MLP."""
    featurizer = Featurizer()
    # The MPNN padding logic ensures output is max(atom_dim, bond_dim)
    atom_dim = featurizer.feature_dim
    # Bond dim is hardcoded to 10 in featurizer.get_dense_bond_features
    bond_dim = 10
    return max(atom_dim, bond_dim)


def load_model():
    input_dim = get_input_dim()
    model = ContextualAtomScalarMLP(input_dim=input_dim, hidden_dims=HIDDEN_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    return model


def load_cache():
    with open(CACHE_FILE, 'rb') as f:
        cache_data = pickle.load(f)
    return cache_data


# Visualization functions are now imported from src.utils.visualization


def visualize_from_cache(model, cache_data, mol_index):
    """Visualize a molecule from the cached dataset"""
    mol_indexs = cache_data['mol_indexs']
    mol_data = cache_data['mol_data']
    X_atoms = cache_data['X_atom_fps']
    atom_rdkit_score = cache_data['atom_rdkit_score']

    # Get atoms for this molecule
    atom_mask = mol_indexs == mol_index
    atom_features = X_atoms[atom_mask]
    atom_contribs = atom_rdkit_score[atom_mask]

    # Get molecule info
    mol_info = mol_data[mol_index]

    # Predict scalars
    atom_features_tensor = torch.FloatTensor(atom_features).to(device)
    with torch.no_grad():
        atom_scalars = model(atom_features_tensor).cpu().numpy()

    if 'smiles' not in mol_info:
        return atom_scalars, atom_contribs

    smiles = mol_info['smiles']
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return atom_scalars, atom_contribs

    # Calculate predictions
    rdkit_logp = mol_info['rdkit_logp']
    predicted_logp = (atom_contribs * atom_scalars).sum()
    exp_logp = mol_info['exp_logp']

    print(f"RDKit LogP: {rdkit_logp:.3f}")
    print(f"Model Predicted LogP: {predicted_logp:.3f}")
    print(f"Experimental LogP: {exp_logp:.3f}")
    print(f"RDKit Error: {abs(rdkit_logp - exp_logp):.3f}")
    print(f"Model Error: {abs(predicted_logp - exp_logp):.3f}")

    # Visualize
    title = f"RDKit: {rdkit_logp:.2f}, Model: {predicted_logp:.2f}, Exp: {exp_logp:.2f}"

    visualize_molecule_with_weights(
        mol, atom_scalars, atom_contribs,
        title=title,
        save_path=f"images/mol_{mol_index}.png"
    )

    return atom_scalars, atom_contribs


def visualize_from_smiles(model, smiles, experimental_logp=None):
    """Visualize a molecule from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Predict scalars
    atom_scalars, atom_contribs = predict_atom_scalars(model, mol, device)

    # Calculate predictions
    rdkit_logp = atom_contribs.sum()
    predicted_logp = (atom_contribs * atom_scalars).sum()

    print(f"\nPredictions for: {smiles}")
    print(f"   RDKit LogP: {rdkit_logp:.3f}")
    print(f"   Model Predicted LogP: {predicted_logp:.3f}")
    if experimental_logp is not None:
        print(f"   Experimental LogP: {experimental_logp:.3f}")
        print(f"   RDKit Error: {abs(rdkit_logp - experimental_logp):.3f}")
        print(f"   Model Error: {abs(predicted_logp - experimental_logp):.3f}")

    # Visualize
    title = f"SMILES: {smiles}\nRDKit: {rdkit_logp:.2f} -> Model: {predicted_logp:.2f}"
    if experimental_logp is not None:
        title += f" (Exp: {experimental_logp:.2f})"

    visualize_molecule_with_weights(
        mol, atom_scalars, atom_contribs,
        title=title,
        save_path=f"molecule_viz_{smiles[:20]}.png"
    )

    return atom_scalars, atom_contribs


def main():
    # Load model
    model = load_model()

    print("\nChoose visualization mode:")
    print("1. Visualize from SMILES string")
    print("2. Browse cached molecules by ID")

    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "1":
        smiles = input("Enter SMILES: ").strip()
        exp_logp_str = input("Enter experimental LogP (optional, press Enter to skip): ").strip()
        exp_logp = float(exp_logp_str) if exp_logp_str else None
        visualize_from_smiles(model, smiles, exp_logp)

    elif choice == "2":
        cache_data = load_cache()
        print(f"\nDataset contains {len(cache_data['mol_data'])} molecules")
        mol_index = int(input(f"Enter molecule ID (0-{len(cache_data['mol_data'])-1}): "))
        if 0 <= mol_index < len(cache_data['mol_data']):
            visualize_from_cache(model, cache_data, mol_index)
        else:
            print(f"Invalid molecule ID. Must be between 0 and {len(cache_data['mol_data'])-1}")

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()