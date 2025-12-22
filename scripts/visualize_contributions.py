"""
Visualize per-atom scalar weights from the trained contextual MLP model.
Shows which atoms are scaled up/down to correct RDKit's LogP predictions.
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Crippen
from src.mlp_regressor.mlp import ContextualAtomScalarMLP
from src.mp_graph.featurizer import Featurizer
from src.mp_graph.simple_mp_graph import MessagePassingGraph

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


def get_atom_features_from_mol(mol):
    """Extract per-atom features from a molecule using MPNN"""
    # 1. Get RDKit contributions
    atom_contribs = np.array([x[0] for x in Crippen._GetAtomContribs(mol)])
    
    # 2. Initialize Featurizer and CMPNN
    cmpnn = MessagePassingGraph(depth=3)
    featurizer = Featurizer()
    
    # 3. Featurize
    atom_feats, bond_feats, adj = featurizer.featurize_molecule(mol)
    
    # 4. Convert to tensors
    atom_feats_t = torch.from_numpy(atom_feats)
    bond_feats_t = torch.from_numpy(bond_feats)
    adj_t = torch.from_numpy(adj)
    
    # 5. Get per-atom embeddings via MPNN
    # Note: We must detach to return numpy array
    atom_embeddings = cmpnn(adj_t, atom_feats_t, bond_feats_t, return_atoms=True)
    
    return atom_embeddings.detach().numpy(), atom_contribs


def predict_atom_scalars(model, mol):
    """Predict scalar weights for each atom in a molecule"""
    atom_features, atom_contribs = get_atom_features_from_mol(mol)

    # Convert to tensor
    atom_features_tensor = torch.FloatTensor(atom_features).to(device)

    # Predict
    with torch.no_grad():
        atom_scalars = model(atom_features_tensor).cpu().numpy()

    return atom_scalars, atom_contribs


def visualize_molecule_3d(mol, atom_scalars, atom_contribs, title="3D Molecule View", vmin=None, vmax=None, save_path=None):
    """
    Create 3D visualization of molecule with atoms colored by their change in hydrophobicity.
    Blue = More Hydrophilic (Predicted < RDKit), Red = More Hydrophobic (Predicted > RDKit)
    """
    # Generate 3D conformer
    mol_3d = Chem.Mol(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)

    # Get conformer
    conf = mol_3d.GetConformer()
    num_atoms = mol_3d.GetNumAtoms()

    # Get atom positions
    positions = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])

    # Calculate difference in contribution: (Predicted - RDKit)
    # Predicted = scalar * RDKit
    # Diff = RDKit * (scalar - 1.0)
    diffs = atom_contribs * (atom_scalars - 1.0)

    # Normalize diffs for coloring (centered at 0)
    # 0 -> White (0.5)
    # <0 -> Blue
    # >0 -> Red
    max_diff = max(abs(diffs.max()), abs(diffs.min()))
    # Cap deviation at some reasonable value so small changes are visible
    max_diff = max(0.1, max_diff) 
    
    # Map diffs to [0, 1] for the colormap: 
    # -max_diff -> 0 (Blue)
    # 0         -> 0.5 (White)
    # +max_diff -> 1 (Red)
    norm_vals = 0.5 + (diffs) / (2 * max_diff)
    norm_vals = np.clip(norm_vals, 0, 1)

    # Get colors (Blue-White-Red)
    colors = plt.cm.bwr(norm_vals)

    # Get bonds
    bonds = []
    for bond in mol_3d.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot bonds
    for bond in bonds:
        i, j = bond
        xs = [positions[i][0], positions[j][0]]
        ys = [positions[i][1], positions[j][1]]
        zs = [positions[i][2], positions[j][2]]
        ax.plot(xs, ys, zs, color='black', linestyle='-', linewidth=2, alpha=0.6)

    # Plot atoms
    for i in range(num_atoms):
        atom = mol_3d.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()

        # Atom size based on contribution magnitude to show importance
        max_contrib = max(abs(atom_contribs.max()), abs(atom_contribs.min()))
        if max_contrib > 0:
            size = 300 + 500 * min(abs(atom_contribs[i]) / max_contrib, 1.0)
        else:
            size = 300

        ax.scatter(positions[i][0], positions[i][1], positions[i][2],
                  c=[colors[i]], s=size, edgecolors='black', linewidth=2,
                  alpha=0.9)

        # Add atom labels
        ax.text(positions[i][0], positions[i][1], positions[i][2],
               f'{symbol}{i}', fontsize=8, ha='center', va='center')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add colorbar
    import matplotlib.cm as cm
    # Create a custom normalization for the colorbar labels
    norm = plt.Normalize(vmin=-max_diff, vmax=max_diff)
    sm = cm.ScalarMappable(cmap=cm.bwr, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Hydrophobicity Shift (Blue=More Hydrophilic, Red=More Hydrophobic)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path_3d = save_path.replace('.png', '_3d.png')
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')

    return fig


def visualize_molecule_with_weights(mol, atom_scalars, atom_contribs,
                                     title="Molecule with Per-Atom Weights",
                                     save_path=None, show_3d=True):
    """
    Visualize molecule with atoms colored by their change in hydrophobicity.

    - Blue:  More Hydrophilic (Predicted < RDKit)
    - White: No Change (Predicted ~= RDKit)
    - Red:   More Hydrophobic (Predicted > RDKit)
    """
    num_atoms = mol.GetNumAtoms()

    # Calculate difference in contribution: (Predicted - RDKit)
    diffs = atom_contribs * (atom_scalars - 1.0)

    # Determine max deviation for symmetric centering around 0
    max_diff = max(abs(diffs.max()), abs(diffs.min()))
    max_diff = max(0.1, max_diff)  # Ensure minimal range to see white
    
    # Map diffs to [0, 1] for the 'bwr' colormap
    # -max_diff -> 0 (Blue)
    # 0         -> 0.5 (White)
    # +max_diff -> 1 (Red)
    norm_vals = 0.5 + (diffs) / (2 * max_diff)
    norm_vals = np.clip(norm_vals, 0, 1)

    # Get colors from bwr colormap
    colors_rgb = plt.cm.bwr(norm_vals)

    # Set atom highlights
    highlight_atoms = list(range(num_atoms))
    highlight_colors = {i: tuple(colors_rgb[i][:3]) for i in range(num_atoms)}

    # Set atom radii based on contribution magnitude (larger = more important)
    atom_radii = {i: 0.3 + 0.2 * min(abs(atom_contribs[i]) / 0.5, 1.0)
                  for i in range(num_atoms)}

    # Draw molecule
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(800, 800)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    # opts.useBWAtomSymbols = True  <-- REMOVED (Not supported in this RDKit version)
    opts.continuousHighlight = False   # Don't bleed highlights into bonds
    opts.fillHighlights = True         # Fill the circles
    opts.bondLineWidth = 2

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=[],
        highlightBondColors=None
    )
    drawer.FinishDrawing()

    # Create figure with molecule and colorbar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={'width_ratios': [3, 1]})

    # Show molecule
    img = drawer.GetDrawingText()
    from PIL import Image
    import io
    pil_img = Image.open(io.BytesIO(img))
    ax1.imshow(pil_img)
    ax1.axis('off')
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Create detailed legend/info panel
    ax2.axis('off')

    # Sort atoms by absolute difference
    abs_diff = np.abs(diffs)
    sorted_by_diff = np.argsort(abs_diff)[::-1]

    info_text = "Per-Atom Analysis\n" + "="*30 + "\n\n"
    info_text += f"RDKit LogP: {atom_contribs.sum():.3f}\n"
    info_text += f"Predicted LogP: {(atom_contribs * atom_scalars).sum():.3f}\n\n"

    info_text += "Most Adjusted Atoms:\n"
    info_text += "(Sorted by shift magnitude)\n\n"

    for idx in sorted_by_diff[:5]:
        diff = diffs[idx]
        direction = "More Hydrophobic" if diff > 0 else "More Hydrophilic"
        label = "[RED]" if diff > 0 else "[BLUE]"
        
        info_text += f"{label} [{idx:2d}] {mol.GetAtomWithIdx(int(idx)).GetSymbol():2s}: "
        info_text += f"{diff:+.3f} ({direction})\n"
        info_text += f"       {atom_contribs[idx]:+.3f} -> {atom_contribs[idx]*atom_scalars[idx]:+.3f}\n"

    info_text += "\nLeast Adjusted Atoms:\n"
    for idx in sorted_by_diff[-3:][::-1]:
        info_text += f"  [{idx:2d}] {mol.GetAtomWithIdx(int(idx)).GetSymbol():2s}: "
        info_text += f"Shift: {diffs[idx]:.3f}\n"

    info_text += "\nColor Legend:\n"
    info_text += "  BLUE : More Hydrophilic (Pred < RDKit)\n"
    info_text += "  WHITE: No Change\n"
    info_text += "  RED  : More Hydrophobic (Pred > RDKit)\n"
    info_text += "\nSize = RDKit contribution\n"

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    # Show 3D visualization if requested
    fig_3d = None
    if show_3d:
        title_3d = f"3D View: {title.split(':')[0] if ':' in title else title}"
        fig_3d = visualize_molecule_3d(mol, atom_scalars, atom_contribs,
                                        title=title_3d, save_path=save_path)

    return fig, fig_3d


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
    atom_scalars, atom_contribs = predict_atom_scalars(model, mol)

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