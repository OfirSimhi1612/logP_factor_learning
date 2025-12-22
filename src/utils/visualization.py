"""
Visualization utilities for per-atom scalar weights and molecular property prediction.

This module provides functions for visualizing molecular structures with per-atom
contributions to lipophilicity predictions, including 2D and 3D representations
with color-coded atoms indicating hydrophobicity adjustments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Crippen
from PIL import Image
import io

from src.mp_graph.featurizer import Featurizer
from src.mp_graph.mp_graph import MessagePassingGraph


def get_atom_features_from_mol(mol):
    """
    Extract per-atom features from a molecule using message passing neural network.

    This function featurizes a molecular structure and applies message passing
    to generate contextualized atom embeddings that capture local chemical
    environment information.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object

    Returns
    -------
    atom_embeddings : numpy.ndarray
        Per-atom embeddings from message passing network, shape (num_atoms, embedding_dim)
    atom_contribs : numpy.ndarray
        RDKit Crippen logP contributions for each atom, shape (num_atoms,)

    Notes
    -----
    The message passing network performs 3 rounds of aggregation, allowing each
    atom to gather information from neighbors up to 3 bonds away.
    """
    # Extract RDKit Crippen contributions
    atom_contribs = np.array([x[0] for x in Crippen._GetAtomContribs(mol)])

    # Initialize featurizer and message passing network
    cmpnn = MessagePassingGraph(depth=3)
    featurizer = Featurizer()

    # Featurize molecule
    atom_feats, bond_feats, adj = featurizer.featurize_molecule(mol)

    # Convert to tensors
    atom_feats_t = torch.from_numpy(atom_feats)
    bond_feats_t = torch.from_numpy(bond_feats)
    adj_t = torch.from_numpy(adj)

    # Generate per-atom embeddings via message passing
    atom_embeddings = cmpnn(adj_t, atom_feats_t, bond_feats_t, return_atoms=True)

    return atom_embeddings.detach().numpy(), atom_contribs


def predict_atom_scalars(model, mol, device):
    """
    Predict scalar adjustment weights for each atom in a molecule.

    This function applies a trained model to predict per-atom scalar multipliers
    that adjust RDKit baseline logP contributions. The model learns to correct
    systematic biases in the Crippen method.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model for per-atom scalar prediction
    mol : rdkit.Chem.Mol
        RDKit molecule object
    device : torch.device
        Device for model inference (CPU or CUDA)

    Returns
    -------
    atom_scalars : numpy.ndarray
        Predicted scalar multipliers for each atom, shape (num_atoms,)
    atom_contribs : numpy.ndarray
        RDKit Crippen logP contributions for each atom, shape (num_atoms,)

    Notes
    -----
    The predicted logP is calculated as: sum(atom_contribs * atom_scalars)
    Scalars close to 1.0 indicate agreement with RDKit baseline, while
    values >1.0 or <1.0 indicate hydrophobic or hydrophilic adjustments.
    """
    # Extract atom features
    atom_features, atom_contribs = get_atom_features_from_mol(mol)

    # Convert to tensor and move to device
    atom_features_tensor = torch.FloatTensor(atom_features).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        atom_scalars = model(atom_features_tensor).cpu().numpy()

    return atom_scalars, atom_contribs


def visualize_molecule_3d(mol, atom_scalars, atom_contribs, title="3D Molecule View",
                          vmin=None, vmax=None, save_path=None):
    """
    Create 3D visualization of molecule with atoms colored by hydrophobicity shift.

    This function generates a three-dimensional representation of the molecular
    structure with atoms color-coded to indicate adjustments to the RDKit baseline
    logP prediction. Blue atoms indicate increased hydrophilicity (predicted < RDKit),
    while red atoms indicate increased hydrophobicity (predicted > RDKit).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    atom_scalars : numpy.ndarray
        Predicted scalar multipliers for each atom
    atom_contribs : numpy.ndarray
        RDKit Crippen logP contributions for each atom
    title : str, optional
        Plot title (default: "3D Molecule View")
    vmin : float, optional
        Minimum value for colormap normalization (default: auto)
    vmax : float, optional
        Maximum value for colormap normalization (default: auto)
    save_path : str, optional
        Path to save the figure (default: None, display only)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure object

    Notes
    -----
    The 3D conformer is generated using RDKit's ETKDG algorithm and optimized
    with the MMFF force field. Atom sizes are proportional to the magnitude
    of their RDKit contributions, emphasizing important atoms.

    Color scheme:
        - Blue (0.0): More hydrophilic than RDKit (scalar < 1.0)
        - White (0.5): No change from RDKit (scalar ≈ 1.0)
        - Red (1.0): More hydrophobic than RDKit (scalar > 1.0)
    """
    # Generate 3D conformer
    mol_3d = Chem.Mol(mol)
    try:
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception as e:
        # Fallback to 2D coordinates if 3D generation fails
        AllChem.Compute2DCoords(mol_3d)

    # Extract atomic coordinates
    conf = mol_3d.GetConformer()
    num_atoms = mol_3d.GetNumAtoms()
    positions = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])

    # Calculate hydrophobicity shift: (Predicted - RDKit)
    # Predicted = scalar * RDKit
    # Difference = RDKit * (scalar - 1.0)
    diffs = atom_contribs * (atom_scalars - 1.0)

    # Normalize differences for color mapping (centered at 0)
    max_diff = max(abs(diffs.max()), abs(diffs.min()))
    max_diff = max(0.1, max_diff)  # Ensure minimal range for visibility

    # Map differences to [0, 1] for blue-white-red colormap
    # -max_diff -> 0 (Blue), 0 -> 0.5 (White), +max_diff -> 1 (Red)
    norm_vals = 0.5 + (diffs) / (2 * max_diff)
    norm_vals = np.clip(norm_vals, 0, 1)

    # Apply blue-white-red colormap
    colors = plt.cm.bwr(norm_vals)

    # Extract bond connectivity
    bonds = []
    for bond in mol_3d.GetBonds():
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Render bonds
    for bond in bonds:
        i, j = bond
        xs = [positions[i][0], positions[j][0]]
        ys = [positions[i][1], positions[j][1]]
        zs = [positions[i][2], positions[j][2]]
        ax.plot(xs, ys, zs, color='black', linestyle='-', linewidth=2, alpha=0.6)

    # Render atoms
    for i in range(num_atoms):
        atom = mol_3d.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()

        # Atom size proportional to contribution magnitude
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

    # Configure axes
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set equal aspect ratio
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
    norm = plt.Normalize(vmin=-max_diff, vmax=max_diff)
    sm = cm.ScalarMappable(cmap=cm.bwr, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Hydrophobicity Shift (Blue=Hydrophilic, Red=Hydrophobic)',
                   rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path_3d = save_path.replace('.png', '_3d.png')
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')

    return fig


def visualize_molecule_with_weights(mol, atom_scalars, atom_contribs,
                                     title="Molecule with Per-Atom Weights",
                                     save_path=None, show_3d=True):
    """
    Visualize molecule with atoms colored by hydrophobicity adjustment.

    This function creates a comprehensive 2D visualization showing how the model
    adjusts each atom's contribution to the logP prediction relative to the RDKit
    baseline. The visualization includes the molecular structure with color-coded
    atoms and an analysis panel detailing the most significant adjustments.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    atom_scalars : numpy.ndarray
        Predicted scalar multipliers for each atom
    atom_contribs : numpy.ndarray
        RDKit Crippen logP contributions for each atom
    title : str, optional
        Plot title (default: "Molecule with Per-Atom Weights")
    save_path : str, optional
        Path to save the figure (default: None, display only)
    show_3d : bool, optional
        Whether to generate accompanying 3D visualization (default: True)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated 2D figure object
    fig_3d : matplotlib.figure.Figure or None
        Generated 3D figure object if show_3d=True, else None

    Notes
    -----
    Color scheme:
        - Blue: More hydrophilic than RDKit (Predicted < RDKit)
        - White: No change from RDKit (Predicted ≈ RDKit)
        - Red: More hydrophobic than RDKit (Predicted > RDKit)

    Atom size is proportional to the magnitude of the RDKit contribution,
    emphasizing atoms that have larger baseline effects on logP.
    """
    num_atoms = mol.GetNumAtoms()

    # Calculate hydrophobicity shift: (Predicted - RDKit)
    diffs = atom_contribs * (atom_scalars - 1.0)

    # Normalize for symmetric color mapping around zero
    max_diff = max(abs(diffs.max()), abs(diffs.min()))
    max_diff = max(0.1, max_diff)  # Ensure minimal range

    # Map to [0, 1] for blue-white-red colormap
    # -max_diff -> 0 (Blue), 0 -> 0.5 (White), +max_diff -> 1 (Red)
    norm_vals = 0.5 + (diffs) / (2 * max_diff)
    norm_vals = np.clip(norm_vals, 0, 1)

    # Apply colormap
    colors_rgb = plt.cm.bwr(norm_vals)

    # Configure atom highlights
    highlight_atoms = list(range(num_atoms))
    highlight_colors = {i: tuple(colors_rgb[i][:3]) for i in range(num_atoms)}

    # Set atom radii proportional to contribution magnitude
    atom_radii = {i: 0.3 + 0.2 * min(abs(atom_contribs[i]) / 0.5, 1.0)
                  for i in range(num_atoms)}

    # Generate 2D molecular drawing
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(800, 800)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.continuousHighlight = False   # Prevent highlight bleeding
    opts.fillHighlights = True         # Fill atom circles
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

    # Create figure with molecule and analysis panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     gridspec_kw={'width_ratios': [3, 1]})

    # Display molecular structure
    img = drawer.GetDrawingText()
    pil_img = Image.open(io.BytesIO(img))
    ax1.imshow(pil_img)
    ax1.axis('off')
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Create analysis panel
    ax2.axis('off')

    # Sort atoms by absolute difference magnitude
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

    # Generate 3D visualization if requested
    fig_3d = None
    if show_3d:
        title_3d = f"3D View: {title.split(':')[0] if ':' in title else title}"
        fig_3d = visualize_molecule_3d(mol, atom_scalars, atom_contribs,
                                        title=title_3d, save_path=save_path)

    return fig, fig_3d
