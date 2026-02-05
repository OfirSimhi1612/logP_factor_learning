# CLAUDE.md

## Project Overview

Deep learning project for predicting molecular lipophilicity (logP) by learning per-atom scalar corrections to RDKit's Crippen-Wildman baseline. The model uses message passing to create contextual atom embeddings, then predicts scalars that adjust each atom's contribution based on molecular context.

**Prediction formula**: `logP = sum(rdkit_contrib[i] * scalar[i])`

## Quick Start

```bash
# Setup
conda env create -f environment.yml
conda activate logp_pred

# Run
jupyter notebook main.ipynb
```

## Project Structure

```
├── main.ipynb              # Primary workflow notebook
├── src/
│   ├── mp_graph/
│   │   ├── featurizer.py   # Molecule → graph (40-dim atom, 10-dim bond features)
│   │   └── mp_graph.py     # Message passing (depth=3, no learnable params)
│   ├── mlp_regressor/
│   │   ├── mlp.py          # Three model variants (see below)
│   │   └── training.py     # Dataset, train_epoch, evaluate functions
│   └── utils/
│       ├── data.py         # Data loading, caching, splits
│       └── visualization.py # 2D/3D molecule visualization
├── data/
│   ├── LogP.csv            # ~14K molecules from DeepChem
│   ├── cache/              # Pickle cache for processed molecules
│   └── splits/             # train/val/test CSVs
└── checkpoints/
    └── prod.pt             # Production model weights
```

## Model Variants (src/mlp_regressor/mlp.py)

1. **ContextualAtomScalarMLP** (main model): Per-atom scalars conditioned on global molecular context via mean pooling
2. **AtomOnlyMLP**: Per-atom scalars from local features only (no context)
3. **ContextOnlyMLP**: Single logP prediction from pooled molecular features (no per-atom)

## Key Functions

```python
# Data loading
from src.utils.data import get_dataloaders, get_features_and_targets
train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

# Model
from src.mlp_regressor.mlp import ContextualAtomScalarMLP
model = ContextualAtomScalarMLP(input_dim=40, hidden_dims=[40, 40, 32])
model.load_state_dict(torch.load('checkpoints/prod.pt'))

# Visualization
from src.utils.visualization import predict_atom_scalars, visualize_molecule_with_weights
atom_scalars, atom_contribs = predict_atom_scalars(model, mol, device)
visualize_molecule_with_weights(mol, atom_scalars, atom_contribs, show_3d=True)
```

## Important Notes

- **Cache**: Delete `data/cache/` if you change featurization or message passing depth
- **Training**: To retrain, rename `checkpoints/prod.pt` to something else; otherwise the notebook loads it and skips training
- **Custom collate**: Molecules have variable atom counts, so DataLoader uses list batching via `collate_molecules`
- **SMILES cleaning**: Pipes (`|`) in SMILES are stripped during preprocessing

## Training Config (defaults in main.ipynb)

- Hidden layers: [40, 40, 32]
- Learning rate: 0.001 with ReduceLROnPlateau
- Batch size: 64
- Early stopping: patience=10
- Loss: MSE on molecule-level logP
