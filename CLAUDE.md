# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning research project focused on predicting molecular lipophilicity (logP) using a **Contextual Atom Scalar MLP** architecture. The key innovation is modeling each atom's contribution to logP as a function of both local atomic environment and global molecular context, rather than using fixed atomic contributions as in traditional methods (like RDKit's Crippen-Wildman).

The model learns scalar multipliers that adjust RDKit's baseline atomic contributions, conditioned on the entire molecular structure via message passing neural networks.

## Development Environment

### Setup
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate logp_pred

# Or using pip
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch: Neural network framework
- RDKit: Molecular featurization and baseline logP calculation
- NumPy, scikit-learn: Data processing
- Seaborn, matplotlib: Visualization

### Running the Project
The primary workflow is through the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

For standalone training/evaluation:
```bash
python main.py
```

## Architecture Deep Dive

### Three-Stage Pipeline

#### 1. Molecular Featurization (`src/mp_graph/`)
- **`featurizer.py`**: Converts RDKit molecules into graph representations
  - Atom features (44-dim): atom type, degree, charge, hybridization, aromaticity, hydrogens, valence
  - Bond features (10-dim): bond type, conjugation, ring membership, stereochemistry
  - Outputs: atom feature matrix, bond feature tensor, adjacency matrix

- **`mp_graph.py`**: Message passing neural network (MPNN)
  - Performs `depth=3` rounds of message passing to create contextualized atom embeddings
  - Each atom aggregates information from neighbors up to 3 bonds away
  - Zero-padding handles variable atom/bond feature dimensions
  - Can return either per-atom embeddings (`return_atoms=True`) or graph-level pooling

#### 2. Contextual Scalar Prediction (`src/mlp_regressor/mlp.py`)
The `ContextualAtomScalarMLP` has three components:

1. **Atom Encoder**: Projects MPNN embeddings through feed-forward layers
2. **Molecular Context Module**: Mean-pools atom encodings and extracts global features
3. **Scalar Predictor**: Concatenates local (atom encoding) + global (molecular context) to predict per-atom scalars

**Critical insight**: Each atom's scalar is conditioned on the entire molecule's structure, not just local environment.

Final prediction: `logP = sum(rdkit_contrib[i] * scalar[i])`

#### 3. Training & Evaluation (`src/mlp_regressor/training.py`)
- **Custom dataset**: `MoleculeDataset` handles variable-length molecules (different atom counts)
- **Custom collate**: Returns list of molecules (not batched tensors) since each molecule has different size
- **Loss**: MSE between predicted and experimental logP at molecule level
- **Evaluation metrics**: RMSE and MAE, stratified by molecular weight (MW > 500 Da for "large" molecules)

### Data Pipeline (`src/utils/data.py`)

**Key functions**:
- `get_dataset()`: Downloads Lipophilicity dataset from DeepChem S3 if not cached
- `get_features_and_targets()`:
  - Processes each molecule through MPNN featurizer
  - Extracts RDKit Crippen atomic contributions as baseline
  - **Uses pickle cache** (`data/cache/processed_molecules_cache.pkl`) to avoid reprocessing
- `create_and_save_splits()`: Saves train/val/test splits to `data/splits/`
- `get_dataloaders()`: Orchestrates the full pipeline

**Important**: The cache file is critical for performance. Delete it if you change featurization or MPNN depth.

### Visualization (`src/utils/visualization.py`)

- `get_atom_features_from_mol()`: Extract features for a single molecule
- `predict_atom_scalars()`: Run inference to get per-atom predictions
- `visualize_molecule_with_weights()`: Creates 2D visualization with atoms colored by hydrophobicity shift
  - Blue = More hydrophilic than RDKit (scalar < 1.0)
  - Red = More hydrophobic than RDKit (scalar > 1.0)
  - White = No change (scalar ≈ 1.0)
- `visualize_molecule_3d()`: 3D conformer with same coloring scheme

## File Organization

```
.
├── main.py                    # Standalone training script
├── main.ipynb                 # Interactive notebook (recommended workflow)
├── src/
│   ├── mlp_regressor/
│   │   ├── mlp.py            # ContextualAtomScalarMLP model
│   │   └── training.py       # Dataset, training loop, evaluation
│   ├── mp_graph/
│   │   ├── featurizer.py     # Molecule → graph conversion
│   │   └── mp_graph.py       # Message passing implementation
│   └── utils/
│       ├── data.py           # Data loading & preprocessing
│       └── visualization.py  # Molecule visualization tools
├── data/
│   ├── LogP.csv              # Downloaded dataset
│   ├── splits/               # Train/val/test CSVs
│   └── cache/                # Processed molecule cache (pickle)
└── checkpoints/
    └── prod.pt               # Trained model weights
```

## Common Development Tasks

### Training a New Model
```python
from src.utils.data import get_dataloaders
from src.mlp_regressor.mlp import ContextualAtomScalarMLP
from src.mlp_regressor.training import train_epoch, evaluate

# Configuration
HIDDEN_LAYERS = [40, 40, 32]
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Get data
train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

# Initialize model
sample_batch = next(iter(train_loader))
input_dim = sample_batch[0]['atom_features'].shape[1]
model = ContextualAtomScalarMLP(input_dim=input_dim, hidden_dims=HIDDEN_LAYERS)
```

### Loading a Trained Model
```python
model.load_state_dict(torch.load('checkpoints/prod.pt'))
model.eval()
```

### Visualizing Predictions
```python
from rdkit import Chem
from src.utils.visualization import predict_atom_scalars, visualize_molecule_with_weights

smiles = "CCO"  # Ethanol
mol = Chem.MolFromSmiles(smiles)
atom_scalars, atom_contribs = predict_atom_scalars(model, mol, device)
fig, fig_3d = visualize_molecule_with_weights(mol, atom_scalars, atom_contribs)
```

### Clearing Cache
If you modify featurization or MPNN:
```bash
rm -rf data/cache/
```

## Important Notes

### Data Handling
- SMILES strings may contain pipes (`|`) for stereochemistry - these are automatically stripped
- Molecular indices (`mol_index`) are critical for aligning atom-level data with molecule-level targets
- Dataset uses custom collate function because molecules have variable atom counts

### Model Training
- Early stopping with patience=10 epochs on validation MSE
- Model saves to `best_model_contextual.pt` (in main.py) or `checkpoints/prod.pt`
- Training loss is averaged per-molecule, not per-atom

### Evaluation Strategy
- Primary benchmark: RDKit Crippen-Wildman logP
- Stratify by molecular weight (MW > 500 Da) to assess performance on larger molecules where RDKit has higher variance
- Sanity checks in `main.py` verify data alignment between molecules, targets, and baselines

### Architecture Constraints
- MPNN depth=3 is hardcoded in data preprocessing (affects cache)
- Atom feature dimension (44) and bond feature dimension (10) are determined by featurizer
- Model input dimension is MPNN output dimension (max of atom/bond feature dims after padding)

### Device Handling
All training/evaluation code uses:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Research Context

The goal is to improve logP prediction accuracy on **large molecules (MW > 500 Da)** where traditional fragment-based methods struggle. The hypothesis is that atomic contributions are context-dependent and can be learned through neural networks that model molecular structure holistically.

Key experimental result to track: percentage RMSE improvement on large molecules compared to RDKit baseline.
