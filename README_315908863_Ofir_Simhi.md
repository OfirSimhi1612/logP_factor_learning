README_315908863_Ofir_Simhi
Ofir Simhi, ID: 315908863

================================================================================
PROJECT: Contextual Atom Scalar Model for Molecular Lipophilicity (logP) Prediction
================================================================================

DESCRIPTION
-----------
This project predicts molecular lipophilicity (logP) using a deep learning model
that learns per-atom scalar corrections to RDKit's Crippen-Wildman baseline.

The model uses message passing neural networks to create contextual atom embeddings,
then predicts scalar multipliers that adjust each atom's contribution based on the
entire molecular structure.

Prediction formula: logP = sum(rdkit_contrib[i] * scalar[i])


================================================================================
DATA FILES
================================================================================

Source: DeepChem Lipophilicity Dataset
URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv
Citation: MoleculeNet benchmark (Wu et al., 2018)
          "MoleculeNet: A Benchmark for Molecular Machine Learning"
          Chemical Science, 2018

The dataset contains ~14,000 molecules with experimental logP measurements.

Data is automatically downloaded when running the code for the first time.
It will be saved to: data/LogP.csv

Generated files (created automatically):
- data/cache/processed_molecules_cache.pkl  - Cached processed molecules
- data/splits/train.csv                     - Training set split
- data/splits/val.csv                       - Validation set split
- data/splits/test.csv                      - Test set split


================================================================================
CODE FILES - EXECUTION ORDER
================================================================================

PRIMARY WORKFLOW: Run main.ipynb from start to finish.

The notebook is organized into sections:
1. Setup and imports
2. Dataset analysis and visualization
3. Data splitting (train/val/test)
4. Baseline (Wildman-Crippen) evaluation
5. Message passing demonstration
6. Model training (or loading pre-trained model)
7. Model evaluation on test set
8. Per-atom contribution visualization

To retrain the model: rename checkpoints/prod.pt before running.


================================================================================
CODE FILES - DETAILED DESCRIPTIONS
================================================================================

main.ipynb
----------
Input: None (downloads data automatically)
Output: Trained model (checkpoints/), visualizations, evaluation metrics
Description: Main Jupyter notebook containing the complete workflow from data
loading through model training and evaluation. Run cells sequentially.


src/mp_graph/featurizer.py
--------------------------
Input: RDKit Mol object
Output: atom_features (N x 40), bond_features (N x N x 10), adjacency (N x N)
Description: Converts molecules into graph representations for neural networks.
- Atom features (40-dim): atom type, degree, charge, hybridization, aromaticity,
  hydrogen count, valence
- Bond features (10-dim): bond type, conjugation, ring membership, stereochemistry


src/mp_graph/mp_graph.py
------------------------
Input: adjacency matrix, atom features, bond features
Output: Contextualized atom embeddings (N x D) or graph-level vector (D,)
Description: Message Passing Neural Network that performs 3 rounds of message
passing to create atom embeddings that capture local chemical environment.
Each atom aggregates information from neighbors up to 3 bonds away.


src/mlp_regressor/mlp.py
------------------------
Input: Atom feature tensor (N x input_dim)
Output: Per-atom scalar predictions (N,)
Description: Contains three model architectures:
1. ContextualAtomScalarMLP - Main model. Predicts per-atom scalars using both
   local atom encoding and global molecular context (mean-pooled atoms).
2. AtomOnlyMLP - Baseline. Per-atom scalars from local features only.
3. ContextOnlyMLP - Baseline. Single logP from pooled molecular features.


src/mlp_regressor/training.py
-----------------------------
Input: Model, DataLoader, optimizer, criterion, device
Output: Training loss, evaluation metrics (predictions, targets, baselines)
Description: Training and evaluation utilities including:
- MoleculeDataset: PyTorch Dataset that groups atoms by molecule
- collate_molecules: Custom collate for variable-size molecules
- train_epoch: Single epoch training loop
- evaluate: Model evaluation returning predictions and metrics


src/utils/data.py
-----------------
Input: None (or DataFrame)
Output: DataLoaders for train/val/test
Description: Data loading and preprocessing pipeline:
- get_dataset(): Downloads Lipophilicity data if not present
- get_features_and_targets(): Processes molecules through MPNN, extracts features
- create_and_save_splits(): Creates reproducible train/val/test splits
- get_dataloaders(): Main entry point returning ready-to-use DataLoaders


src/utils/visualization.py
--------------------------
Input: RDKit Mol, atom scalars, atom contributions
Output: Matplotlib figures (2D and 3D molecular visualizations)
Description: Visualization utilities for interpreting model predictions:
- predict_atom_scalars(): Run inference on a molecule
- visualize_molecule_with_weights(): 2D molecule with color-coded atoms
- visualize_molecule_3d(): 3D conformer visualization
Color scheme: Blue = more hydrophilic, White = no change, Red = more hydrophobic


================================================================================
FILE STRUCTURE
================================================================================

logP_factor_learning/
├── main.ipynb                 # Primary workflow notebook - RUN THIS
├── README_315908863_Ofir_Simhi # This file
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
│
├── src/
│   ├── mp_graph/
│   │   ├── featurizer.py     # Molecule to graph conversion
│   │   └── mp_graph.py       # Message passing neural network
│   ├── mlp_regressor/
│   │   ├── mlp.py            # Model architectures
│   │   └── training.py       # Training and evaluation functions
│   └── utils/
│       ├── data.py           # Data loading and preprocessing
│       └── visualization.py  # Molecular visualization
│
├── data/                      # Created automatically
│   ├── LogP.csv              # Downloaded dataset
│   ├── cache/                # Processed molecule cache
│   └── splits/               # Train/val/test CSV files
│
└── checkpoints/
    └── prod.pt               # Pre-trained model weights


================================================================================
SETUP AND INSTALLATION
================================================================================

Option 1 - Conda (recommended):
    conda env create -f environment.yml
    conda activate logp_pred
    jupyter notebook main.ipynb

Option 2 - pip:
    pip install -r requirements.txt
    jupyter notebook main.ipynb


================================================================================
DEPENDENCIES
================================================================================

- Python 3.8+
- PyTorch
- RDKit
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- requests

See requirements.txt for exact versions.


================================================================================
NOTES
================================================================================

- All file paths in the code are relative to the project root
- Data is downloaded automatically on first run
- To retrain the model, rename or delete checkpoints/prod.pt
- Cache file (data/cache/) can be deleted to reprocess molecules
- Training takes ~5-10 minutes on CPU, ~2-3 minutes on GPU
