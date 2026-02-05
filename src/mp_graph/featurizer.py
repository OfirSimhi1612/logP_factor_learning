"""
Atom featurizer for molecular graphs.
Extracts features from RDKit molecules for use in message passing.
"""

import numpy as np
from rdkit import Chem


class Featurizer:
    """
    Molecular featurizer that converts RDKit molecules into graph representations.

    Extracts atom features (40-dim), bond features (10-dim), and adjacency matrices
    for use in message passing neural networks.
    """

    def __init__(self):
        # Define feature vocabularies
        self.atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Other"]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]

        # feature dimensions
        self.atom_type_dim = len(self.atom_types)
        self.degree_dim = 6  # 0-5
        self.charge_dim = 5  # -2, -1, 0, 1, 2
        self.hybrid_dim = len(self.hybridizations)
        self.aromatic_dim = 1
        self.h_dim = 5  # 0-4
        self.valence_dim = 7  # 0-6

        self.feature_dim = (
            self.atom_type_dim
            + self.degree_dim
            + self.charge_dim
            + self.hybrid_dim
            + self.aromatic_dim
            + self.h_dim
            + self.valence_dim
        )

    def one_hot(self, value, vocab):
        """Create one-hot encoding"""
        vec = np.zeros(len(vocab))
        if value in vocab:
            vec[vocab.index(value)] = 1
        else:
            vec[-1] = 1  # Use last index for unknown
        return vec

    def featurize_atom(self, atom):
        """
        Featurize a single atom.

        Args:
            atom: RDKit Atom object

        Returns:
            features: numpy array of atom features
        """
        features = []

        # Atom type (one-hot)
        symbol = atom.GetSymbol()
        if symbol not in self.atom_types[:-1]:
            symbol = "Other"
        features.extend(self.one_hot(symbol, self.atom_types))

        # Degree (one-hot, capped at 5)
        degree = min(atom.GetDegree(), 5)
        features.extend(self.one_hot(degree, list(range(6))))

        # Formal charge (one-hot)
        charge = atom.GetFormalCharge()
        charge = max(-2, min(2, charge))  # Clamp to [-2, 2]
        features.extend(self.one_hot(charge, [-2, -1, 0, 1, 2]))

        # Hybridization (one-hot)
        hybrid = atom.GetHybridization()
        features.extend(self.one_hot(hybrid, self.hybridizations))

        # Aromaticity (binary)
        features.append(float(atom.GetIsAromatic()))

        # Number of hydrogens (one-hot, capped at 4)
        num_h = min(atom.GetTotalNumHs(), 4)
        features.extend(self.one_hot(num_h, list(range(5))))

        # Valence (one-hot, capped at 6)
        valence = min(atom.GetTotalValence(), 6)
        features.extend(self.one_hot(valence, list(range(7))))

        return np.array(features, dtype=np.float32)

    def featurize_atoms(self, mol):
        """
        Featurize all atoms in a molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            atom_features: numpy array of shape (num_atoms, feature_dim)
        """
        if mol is None:
            raise ValueError("Invalid molecule")

        num_atoms = mol.GetNumAtoms()
        atom_features = np.zeros((num_atoms, self.feature_dim), dtype=np.float32)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features[i] = self.featurize_atom(atom)

        return atom_features

    def get_adjacency_matrix(self, mol):
        """
        Get adjacency matrix from molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            adj: numpy array of shape (num_atoms, num_atoms)
                 adj[i,j] = 1 if there's a bond from i to j
        """
        num_atoms = mol.GetNumAtoms()
        adj = np.zeros((num_atoms, num_atoms), dtype=np.float32)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Undirected graph - add both directions
            adj[i, j] = 1
            adj[j, i] = 1

        return adj

    def featurize_bond(self, bond):
        """
        Featurize a single bond.

        Features:
        - Bond type (4): SINGLE, DOUBLE, TRIPLE, AROMATIC
        - Conjugated (1): binary
        - In ring (1): binary
        - Stereo (4): STEREONONE, STEREOZ, STEREOE, STEREOANY

        Total: 10 dimensions

        Args:
            bond: RDKit Bond object

        Returns:
            features: numpy array of bond features (10-dimensional)
        """
        features = []

        # Bond type (one-hot)
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        features.extend(self.one_hot(bond.GetBondType(), bond_types))

        # Conjugated (binary)
        features.append(float(bond.GetIsConjugated()))

        # In ring (binary)
        features.append(float(bond.IsInRing()))

        # Stereo (one-hot)
        stereo_types = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOANY,
        ]
        features.extend(self.one_hot(bond.GetStereo(), stereo_types))

        return np.array(features, dtype=np.float32)

    def get_bond_features(self, mol):
        """
        Get dense bond features tensor.

        Args:
            mol: RDKit Mol object

        Returns:
            bond_tensor: numpy array of shape (num_atoms, num_atoms, bond_feature_dim)
        """
        num_atoms = mol.GetNumAtoms()
        # Get feature dimension from a dummy bond or hardcoded
        # We know it's 10 from featurize_bond
        bond_dim = 10

        bond_tensor = np.zeros((num_atoms, num_atoms, bond_dim), dtype=np.float32)

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            feat = self.featurize_bond(bond)

            # Add features to both directions (undirected graph)
            bond_tensor[u, v] = feat
            bond_tensor[v, u] = feat

        return bond_tensor

    def featurize_molecule(self, mol):
        """
        Featurize a complete molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            atom_features: numpy array of shape (num_atoms, atom_feature_dim)
            bond_features: numpy array of shape (num_atoms, num_atoms, bond_feature_dim)
            adj: numpy array of shape (num_atoms, num_atoms)
        """
        atom_features = self.featurize_atoms(mol)
        bond_features = self.get_bond_features(mol)
        adj = self.get_adjacency_matrix(mol)
        return atom_features, bond_features, adj
