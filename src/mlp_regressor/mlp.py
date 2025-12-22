import torch
import torch.nn as nn


class ContextualAtomScalarMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(ContextualAtomScalarMLP, self).__init__()

        atom_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            atom_layers.append(nn.Linear(prev_dim, hidden_dim))
            atom_layers.append(nn.ReLU())
            atom_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.atom_encoder = nn.Sequential(*atom_layers)
        atom_encoding_dim = prev_dim

        self.mol_context = nn.Sequential(
            nn.Linear(atom_encoding_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        mol_context_dim = hidden_dims[-1]

        self.scalar_predictor = nn.Sequential(
            nn.Linear(atom_encoding_dim + mol_context_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, atom_features):
        """
        Args:
            atom_features: (num_atoms, input_dim) - all atoms in a molecule
        Returns:
            scalars: (num_atoms,) - one scalar per atom, context-aware!
        """
        atom_encodings = self.atom_encoder(atom_features)  # (num_atoms, encoding_dim)

        mol_context = torch.mean(atom_encodings, dim=0, keepdim=True)  # (1, encoding_dim)
        mol_context = self.mol_context(mol_context)  # (1, context_dim)

        mol_context_broadcast = mol_context.expand(atom_encodings.size(0), -1)  # (num_atoms, context_dim)

        atom_with_context = torch.cat([atom_encodings, mol_context_broadcast], dim=1)  # (num_atoms, encoding_dim + context_dim)

        scalars = self.scalar_predictor(atom_with_context)  # (num_atoms, 1)

        return scalars.squeeze(-1)  # (num_atoms,)
