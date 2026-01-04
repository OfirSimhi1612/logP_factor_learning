import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from src.mp_graph.featurizer import Featurizer

class MessagePassingGraph(nn.Module):
    def __init__(self, depth: int = 3):
        """
        Args:
            depth: Number of message passing steps.
        """
        super().__init__()
        self.depth = depth
        
    def forward(self, 
                adj_matrix: torch.Tensor,
                atom_features: torch.Tensor, 
                bond_features: torch.Tensor, 
                readout_pooling: str = 'sum', 
                global_features: torch.Tensor = None,
                return_atoms: bool = False) -> torch.Tensor:
        """
        Runs message passing and returns a representation.
        
        Args:
            adj_matrix: (N, N) - Adjacency matrix.
            atom_features: (N, D_atom) - Input features.
            bond_features: (N, N, D_bond) - Input features.
            readout_pooling: 'sum', 'mean', or 'max'.
            global_features: Optional (G_dim,) tensor.
            return_atoms: If True, returns (N, d_max) tensor of atom embeddings.
            
        Returns:
            torch.Tensor: Either (N, d_max) or (d_max + G_dim,).
        """
        num_atoms = adj_matrix.shape[0]
        d_atom = atom_features.shape[1]
        d_bond = bond_features.shape[2]
        d_max = max(d_atom, d_bond)
        
        # Mask: (N, N, 1)
        mask = (adj_matrix > 0).float().unsqueeze(-1)
        
        # 0. Zero-Pad Inputs to Match Dimensions
        if d_atom < d_max:
            current_atoms = F.pad(atom_features, (0, d_max - d_atom))
        else:
            current_atoms = atom_features
            
        if d_bond < d_max:
            current_messages = F.pad(bond_features, (0, d_max - d_bond))
        else:
            current_messages = bond_features
        
        # Keep initial aligned bonds for residual
        initial_bonds = current_messages.clone()

        # Atom identity injection
        atom_broadcast = current_atoms.unsqueeze(1).expand(-1, num_atoms, -1)
        current_messages = (current_messages + atom_broadcast) * mask

        # --- Message Passing Loop ---
        for _ in range(self.depth):
            incoming_sum = torch.sum(current_messages, dim=0)
            current_atoms = current_atoms + incoming_sum
            total_broad = incoming_sum.unsqueeze(1).expand(-1, num_atoms, -1)
            reverse_messages = current_messages.transpose(0, 1)
            new_messages = total_broad - reverse_messages
            current_messages = (new_messages + initial_bonds + atom_broadcast) * mask

        # --- Output Selection ---
        if return_atoms:
            return current_atoms

        # --- Readout (Pool atoms only) ---
        if readout_pooling == 'sum':
            graph_feature = torch.sum(current_atoms, dim=0)
        elif readout_pooling == 'mean':
            graph_feature = torch.mean(current_atoms, dim=0)
        elif readout_pooling == 'max':
            graph_feature, _ = torch.max(current_atoms, dim=0)
        else:
            raise ValueError(f"Unsupported readout_pooling: {readout_pooling}")

        # Concatenate global features if provided
        if global_features is not None:
            graph_feature = torch.cat([graph_feature, global_features], dim=0)

        return graph_feature

# --- Usage Example ---
if __name__ == "__main__":
    smiles = "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"
    feturizer = Featurizer()
    atom_features, bond_features, adj = feturizer.featurize_molecule(Chem.MolFromSmiles(smiles))
    
    # Convert numpy to torch
    atom_features = torch.from_numpy(atom_features)
    bond_features = torch.from_numpy(bond_features)
    adj = torch.from_numpy(adj)
    
    # Initialize model (No dimension args needed!)
    cmpnn = MessagePassingGraph(depth=3)
    
    # Run forward pass
    # It will auto-pad smaller features to match larger ones
    vec1 = cmpnn(adj, atom_features, bond_features, readout_pooling='mean')
    print("Vector 1 Shape:", vec1) 

    global_f = torch.randn(10)
    vec2 = cmpnn(adj, atom_features, bond_features, readout_pooling='sum', global_features=global_f)
    print("Vector 2 Shape:", vec2.shape)