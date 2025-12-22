import numpy as np
from typing import Tuple


class MessagePassingGraph:
    def __init__(self, atom_features: np.ndarray, bond_features: np.ndarray, adj_matrix: np.ndarray,
                 num_timesteps: int = 3, bond_message_mode: str = 'tiling'):
        """
        Message Passing Graph with bond-aware communication.

        Args:
            atom_features: Shape (num_atoms, atom_feature_dim)
            bond_features: Shape (num_bonds, bond_feature_dim)
            adj_matrix: Shape (num_atoms, num_atoms)
            num_timesteps: Number of message passing iterations
            bond_message_mode: How to use bonds in messages
                - 'tiling': Repeat bond pattern (good for one-hot)
                - 'statistics': Use mean/std as scalars
                - 'attention': Normalize as probability weights
                - 'truncate': Simple dimension matching
        """
        self.num_atoms = atom_features.shape[0]
        self.atom_feature_dim = atom_features.shape[1]
        self.num_bonds = bond_features.shape[0]
        self.bond_feature_dim = bond_features.shape[1]
        self.num_timesteps = num_timesteps
        self.bond_message_mode = bond_message_mode

        self.initial_atoms_features = atom_features.copy()
        self.initial_bonds_features = bond_features.copy()

        self.atoms_hidden_states = atom_features.copy()
        self.bond_states = bond_features.copy()
        self.adj = adj_matrix.copy()

        # Initialize edges and mappings
        # Each undirected bond creates 2 directed edges (to prevent echoing)
        self.edges = []
        self.edge_to_bond_map = {}  # edge_idx -> bond_idx
        self.bond_to_atoms = {}     # bond_idx -> (atom_i, atom_j)
        self.edge_to_reverse_edge = {}  # edge_idx -> reverse_edge_idx (for anti-echo)

        bond_idx = 0
        for i in range(self.num_atoms):
            for j in range(i+1, self.num_atoms):  # Upper triangle only
                if self.adj[i, j] > 0:
                    self.bond_to_atoms[bond_idx] = (i, j)

                    # Create 2 directed edges for this bond
                    edge_fwd = len(self.edges)
                    self.edges.append((i, j))
                    self.edge_to_bond_map[edge_fwd] = bond_idx

                    edge_bwd = len(self.edges)
                    self.edges.append((j, i))
                    self.edge_to_bond_map[edge_bwd] = bond_idx

                    # Map reverse edges (for anti-echo mechanism)
                    self.edge_to_reverse_edge[edge_fwd] = edge_bwd
                    self.edge_to_reverse_edge[edge_bwd] = edge_fwd

                    bond_idx += 1

        self.messages = [{} for _ in range(num_timesteps)]

        # Initialize edge messages for anti-echo mechanism
        # Store what each edge sends (used to subtract reverse messages)
        num_edges = len(self.edges)
        self.edge_messages = np.zeros((num_edges, self.atom_feature_dim), dtype=np.float32)

        # Initialize with bond-modulated destination atom features
        for edge_idx, (src, dst) in enumerate(self.edges):
            bond_idx = self.edge_to_bond_map[edge_idx]
            bond_feats = self.bond_states[bond_idx]

            # Initial edge message: destination atom modulated by bond
            dest_state = self.atoms_hidden_states[dst].copy()
            self.edge_messages[edge_idx] = self._apply_bond_modulation(dest_state, bond_feats)

    def _apply_bond_modulation(self, atom_msg: np.ndarray, bond_feats: np.ndarray) -> np.ndarray:
        """
        Apply bond feature modulation to atom message.

        Args:
            atom_msg: Atom features/message
            bond_feats: Bond features

        Returns:
            Bond-modulated message
        """
        if self.bond_message_mode == 'tiling':
            if self.bond_feature_dim <= self.atom_feature_dim:
                repeats = (self.atom_feature_dim // self.bond_feature_dim) + 1
                bond_weights = np.tile(bond_feats, repeats)[:self.atom_feature_dim]
            else:
                bond_weights = bond_feats[:self.atom_feature_dim]
            return atom_msg * (1.0 + bond_weights)

        elif self.bond_message_mode == 'statistics':
            bond_mean = np.mean(bond_feats)
            bond_std = np.std(bond_feats)
            return atom_msg * (1.0 + bond_mean) * (1.0 + 0.1 * bond_std)

        elif self.bond_message_mode == 'attention':
            bond_attention = bond_feats / (np.sum(np.abs(bond_feats)) + 1e-8)
            if self.bond_feature_dim <= self.atom_feature_dim:
                repeats = (self.atom_feature_dim // self.bond_feature_dim) + 1
                bond_weights = np.tile(bond_attention, repeats)[:self.atom_feature_dim]
            else:
                bond_weights = bond_attention[:self.atom_feature_dim]
            return atom_msg * (1.0 + bond_weights)

        elif self.bond_message_mode == 'truncate':
            if self.bond_feature_dim <= self.atom_feature_dim:
                bond_weights = np.zeros(self.atom_feature_dim)
                bond_weights[:self.bond_feature_dim] = bond_feats
            else:
                bond_weights = bond_feats[:self.atom_feature_dim]
            return atom_msg * (1.0 + bond_weights)

        else:
            raise ValueError(f"Unknown bond_message_mode: {self.bond_message_mode}")

    def communicate(self, timestep: int):
        """
        Send messages along edges with anti-echo mechanism.

        Anti-echo: For edge src->dst, we propagate dst's info to src,
        but subtract what dst previously sent to src to avoid echoing.

        Args:
            timestep: Current timestep
        """
        current_messages = {}

        for edge_idx, (src, dst) in enumerate(self.edges):
            # Step 1: Get destination atom's current state
            # (we're propagating dst's info to src)
            dest_state = self.atoms_hidden_states[dst].copy()

            # Step 2: ANTI-ECHO - subtract what dst already told src
            # Get the reverse edge (dst->src) message from previous iteration
            rev_edge_idx = self.edge_to_reverse_edge[edge_idx]
            prev_reverse_msg = self.edge_messages[rev_edge_idx]

            # Simple subtraction - this is the key to anti-echo!
            echo_removed = dest_state - prev_reverse_msg

            # Step 3: Apply bond modulation to the echo-removed message
            bond_idx = self.edge_to_bond_map[edge_idx]
            bond_feats = self.bond_states[bond_idx]
            message = self._apply_bond_modulation(echo_removed, bond_feats)

            current_messages[(src, dst)] = message

        self.messages[timestep] = current_messages

    def aggregate(self, timestep: int) -> np.ndarray:
        """
        Aggregation step: Each atom aggregates messages from its neighbors.

        For each atom, we aggregate incoming messages (messages sent TO this atom).
        This prevents echoing because we only consider directed edges.

        Args:
            timestep: Current timestep

        Returns:
            aggregated_messages: Shape (num_atoms, atom_feature_dim)
                                Aggregated messages for each atom
        """
        aggregated = np.zeros((self.num_atoms, self.atom_feature_dim), dtype=np.float32)

        current_messages = self.messages[timestep]

        for atom_idx in range(self.num_atoms):
            # Collect all messages sent TO this atom
            incoming_messages = []

            for (src, dst), message in current_messages.items():
                if dst == atom_idx:
                    incoming_messages.append(message)

            # Aggregate using sum (could also use mean, max, etc.)
            if len(incoming_messages) > 0:
                # Sum aggregation
                aggregated[atom_idx] = np.sum(incoming_messages, axis=0)
                # Normalize by number of neighbors
                aggregated[atom_idx] /= len(incoming_messages)

        return aggregated

    def update(self, aggregated_messages: np.ndarray):
        """
        Update step: Update each atom's hidden state based on aggregated messages.

        Non-learnable update: simple addition/concatenation of current state
        and incoming messages.

        Args:
            aggregated_messages: Aggregated messages, shape (num_atoms, atom_feature_dim)
        """
        # Simple non-learnable update: add aggregated messages to current state
        # and normalize
        self.atoms_hidden_states = self.atoms_hidden_states + aggregated_messages

        # Normalize to prevent explosion
        norms = np.linalg.norm(self.atoms_hidden_states, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        self.atoms_hidden_states = self.atoms_hidden_states / norms

    def update_bond_states(self):
        """
        Update bond states based on connected atom states.
        Non-learnable aggregation: mean of endpoint atoms with residual connection.
        """
        for bond_idx in range(self.num_bonds):
            atom_i, atom_j = self.bond_to_atoms[bond_idx]

            # Mean of endpoint atoms (truncate to bond dimension)
            atom_avg = (self.atoms_hidden_states[atom_i] +
                       self.atoms_hidden_states[atom_j]) / 2.0
            atom_avg_truncated = atom_avg[:self.bond_feature_dim]

            # Residual connection: preserve initial bond features
            alpha = 0.5
            new_bond_state = (alpha * self.initial_bonds_features[bond_idx] +
                            (1 - alpha) * atom_avg_truncated)

            # Normalize
            norm = np.linalg.norm(new_bond_state)
            if norm > 0:
                new_bond_state = new_bond_state / norm

            self.bond_states[bond_idx] = new_bond_state

    def update_edge_messages(self):
        """
        Update edge messages after atom states have been updated.

        Remember what each edge just sent for the anti-echo mechanism next iteration.
        """
        for edge_idx, (src, dst) in enumerate(self.edges):
            # Get current destination state
            dest_state = self.atoms_hidden_states[dst].copy()

            # Apply bond modulation
            bond_idx = self.edge_to_bond_map[edge_idx]
            bond_feats = self.bond_states[bond_idx]

            # Store for next iteration (no anti-echo here - just store what we're sending)
            self.edge_messages[edge_idx] = self._apply_bond_modulation(dest_state, bond_feats)

    def run(self) -> np.ndarray:
        """
        Run message passing with anti-echo mechanism.

        Returns:
            final_features: Final atom features after message passing,
                        shape (num_atoms, atom_feature_dim)
        """
        for t in range(self.num_timesteps):
            # 1. Communicate: send anti-echo messages along directed edges
            self.communicate(t)

            # 2. Aggregate: collect incoming messages for each atom
            aggregated = self.aggregate(t)

            # 3. Update atoms: update hidden states based on messages
            self.update(aggregated)

            # 4. Update bonds: synchronize bond states with connected atoms
            self.update_bond_states()

            # 5. Update edge messages for next iteration (anti-echo mechanism)
            self.update_edge_messages()

        return self.atoms_hidden_states

    def get_readout(self, method: str = 'sum') -> np.ndarray:
        """
        Graph-level readout: aggregate all atom features to get graph representation.

        Args:
            method: Aggregation method ('sum', 'mean', 'max')

        Returns:
            graph_features: Graph-level feature vector (atom_feature_dim,)
        """
        if method == 'sum':
            return np.sum(self.atoms_hidden_states, axis=0)
        elif method == 'mean':
            return np.mean(self.atoms_hidden_states, axis=0)
        elif method == 'max':
            return np.max(self.atoms_hidden_states, axis=0)
        else:
            raise ValueError(f"Unknown readout method: {method}")

    def get_bond_aware_readout(self,
                              atom_method: str = 'mean',
                              bond_method: str = 'mean',
                              combination: str = 'concat') -> np.ndarray:
        """
        Bond-aware graph readout combining atom and bond information.

        Args:
            atom_method: 'sum', 'mean', 'max'
            bond_method: 'sum', 'mean', 'max'
            combination: How to combine atom and bond representations
                - 'concat': Concatenate without padding (atom_dim + bond_dim)
                - 'sum': Element-wise sum (requires padding to atom_dim)
                - 'product': Element-wise product (requires padding to atom_dim)
                - 'all': All combinations for maximum expressiveness

        Returns:
            graph_features: Graph-level feature vector
        """
        # Aggregate atoms
        if atom_method == 'sum':
            atom_repr = np.sum(self.atoms_hidden_states, axis=0)
        elif atom_method == 'mean':
            atom_repr = np.mean(self.atoms_hidden_states, axis=0)
        elif atom_method == 'max':
            atom_repr = np.max(self.atoms_hidden_states, axis=0)
        else:
            raise ValueError(f"Unknown atom_method: {atom_method}")

        # Aggregate bonds
        if self.num_bonds > 0:
            if bond_method == 'sum':
                bond_repr = np.sum(self.bond_states, axis=0)
            elif bond_method == 'mean':
                bond_repr = np.mean(self.bond_states, axis=0)
            elif bond_method == 'max':
                bond_repr = np.max(self.bond_states, axis=0)
            else:
                raise ValueError(f"Unknown bond_method: {bond_method}")
        else:
            # No bonds: return atom-only representation
            return atom_repr

        # Combine based on method
        if combination == 'concat':
            # Simply concatenate - NO padding needed!
            # Output: (atom_feature_dim + bond_feature_dim,)
            return np.concatenate([atom_repr, bond_repr])

        elif combination == 'sum' or combination == 'product':
            # For element-wise ops, need dimension alignment via padding
            if len(bond_repr) < len(atom_repr):
                bond_repr_aligned = np.zeros(len(atom_repr))
                bond_repr_aligned[:len(bond_repr)] = bond_repr
            else:
                bond_repr_aligned = bond_repr[:len(atom_repr)]

            if combination == 'sum':
                return atom_repr + bond_repr_aligned
            else:  # product
                return atom_repr * bond_repr_aligned

        elif combination == 'all':
            # Maximum expressiveness: combine multiple ways
            # Align for element-wise ops
            if len(bond_repr) < len(atom_repr):
                bond_aligned = np.zeros(len(atom_repr))
                bond_aligned[:len(bond_repr)] = bond_repr
            else:
                bond_aligned = bond_repr[:len(atom_repr)]

            return np.concatenate([
                atom_repr,                              # Atom-only
                bond_repr,                              # Bond-only (original dim)
                atom_repr + bond_aligned,               # Sum (aligned)
                atom_repr * bond_aligned,               # Product (aligned)
                np.maximum(atom_repr, bond_aligned)     # Max (aligned)
            ])
        else:
            raise ValueError(f"Unknown combination: {combination}")
        
