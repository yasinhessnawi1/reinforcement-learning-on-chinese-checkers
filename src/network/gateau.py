"""
gateau.py — Graph Attention Network (GATEAU) for Chinese Checkers.

Treats the hex board as a graph where:
  - Nodes = 121 board cells
  - Edges = hex adjacency (each cell connects to up to 6 neighbors)

Unlike ResNet (which treats the board as a 17x17 grid with many empty cells),
GATEAU operates directly on the hex topology — no wasted computation on
invalid cells, and natural handling of the star-shaped board.

Architecture based on:
  - Cazenave (2020) "Improving Monte Carlo Tree Search with Graph Neural Networks"
  - Veličković et al. (2018) "Graph Attention Networks (GAT)"

Each node feature vector contains:
  - Has agent pin (binary)
  - Has opponent pin (binary)
  - Is goal cell (binary)
  - Is home cell (binary)
  - Normalized axial coordinates (q, r)

The model applies N layers of graph attention (message-passing on hex edges),
then reads out:
  - Policy: per-pin attention over destination nodes
  - Value: global graph pooling → MLP → tanh

This is the pure-PyTorch implementation (no PyG dependency) for portability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_hex_adjacency(num_cells: int = 121) -> torch.Tensor:
    """Build adjacency list for the hex board.

    Returns edge_index: (2, num_edges) tensor of directed edges.
    Each cell connects to its 6 hex neighbors (if they exist).

    The hex board uses axial coordinates with R=4, giving 121 cells.
    Axial neighbors: (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    """
    # Build coordinate map from the actual board
    # Axial coordinates range from -8 to 8, but only 121 are valid
    # We'll build this lazily from the board the first time it's needed
    # For now, use a precomputed approach

    # Hex neighbor offsets in axial coordinates
    HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    # We need the actual board to map cell indices to axial coords
    # Import here to avoid circular dependency
    from src.env.board_wrapper import BoardWrapper
    bw = BoardWrapper(colours=["red", "blue"])
    board = bw.board

    # Build index -> (q, r) mapping
    idx_to_qr = {}
    qr_to_idx = {}
    for idx, cell in enumerate(board.cells):
        idx_to_qr[idx] = (cell.q, cell.r)
        qr_to_idx[(cell.q, cell.r)] = idx

    # Build edge list
    src_list = []
    dst_list = []
    for idx in range(num_cells):
        q, r = idx_to_qr[idx]
        for dq, dr in HEX_DIRS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in qr_to_idx:
                neighbor_idx = qr_to_idx[(nq, nr)]
                src_list.append(idx)
                dst_list.append(neighbor_idx)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


# Cached adjacency (built once, reused)
_CACHED_EDGE_INDEX: torch.Tensor | None = None
_CACHED_QR_COORDS: np.ndarray | None = None


def get_hex_graph() -> tuple[torch.Tensor, np.ndarray]:
    """Get cached hex adjacency and coordinate arrays.

    Returns (edge_index, qr_coords) where:
      edge_index: (2, num_edges) long tensor
      qr_coords: (121, 2) float array of normalized (q, r) coordinates
    """
    global _CACHED_EDGE_INDEX, _CACHED_QR_COORDS

    if _CACHED_EDGE_INDEX is None:
        _CACHED_EDGE_INDEX = build_hex_adjacency()

        from src.env.board_wrapper import BoardWrapper
        bw = BoardWrapper(colours=["red", "blue"])
        coords = np.zeros((121, 2), dtype=np.float32)
        for idx, cell in enumerate(bw.board.cells):
            coords[idx] = [cell.q / 8.0, cell.r / 8.0]  # normalize to [-1, 1]
        _CACHED_QR_COORDS = coords

    return _CACHED_EDGE_INDEX, _CACHED_QR_COORDS


class GATLayer(nn.Module):
    """Graph Attention Network layer (Veličković et al. 2018).

    Pure PyTorch implementation — no PyG dependency.

    Computes attention-weighted message passing on the hex graph:
      h_i' = sum_j alpha_ij * W * h_j
    where alpha_ij are learned attention coefficients.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        # Per-head projection
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)

        # Attention coefficients (per head)
        self.a_src = nn.Parameter(torch.Tensor(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(n_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, in_features) — node features
        edge_index : (2, E) — edge indices (shared across batch)

        Returns
        -------
        (B, N, out_features * n_heads) if concat, else (B, N, out_features)
        """
        B, N, _ = x.shape
        H = self.n_heads
        D = self.out_features

        # Project all nodes: (B, N, H*D)
        h = self.W(x).reshape(B, N, H, D)  # (B, N, H, D)

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        # Compute attention scores
        # a_src * h_src + a_dst * h_dst for each edge
        h_src = h[:, src_idx]  # (B, E, H, D)
        h_dst = h[:, dst_idx]  # (B, E, H, D)

        # Attention: (B, E, H)
        e_src = (h_src * self.a_src.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        e_dst = (h_dst * self.a_dst.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)  # (B, E, H)

        # Softmax over neighbors (scatter-based)
        # For efficiency, compute per-destination softmax
        alpha = self._edge_softmax(e, dst_idx, N)  # (B, E, H)
        alpha = self.dropout(alpha)

        # Message passing: sum of attention-weighted messages
        messages = alpha.unsqueeze(-1) * h_src  # (B, E, H, D)
        out = torch.zeros(B, N, H, D, device=x.device)
        dst_expanded = dst_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, D)
        out.scatter_add_(1, dst_expanded, messages)

        if self.concat:
            return out.reshape(B, N, H * D)
        else:
            return out.mean(dim=2)  # (B, N, D)

    def _edge_softmax(self, e: torch.Tensor, dst_idx: torch.Tensor, N: int) -> torch.Tensor:
        """Compute softmax of edge scores grouped by destination node.

        Parameters
        ----------
        e : (B, E, H) — raw attention scores
        dst_idx : (E,) — destination node indices
        N : int — number of nodes
        """
        B, E, H = e.shape

        # Max per destination for numerical stability
        max_e = torch.full((B, N, H), -1e9, device=e.device)
        dst_exp = dst_idx.unsqueeze(0).unsqueeze(-1).expand(B, E, H)
        max_e.scatter_reduce_(1, dst_exp, e, reduce='amax', include_self=False)
        e_shifted = e - max_e.gather(1, dst_exp)

        # Exp and sum per destination
        exp_e = torch.exp(e_shifted)
        sum_exp = torch.zeros(B, N, H, device=e.device)
        sum_exp.scatter_add_(1, dst_exp, exp_e)

        # Normalize
        return exp_e / (sum_exp.gather(1, dst_exp) + 1e-10)


class GATEAUBlock(nn.Module):
    """GATEAU residual block: GAT + skip connection + layer norm."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads

        self.gat = GATLayer(d_model, head_dim, n_heads, dropout, concat=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # GAT + residual
        x = x + self.gat(self.norm1(x), edge_index)
        # FF + residual
        x = x + self.ff(self.norm2(x))
        return x


class GATEAUNet(nn.Module):
    """GATEAU Network for Chinese Checkers.

    Graph attention on the 121-node hex board with policy and value heads.

    Per-node features (extracted from grid observation):
      - has_agent_pin: binary
      - has_opponent_pin: binary
      - is_goal: binary
      - is_home: binary
      - q_coord, r_coord: normalized axial coordinates

    Parameters
    ----------
    d_model : int
        Hidden dimension per node.
    n_heads : int
        GAT attention heads.
    n_layers : int
        Number of GATEAU blocks.
    dropout : float
    num_cells : int
        Board cells (121).
    num_pins : int
        Pins per player (10).
    num_actions : int
        Action space (1210).
    use_auxiliary_head : bool
        Predict pins_in_goal as auxiliary.
    """

    BOARD_SIZE: int = 17  # compatibility with AlphaZeroNet wrapper

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        num_cells: int = 121,
        num_pins: int = 10,
        num_actions: int = 1210,
        use_auxiliary_head: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cells = num_cells
        self.num_pins = num_pins
        self.num_actions = num_actions
        self.use_auxiliary_head = use_auxiliary_head

        # Node feature embedding (6 features → d_model)
        self.node_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # GATEAU blocks
        self.blocks = nn.ModuleList([
            GATEAUBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Policy head: for each action (pin_i, dest_j), compute score
        # using pin node representation and destination node representation
        self.policy_pin_proj = nn.Linear(d_model, d_model, bias=False)
        self.policy_dest_proj = nn.Linear(d_model, d_model, bias=False)
        self.policy_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model)))

        # Value head: global graph pooling
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

        # Auxiliary head
        if use_auxiliary_head:
            self.aux_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        # Register hex graph (not a parameter, but persisted)
        edge_index, qr_coords = get_hex_graph()
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('qr_coords', torch.tensor(qr_coords))

    def _build_grid_to_cell_map(self) -> None:
        """Precompute mapping from cell index to grid (row, col). Called once."""
        if hasattr(self, '_cell_rows'):
            return
        qr = self.qr_coords  # (121, 2)
        cols = torch.round(qr[:, 0] * 8 + 8).long()  # (121,)
        rows = torch.round(qr[:, 1] * 8 + 8).long()  # (121,)
        # Clamp to valid grid range
        rows = rows.clamp(0, 16)
        cols = cols.clamp(0, 16)
        self.register_buffer('_cell_rows', rows)
        self.register_buffer('_cell_cols', cols)

    def _extract_node_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert grid observation to per-node features.

        Fully vectorized — uses precomputed cell-to-grid mapping for
        fast indexed gathering. No Python loops.

        Parameters
        ----------
        obs : (B, 10, 17, 17) — grid observation

        Returns
        -------
        (B, 121, 6) — per-node feature vectors
        """
        self._build_grid_to_cell_map()

        B = obs.shape[0]
        device = obs.device

        agent_map = obs[:, 0]   # (B, 17, 17)
        opp_map = obs[:, 1]
        goal_map = obs[:, 2]
        home_map = obs[:, 3]

        rows = self._cell_rows  # (121,)
        cols = self._cell_cols  # (121,)
        qr = self.qr_coords    # (121, 2)

        # Gather features for all 121 cells at once using advanced indexing
        # obs[:, ch, rows, cols] gives (B, 121) for each channel
        has_agent = agent_map[:, rows, cols]   # (B, 121)
        has_opp = opp_map[:, rows, cols]       # (B, 121)
        is_goal = goal_map[:, rows, cols]      # (B, 121)
        is_home = home_map[:, rows, cols]      # (B, 121)

        # Coordinates: broadcast (121, 2) across batch
        q_coords = qr[:, 0].unsqueeze(0).expand(B, -1)  # (B, 121)
        r_coords = qr[:, 1].unsqueeze(0).expand(B, -1)  # (B, 121)

        # Stack: (B, 121, 6)
        features = torch.stack([has_agent, has_opp, is_goal, is_home, q_coords, r_coords], dim=-1)

        return features

    def _find_agent_pin_nodes(self, node_features: torch.Tensor) -> torch.Tensor:
        """Find which nodes contain agent pins. Vectorized.

        Returns (B, 10) tensor of node indices for agent pins.
        Uses topk on the has_agent_pin channel.
        """
        # has_agent_pin is channel 0: (B, 121)
        pin_values = node_features[:, :, 0]  # (B, 121)
        # topk gives us the indices of the top-10 values (the pin positions)
        _, pin_nodes = torch.topk(pin_values, k=self.num_pins, dim=1)  # (B, 10)
        return pin_nodes

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass compatible with AlphaZeroNet wrapper.

        Parameters
        ----------
        obs : (B, 10, 17, 17) — grid observation

        Returns
        -------
        policy_logits : (B, 1210)
        value : (B, 1)
        """
        B = obs.shape[0]
        device = obs.device

        # Extract per-node features
        node_feats = self._extract_node_features(obs)  # (B, 121, 6)

        # Encode nodes
        h = self.node_encoder(node_feats)  # (B, 121, d_model)

        # Move edge_index to correct device
        edge_idx = self.edge_index.to(device)

        # GATEAU message-passing layers
        for block in self.blocks:
            h = block(h, edge_idx)

        h = self.final_norm(h)  # (B, 121, d_model)

        # --- Policy head ---
        # Find agent pin node indices
        pin_nodes = self._find_agent_pin_nodes(node_feats)  # (B, 10)

        # Gather pin representations
        pin_idx = pin_nodes.unsqueeze(-1).expand(-1, -1, self.d_model)
        pin_reprs = torch.gather(h, 1, pin_idx)  # (B, 10, d_model)

        # Score each (pin, destination) pair
        pin_proj = self.policy_pin_proj(pin_reprs)   # (B, 10, d_model)
        dest_proj = self.policy_dest_proj(h)          # (B, 121, d_model)

        # (B, 10, d_model) @ (B, d_model, 121) = (B, 10, 121)
        logits = torch.bmm(pin_proj, dest_proj.transpose(1, 2)) * self.policy_scale
        policy_logits = logits.reshape(B, -1)  # (B, 1210)

        # --- Value head ---
        # Global mean pooling over all nodes
        graph_repr = h.mean(dim=1)  # (B, d_model)
        value = self.value_head(graph_repr)  # (B, 1)

        return policy_logits, value

    def forward_with_auxiliary(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with auxiliary pins_in_goal prediction."""
        B = obs.shape[0]
        device = obs.device

        node_feats = self._extract_node_features(obs)
        h = self.node_encoder(node_feats)
        edge_idx = self.edge_index.to(device)

        for block in self.blocks:
            h = block(h, edge_idx)
        h = self.final_norm(h)

        pin_nodes = self._find_agent_pin_nodes(node_feats)
        pin_idx = pin_nodes.unsqueeze(-1).expand(-1, -1, self.d_model)
        pin_reprs = torch.gather(h, 1, pin_idx)

        pin_proj = self.policy_pin_proj(pin_reprs)
        dest_proj = self.policy_dest_proj(h)
        logits = torch.bmm(pin_proj, dest_proj.transpose(1, 2)) * self.policy_scale
        policy_logits = logits.reshape(B, -1)

        graph_repr = h.mean(dim=1)
        value = self.value_head(graph_repr)

        aux_pred = torch.zeros(B, 1, device=device)
        if self.use_auxiliary_head:
            aux_pred = self.aux_head(graph_repr)

        return policy_logits, value, aux_pred
