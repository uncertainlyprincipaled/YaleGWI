import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EGNNLayer(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network layer.
    Handles receiver geometry and maintains SE(2/3) equivariance.
    """
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 coords_dim: int = 2,
                 act_fn: nn.Module = nn.SiLU(),
                 attention: bool = True):
        """
        Initialize EGNN layer.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for MLPs
            coords_dim: Dimension of coordinate space (2 for 2D, 3 for 3D)
            act_fn: Activation function
            attention: Whether to use attention mechanism
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.coords_dim = coords_dim
        self.act_fn = act_fn
        self.attention = attention
        
        # Node feature MLPs
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Edge feature MLPs
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim + 1, hidden_dim),  # +1 for distance
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, coords_dim)
        )
        
        # Attention mechanism
        if attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, 
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of EGNN layer.
        
        Args:
            h: Node features [num_nodes, node_dim]
            x: Node coordinates [num_nodes, coords_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Tuple of (updated_node_features, updated_coordinates)
        """
        num_nodes = h.shape[0]
        num_edges = edge_index.shape[1]
        
        # Initialize edge features if not provided
        if edge_attr is None:
            edge_attr = torch.zeros(num_edges, self.edge_dim, device=h.device)
        
        # Get source and target nodes
        row, col = edge_index
        
        # Compute relative distances
        rel_x = x[row] - x[col]
        dist = torch.norm(rel_x, dim=-1, keepdim=True)
        
        # Concatenate features for edge MLP
        edge_input = torch.cat([
            edge_attr,
            h[row],
            h[col],
            dist
        ], dim=-1)
        
        # Process edge features
        edge_out = self.edge_mlp(edge_input)
        
        # Apply attention if enabled
        if self.attention:
            attention_weights = torch.sigmoid(self.attention_mlp(edge_out))
            edge_out = edge_out * attention_weights
        
        # Aggregate edge features to nodes
        node_aggr = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        node_aggr.index_add_(0, row, edge_out)
        
        # Update node features
        node_input = torch.cat([h, node_aggr], dim=-1)
        h_out = h + self.node_mlp(node_input)
        
        # Update coordinates
        coord_aggr = torch.zeros(num_nodes, self.coords_dim, device=h.device)
        coord_out = self.coord_mlp(edge_out)
        coord_aggr.index_add_(0, row, coord_out * rel_x / (dist + 1e-8))
        x_out = x + coord_aggr
        
        return h_out, x_out

class EGNN(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network for receiver geometry.
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 coords_dim: int = 2,
                 output_dim: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Initialize EGNN.
        
        Args:
            node_dim: Input node feature dimension
            edge_dim: Input edge feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of EGNN layers
            coords_dim: Coordinate dimension (2 for 2D, 3 for 3D)
            output_dim: Output dimension (if None, same as node_dim)
            dropout: Dropout rate
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.coords_dim = coords_dim
        self.output_dim = output_dim or node_dim
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=node_dim if i == 0 else hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                coords_dim=coords_dim
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of EGNN.
        
        Args:
            h: Node features [num_nodes, node_dim]
            x: Node coordinates [num_nodes, coords_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, output_dim]
        """
        # Process through EGNN layers
        for i, layer in enumerate(self.layers):
            h_new, x_new = layer(h, x, edge_index, edge_attr)
            
            # Apply layer normalization and residual connection
            if i > 0:  # Skip for first layer since input dim might be different
                h_new = self.layer_norms[i](h_new)
                h = h + self.dropout(h_new)
            else:
                h = h_new
            
            x = x_new
        
        # Output projection
        out = self.output_proj(h)
        
        return out

class ReceiverGeometryEncoder(nn.Module):
    """
    Encoder for receiver geometry using EGNN.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 coords_dim: int = 2,
                 dropout: float = 0.1):
        """
        Initialize receiver geometry encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of EGNN layers
            coords_dim: Coordinate dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.coords_dim = coords_dim
        
        # Feature projection
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # EGNN for geometry processing
        self.egnn = EGNN(
            node_dim=hidden_dim,
            edge_dim=0,  # No edge features initially
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            coords_dim=coords_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def create_receiver_graph(self, 
                             receiver_positions: torch.Tensor,
                             max_neighbors: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph connectivity for receivers.
        
        Args:
            receiver_positions: Receiver coordinates [num_receivers, coords_dim]
            max_neighbors: Maximum number of neighbors per node
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        num_receivers = receiver_positions.shape[0]
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(receiver_positions, receiver_positions)
        
        # Create edges to nearest neighbors
        edge_list = []
        for i in range(num_receivers):
            # Get distances to other receivers
            distances = dist_matrix[i]
            
            # Find nearest neighbors (excluding self)
            _, indices = torch.topk(distances, min(max_neighbors + 1, num_receivers), largest=False)
            
            # Add edges (excluding self-connection)
            for j in indices[1:]:  # Skip first (self)
                edge_list.append([i, j.item()])
                edge_list.append([j.item(), i])  # Bidirectional
        
        if edge_list:
            edge_index = torch.tensor(edge_list, device=receiver_positions.device).t()
        else:
            edge_index = torch.zeros((2, 0), device=receiver_positions.device, dtype=torch.long)
        
        # No edge attributes for now
        edge_attr = None
        
        return edge_index, edge_attr
    
    def forward(self,
                features: torch.Tensor,
                receiver_positions: torch.Tensor,
                max_neighbors: int = 8) -> torch.Tensor:
        """
        Forward pass of receiver geometry encoder.
        
        Args:
            features: Input features [batch_size, num_receivers, input_dim]
            receiver_positions: Receiver positions [batch_size, num_receivers, coords_dim]
            max_neighbors: Maximum neighbors for graph construction
            
        Returns:
            Encoded features [batch_size, num_receivers, hidden_dim]
        """
        batch_size, num_receivers, _ = features.shape
        
        # Process each batch item
        outputs = []
        for b in range(batch_size):
            # Project features
            h = self.feature_proj(features[b])  # [num_receivers, hidden_dim]
            x = receiver_positions[b]  # [num_receivers, coords_dim]
            
            # Create graph
            edge_index, edge_attr = self.create_receiver_graph(x, max_neighbors)
            
            # Process through EGNN
            h_out = self.egnn(h, x, edge_index, edge_attr)
            
            # Project output
            h_out = self.output_proj(h_out)
            
            outputs.append(h_out)
        
        # Stack batch outputs
        output = torch.stack(outputs, dim=0)  # [batch_size, num_receivers, hidden_dim]
        
        return output 