import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

def create_pyg_data(triplets: List[Dict], embedding_dim: int = 128) -> Tuple[Data, Dict[str, int]]:
    """Convert triplets to PyTorch Geometric Data object."""
    # Create entity to index mapping
    entities = set()
    for triplet in triplets:
        entities.add(triplet['subject'])
        entities.add(triplet['object'])
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
    
    # Create node features (random initialization for now)
    num_nodes = len(entities)
    x = torch.randn(num_nodes, embedding_dim)
    
    # Create edge index
    edge_index = []
    for triplet in triplets:
        src_idx = entity_to_idx[triplet['subject']]
        dst_idx = entity_to_idx[triplet['object']]
        edge_index.append([src_idx, dst_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index), entity_to_idx

def train_graphsage(data: Data, 
                   hidden_channels: int = 256,
                   out_channels: int = 128,
                   num_layers: int = 2,
                   epochs: int = 100) -> GraphSAGE:
    """Train GraphSAGE model on the graph data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Use reconstruction loss
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    return model

def get_node_embeddings(model: GraphSAGE, data: Data, entity_to_idx: Dict[str, int]) -> Dict[str, List[float]]:
    """Get node embeddings from trained GraphSAGE model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    # Convert to numpy and create dictionary mapping entities to embeddings
    embeddings = out.cpu().numpy()
    return {entity: embeddings[idx].tolist() for entity, idx in entity_to_idx.items()}

def generate_graphsage_embeddings(triplets: List[Dict], 
                                embedding_dim: int = 128,
                                hidden_channels: int = 256,
                                num_layers: int = 2,
                                epochs: int = 100) -> Dict[str, List[float]]:
    """Generate GraphSAGE embeddings for all entities in the triplets."""
    logger.info("Converting triplets to PyTorch Geometric format...")
    data, entity_to_idx = create_pyg_data(triplets, embedding_dim)
    
    logger.info("Training GraphSAGE model...")
    model = train_graphsage(data, hidden_channels, embedding_dim, num_layers, epochs)
    
    logger.info("Generating node embeddings...")
    embeddings = get_node_embeddings(model, data, entity_to_idx)
    
    return embeddings 