# PEARL Tutorial: Step-by-Step Guide

This tutorial walks you through using PEARL (Positional Encodings Augmented with Representation Learning) for graph neural networks.

## Table of Contents

1. [Installation](#installation)
2. [Basic Concepts](#basic-concepts)
3. [Quick Start](#quick-start)
4. [Understanding PEARL](#understanding-pearl)
5. [Training Models](#training-models)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)

## Installation

### Step 1: Install Dependencies

```bash
cd neural-networks
pip install -r requirements.txt
```

### Step 2: Verify Installation

```python
import torch
import torch_geometric
print(f"PyTorch version: {torch.__version__}")
print(f"PyG version: {torch_geometric.__version__}")
```

## Basic Concepts

### What are Positional Encodings?

In graph learning, **positional encodings** provide information about a node's position in the graph structure. Unlike sequences (where position = index), graphs don't have a natural ordering, so we need to compute PEs from the graph structure.

### Why PEARL?

Traditional methods (Laplacian eigenvectors) are expensive:
- **Eigendecomposition**: O(n³) complexity
- **Memory intensive**: Stores n×n matrices
- **Not scalable**: Infeasible for graphs with >10,000 nodes

PEARL solves this by:
- **Using GNNs**: O(n) complexity for sparse graphs
- **Learning PEs**: End-to-end trainable
- **Faster**: 10-100× speedup

## Quick Start

### Example 1: Generate Positional Encodings

```python
import torch
from pearl import PEARL

# Create a simple graph
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],  # source nodes
    [1, 0, 2, 1, 3, 2, 0, 3]   # target nodes
])
num_nodes = 4

# Create PEARL model
pearl = PEARL(
    hidden_dim=64,   # Internal dimension
    pe_dim=16,       # Output PE dimension
    num_layers=3     # Number of GNN layers
)

# Generate PEs
pearl.eval()
with torch.no_grad():
    positional_encodings = pearl(edge_index, num_nodes)

print(f"Shape: {positional_encodings.shape}")  # [4, 16]
print(f"PEs:\n{positional_encodings}")
```

**Output:**
```
Shape: torch.Size([4, 16])
PEs:
tensor([[-0.1234,  0.5678, ...],  # Node 0
        [ 0.2345, -0.6789, ...],  # Node 1
        [-0.3456,  0.7890, ...],  # Node 2
        [ 0.4567, -0.8901, ...]])  # Node 3
```

### Example 2: Use PEARL in a GNN

```python
from gnn_with_pearl import GNNWithPEARL
from torch_geometric.datasets import KarateClub
import torch.nn.functional as F

# Load dataset
dataset = KarateClub()
data = dataset[0]

# Create GNN with PEARL
model = GNNWithPEARL(
    input_dim=data.x.size(1),
    hidden_dim=64,
    output_dim=dataset.num_classes,
    pe_dim=16,
    num_layers=3,
    gnn_type='gcn',
    use_pearl=True  # Enable PEARL
)

# Forward pass
model.eval()
with torch.no_grad():
    output = model(data.x, data.edge_index)
    predictions = output.argmax(dim=1)

print(f"Predictions: {predictions}")
```

### Example 3: Train a Model

```python
# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss (example with all nodes)
    loss = F.cross_entropy(out, data.y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## Understanding PEARL

### Architecture Components

#### 1. Initialization

PEARL starts by initializing node features:

```python
# Random initialization (default)
pearl_random = PEARL(init_mode='random')

# Basis initialization (for regular graphs)
pearl_basis = PEARL(init_mode='basis')
```

**When to use:**
- `random`: Social networks, citation graphs, general purpose
- `basis`: Grid graphs, molecular structures, regular patterns

#### 2. Statistical Pooling

Ensures permutation equivariance by computing statistics:

```python
from pearl import StatisticalPooling

pooling = StatisticalPooling(mode='all')  # mean + max + std
# pooling = StatisticalPooling(mode='mean')  # only mean
```

#### 3. Message Passing

GNN layers process and propagate information:

```python
pearl = PEARL(
    num_layers=2,  # Shallow: fast, less expressive
    num_layers=3,  # Recommended: good balance
    num_layers=4   # Deep: slow, more expressive
)
```

#### 4. Output Normalization

L2 normalization for stable PEs:

```python
# Automatically applied in PEARL
pe = F.normalize(pe, p=2, dim=1)
```

### Parameters Explained

```python
pearl = PEARL(
    hidden_dim=64,      # Internal GNN dimension
                        # Larger = more expressive, slower
                        # Recommended: 32-128
    
    pe_dim=32,          # Output PE dimension
                        # Should be ≤ hidden_dim
                        # Recommended: 16-64
    
    num_layers=3,       # Number of message-passing layers
                        # More layers = larger receptive field
                        # Recommended: 2-4
    
    init_mode='random', # Initialization strategy
                        # 'random' or 'basis'
    
    pooling_mode='all', # Statistical pooling
                        # 'mean', 'max', 'std', or 'all'
    
    dropout=0.1,        # Dropout rate (0.0 to disable)
                        # Higher for small graphs
    
    use_batch_norm=True # Batch normalization
                        # Helps training stability
)
```

## Training Models

### Graph Classification

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Load dataset
dataset = TUDataset(root='./data', name='MUTAG')

# Split dataset
train_dataset = dataset[:150]
test_dataset = dataset[150:]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Create model
model = GNNWithPEARL(
    input_dim=dataset.num_features or 1,
    hidden_dim=64,
    output_dim=dataset.num_classes,
    pe_dim=32,
    task='graph'  # Important!
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")
```

### Node Classification

```python
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Create model
model = GNNWithPEARL(
    input_dim=dataset.num_features,
    hidden_dim=64,
    output_dim=dataset.num_classes,
    pe_dim=32,
    task='node'  # Important!
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    # Evaluate
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
        print(f"Epoch {epoch}: Test Acc = {acc:.4f}")
```

### Using Command-Line Training

```bash
# Graph classification on MUTAG
python train.py --dataset MUTAG --epochs 200 --hidden_dim 64 --pe_dim 32

# Node classification on Cora
python train.py --dataset Cora --epochs 200 --lr 0.01

# Disable PEARL (ablation)
python train.py --dataset MUTAG --no_pearl

# Use different GNN types
python train.py --dataset PROTEINS --gnn_type gat
python train.py --dataset PROTEINS --gnn_type gin

# Use Graph Transformer
python train.py --dataset MUTAG --model_type transformer --num_heads 8
```

## Advanced Usage

### Custom PEARL Initialization

```python
from pearl import PEARL

class CustomPEARL(PEARL):
    def initialize_features(self, num_nodes, device, edge_index=None):
        # Custom initialization
        features = torch.randn(num_nodes, self.hidden_dim, device=device)
        
        # Example: Scale by node degree
        if edge_index is not None:
            from torch_geometric.utils import degree
            deg = degree(edge_index[1], num_nodes, dtype=features.dtype)
            features = features * (1 + deg.unsqueeze(1))
        
        return F.normalize(features, p=2, dim=1)

# Use custom PEARL
custom_pearl = CustomPEARL(hidden_dim=64, pe_dim=32)
```

### Ensemble Models

```python
from gnn_with_pearl import EnsemblePEARL

# Create ensemble of 3 models with different initializations
ensemble = EnsemblePEARL(
    input_dim=dataset.num_features,
    hidden_dim=64,
    output_dim=dataset.num_classes,
    pe_dim=32,
    num_models=3
)

# Predictions are automatically averaged
output = ensemble(data.x, data.edge_index, data.batch)
```

### Transfer Learning

```python
# Pre-train PEARL on large dataset
pearl_pretrained = PEARL(hidden_dim=64, pe_dim=32)
# ... train on large graphs ...

# Transfer to new task
model = GNNWithPEARL(...)
model.pearl = pearl_pretrained  # Use pre-trained PEARL
model.pearl.requires_grad_(False)  # Freeze PEARL

# Fine-tune only GNN layers
optimizer = torch.optim.Adam(
    [p for n, p in model.named_parameters() if 'pearl' not in n],
    lr=0.001
)
```

### Visualizing PEs

```python
from utils import visualize_positional_encodings, visualize_graph_with_pe

# Generate PEs
pe = pearl(data.edge_index, data.num_nodes)

# Visualize in 2D using t-SNE
visualize_positional_encodings(
    pe, 
    labels=data.y,
    method='tsne',
    save_path='pe_visualization.png'
)

# Visualize on graph
visualize_graph_with_pe(
    data,
    pe,
    pe_dim_to_visualize=0,
    save_path='graph_with_pe.png'
)
```

### Analyzing Stability

```python
from utils import analyze_pe_stability

# Test stability to perturbations
stability = analyze_pe_stability(
    pearl,
    data.edge_index,
    data.num_nodes,
    num_perturbations=20,
    noise_level=0.1  # 10% edge perturbation
)

print(f"Mean cosine similarity: {stability['mean_cosine_similarity']:.4f}")
print(f"Mean L2 distance: {stability['mean_l2_distance']:.4f}")
```

### Measuring Complexity

```python
from utils import measure_pe_complexity, plot_complexity_analysis

# Test on different graph sizes
results = measure_pe_complexity(
    pearl,
    num_nodes_list=[100, 500, 1000, 5000, 10000]
)

# Plot results
plot_complexity_analysis(results, save_path='complexity.png')
```

## Best Practices

### 1. Choose the Right Parameters

**For small graphs (<1,000 nodes):**
```python
pearl = PEARL(hidden_dim=32, pe_dim=16, num_layers=2)
```

**For medium graphs (1,000-10,000 nodes):**
```python
pearl = PEARL(hidden_dim=64, pe_dim=32, num_layers=3)
```

**For large graphs (>10,000 nodes):**
```python
pearl = PEARL(hidden_dim=64, pe_dim=32, num_layers=2, dropout=0.3)
```

### 2. Initialization Strategy

```python
# For diverse, irregular graphs (social networks)
pearl = PEARL(init_mode='random')

# For regular structures (grids, molecules)
pearl = PEARL(init_mode='basis')

# For graphs with important degree information
pearl = PEARLWithLaplacianInit(...)
```

### 3. GNN Type Selection

```python
# Fast baseline (recommended to start)
model = GNNWithPEARL(gnn_type='gcn')

# Better for heterophilic graphs
model = GNNWithPEARL(gnn_type='gat')

# Maximum expressive power
model = GNNWithPEARL(gnn_type='gin')
```

### 4. Training Tips

```python
# Use learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=20
)

# Early stopping
best_val_acc = 0
patience = 50
counter = 0

for epoch in range(epochs):
    # ... train ...
    val_acc = evaluate(model, val_loader)
    
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        # Save model
    else:
        counter += 1
        if counter >= patience:
            break
```

### 5. Memory Optimization

For large graphs:

```python
# Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

# Reduce hidden dimensions
model = GNNWithPEARL(hidden_dim=32, pe_dim=16)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(data.x, data.edge_index)
    loss = criterion(output, labels)
scaler.scale(loss).backward()
```

## Common Issues and Solutions

### Issue 1: Out of Memory

**Solution:**
```python
# Reduce dimensions
pearl = PEARL(hidden_dim=32, pe_dim=16, num_layers=2)

# Use CPU for very large graphs
pearl = pearl.cpu()
```

### Issue 2: Slow Training

**Solution:**
```python
# Reduce layers
pearl = PEARL(num_layers=2)

# Use faster GNN
model = GNNWithPEARL(gnn_type='gcn')  # instead of 'gat'

# Disable batch norm if not needed
pearl = PEARL(use_batch_norm=False)
```

### Issue 3: Poor Performance

**Solution:**
```python
# Increase PE dimension
pearl = PEARL(pe_dim=64)

# Add more layers
pearl = PEARL(num_layers=4)

# Try different initialization
pearl = PEARL(init_mode='basis')  # or try Laplacian

# Tune dropout
model = GNNWithPEARL(dropout=0.3)  # try 0.3, 0.5, 0.7
```

## Next Steps

1. **Run Examples**: `python examples.py --example all`
2. **Try Quickstart**: `python quickstart.py`
3. **Train on Your Data**: Modify `train.py` for custom datasets
4. **Read Paper**: See `PAPER_SUMMARY.md` for theoretical details
5. **Experiment**: Try different architectures and hyperparameters

## Resources

- **README.md**: Full documentation
- **PAPER_SUMMARY.md**: Detailed paper analysis
- **examples.py**: Comprehensive examples
- **quickstart.py**: Minimal working example
- **train.py**: Full training pipeline

## Getting Help

If you encounter issues:
1. Check this tutorial
2. Read the README
3. Run examples to see working code
4. Check hyperparameters match your graph size

---

Happy graph learning! 🎉

