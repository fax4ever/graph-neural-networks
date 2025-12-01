"""
Training script for GNN models with PEARL positional encodings

Supports various datasets and tasks including:
- Graph classification (molecular property prediction)
- Node classification (citation networks)
- Link prediction
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid, QM9
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
from typing import Dict, Tuple

from gnn_with_pearl import GNNWithPEARL, TransformerWithPEARL, EnsemblePEARL
from pearl import PEARL


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_dataset(name: str, root: str = './data'):
    """
    Load dataset by name
    
    Args:
        name: Dataset name
        root: Root directory for datasets
        
    Returns:
        dataset, task_type, num_features, num_classes
    """
    if name in ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(root=root, name=name)
        task = 'graph'
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        
    elif name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=root, name=name)
        task = 'node'
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        
    elif name == 'QM9':
        dataset = QM9(root=root)
        task = 'graph'
        num_features = dataset.num_features
        num_classes = 1  # Regression task
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset, task, num_features, num_classes


def train_epoch(model, loader, optimizer, device, task='graph'):
    """
    Train for one epoch
    
    Args:
        model: GNN model
        loader: Data loader
        optimizer: Optimizer
        device: Device to train on
        task: Task type ('graph' or 'node')
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        if task == 'graph':
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
        else:  # node classification
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, task='graph', split_mask=None):
    """
    Evaluate model
    
    Args:
        model: GNN model
        loader: Data loader
        device: Device
        task: Task type
        split_mask: For node classification (train/val/test mask)
        
    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        
        if task == 'graph':
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        else:  # node classification
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            if split_mask is not None:
                mask = getattr(data, split_mask)
                correct += (pred[mask] == data.y[mask]).sum().item()
                total += mask.sum().item()
            else:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    
    return correct / total if total > 0 else 0


def train_model(args):
    """
    Main training function
    
    Args:
        args: Training arguments
    """
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, task, num_features, num_classes = load_dataset(args.dataset, args.data_root)
    print(f"Dataset: {len(dataset)} graphs, {num_features} features, {num_classes} classes")
    print(f"Task type: {task}")
    
    # Handle datasets with no node features
    if num_features == 0:
        # Use degree as features
        print("No node features found, using node degree as features")
        from torch_geometric.utils import degree
        max_degree = 0
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        
        # One-hot encode degrees
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            data.x = F.one_hot(d, num_classes=max_degree + 1).float()
        
        num_features = max_degree + 1
    
    # Split dataset
    if task == 'graph':
        # Random split for graph classification
        num_train = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - num_train - num_val
        
        train_dataset = dataset[:num_train]
        val_dataset = dataset[num_train:num_train + num_val]
        test_dataset = dataset[num_train + num_val:]
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
    else:  # node classification
        # Use predefined splits
        train_loader = DataLoader([dataset[0]], batch_size=1)
        val_loader = DataLoader([dataset[0]], batch_size=1)
        test_loader = DataLoader([dataset[0]], batch_size=1)
    
    # Create model
    print(f"Creating {args.model_type} model with PEARL")
    
    if args.model_type == 'gnn':
        model = GNNWithPEARL(
            input_dim=num_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            pe_dim=args.pe_dim,
            num_layers=args.num_layers,
            gnn_type=args.gnn_type,
            pe_init_mode=args.pe_init_mode,
            pe_num_layers=args.pe_num_layers,
            dropout=args.dropout,
            task=task,
            use_pearl=args.use_pearl
        ).to(device)
        
    elif args.model_type == 'transformer':
        model = TransformerWithPEARL(
            input_dim=num_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            pe_dim=args.pe_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            pe_num_layers=args.pe_num_layers,
            dropout=args.dropout,
            task=task
        ).to(device)
        
    elif args.model_type == 'ensemble':
        model = EnsemblePEARL(
            input_dim=num_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            pe_dim=args.pe_dim,
            num_models=args.num_ensemble,
            num_layers=args.num_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            task=task
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        loss = train_epoch(model, train_loader, optimizer, device, task)
        
        # Evaluate
        if task == 'graph':
            train_acc = evaluate(model, train_loader, device, task)
            val_acc = evaluate(model, val_loader, device, task)
            test_acc = evaluate(model, test_loader, device, task)
        else:
            train_acc = evaluate(model, train_loader, device, task, 'train_mask')
            val_acc = evaluate(model, val_loader, device, task, 'val_mask')
            test_acc = evaluate(model, test_loader, device, task, 'test_mask')
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
            
            # Save checkpoint
            if args.save_model:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                }, f'checkpoints/{args.dataset}_{args.model_type}_best.pt')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % args.log_interval == 0:
            print(f'Epoch {epoch:03d}: Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Final results
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {best_test_acc:.4f}")
    
    return best_val_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser(description='Train GNN with PEARL')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='MUTAG',
                       help='Dataset name (MUTAG, PROTEINS, Cora, etc.)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='gnn',
                       choices=['gnn', 'transformer', 'ensemble'],
                       help='Model architecture type')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'gin'],
                       help='GNN layer type')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (for transformer)')
    parser.add_argument('--num_ensemble', type=int, default=3,
                       help='Number of ensemble models')
    
    # PEARL arguments
    parser.add_argument('--use_pearl', action='store_true', default=True,
                       help='Use PEARL positional encodings')
    parser.add_argument('--no_pearl', dest='use_pearl', action='store_false',
                       help='Disable PEARL (for ablation)')
    parser.add_argument('--pe_dim', type=int, default=32,
                       help='Positional encoding dimension')
    parser.add_argument('--pe_num_layers', type=int, default=2,
                       help='Number of PEARL layers')
    parser.add_argument('--pe_init_mode', type=str, default='random',
                       choices=['random', 'basis'],
                       help='PEARL initialization mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_model', action='store_true',
                       help='Save best model checkpoint')
    
    args = parser.parse_args()
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()

