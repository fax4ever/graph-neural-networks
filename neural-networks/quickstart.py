#!/usr/bin/env python3
"""
Quick start script for PEARL

Demonstrates basic usage with minimal setup.
"""

import torch
from torch_geometric.datasets import KarateClub
import torch.nn.functional as F

from pearl import PEARL
from gnn_with_pearl import GNNWithPEARL


def main():
    print("=" * 70)
    print("PEARL Quick Start Demo")
    print("=" * 70)
    print()
    
    # Load a simple dataset
    print("1. Loading Karate Club dataset...")
    dataset = KarateClub()
    data = dataset[0]
    
    print(f"   Graph: {data.num_nodes} nodes, {data.num_edges} edges, {dataset.num_classes} classes")
    print()
    
    # Create PEARL model
    print("2. Creating PEARL model...")
    pearl_model = PEARL(
        hidden_dim=64,
        pe_dim=16,
        num_layers=3,
        init_mode='random'
    )
    print(f"   PEARL configured: {sum(p.numel() for p in pearl_model.parameters()):,} parameters")
    print()
    
    # Generate positional encodings
    print("3. Generating positional encodings...")
    pearl_model.eval()
    with torch.no_grad():
        pe = pearl_model(data.edge_index, data.num_nodes)
    
    print(f"   Generated PE shape: {pe.shape}")
    print(f"   PE stats - Mean: {pe.mean():.4f}, Std: {pe.std():.4f}")
    print()
    
    # Create and train a GNN with PEARL
    print("4. Creating GNN with PEARL...")
    model = GNNWithPEARL(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=dataset.num_classes,
        pe_dim=16,
        num_layers=2,
        gnn_type='gcn',
        dropout=0.5,
        task='node',
        use_pearl=True
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   GNN model: {num_params:,} parameters")
    print()
    
    # Simple train/test split
    num_train = int(0.7 * data.num_nodes)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask = ~train_mask
    
    # Train
    print("5. Training GNN for 100 epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                train_acc = (pred[train_mask] == data.y[train_mask]).sum() / train_mask.sum()
                test_acc = (pred[test_mask] == data.y[test_mask]).sum() / test_mask.sum()
            print(f"   Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
                  f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
            model.train()
    print()
    
    # Final evaluation
    print("6. Final Evaluation...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
    
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")
    print()
    
    print("=" * 70)
    print("✓ Quick start completed successfully!")
    print()
    print("Next steps:")
    print("  - Run examples: python examples.py --example all")
    print("  - Train on datasets: python train.py --dataset MUTAG")
    print("  - Read README.md for detailed documentation")
    print("=" * 70)


if __name__ == '__main__':
    main()

