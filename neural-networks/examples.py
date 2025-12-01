"""
Example usage of PEARL positional encodings

Demonstrates various use cases and comparisons.
"""

import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import degree
import torch.nn.functional as F

from pearl import PEARL, PEARLWithLaplacianInit
from gnn_with_pearl import GNNWithPEARL
from utils import (
    visualize_positional_encodings,
    visualize_graph_with_pe,
    analyze_pe_stability,
    measure_pe_complexity,
    plot_complexity_analysis,
    compute_expressive_power_score
)


def example_basic_pearl():
    """
    Example 1: Basic PEARL usage on Karate Club graph
    """
    print("=" * 60)
    print("Example 1: Basic PEARL Positional Encodings")
    print("=" * 60)
    
    # Load Karate Club graph
    dataset = KarateClub()
    data = dataset[0]
    
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Create PEARL model
    pearl_model = PEARL(
        hidden_dim=64,
        pe_dim=16,
        num_layers=3,
        init_mode='random',
        dropout=0.0
    )
    
    # Generate positional encodings
    pearl_model.eval()
    with torch.no_grad():
        pe = pearl_model(data.edge_index, data.num_nodes)
    
    print(f"Positional encodings shape: {pe.shape}")
    print(f"PE statistics - Mean: {pe.mean():.4f}, Std: {pe.std():.4f}")
    
    # Visualize
    visualize_positional_encodings(
        pe, 
        labels=data.y,
        title='PEARL PE on Karate Club Graph'
    )
    
    # Visualize graph with PE
    visualize_graph_with_pe(
        data, 
        pe, 
        pe_dim_to_visualize=0,
        title='Karate Club - PE Dimension 0'
    )
    
    print("\n✓ Basic PEARL example completed\n")


def example_initialization_comparison():
    """
    Example 2: Compare different initialization strategies
    """
    print("=" * 60)
    print("Example 2: Comparing Initialization Strategies")
    print("=" * 60)
    
    dataset = KarateClub()
    data = dataset[0]
    
    # Random initialization
    pearl_random = PEARL(hidden_dim=64, pe_dim=16, init_mode='random')
    pearl_random.eval()
    
    # Basis initialization
    pearl_basis = PEARL(hidden_dim=64, pe_dim=16, init_mode='basis')
    pearl_basis.eval()
    
    # Laplacian initialization
    pearl_laplacian = PEARLWithLaplacianInit(hidden_dim=64, pe_dim=16)
    pearl_laplacian.eval()
    
    # Generate PEs
    with torch.no_grad():
        pe_random = pearl_random(data.edge_index, data.num_nodes)
        pe_basis = pearl_basis(data.edge_index, data.num_nodes)
        pe_laplacian = pearl_laplacian(data.edge_index, data.num_nodes)
    
    print("Initialization comparison:")
    print(f"  Random    - Mean: {pe_random.mean():.4f}, Std: {pe_random.std():.4f}")
    print(f"  Basis     - Mean: {pe_basis.mean():.4f}, Std: {pe_basis.std():.4f}")
    print(f"  Laplacian - Mean: {pe_laplacian.mean():.4f}, Std: {pe_laplacian.std():.4f}")
    
    # Compute expressive power
    for name, pe in [('Random', pe_random), ('Basis', pe_basis), ('Laplacian', pe_laplacian)]:
        metrics = compute_expressive_power_score(pe)
        print(f"\n{name} expressiveness:")
        print(f"  Effective rank: {metrics['effective_rank']:.2f}")
        print(f"  Spectral gap: {metrics['spectral_gap']:.4f}")
    
    print("\n✓ Initialization comparison completed\n")


def example_stability_analysis():
    """
    Example 3: Analyze PEARL stability to graph perturbations
    """
    print("=" * 60)
    print("Example 3: Stability Analysis")
    print("=" * 60)
    
    dataset = KarateClub()
    data = dataset[0]
    
    # Create PEARL model
    pearl_model = PEARL(hidden_dim=64, pe_dim=16, num_layers=3)
    pearl_model.eval()
    
    # Analyze stability
    print("Testing stability with 10% edge perturbations...")
    stability_results = analyze_pe_stability(
        pearl_model,
        data.edge_index,
        data.num_nodes,
        num_perturbations=20,
        noise_level=0.1
    )
    
    print(f"\nStability results:")
    print(f"  Mean cosine similarity: {stability_results['mean_cosine_similarity']:.4f} "
          f"(±{stability_results['std_cosine_similarity']:.4f})")
    print(f"  Mean L2 distance: {stability_results['mean_l2_distance']:.4f} "
          f"(±{stability_results['std_l2_distance']:.4f})")
    
    print("\n✓ Stability analysis completed\n")


def example_complexity_analysis():
    """
    Example 4: Measure computational complexity
    """
    print("=" * 60)
    print("Example 4: Computational Complexity Analysis")
    print("=" * 60)
    
    # Create PEARL model
    pearl_model = PEARL(hidden_dim=64, pe_dim=16, num_layers=2)
    pearl_model.eval()
    
    # Test on different graph sizes
    num_nodes_list = [50, 100, 200, 500, 1000, 2000]
    
    print("Measuring complexity for different graph sizes...")
    complexity_results = measure_pe_complexity(pearl_model, num_nodes_list)
    
    print("\nComplexity results:")
    for n, t in zip(complexity_results['num_nodes'], complexity_results['times']):
        print(f"  {n:4d} nodes: {t:.4f} seconds")
    
    # Plot results
    plot_complexity_analysis(complexity_results)
    
    print("\n✓ Complexity analysis completed\n")


def example_gnn_training():
    """
    Example 5: Train GNN with PEARL on node classification
    """
    print("=" * 60)
    print("Example 5: GNN Training with PEARL")
    print("=" * 60)
    
    # Load dataset
    dataset = KarateClub()
    data = dataset[0]
    
    # Create features if not present
    if data.x is None:
        data.x = degree(data.edge_index[1], data.num_nodes, dtype=torch.float).unsqueeze(1)
    
    print(f"Training on Karate Club:")
    print(f"  Nodes: {data.num_nodes}, Features: {data.x.size(1)}, Classes: {data.y.max().item() + 1}")
    
    # Create model with PEARL
    model = GNNWithPEARL(
        input_dim=data.x.size(1),
        hidden_dim=32,
        output_dim=data.y.max().item() + 1,
        pe_dim=8,
        num_layers=2,
        gnn_type='gcn',
        dropout=0.5,
        task='node',
        use_pearl=True
    )
    
    # Create train/val/test masks
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 60% train, 20% val, 20% test
    perm = torch.randperm(num_nodes)
    train_mask[perm[:int(0.6 * num_nodes)]] = True
    val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
    test_mask[perm[int(0.8 * num_nodes):]] = True
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
        val_acc = (pred[val_mask] == data.y[val_mask]).sum().item() / val_mask.sum().item()
        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
        
        return train_acc, val_acc, test_acc
    
    # Train
    print("\nTraining...")
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(1, 201):
        loss = train()
        train_acc, val_acc, test_acc = test()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}: Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {best_test_acc:.4f}")
    
    print("\n✓ GNN training completed\n")


def example_ablation_study():
    """
    Example 6: Ablation study - with and without PEARL
    """
    print("=" * 60)
    print("Example 6: Ablation Study (PEARL vs No PEARL)")
    print("=" * 60)
    
    dataset = KarateClub()
    data = dataset[0]
    
    if data.x is None:
        data.x = degree(data.edge_index[1], data.num_nodes, dtype=torch.float).unsqueeze(1)
    
    results = {}
    
    for use_pearl in [False, True]:
        name = "With PEARL" if use_pearl else "Without PEARL"
        print(f"\nTraining {name}...")
        
        model = GNNWithPEARL(
            input_dim=data.x.size(1),
            hidden_dim=32,
            output_dim=data.y.max().item() + 1,
            pe_dim=8,
            num_layers=2,
            gnn_type='gcn',
            dropout=0.5,
            task='node',
            use_pearl=use_pearl
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Simple train/test split
        num_train = int(0.7 * data.num_nodes)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[:num_train] = True
        test_mask = ~train_mask
        
        # Train for 100 epochs
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
        
        results[name] = test_acc
        print(f"  Test accuracy: {test_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Ablation Study Results:")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:20s}: {acc:.4f}")
    
    improvement = (results["With PEARL"] - results["Without PEARL"]) / results["Without PEARL"] * 100
    print(f"\n  Improvement with PEARL: {improvement:+.2f}%")
    
    print("\n✓ Ablation study completed\n")


def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("PEARL - Positional Encodings with Graph Neural Networks")
    print("Running All Examples")
    print("=" * 60 + "\n")
    
    example_basic_pearl()
    example_initialization_comparison()
    example_stability_analysis()
    example_complexity_analysis()
    example_gnn_training()
    example_ablation_study()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PEARL Examples')
    parser.add_argument('--example', type=str, default='all',
                       choices=['all', 'basic', 'init', 'stability', 
                               'complexity', 'training', 'ablation'],
                       help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example == 'all':
        run_all_examples()
    elif args.example == 'basic':
        example_basic_pearl()
    elif args.example == 'init':
        example_initialization_comparison()
    elif args.example == 'stability':
        example_stability_analysis()
    elif args.example == 'complexity':
        example_complexity_analysis()
    elif args.example == 'training':
        example_gnn_training()
    elif args.example == 'ablation':
        example_ablation_study()

