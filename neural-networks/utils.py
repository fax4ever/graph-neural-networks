"""
Utility functions for PEARL and GNN experiments
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import Optional, List


def visualize_positional_encodings(
    pe: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = 'tsne',
    title: str = 'PEARL Positional Encodings',
    save_path: Optional[str] = None
):
    """
    Visualize positional encodings in 2D
    
    Args:
        pe: Positional encodings [num_nodes, pe_dim]
        labels: Optional node labels for coloring
        method: Dimensionality reduction method ('tsne' or 'pca')
        title: Plot title
        save_path: Path to save figure
    """
    pe_np = pe.detach().cpu().numpy()
    
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    pe_2d = reducer.fit_transform(pe_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        scatter = plt.scatter(pe_2d[:, 0], pe_2d[:, 1], c=labels_np, 
                            cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Node Label')
    else:
        plt.scatter(pe_2d[:, 0], pe_2d[:, 1], alpha=0.7, s=50)
    
    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_positional_encodings(
    pearl_pe: torch.Tensor,
    baseline_pe: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Compare PEARL with baseline positional encodings
    
    Args:
        pearl_pe: PEARL positional encodings
        baseline_pe: Baseline PE (e.g., Laplacian eigenvectors)
        labels: Optional node labels
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, pe, name in zip(axes, [pearl_pe, baseline_pe], ['PEARL', 'Baseline']):
        pe_np = pe.detach().cpu().numpy()
        
        # Use PCA for visualization
        pca = PCA(n_components=2)
        pe_2d = pca.fit_transform(pe_np)
        
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            scatter = ax.scatter(pe_2d[:, 0], pe_2d[:, 1], c=labels_np,
                               cmap='tab10', alpha=0.7, s=50)
            if name == 'Baseline':
                plt.colorbar(scatter, ax=ax, label='Node Label')
        else:
            ax.scatter(pe_2d[:, 0], pe_2d[:, 1], alpha=0.7, s=50)
        
        ax.set_title(f'{name} Positional Encodings')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_graph_with_pe(
    data,
    pe: torch.Tensor,
    pe_dim_to_visualize: int = 0,
    title: str = 'Graph with PEARL PE',
    save_path: Optional[str] = None
):
    """
    Visualize graph with nodes colored by a PE dimension
    
    Args:
        data: PyG data object
        pe: Positional encodings [num_nodes, pe_dim]
        pe_dim_to_visualize: Which PE dimension to visualize
        title: Plot title
        save_path: Path to save figure
    """
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Get PE values for coloring
    pe_values = pe[:, pe_dim_to_visualize].detach().cpu().numpy()
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=pe_values,
        node_size=500,
        cmap='viridis',
        alpha=0.8
    )
    
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.colorbar(nodes, label=f'PE Dimension {pe_dim_to_visualize}')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_pe_stability(
    model,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_perturbations: int = 10,
    noise_level: float = 0.1
) -> dict:
    """
    Analyze stability of PEARL encodings to graph perturbations
    
    Args:
        model: PEARL model
        edge_index: Original graph
        num_nodes: Number of nodes
        num_perturbations: Number of perturbations to test
        noise_level: Fraction of edges to perturb
        
    Returns:
        Dictionary with stability metrics
    """
    model.eval()
    device = edge_index.device
    
    # Get original PE
    with torch.no_grad():
        pe_original = model(edge_index, num_nodes)
    
    # Perturb graph and compute PEs
    pe_perturbed_list = []
    
    for _ in range(num_perturbations):
        # Random edge perturbation
        num_edges = edge_index.size(1)
        num_to_remove = int(num_edges * noise_level)
        
        # Remove random edges
        perm = torch.randperm(num_edges, device=device)
        keep_idx = perm[num_to_remove:]
        edge_index_perturbed = edge_index[:, keep_idx]
        
        # Add random edges
        num_to_add = num_to_remove
        new_edges = torch.randint(0, num_nodes, (2, num_to_add), device=device)
        edge_index_perturbed = torch.cat([edge_index_perturbed, new_edges], dim=1)
        
        # Compute PE
        with torch.no_grad():
            pe_perturbed = model(edge_index_perturbed, num_nodes)
            pe_perturbed_list.append(pe_perturbed)
    
    # Compute stability metrics
    pe_perturbed_stack = torch.stack(pe_perturbed_list)
    
    # Average cosine similarity with original
    cos_sims = []
    for pe_p in pe_perturbed_list:
        cos_sim = F.cosine_similarity(pe_original, pe_p, dim=1).mean().item()
        cos_sims.append(cos_sim)
    
    # Average L2 distance
    l2_dists = []
    for pe_p in pe_perturbed_list:
        l2_dist = torch.norm(pe_original - pe_p, p=2, dim=1).mean().item()
        l2_dists.append(l2_dist)
    
    return {
        'mean_cosine_similarity': np.mean(cos_sims),
        'std_cosine_similarity': np.std(cos_sims),
        'mean_l2_distance': np.mean(l2_dists),
        'std_l2_distance': np.std(l2_dists),
        'cosine_similarities': cos_sims,
        'l2_distances': l2_dists
    }


def measure_pe_complexity(model, num_nodes_list: List[int]) -> dict:
    """
    Measure computational complexity of PEARL
    
    Args:
        model: PEARL model
        num_nodes_list: List of graph sizes to test
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    times = []
    
    for num_nodes in num_nodes_list:
        # Create random graph
        num_edges = num_nodes * 5  # Average degree of 10
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        
        # Warm up
        with torch.no_grad():
            _ = model(edge_index, num_nodes)
        
        # Measure time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            _ = model(edge_index, num_nodes)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        times.append(elapsed)
    
    return {
        'num_nodes': num_nodes_list,
        'times': times
    }


def plot_complexity_analysis(complexity_results: dict, save_path: Optional[str] = None):
    """
    Plot complexity analysis results
    
    Args:
        complexity_results: Results from measure_pe_complexity
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    num_nodes = complexity_results['num_nodes']
    times = complexity_results['times']
    
    plt.plot(num_nodes, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('PEARL Computational Complexity', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Fit linear curve for reference
    coeffs = np.polyfit(num_nodes, times, 1)
    fit_line = np.poly1d(coeffs)
    plt.plot(num_nodes, fit_line(num_nodes), '--', 
            label=f'Linear fit: {coeffs[0]:.2e}n + {coeffs[1]:.2e}',
            alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_expressive_power_score(pe: torch.Tensor) -> dict:
    """
    Compute metrics related to expressive power of PEs
    
    Args:
        pe: Positional encodings [num_nodes, pe_dim]
        
    Returns:
        Dictionary with expressiveness metrics
    """
    pe_np = pe.detach().cpu().numpy()
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(pe_np, metric='euclidean'))
    
    # Metrics
    metrics = {
        'mean_pairwise_distance': dist_matrix.mean(),
        'std_pairwise_distance': dist_matrix.std(),
        'min_pairwise_distance': dist_matrix[dist_matrix > 0].min(),
        'effective_rank': compute_effective_rank(pe),
        'spectral_gap': compute_spectral_gap(pe)
    }
    
    return metrics


def compute_effective_rank(tensor: torch.Tensor) -> float:
    """
    Compute effective rank using singular values
    
    Args:
        tensor: Input tensor
        
    Returns:
        Effective rank
    """
    _, S, _ = torch.svd(tensor)
    S_normalized = S / S.sum()
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()
    return effective_rank


def compute_spectral_gap(tensor: torch.Tensor) -> float:
    """
    Compute spectral gap (difference between top 2 singular values)
    
    Args:
        tensor: Input tensor
        
    Returns:
        Spectral gap
    """
    _, S, _ = torch.svd(tensor)
    if len(S) >= 2:
        gap = (S[0] - S[1]).item()
    else:
        gap = S[0].item()
    return gap


def save_results(results: dict, filename: str):
    """
    Save experimental results to file
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    import json
    
    # Convert numpy/torch to python types
    def convert(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert(results)
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filename}")

