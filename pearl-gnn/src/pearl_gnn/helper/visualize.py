"""Visualize graphs from PyTorch Geometric Data objects."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data


def data_to_networkx(data: Data) -> nx.Graph:
    """Convert a PyTorch Geometric Data object to a NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    return G


def plot_graph(
    data: Data,
    ax: plt.Axes | None = None,
    title: str | None = None,
    node_size: int = 15,
    node_color: str = "steelblue",
    edge_color: str = "gray",
    seed: int = 42,
) -> plt.Axes:
    """Plot a single PyTorch Geometric Data object.

    Args:
        data: A PyTorch Geometric Data object.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        title: Custom title. If None, generates from graph stats.
        node_size: Size of nodes in the plot.
        node_color: Color of nodes.
        edge_color: Color of edges.
        seed: Random seed for layout reproducibility.

    Returns:
        The matplotlib Axes object.
    """
    _, ax = plt.subplots()

    G = data_to_networkx(data)

    # Graph statistics
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2  # Undirected, so divide by 2
    label = data.y.item() if data.y is not None else "N/A"
    avg_degree = sum(dict(G.degree()).values()) / num_nodes

    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=seed, k=0.5, iterations=50)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.3, edge_color=edge_color)

    if title is None:
        title = f"Nodes: {num_nodes} | Edges: {num_edges}\nAvg degree: {avg_degree:.1f} | Label: {label}"

    ax.set_title(title, fontsize=12)
    ax.axis("off")


def plot(data: Data, save_path: str):
    """Plot a single PyTorch Geometric Data object and save it to a file in the 'images' directory."""
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    plot_graph(data)
    plt.savefig(images_dir / save_path, dpi=150)
    plt.close()