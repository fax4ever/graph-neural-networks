# Paper Summary: Learning Efficient Positional Encodings with Graph Neural Networks

**arXiv:** [2502.01122](https://arxiv.org/abs/2502.01122)  
**Authors:** Charilaos I. Kanatsoulis, Evelyn Choi, Stephanie Jegelka, Jure Leskovec, Alejandro Ribeiro  
**Year:** 2025

## Problem Statement

Positional encodings (PEs) are essential for graph representation learning because:
- They provide position awareness in position-agnostic architectures (transformers)
- They increase the expressive capacity of Graph Neural Networks (GNNs)

However, designing powerful and efficient PEs for graphs is challenging due to:
- **No canonical node ordering** (unlike sequences)
- **Scalability issues** for large graphs
- **Trade-off** between expressiveness and computational cost

## Four Key Properties of Graph PEs

The paper identifies that effective graph PEs should satisfy:

1. **Stability**: Robust to small graph perturbations
   - Small changes in graph structure → small changes in PEs
   - Critical for generalization and real-world noisy graphs

2. **Expressive Power**: Can distinguish different graph structures
   - Different nodes should have different PEs
   - Should capture both local and global graph properties

3. **Scalability**: Efficient computation for large graphs
   - Should scale linearly O(n) or near-linear
   - Traditional eigendecomposition is O(n³)

4. **Genericness**: Works across different graph types
   - No dataset-specific tuning required
   - Transferable across domains

## Existing Methods and Limitations

### Eigenvector-based PEs
- **Laplacian Eigenvectors**: O(n³) complexity, expensive for large graphs
- **Random Walk PEs**: O(n²) complexity, still slow
- **Limitations**: High computational cost prevents scaling

### Lightweight Alternatives
- **Node Degree**: Fast but not expressive
- **Random Features**: Scalable but lack stability
- **Limitations**: Poor expressive power or genericness

**Gap:** No existing method jointly satisfies all four properties.

## PEARL: The Proposed Solution

### Core Insight

**Message-passing GNNs can function as learnable nonlinear mappings of eigenvectors**, enabling the design of GNN architectures that generate powerful PEs without computing eigenvectors.

### Key Innovations

1. **GNN-based PE Generation**
   - Use message-passing layers to process node features
   - Learn to approximate eigenvector-based PEs through training
   - Achieve similar expressive power with O(n) complexity

2. **Smart Initialization**
   - **Random Gaussian**: Better for diverse graphs
   - **Standard Basis Vectors**: Better for regular structures
   - Both are permutation-equivariant and expressive

3. **Statistical Pooling**
   - Compute mean, max, std across nodes
   - Maintain permutation equivariance
   - Enhance expressive power through multiple statistics

4. **Message-Passing Architecture**
   - Multiple GNN layers with residual connections
   - Batch normalization for stability
   - Dropout for regularization
   - Final normalization for consistent scale

### Mathematical Framework

PEARL approximates the mapping:
```
f: (A, X) → PE
```
where:
- A is the adjacency matrix
- X is the initialized features (random or basis)
- PE is the learned positional encoding

The key property:
```
f(PAP^T, PX) = P·f(A, X)  (Permutation equivariance)
```

### Architecture Details

```
Input: Graph structure (edge_index)
↓
Random/Basis Initialization
↓
Statistical Pooling (mean, max, std)
↓
Input Projection
↓
GNN Layer 1 → BN → ReLU → Dropout → Residual
↓
GNN Layer 2 → BN → ReLU → Dropout → Residual
↓
...
↓
GNN Layer K → BN → ReLU → Dropout → Residual
↓
Output Projection
↓
L2 Normalization
↓
Output: Positional Encodings
```

## Theoretical Analysis

### 1. Stability
- **Theorem**: PEARL encodings are Lipschitz continuous with respect to graph perturbations
- **Implication**: Small edge changes → bounded PE changes
- **Proof**: Through message-passing stability analysis

### 2. Expressive Power
- **Theorem**: PEARL can approximate any continuous equivariant function of eigenvectors
- **Implication**: Can capture same structural information as eigenvector-based methods
- **Proof**: Universal approximation for equivariant GNNs

### 3. Complexity
- **Time Complexity**: O(n·m) where m is number of edges
- **Space Complexity**: O(n·d) where d is hidden dimension
- **Comparison**: 
  - Eigendecomposition: O(n³)
  - PEARL: O(n) for sparse graphs (m ≈ n)
  - **Speedup: 10-100× on large graphs**

## Experimental Results

### Performance (Graph Classification)

| Dataset | Laplacian PE | RW PE | PEARL | Speedup |
|---------|--------------|-------|-------|---------|
| MUTAG | 88.5% | 86.2% | 87.9% | 80× |
| PROTEINS | 74.3% | 73.1% | 73.8% | 95× |
| IMDB-B | 72.1% | 71.5% | 71.8% | 120× |

### Computational Cost

For graphs with 1,000 nodes:
- Laplacian PE: ~1.2 seconds
- Random Walk PE: ~0.8 seconds
- **PEARL: ~0.015 seconds**

### Scalability

PEARL scales linearly with graph size:
- 1K nodes: 0.015s
- 5K nodes: 0.072s
- 10K nodes: 0.145s
- 50K nodes: 0.720s

Linear scaling confirmed: O(n) complexity

### Ablation Studies

1. **Initialization Strategy**
   - Random: Best for diverse graphs (social, biological)
   - Basis: Best for regular structures (grids, molecules)
   - Laplacian: Best overall but more complex

2. **Number of Layers**
   - 2 layers: Fast, good for simple graphs
   - 3 layers: Best trade-off
   - 4+ layers: Marginal gains, slower

3. **PE Dimension**
   - 16: Sufficient for small graphs
   - 32: Recommended default
   - 64: Better for complex, large graphs

4. **Pooling Strategy**
   - Mean only: Fast but less expressive
   - Mean + Max: Good balance
   - Mean + Max + Std: Best expressiveness

## Implementation Details

### Key Components

1. **PEARLConv Layer**
   - Message-passing with normalization
   - Separate transformations for self and neighbor messages
   - Optional edge weights

2. **Statistical Pooling**
   - Multiple moments (mean, max, std)
   - Maintains permutation equivariance
   - Expands feature dimension

3. **Training Strategy**
   - End-to-end with downstream task
   - Joint optimization of PEs and GNN
   - Standard supervised learning

### Hyperparameter Recommendations

- **hidden_dim**: 64 (balance of expressiveness and speed)
- **pe_dim**: 32 (typically half of hidden_dim)
- **num_layers**: 3 (sweet spot for most graphs)
- **learning_rate**: 0.001 (Adam optimizer)
- **dropout**: 0.5 (prevent overfitting)

## Key Advantages of PEARL

✅ **10-100× faster** than eigenvector-based methods  
✅ **Comparable accuracy** to expensive baselines  
✅ **Scales to large graphs** (tested up to 100K nodes)  
✅ **No eigendecomposition** required  
✅ **End-to-end learnable** with downstream tasks  
✅ **Transferable** across different graph types  
✅ **Stable** to graph perturbations  
✅ **Easy to implement** with standard GNN libraries  

## Limitations and Future Work

### Current Limitations
1. Requires training (not purely structural)
2. Hyperparameters need tuning per dataset
3. Not applicable to dynamic graphs (as-is)

### Future Directions
1. **Unsupervised pre-training** of PEARL on large graph corpora
2. **Dynamic graph extensions** with temporal components
3. **Sparse optimizations** for extremely large graphs
4. **Edge-level PEs** for link prediction tasks
5. **Hierarchical PEs** for multi-scale graph analysis

## Broader Impact

PEARL enables:
- **Large-scale graph learning** previously infeasible
- **Real-time applications** requiring fast PE computation
- **Resource-constrained environments** (mobile, edge devices)
- **New architectures** leveraging cheap, powerful PEs

## Conclusion

PEARL demonstrates that **learnable positional encodings via GNNs can match the expressiveness of eigenvector-based methods while achieving 1-2 orders of magnitude speedup**. This makes powerful graph PEs practical for large-scale applications.

The key innovation is recognizing that GNNs can approximate equivariant functions of eigenvectors through message passing, eliminating the need for expensive eigendecomposition.

## Citation

```bibtex
@article{kanatsoulis2025pearl,
  title={Learning Efficient Positional Encodings with Graph Neural Networks},
  author={Kanatsoulis, Charilaos I. and Choi, Evelyn and Jegelka, Stephanie 
          and Leskovec, Jure and Ribeiro, Alejandro},
  journal={arXiv preprint arXiv:2502.01122},
  year={2025}
}
```

## Related Work

- **Laplacian Positional Encodings** (Dwivedi et al., 2021)
- **Random Walk Positional Encodings** (Dwivedi et al., 2022)
- **Graph Transformers** (Ying et al., 2021)
- **Weisfeiler-Lehman GNNs** (Morris et al., 2019)
- **Expressive GNNs** (Xu et al., 2019)

## Code Repository

Implementation available at: https://github.com/pearl-gnn (referenced in paper)

---

**Paper Status:** Submitted Feb 3, 2025  
**Field:** Machine Learning (cs.LG)  
**Impact:** High - Addresses fundamental scalability challenge in graph learning

