"""
PEARL: Positional Encodings Augmented with Representation Learning

A PyTorch implementation of learnable positional encodings for graphs.
"""

from .pearl import PEARL, PEARLWithLaplacianInit, PEARLConv, StatisticalPooling
from .gnn_with_pearl import GNNWithPEARL, TransformerWithPEARL, EnsemblePEARL

__version__ = '1.0.0'
__author__ = 'PEARL Research Team'

__all__ = [
    'PEARL',
    'PEARLWithLaplacianInit',
    'PEARLConv',
    'StatisticalPooling',
    'GNNWithPEARL',
    'TransformerWithPEARL',
    'EnsemblePEARL',
]

