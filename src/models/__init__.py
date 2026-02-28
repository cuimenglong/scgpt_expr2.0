"""
Model definitions for scGPT perturbation experiments
"""

from .baseline import scGPTBaseline
from .protein import scGPTWithVirtualProtein
from .ppi import scGPTWithVirtualProteinAndPPI
from .target_bias import scGPTWithTargetBias
from .metaselection import scGPTWithMetadata

__all__ = [
    'scGPTBaseline',
    'scGPTWithVirtualProtein', 
    'scGPTWithVirtualProteinAndPPI',
    'scGPTWithTargetBias',
    'scGPTWithMetadata'
]
