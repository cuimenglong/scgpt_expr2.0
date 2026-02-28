"""
Gene Processor Factory - Handles different gene processing strategies for different experiments
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from typing import List, Dict, Optional, Set
from abc import ABC, abstractmethod

# Import utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GeneProcessor(ABC):
    """Abstract base class for gene processors"""
    
    @abstractmethod
    def process(self, adata, config: Dict) -> List[str]:
        """
        Process genes from adata, return final gene list
        
        Args:
            adata: Scanpy AnnData object
            config: Configuration dictionary
            
        Returns:
            List of selected gene names
        """
        pass
    
    @abstractmethod
    def get_inference_genes(self, checkpoint_dir: str) -> List[str]:
        """Get gene list for inference from checkpoint"""
        pass
    
    def save_gene_list(self, gene_list: List[str], save_path: str):
        """Save gene list to CSV"""
        df = pd.DataFrame({'gene': gene_list})
        df.to_csv(save_path, index=False)
        print(f"Gene list saved to: {save_path}")


class BaselineGeneProcessor(GeneProcessor):
    """
    Baseline processor: Uses only HVG (Highly Variable Genes)
    """
    
    def process(self, adata, config: Dict) -> List[str]:
        n_hvg = config.get('n_hvg', 2000)
        
        adata.var_names = [g.upper() for g in adata.var_names]
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            flavor='seurat_v3', 
            check_values=False
        )
        
        hvg_genes = adata.var_names[adata.var.highly_variable].tolist()
        print(f"[BaselineGeneProcessor] Selected {len(hvg_genes)} HVG genes")
        
        return hvg_genes
    
    def get_inference_genes(self, checkpoint_dir: str) -> List[str]:
        genes_path = os.path.join(checkpoint_dir, "selected_genes.csv")
        df = pd.read_csv(genes_path)
        return df['gene'].tolist()


class ProteinGeneProcessor(GeneProcessor):
    """
    Protein processor: HVG + ESM embeddings
    Filters genes to those with ESM protein embeddings
    """
    
    def process(self, adata, config: Dict) -> List[str]:
        n_hvg = config.get('n_hvg', 2000)
        esm_path = config.get('esm_path', '')
        
        # 1. HVG selection
        adata.var_names = [g.upper() for g in adata.var_names]
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            flavor='seurat_v3', 
            check_values=False
        )
        hvg_genes = set(adata.var_names[adata.var.highly_variable].tolist())
        print(f"[ProteinGeneProcessor] HVG genes: {len(hvg_genes)}")
        
        # 2. Load ESM embeddings
        esm_genes = set()
        if esm_path and os.path.exists(esm_path):
            esm_data = torch.load(esm_path)
            esm_genes = set(esm_data.keys())
            print(f"[ProteinGeneProcessor] ESM embedded genes: {len(esm_genes)}")
        
        # 3. Get intersection
        final_genes = list(hvg_genes & esm_genes)
        print(f"[ProteinGeneProcessor] Final genes (HVG ∩ ESM): {len(final_genes)}")
        
        return final_genes
    
    def get_inference_genes(self, checkpoint_dir: str) -> List[str]:
        genes_path = os.path.join(checkpoint_dir, "selected_genes.csv")
        df = pd.read_csv(genes_path)
        return df['gene'].tolist()


class PPIGeneProcessor(GeneProcessor):
    """
    PPI processor: HVG + ESM + PPI Network
    Filters genes to those with ESM embeddings, then builds PPI network
    Uses local TSV file and mygene for ID translation
    """
    
    def __init__(self):
        self.ppi_stats = None
        self.ppi_adjacency = None
        
    def process(self, adata, config: Dict) -> List[str]:
        n_hvg = config.get('n_hvg', 2000)
        esm_path = config.get('esm_path', '')
        ppi_tsv_path = config.get('ppi_tsv_path', None)  # Local PPI TSV file
        ppi_cache_dir = config.get('ppi_cache_dir', './data')
        
        # 1. HVG selection
        adata.var_names = [g.upper() for g in adata.var_names]
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            flavor='seurat_v3', 
            check_values=False
        )
        hvg_genes = set(adata.var_names[adata.var.highly_variable].tolist())
        print(f"[PPIGeneProcessor] HVG genes: {len(hvg_genes)}")
        
        # 2. Load ESM embeddings
        esm_genes = set()
        if esm_path and os.path.exists(esm_path):
            esm_data = torch.load(esm_path)
            esm_genes = set(esm_data.keys())
            print(f"[PPIGeneProcessor] ESM embedded genes: {len(esm_genes)}")
        
        # 3. Get ESM covered genes
        esm_covered = hvg_genes & esm_genes
        print(f"[PPIGeneProcessor] ESM covered genes: {len(esm_covered)}")
        
        # 4. Build PPI network using new PPINetworkLoader
        if config.get('use_ppi', True):
            from ..utils.ppi_utils import PPINetworkLoader
            
            # Initialize with local TSV path if provided
            ppi_loader = PPINetworkLoader(tsv_path=ppi_tsv_path, cache_dir=ppi_cache_dir)
            
            # Load and translate PPI network
            selected_genes_list = list(esm_covered)
            ppi_loader.load_and_translate(selected_genes_list)
            
            # Get PPI statistics
            self.ppi_stats = ppi_loader.get_ppi_statistics(selected_genes_list)
            print(f"[PPIGeneProcessor] PPI Statistics:")
            print(f"  - Total proteins: {self.ppi_stats['total_proteins']}")
            print(f"  - Proteins in PPI: {self.ppi_stats['proteins_in_ppi']}")
            print(f"  - PPI coverage: {self.ppi_stats.get('coverage_rate', 0):.1f}%")
            print(f"  - PPI edges: {self.ppi_stats.get('ppi_edges', 0)}")
            
            # Create PPI adjacency matrix
            self.ppi_adjacency = ppi_loader.create_adjacency_matrix(
                selected_genes_list, 
                include_self_loop=False
            )
        
        return list(esm_covered)
    
    def get_inference_genes(self, checkpoint_dir: str) -> List[str]:
        genes_path = os.path.join(checkpoint_dir, "selected_genes.csv")
        df = pd.read_csv(genes_path)
        return df['gene'].tolist()
    
    def get_ppi_data(self) -> tuple:
        """Get PPI statistics and adjacency matrix"""
        return self.ppi_stats, self.ppi_adjacency


class GeneProcessorFactory:
    """
    Factory class for creating gene processors
    """
    
    _processors = {
        'baseline': BaselineGeneProcessor,
        'protein': ProteinGeneProcessor,
        'ppi': PPIGeneProcessor,
        'target_bias': BaselineGeneProcessor,  # Same as baseline - uses HVG
        'metaselection': BaselineGeneProcessor,  # Same as baseline - uses HVG
    }
    
    @classmethod
    def get_processor(cls, experiment_type: str) -> GeneProcessor:
        """
        Get gene processor by experiment type
        
        Args:
            experiment_type: Type of experiment (baseline, protein, ppi, etc.)
            
        Returns:
            GeneProcessor instance
        """
        processor_class = cls._processors.get(experiment_type.lower(), BaselineGeneProcessor)
        print(f"[GeneProcessorFactory] Using processor: {processor_class.__name__}")
        return processor_class()
    
    @classmethod
    def register_processor(cls, name: str, processor_class: type):
        """Register a new processor type"""
        cls._processors[name.lower()] = processor_class


# Export
__all__ = [
    'GeneProcessor',
    'BaselineGeneProcessor', 
    'ProteinGeneProcessor',
    'PPIGeneProcessor',
    'GeneProcessorFactory'
]
