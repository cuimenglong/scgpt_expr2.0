"""
Gene Processor Factory - Handles different gene processing strategies for different experiments
"""

import os
import sys
import json
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


def load_esm_embeddings(esm_path: str) -> Dict[str, torch.Tensor]:
    """
    Load ESM embeddings from file.
    Supports two formats:
    1. Dict format: {gene_name: embedding_tensor} - legacy format
    2. Tensor format: (N, 1280) tensor + _genes.json mapping - new format
    
    Args:
        esm_path: Path to ESM embeddings file
        
    Returns:
        Dictionary mapping gene names to embedding tensors
    """
    if not os.path.exists(esm_path):
        return {}
    
    esm_data = torch.load(esm_path)
    
    # Check if it's a dict (legacy format)
    if isinstance(esm_data, dict):
        return esm_data
    
    # Check if it's a tensor with accompanying JSON mapping
    if isinstance(esm_data, torch.Tensor):
        # Try to load gene mapping
        mapping_path = esm_path.replace('.pt', '_genes.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                gene_to_idx = json.load(f)
            
            # Convert tensor to dict
            esm_dict = {}
            for gene_name, idx in gene_to_idx.items():
                if idx < esm_data.shape[0]:
                    esm_dict[gene_name] = esm_data[idx]
            return esm_dict
        else:
            print(f"Warning: ESM file is tensor but no _genes.json mapping found at {mapping_path}")
            return {}
    
    return {}


class BaselineGeneProcessor(GeneProcessor):
    """
    Baseline processor: Uses HVG genes from data, but creates model with full scGPT vocab
    This allows loading pretrained weights while using only data-relevant genes for training
    """
    
    def process(self, adata, config: Dict) -> List[str]:
        n_hvg = config.get('n_hvg', 2000)
        vocab_path = config.get('vocab_path', './scgpt/tokenizer/default_gene_vocab.json')
        
        # Load scGPT vocab (for weight loading, not for model input)
        self.vocab_dict = {}
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocab_dict = json.load(f)
            scgpt_genes = [g for g in self.vocab_dict.keys() if g not in {'<pad>', '<unk>', '<eos>'}]
            print(f"[BaselineGeneProcessor] scGPT vocab loaded: {len(scgpt_genes)} genes")
        else:
            print(f"[BaselineGeneProcessor] WARNING: Vocab file not found")
        
        # HVG selection
        adata.var_names = [g.upper() for g in adata.var_names]
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            flavor='seurat_v3', 
            check_values=False
        )
        
        hvg_genes = adata.var_names[adata.var.highly_variable].tolist()
        
        # Filter to genes that exist in vocab (if vocab loaded)
        if self.vocab_dict:
            final_genes = [g for g in hvg_genes if g in self.vocab_dict]
            missing = len(hvg_genes) - len(final_genes)
            if missing > 0:
                print(f"[BaselineGeneProcessor] {missing} HVG genes not in vocab, using {len(final_genes)}")
        else:
            final_genes = hvg_genes
        
        print(f"[BaselineGeneProcessor] Selected {len(final_genes)} HVG genes")
        
        return final_genes
    
    def get_vocab_dict(self) -> Dict:
        """Get full vocab dict for model construction"""
        return getattr(self, 'vocab_dict', {})
    
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
            esm_data = load_esm_embeddings(esm_path)
            esm_genes = set(esm_data.keys())
            print(f"[ProteinGeneProcessor] ESM embedded genes: {len(esm_genes)}")
        else:
            print(f"[ProteinGeneProcessor] WARNING: ESM file not found at {esm_path}")
            print("[ProteinGeneProcessor] Will use all HVG genes (no ESM filtering)")
        
        # 3. Get intersection
        if esm_genes:
            final_genes = list(hvg_genes & esm_genes)
        else:
            final_genes = list(hvg_genes)  # Use all HVG if no ESM
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
            esm_data = load_esm_embeddings(esm_path)
            esm_genes = set(esm_data.keys())
            print(f"[PPIGeneProcessor] ESM embedded genes: {len(esm_genes)}")
        else:
            print(f"[PPIGeneProcessor] WARNING: ESM file not found at {esm_path}")
            print("[PPIGeneProcessor] Will use all HVG genes (no ESM filtering)")
        
        # 3. Get ESM covered genes
        if esm_genes:
            esm_covered = hvg_genes & esm_genes
        else:
            esm_covered = hvg_genes  # Use all HVG if no ESM
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


class EnhancedGeneProcessor(PPIGeneProcessor):
    """
    Enhanced processor: HVG + ESM + Filtered PPI + Pathway priors.
    """

    def __init__(self):
        super().__init__()
        self.pathway_adjacency = None
        self.pathway_names = None

    def process(self, adata, config: Dict) -> List[str]:
        selected_genes = super().process(adata, config)

        # Re-create PPI with filtering if requested
        if config.get('filter_ppi', False):
            from ..utils.ppi_utils import PPINetworkLoader
            ppi_loader = PPINetworkLoader(
                tsv_path=config.get('ppi_tsv_path'),
                cache_dir=config.get('ppi_cache_dir', './data')
            )
            ppi_loader.load_and_translate(
                selected_genes,
                min_evidence_count=config.get('ppi_min_evidence', 2),
                interaction_types=config.get('ppi_interaction_types', None)
            )
            self.ppi_stats = ppi_loader.get_ppi_statistics(selected_genes)
            print(f"[EnhancedGeneProcessor] Filtered PPI: {self.ppi_stats['ppi_edges']} edges")

            if config.get('ppi_weighted', True):
                self.ppi_adjacency = ppi_loader.create_weighted_adjacency_matrix(
                    selected_genes, normalize=True, include_self_loop=False
                )
            else:
                self.ppi_adjacency = ppi_loader.create_adjacency_matrix(
                    selected_genes, include_self_loop=False
                )

        # Build pathway data
        if config.get('use_pathways', True):
            try:
                from ..utils.pathway_utils import PathwayLoader
                pathway_loader = PathwayLoader(cache_dir=config.get('pathway_cache_dir', './data'))
                gmt_path = config.get('pathway_gmt_path', None)
                if gmt_path:
                    pathway_loader.load_from_gmt(gmt_path)
                else:
                    pathway_loader.load_msigdb(
                        collection=config.get('pathway_collection', 'C2'),
                        subcollection=config.get('pathway_subcollection', 'CP:KEGG')
                    )
                self.pathway_adjacency = pathway_loader.build_co_pathway_adjacency(selected_genes)
                _, self.pathway_names = pathway_loader.build_pathway_matrix(selected_genes)
                print(f"[EnhancedGeneProcessor] Pathways: {len(self.pathway_names)} active, "
                      f"{np.count_nonzero(self.pathway_adjacency) // 2} co-pathway pairs")
            except Exception as e:
                print(f"[EnhancedGeneProcessor] Pathway loading failed: {e}, continuing without")

        return selected_genes

    def get_pathway_data(self):
        return self.pathway_adjacency, self.pathway_names


class GeneProcessorFactory:
    """
    Factory class for creating gene processors
    """

    _processors = {
        'baseline': BaselineGeneProcessor,
        'protein': ProteinGeneProcessor,
        'ppi': PPIGeneProcessor,
        'target_bias': BaselineGeneProcessor,
        'metaselection': BaselineGeneProcessor,
        'enhanced': EnhancedGeneProcessor,
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
    'EnhancedGeneProcessor',
    'GeneProcessorFactory'
]
