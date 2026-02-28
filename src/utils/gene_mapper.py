"""
Gene Symbol Mapper Utility
Maps Ensembl IDs, lncRNAs, etc. to gene symbols using MyGene.info API
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Set
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GeneSymbolMapper:
    """Maps various gene identifiers to official gene symbols"""
    
    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.ensembl_to_symbol = {}
        self.alias_to_symbol = {}
        self.symbol_to_symbol = {}
        
    def load_builtin_mappings(self):
        """Load built-in mappings for common gene types"""
        
        # Common lncRNA name mappings
        lncrna_mappings = {
            'LINC00113': 'LINC00113', 'MEG3': 'MEG3', 'MEG8': 'MEG8',
            'XACT': 'XACT', 'PEG10': 'PEG10', 'H19': 'H19', 'MAL': 'MAL',
            'RMST': 'RMST', 'CARMN': 'CARMN', 'MIR100HG': 'MIR100HG',
            'MIR99AHG': 'MIR99AHG', 'MIR7-3HG': 'MIR7-3HG', 'MIR493HG': 'MIR493HG',
        }
        
        for alias, symbol in lncrna_mappings.items():
            self.alias_to_symbol[alias.upper()] = symbol
            self.symbol_to_symbol[symbol.upper()] = symbol
    
    def query_mygene(self, gene_ids: List[str], batch_size: int = 1000) -> Dict:
        """Query MyGene.info for gene symbol mappings"""
        
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            
            # Find Ensembl IDs that need mapping
            ensembl_ids = [g for g in gene_ids if g.startswith('ENSG')]
            logger.info(f"Querying MyGene.info for {len(ensembl_ids)} Ensembl IDs...")
            
            # Query in batches
            all_results = []
            for i in range(0, len(ensembl_ids), batch_size):
                batch = ensembl_ids[i:i+batch_size]
                results = mg.querymany(
                    batch, 
                    scopes='ensemblgene', 
                    fields='symbol', 
                    species='human', 
                    verbose=False
                )
                all_results.extend(results)
                logger.info(f"  Processed {min(i+batch_size, len(ensembl_ids))}/{len(ensembl_ids)}")
            
            # Build mapping from results
            for r in all_results:
                if 'notfound' not in r or not r.get('notfound', False):
                    if 'query' in r and 'symbol' in r:
                        self.ensembl_to_symbol[r['query'].upper()] = r['symbol']
                        
            logger.info(f"Found {len(self.ensembl_to_symbol)} Ensembl -> Symbol mappings")
            return self.ensembl_to_symbol
            
        except ImportError:
            logger.warning("MyGene.info not available. Install with: pip install mygene")
            return {}
    
    def map_gene(self, gene_name: str) -> str:
        """Map a gene name to its canonical symbol"""
        
        if not gene_name or pd.isna(gene_name):
            return gene_name
            
        gene = str(gene_name).strip()
        gene_upper = gene.upper()
        
        # Already a valid symbol
        if gene_upper in self.symbol_to_symbol:
            return self.symbol_to_symbol[gene_upper]
            
        # Check alias mapping
        if gene_upper in self.alias_to_symbol:
            return self.alias_to_symbol[gene_upper]
            
        # Check Ensembl ID mapping
        if gene_upper in self.ensembl_to_symbol:
            return self.ensembl_to_symbol[gene_upper]
            
        # Return original if no mapping found
        return gene
        
    def map_gene_list(self, gene_list: List[str]) -> List[str]:
        """Map a list of gene names"""
        return [self.map_gene(g) for g in gene_list]


def map_h5ad_genes(h5ad_path: str, output_path: Optional[str] = None) -> str:
    """
    Map genes in an h5ad file to canonical symbols
    
    Args:
        h5ad_path: Path to input h5ad file
        output_path: Path to output file (if None, overwrites input)
        
    Returns:
        Path to output file
    """
    import scanpy as sc
    
    logger.info(f"Processing: {h5ad_path}")
    
    # Initialize mapper
    mapper = GeneSymbolMapper()
    mapper.load_builtin_mappings()
    
    # Read h5ad file
    adata = sc.read_h5ad(h5ad_path)
    original_genes = adata.var_names.tolist()
    
    # Query MyGene.info
    ensembl_ids = [g for g in original_genes if g.startswith('ENSG')]
    if ensembl_ids:
        mapper.query_mygene(ensembl_ids)
    
    # Map genes
    mapped_genes = mapper.map_gene_list(original_genes)
    
    # Update var_names
    adata.var_names = mapped_genes
    
    # Determine output path
    if output_path is None:
        output_path = h5ad_path
        backup_path = h5ad_path + ".backup"
        logger.info(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(h5ad_path, backup_path)
    
    # Save
    adata.write_h5ad(output_path)
    logger.info(f"Saved to: {output_path}")
    
    return output_path
