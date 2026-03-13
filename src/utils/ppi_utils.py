"""
PPI Network Loader Module
Load Protein-Protein Interaction network from local TSV file
Based on the new reference code that supports local interaction network
"""

import os
import json
import pandas as pd
import numpy as np
import logging
import requests
import time
from typing import Dict, Set, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PPINetworkLoader:
    """Load and process PPI network from local TSV file"""
    
    def __init__(self, tsv_path: str = None, cache_dir: str = "./data"):
        """
        Initialize PPI Network Loader
        
        Args:
            tsv_path: Path to local PPI TSV file (e.g., omnipath_ppi.tsv)
            cache_dir: Directory for caching downloaded data
        """
        self.tsv_path = tsv_path
        self.cache_dir = cache_dir
        self.ppi_network = None  # Dict[gene, Set[neighbor_genes]]
        self.protein_pairs = set()
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_uniprot_mapping_via_mygene(self, uniprot_ids: List[str]) -> Dict[str, str]:
        """
        Use mygene library for ID translation (most stable)
        
        Args:
            uniprot_ids: List of UniProt IDs
            
        Returns:
            Dictionary mapping UniProt ID to Gene Symbol
        """
        try:
            import mygene
            logger.info(f"Using mygene library for ID translation ({len(uniprot_ids)} IDs)...")
            mg = mygene.MyGeneInfo()
            
            # Batch processing for large lists
            mapping = {}
            batch_size = 2000
            
            for i in range(0, len(uniprot_ids), batch_size):
                batch = uniprot_ids[i:i+batch_size]
                results = mg.querymany(
                    batch, 
                    scopes='uniprot', 
                    fields='symbol', 
                    species='human',
                    verbose=False
                )
                
                for res in results:
                    # Check if the query was successfully matched
                    if 'query' in res:
                        if 'notfound' in res and res['notfound']:
                            continue
                        if 'symbol' in res:
                            mapping[res['query']] = res['symbol']
                
                logger.info(f"Translated {len(mapping)} IDs so far...")
            
            logger.info(f"Successfully translated {len(mapping)} UniProt IDs to gene symbols")
            return mapping
        except ImportError:
            logger.warning("mygene library not installed, trying alternative methods...")
            return {}
    
    def _get_uniprot_mapping_via_api(self, uniprot_ids: List[str]) -> Dict[str, str]:
        """
        Direct UniProt REST API translation (no extra library needed)
        Note: For large datasets, mygene is recommended
        """
        logger.info("Trying UniProt REST API for translation...")
        # Simplified implementation - for production, consider using mygene
        return {}
    
    def download_omnipath(self, organism: str = "human", save_path: str = None) -> str:
        """
        Download PPI data from OmniPath database
        
        Args:
            organism: Organism name, default: human
            save_path: Path to save the downloaded file
            
        Returns:
            Path to the downloaded file
        """
        if save_path is None:
            save_path = os.path.join(self.cache_dir, "omnipath_ppi.tsv")
        
        if os.path.exists(save_path):
            logger.info(f"Using cached file: {save_path}")
            return save_path
        
        logger.info("Downloading PPI data from OmniPath database...")
        
        # Try multiple URLs for better reliability
        urls = [
            # OmniPath direct download
            f"https://omnipathdb.org/interactions?organism=9606&format=tsv",
            f"https://omnipathdb.org/interactions?datasets=omnipath&organism=9606&format=tsv",
            # Alternative format
            f"https://omnipathdb.org/interactions?organism={organism}&format=tsv",
        ]
        
        for url in urls:
            try:
                logger.info(f"Requesting: {url}")
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                
                # Check if we got actual data
                if len(response.text.strip()) < 100:
                    logger.warning(f"Got empty or too small response from {url}")
                    continue
                
                # Save as TSV
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"PPI data saved to: {save_path}")
                return save_path
                
            except Exception as e:
                logger.warning(f"Download failed from {url}: {e}")
                continue
        
        raise RuntimeError("Cannot download PPI data from OmniPath. Please check network connection.")
    
    def load_and_translate(self, selected_genes: List[str],
                          min_evidence_count: int = 0,
                          interaction_types: Optional[List[str]] = None) -> None:
        """
        Load PPI network and translate to gene symbols

        Args:
            selected_genes: List of selected gene symbols
            min_evidence_count: Minimum literature references to keep (0 = no filtering)
            interaction_types: List of interaction types to keep (None = all)
        """
        tsv_path = self.tsv_path
        if tsv_path is None:
            tsv_path = os.path.join(self.cache_dir, "omnipath_ppi.tsv")

        if not os.path.exists(tsv_path):
            logger.info("PPI file not found locally, downloading from OmniPath...")
            tsv_path = self.download_omnipath(save_path=tsv_path)

        logger.info(f"Loading PPI data from: {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
        logger.info(f"Loaded {len(df)} interaction records")

        # --- Filtering by evidence count ---
        if min_evidence_count > 0:
            ref_col = None
            for col_name in ['n_references', 'n_resources', 'references']:
                if col_name in df.columns:
                    ref_col = col_name
                    break
            if ref_col:
                before = len(df)
                df = df[df[ref_col] >= min_evidence_count].copy()
                logger.info(f"Filtered by {ref_col} >= {min_evidence_count}: {before} -> {len(df)}")
            else:
                logger.warning("No evidence count column found, skipping evidence filtering")

        # --- Filtering by interaction type ---
        if interaction_types:
            type_col = None
            for col_name in ['type', 'interaction_type', 'consensus_direction']:
                if col_name in df.columns:
                    type_col = col_name
                    break
            if type_col:
                before = len(df)
                df = df[df[type_col].isin(interaction_types)].copy()
                logger.info(f"Filtered by {type_col} in {interaction_types}: {before} -> {len(df)}")
            else:
                logger.warning("No interaction type column found, skipping type filtering")

        # Store filtered DataFrame for weighted adjacency
        self.interaction_df = df

        if 'source_genesymbol' in df.columns and 'target_genesymbol' in df.columns:
            logger.info("TSV contains gene symbol columns, using directly.")
            uniprot_to_symbol = {}
            use_existing_symbols = True
        else:
            use_existing_symbols = False
            all_uniprots = list(
                set(df['source'].unique().astype(str)) |
                set(df['target'].unique().astype(str))
            )
            logger.info(f"Found {len(all_uniprots)} unique UniProt IDs, translating...")
            uniprot_to_symbol = self._get_uniprot_mapping_via_mygene(all_uniprots)
            if not uniprot_to_symbol:
                raise RuntimeError(
                    "All ID translation methods failed. Please check network connection "
                    "or install mygene: pip install mygene"
                )

        self.ppi_network = {}
        self.protein_pairs = set()
        self._edge_weights = {}
        selected_genes_upper = {str(g).upper().strip() for g in selected_genes}
        logger.info(f"Looking for PPI interactions among {len(selected_genes_upper)} genes")

        sample_genes = list(selected_genes_upper)[:10]
        logger.info(f"Sample selected genes: {sample_genes}")

        if use_existing_symbols:
            ppi_sample = df[['source_genesymbol', 'target_genesymbol']].dropna().head(5)
            logger.info(f"Sample PPI genes from TSV: {ppi_sample.values.tolist()}")

        # Determine weight column
        weight_col = None
        for col_name in ['n_references', 'n_resources']:
            if col_name in df.columns:
                weight_col = col_name
                break

        for _, row in df.iterrows():
            if use_existing_symbols:
                s_src_raw = row['source_genesymbol']
                s_tgt_raw = row['target_genesymbol']
            else:
                s_src_raw = uniprot_to_symbol.get(str(row['source']))
                s_tgt_raw = uniprot_to_symbol.get(str(row['target']))

            if pd.isna(s_src_raw) or pd.isna(s_tgt_raw):
                continue

            s_src = str(s_src_raw).upper().strip()
            s_tgt = str(s_tgt_raw).upper().strip()

            if s_src in selected_genes_upper and s_tgt in selected_genes_upper:
                if s_src == s_tgt:
                    continue
                if s_src not in self.ppi_network:
                    self.ppi_network[s_src] = set()
                if s_tgt not in self.ppi_network:
                    self.ppi_network[s_tgt] = set()

                self.ppi_network[s_src].add(s_tgt)
                self.ppi_network[s_tgt].add(s_src)

                pair_key = tuple(sorted((s_src, s_tgt)))
                self.protein_pairs.add(pair_key)

                if weight_col and pd.notna(row.get(weight_col)):
                    w = float(row[weight_col])
                    self._edge_weights[pair_key] = max(self._edge_weights.get(pair_key, 0), w)
                else:
                    self._edge_weights[pair_key] = max(self._edge_weights.get(pair_key, 0), 1.0)

        logger.info(
            f"PPI network built: {len(self.ppi_network)} genes, "
            f"{len(self.protein_pairs)} interactions"
        )
    
    def get_ppi_statistics(self, selected_genes: List[str] = None) -> Dict:
        """
        Calculate PPI statistics
        
        Args:
            selected_genes: Optional list of selected genes for coverage calculation
            
        Returns:
            Dictionary with statistics
        """
        if selected_genes is not None:
            total = len(selected_genes)
        else:
            total = 0
            
        in_ppi = len(self.ppi_network) if self.ppi_network else 0
        edges = len(self.protein_pairs)
        
        stats = {
            'total_proteins': in_ppi,
            'proteins_in_ppi': in_ppi,
            'ppi_edges': edges,
            'coverage_rate': (in_ppi / total * 100) if total > 0 else 0.0
        }
        
        logger.info(f"PPI Statistics: {stats}")
        return stats
    
    def create_adjacency_matrix(
        self, 
        gene_list: List[str], 
        include_self_loop: bool = True
    ) -> np.ndarray:
        """
        Create adjacency matrix for PPI network
        
        Args:
            gene_list: List of genes in order
            include_self_loop: Whether to include self-loops
            
        Returns:
            Adjacency matrix (n_genes x n_genes)
        """
        num_genes = len(gene_list)
        gene_to_idx = {str(g).upper().strip(): i for i, g in enumerate(gene_list)}
        
        adj = np.zeros((num_genes, num_genes), dtype=np.float32)
        
        if self.ppi_network:
            for s1, neighbors in self.ppi_network.items():
                if s1 in gene_to_idx:
                    for s2 in neighbors:
                        if s2 in gene_to_idx:
                            adj[gene_to_idx[s1], gene_to_idx[s2]] = 1.0
        
        if include_self_loop:
            np.fill_diagonal(adj, 1.0)
        
        return adj
    
    def create_weighted_adjacency_matrix(
        self,
        gene_list: List[str],
        normalize: bool = True,
        include_self_loop: bool = False
    ) -> np.ndarray:
        """
        Create confidence-weighted adjacency matrix (continuous values).

        Args:
            gene_list: List of genes in order
            normalize: If True, scale weights to [0, 1]
            include_self_loop: Whether to include self-loops

        Returns:
            Weighted adjacency matrix (n_genes x n_genes)
        """
        num_genes = len(gene_list)
        gene_to_idx = {str(g).upper().strip(): i for i, g in enumerate(gene_list)}

        adj = np.zeros((num_genes, num_genes), dtype=np.float32)

        if self.ppi_network and hasattr(self, '_edge_weights'):
            for pair_key, weight in self._edge_weights.items():
                s1, s2 = pair_key
                if s1 in gene_to_idx and s2 in gene_to_idx:
                    adj[gene_to_idx[s1], gene_to_idx[s2]] = weight
                    adj[gene_to_idx[s2], gene_to_idx[s1]] = weight
        elif self.ppi_network:
            for s1, neighbors in self.ppi_network.items():
                if s1 in gene_to_idx:
                    for s2 in neighbors:
                        if s2 in gene_to_idx:
                            adj[gene_to_idx[s1], gene_to_idx[s2]] = 1.0

        if normalize:
            max_val = adj.max()
            if max_val > 0:
                adj = adj / max_val

        if include_self_loop:
            np.fill_diagonal(adj, 1.0)

        n_edges = np.count_nonzero(adj) // 2
        logger.info(f"Weighted adjacency: {num_genes} genes, {n_edges} edges, max={adj.max():.3f}")
        return adj

    def create_bidirectional_attention_mask(
        self,
        gene_list: List[str]
    ) -> np.ndarray:
        """
        Create bidirectional attention mask for PPI network
        
        Args:
            gene_list: List of genes in order
            
        Returns:
            Attention mask (n_genes x n_genes)
            0 = allow attention, -inf = mask out
        """
        adj = self.create_adjacency_matrix(gene_list, include_self_loop=False)
        
        # Convert to attention mask (0 = allow, -inf = block)
        attention_mask = np.where(adj > 0, 0.0, -1e9)
        
        return attention_mask
    
    def get_protein_to_protein_indices(
        self, 
        gene_list: List[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Get indices of protein pairs for PPI attention
        
        Args:
            gene_list: List of genes
            
        Returns:
            Tuple of (source_indices, target_indices)
        """
        source_indices = []
        target_indices = []
        
        gene_to_idx = {str(g).upper().strip(): i for i, g in enumerate(gene_list)}
        
        for pair in self.protein_pairs:
            if pair[0] in gene_to_idx and pair[1] in gene_to_idx:
                source_indices.append(gene_to_idx[pair[0]])
                target_indices.append(gene_to_idx[pair[1]])
        
        return source_indices, target_indices


def load_ppi_network(
    gene_list: List[str], 
    tsv_path: str = None,
    cache_dir: str = "./data",
    save_dir: Optional[str] = None
) -> Tuple[Dict, np.ndarray, Dict]:
    """
    Convenience function to load PPI network and create adjacency matrix
    
    Args:
        gene_list: List of gene symbols
        tsv_path: Path to local PPI TSV file
        cache_dir: Directory for caching PPI data
        save_dir: Optional directory to save outputs
        
    Returns:
        Tuple of (ppi_network, adjacency_matrix, statistics)
    """
    loader = PPINetworkLoader(tsv_path=tsv_path, cache_dir=cache_dir)
    
    # Load and build network
    loader.load_and_translate(gene_list)
    
    # Get statistics
    stats = loader.get_ppi_statistics(gene_list)
    
    # Create adjacency matrix
    adj_matrix = loader.create_adjacency_matrix(gene_list)
    
    # Save if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save adjacency matrix
        np.save(os.path.join(save_dir, "ppi_adjacency.npy"), adj_matrix)
        
        # Save statistics
        with open(os.path.join(save_dir, "ppi_statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"PPI data saved to {save_dir}")
    
    return loader.ppi_network, adj_matrix, stats
