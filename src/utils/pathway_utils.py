"""
Pathway Utilities for loading KEGG/Reactome/MSigDB pathway gene sets
and constructing pathway-level attention bias matrices.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Set, Optional, Tuple

logger = logging.getLogger(__name__)


class PathwayLoader:
    """Load pathway gene sets and build co-pathway adjacency matrices."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        self.pathway_gene_sets: Dict[str, Set[str]] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def load_from_gmt(self, gmt_path: str) -> None:
        """
        Parse a local GMT file into pathway_gene_sets dict.
        GMT format: pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
        """
        if not os.path.exists(gmt_path):
            raise FileNotFoundError(f"GMT file not found: {gmt_path}")

        self.pathway_gene_sets = {}
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                pathway_name = parts[0]
                genes = {g.upper().strip() for g in parts[2:] if g.strip()}
                if genes:
                    self.pathway_gene_sets[pathway_name] = genes

        logger.info(f"Loaded {len(self.pathway_gene_sets)} pathways from {gmt_path}")

    def load_msigdb(self, collection: str = "C2", subcollection: str = "CP:KEGG",
                    species: str = "Homo sapiens") -> None:
        """
        Download pathway gene sets from MSigDB via gseapy.
        Falls back to loading a cached GMT file if available.
        """
        cache_file = os.path.join(self.cache_dir, f"msigdb_{collection}_{subcollection.replace(':', '_')}.gmt")

        if os.path.exists(cache_file):
            logger.info(f"Loading cached pathway file: {cache_file}")
            self.load_from_gmt(cache_file)
            return

        try:
            import gseapy as gp
            logger.info(f"Downloading MSigDB {collection}/{subcollection} via gseapy...")

            gene_sets = gp.get_library(name=subcollection.replace("CP:", "").replace(":", "_"),
                                       organism="Human")
            self.pathway_gene_sets = {}
            for pathway_name, genes in gene_sets.items():
                self.pathway_gene_sets[pathway_name] = {g.upper() for g in genes}

            # Cache as GMT
            self._save_as_gmt(cache_file)
            logger.info(f"Downloaded {len(self.pathway_gene_sets)} pathways, cached to {cache_file}")

        except Exception as e:
            logger.warning(f"gseapy download failed: {e}")
            # Try alternative: direct KEGG download
            try:
                self._load_kegg_fallback()
            except Exception as e2:
                logger.warning(f"KEGG fallback also failed: {e2}")
                logger.warning("No pathway data available. Continuing without pathways.")

    def _load_kegg_fallback(self) -> None:
        """Fallback: load KEGG pathways via gseapy library names."""
        import gseapy as gp
        available = gp.get_library_name(organism="Human")
        kegg_libs = [lib for lib in available if 'KEGG' in lib.upper()]
        if kegg_libs:
            gene_sets = gp.get_library(name=kegg_libs[0], organism="Human")
            self.pathway_gene_sets = {
                name: {g.upper() for g in genes}
                for name, genes in gene_sets.items()
            }
            logger.info(f"Loaded {len(self.pathway_gene_sets)} pathways from {kegg_libs[0]}")

    def _save_as_gmt(self, path: str) -> None:
        """Save pathway_gene_sets as GMT file."""
        with open(path, 'w') as f:
            for name, genes in self.pathway_gene_sets.items():
                f.write(f"{name}\tna\t" + "\t".join(sorted(genes)) + "\n")

    def build_pathway_matrix(self, gene_list: List[str],
                             min_genes_in_pathway: int = 3) -> Tuple[np.ndarray, List[str]]:
        """
        Build binary membership matrix (n_genes x n_pathways).
        Only includes pathways with >= min_genes_in_pathway overlap with gene_list.

        Returns:
            matrix: np.ndarray (n_genes, n_pathways)
            pathway_names: list of pathway names
        """
        gene_set = {g.upper() for g in gene_list}
        gene_to_idx = {g.upper(): i for i, g in enumerate(gene_list)}

        valid_pathways = []
        for name, genes in self.pathway_gene_sets.items():
            overlap = genes & gene_set
            if len(overlap) >= min_genes_in_pathway:
                valid_pathways.append((name, overlap))

        n_genes = len(gene_list)
        n_pathways = len(valid_pathways)
        matrix = np.zeros((n_genes, n_pathways), dtype=np.float32)
        pathway_names = []

        for j, (name, overlap_genes) in enumerate(valid_pathways):
            pathway_names.append(name)
            for gene in overlap_genes:
                if gene in gene_to_idx:
                    matrix[gene_to_idx[gene], j] = 1.0

        logger.info(f"Pathway matrix: {n_genes} genes x {n_pathways} pathways "
                     f"(filtered from {len(self.pathway_gene_sets)} total)")
        return matrix, pathway_names

    def build_co_pathway_adjacency(self, gene_list: List[str],
                                    min_shared_pathways: int = 1,
                                    min_genes_in_pathway: int = 3) -> np.ndarray:
        """
        Build co-pathway adjacency matrix (n_genes x n_genes).
        Entry (i,j) = normalized number of shared pathways between gene i and gene j.

        Returns:
            np.ndarray (n_genes, n_genes) float32, values in [0, 1]
        """
        matrix, _ = self.build_pathway_matrix(gene_list, min_genes_in_pathway)

        # Co-pathway = matrix @ matrix.T (counts shared pathways)
        co_pathway = matrix @ matrix.T

        # Zero out diagonal (no self-loops from pathways)
        np.fill_diagonal(co_pathway, 0)

        # Filter by minimum shared pathways
        co_pathway[co_pathway < min_shared_pathways] = 0

        # Normalize to [0, 1]
        max_val = co_pathway.max()
        if max_val > 0:
            co_pathway = co_pathway / max_val

        n_edges = np.count_nonzero(co_pathway) // 2
        logger.info(f"Co-pathway adjacency: {len(gene_list)} genes, "
                     f"{n_edges} gene pairs sharing pathways")
        return co_pathway.astype(np.float32)

    def get_pathway_attention_bias(self, gene_list: List[str],
                                    bias_scale: float = 1.0) -> np.ndarray:
        """
        Construct attention bias from co-pathway membership.
        Returns (n_genes, n_genes) float bias matrix.
        """
        co_pathway = self.build_co_pathway_adjacency(gene_list)
        return (co_pathway * bias_scale).astype(np.float32)
