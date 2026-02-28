"""
Data processing modules for scGPT perturbation experiments
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Set
import scanpy as sc
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Add parent directory to path for scgpt imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class H5ADPerturbationDataset(Dataset):
    """
    Dataset for perturbation prediction from h5ad file
    Supports ChemBERTa drug embeddings and control pool matching
    """
    def __init__(
        self,
        adata,
        drug_to_target_nodes: Dict[str, List[int]] = {},
        control_drug_name: str = "DMSO_TF",
        device: str = "cuda",
        compute_drug_embeddings: bool = False,
        use_knn_matching: bool = False,    # NEW: Use KNN for control matching
        metadata_cols: Optional[List[str]] = None  # NEW: Metadata columns for KNN
    ):
        # Gene filtering
        if 'selected' in adata.var.columns:
            mask = adata.var['selected'] == True
            self.adata = adata[:, mask].copy()
        else:
            self.adata = adata.copy()
            
        self.selected_gene_names = self.adata.var_names.tolist()
        self.use_knn_matching = use_knn_matching
        self.metadata_cols = metadata_cols or []
        
        # Precompute cell_line indices
        if 'cell_line_idx' not in self.adata.obs.columns:
            self.adata.obs['cell_line_idx'] = self.adata.obs['cell_line'].astype('category').cat.codes
            
        # Extract perturbation indices
        self.perturb_indices = np.where(self.adata.obs['drug'] != control_drug)[0]
        self.control_drug_name = control_drug_name
        self.drug_to_target_nodes = drug_to_target_nodes
        self.device = device
        
        # Build control pool
        self.control_pool = {}
        all_obs = self.adata.obs
        
        if use_knn_matching and metadata_cols:
            # KNN-based control matching (for metaselection experiment)
            print("Building KNN indices for metadata matching...")
            for cl in all_obs['cell_line'].unique():
                ctrl_mask = (all_obs['cell_line'] == cl) & (all_obs['drug'] == control_drug)
                ctrl_indices = np.where(ctrl_mask)[0]
                
                if len(ctrl_indices) > 0:
                    # Build KNN index using metadata features
                    ctrl_meta_features = all_obs.iloc[ctrl_indices][self.metadata_cols].values
                    knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(ctrl_meta_features)
                    self.control_pool[cl] = {
                        "indices": ctrl_indices,
                        "knn": knn
                    }
        else:
            # Simple control matching by cell line (random)
            for cl in all_obs['cell_line'].unique():
                indices = np.where((all_obs['cell_line'] == cl) & (all_obs['drug'] == control_drug))[0]
                if len(indices) > 0:
                    self.control_pool[cl] = indices
        
        # Precompute ChemBERTa embeddings if requested
        self.drug_embeddings = {}
        if compute_drug_embeddings:
            self.drug_embeddings = self._precompute_drug_embeddings()

    def _precompute_drug_embeddings(self, embedding_dim: int = 768) -> Dict[str, torch.Tensor]:
        """
        Precompute ChemBERTa embeddings for all unique drugs
        
        Args:
            embedding_dim: Dimension of drug embeddings (768 for ChemBERTa-77M-MLM)
            
        Returns:
            Dictionary mapping drug name to embedding tensor
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            print("Warning: transformers not installed. Using random embeddings.")
            return {}
        
        # Get unique drug-SMILES pairs
        if 'canonical_smiles' not in self.adata.obs.columns:
            print("Warning: 'canonical_smiles' not in obs. Using random embeddings.")
            return {}
            
        drug_smiles_df = self.adata.obs[['drug', 'canonical_smiles']].drop_duplicates()
        
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", use_safetensors=True).to(self.device)
        model.eval()
        
        embeddings_dict = {}
        
        print(f"Precomputing ChemBERTa embeddings for {len(drug_smiles_df)} drugs...")
        
        with torch.no_grad():
            for _, row in tqdm(drug_smiles_df.iterrows(), total=len(drug_smiles_df)):
                drug_name = row['drug']
                smiles = row['canonical_smiles']
                
                # Handle control drug or empty SMILES
                if drug_name == self.control_drug_name or not isinstance(smiles, str) or smiles == "":
                    embeddings_dict[drug_name] = torch.zeros(embedding_dim)
                    continue
                
                # Tokenize and move to device
                inputs = tokenizer(smiles, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=128).to(self.device)
                
                # Forward pass
                outputs = model(**inputs)
                
                # Get CLS token representation
                emb = outputs.last_hidden_state[0, 0, :].cpu()
                embeddings_dict[drug_name] = emb
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        return embeddings_dict
    
    def _get_expression(self, idx: int) -> torch.Tensor:
        """Get gene expression for a given index"""
        expr = self.adata.X[idx]
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        return torch.tensor(expr, dtype=torch.float32).squeeze()
    
    def __len__(self):
        return len(self.perturb_indices)
    
    def __getitem__(self, idx: int):
        p_idx = self.perturb_indices[idx]
        row = self.adata.obs.iloc[p_idx]
        
        p_cl = row['cell_line']
        p_drug = row['drug']
        
        # Get perturbation expression
        p_gene = self._get_expression(p_idx)
        
        # Get control expression with KNN matching or random
        if self.use_knn_matching and self.metadata_cols and p_cl in self.control_pool:
            # KNN-based matching: find most similar control cell by metadata
            pool = self.control_pool[p_cl]
            dist, indices = pool["knn"].kneighbors(row[self.metadata_cols].values.reshape(1, -1))
            c_idx = pool["indices"][indices[0][0]]
        elif p_cl in self.control_pool:
            # Simple random matching by cell line
            if isinstance(self.control_pool[p_cl], dict):
                c_idx = np.random.choice(self.control_pool[p_cl]["indices"])
            else:
                c_idx = np.random.choice(self.control_pool[p_cl])
        else:
            c_idx = np.random.choice(list(range(len(self.adata))))
        
        c_gene = self._get_expression(c_idx)
        
        # Get metadata if using KNN
        metadata = None
        if self.use_knn_matching and self.metadata_cols:
            metadata = torch.tensor(row[self.metadata_cols].values.astype(np.float32))
        
        # Get target nodes and cell type
        target_nodes = torch.tensor(self.drug_to_target_nodes.get(p_drug, []), dtype=torch.long)
        cell_type_id = int(row['cell_line_idx'])
        
        # Get drug embedding (precomputed or zeros)
        drug_emb = self.drug_embeddings.get(p_drug, torch.zeros(768))
        
        result = {
            'c_gene': c_gene,          # Control gene expression
            'p_gene': p_gene,          # Perturbation gene expression  
            'cell_type_id': cell_type_id,
            'drug_emb': drug_emb,      # ChemBERTa embedding
            'target_nodes': target_nodes,  # Target gene indices for target_bias model
            'drug': p_drug
        }
        
        # Add metadata if available
        if metadata is not None:
            result['metadata'] = metadata
        
        return result


class SimplePerturbationDataset(Dataset):
    """
    Simplified dataset for basic perturbation prediction
    """
    def __init__(
        self,
        adata,
        control_drug_name: str = "DMSO_TF",
        device: str = "cuda"
    ):
        # Gene filtering
        if 'selected' in adata.var.columns:
            mask = adata.var['selected'] == True
            self.adata = adata[:, mask].copy()
        else:
            self.adata = adata.copy()
            
        self.selected_gene_names = self.adata.var_names.tolist()
        
        # Precompute cell_line indices
        if 'cell_line_idx' not in self.adata.obs.columns:
            self.adata.obs['cell_line_idx'] = self.adata.obs['cell_line'].astype('category').cat.codes
            
        # Extract perturbation indices
        self.perturb_indices = np.where(self.adata.obs['drug'] != control_drug)[0]
        self.control_drug_name = control_drug_name
        self.device = device
        
        # Build control pool
        self.control_pool = {}
        all_obs = self.adata.obs
        for cl in all_obs['cell_line'].unique():
            indices = np.where((all_obs['cell_line'] == cl) & (all_obs['drug'] == control_drug))[0]
            if len(indices) > 0:
                self.control_pool[cl] = indices

    def __len__(self):
        return len(self.perturb_indices)
    
    def __getitem__(self, idx: int):
        p_idx = self.perturb_indices[idx]
        row = self.adata.obs.iloc[p_idx]
        
        p_cl = row['cell_line']
        
        # Get perturbation expression
        p_gene = self._get_expression(p_idx)
        
        # Get control expression
        if p_cl in self.control_pool:
            c_idx = np.random.choice(self.control_pool[p_cl])
        else:
            c_idx = np.random.choice(list(range(len(self.adata))))
        c_gene = self._get_expression(c_idx)
        
        cell_type_id = int(row['cell_line_idx'])
        
        return {
            'gene_expr': p_gene,
            'dmso_expr': c_gene,
            'drug': row['drug'],
            'cell_line_idx': cell_type_id
        }
    
    def _get_expression(self, idx: int) -> torch.Tensor:
        expr = self.adata.X[idx]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        return torch.tensor(expr, dtype=torch.float32)


def load_adata_and_filter_genes(
    adata_path: str,
    n_hvg: int = 2000,
    gene_list: Optional[List[str]] = None,
    gene_upper: bool = True
) -> tuple:
    """
    Load h5ad and filter genes
    
    Args:
        adata_path: Path to h5ad file
        n_hvg: Number of highly variable genes to select
        gene_list: If provided, use this gene list instead of HVG
        gene_upper: Whether to convert gene names to uppercase
        
    Returns:
        adata: Filtered adata
        selected_genes: List of selected gene names
    """
    adata = sc.read_h5ad(adata_path)
    
    if gene_upper:
        adata.var_names = [g.upper() for g in adata.var_names]
    
    if gene_list is not None:
        # Filter to specific gene list
        gene_set = set(g.upper() if gene_upper else g for g in gene_list)
        existing_genes = [g for g in adata.var_names if (g.upper() if gene_upper else g) in gene_set]
        adata = adata[:, existing_genes].copy()
        selected_genes = existing_genes
    else:
        # Use HVG
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            flavor='seurat_v3', 
            check_values=False
        )
        adata = adata[:, adata.var.highly_variable].copy()
        selected_genes = adata.var_names.tolist()
    
    return adata, selected_genes


def load_drug_metadata(drug_meta_path: str) -> pd.DataFrame:
    """Load drug metadata"""
    if drug_meta_path.endswith('.parquet'):
        return pd.read_parquet(drug_meta_path)
    elif drug_meta_path.endswith('.csv'):
        return pd.read_csv(drug_meta_path)
    else:
        raise ValueError(f"Unsupported file format: {drug_meta_path}")


def prepare_inference_data(
    test_h5ad_path: str,
    drug_meta_path: str,
    selected_genes_path: str,
    use_file_hvg: bool = True
) -> tuple:
    """
    Prepare data for inference
    
    Returns:
        adata_full, core_X, train_gene_list, drug_meta, core_gene_idx_in_full, valid_mask_in_core
    """
    # Load data
    adata_full = sc.read_h5ad(test_h5ad_path)
    adata_full.var_names = [g.upper() for g in adata_full.var_names]
    
    # Load selected genes
    selected_genes_df = pd.read_csv(selected_genes_path)
    train_gene_list = selected_genes_df['gene'].astype(str).str.upper().tolist()
    
    # Load drug metadata
    drug_meta = load_drug_metadata(drug_meta_path)
    
    # Filter genes
    if use_file_hvg and 'highly_variable' in adata_full.var.columns:
        HVG_mask = adata_full.var['highly_variable'].values
        all_genes_upper = adata_full.var_names.str.upper()
        core_gene_mask = np.isin(all_genes_upper, train_gene_list)
        valid_mask = HVG_mask & core_gene_mask
    else:
        all_genes_upper = adata_full.var_names.str.upper()
        core_gene_mask = np.isin(all_genes_upper, train_gene_list)
        valid_mask = core_gene_mask
    
    # Get core gene indices
    core_gene_idx_in_full = np.where(valid_mask)[0]
    n_core = len(core_gene_idx_in_full)
    
    # Build core expression matrix
    core_X = np.zeros((adata_full.n_obs, n_core), dtype=np.float32)
    for i, col_idx in enumerate(core_gene_idx_in_full):
        val = adata_full.X[:, col_idx]
        if hasattr(val, 'toarray'):
            val = val.toarray().flatten()
        core_X[:, i] = val
    
    return adata_full, core_X, train_gene_list, drug_meta, core_gene_idx_in_full, valid_mask
