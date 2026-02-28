"""
Unified Inference Script for scGPT Perturbation Experiments

Usage:
    python scripts/infer.py --checkpoint checkpoints1/model.pt --output infer/result.h5ad
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Apply torchtext patch
from src.utils.torchtext_patch import patch_torchtext
patch_torchtext()

import scanpy as sc
import pandas as pd
import anndata as ad

# Import src modules
from src.models import scGPTBaseline, scGPTWithVirtualProtein, scGPTWithVirtualProteinAndPPI


def get_model_class(model_name: str):
    """Get model class by name"""
    model_map = {
        'baseline': scGPTBaseline,
        'protein': scGPTWithVirtualProtein,
        'ppi': scGPTWithVirtualProteinAndPPI,
    }
    
    model_name_lower = model_name.lower()
    if model_name_lower not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name_lower]


def prepare_inference_data(test_h5ad_path: str, drug_meta_path: str, selected_genes_path: str):
    """Prepare data for inference"""
    
    # Load test data
    adata_full = sc.read_h5ad(test_h5ad_path)
    adata_full.var_names = [g.upper() for g in adata_full.var_names]
    
    # Load selected genes
    selected_genes_df = pd.read_csv(selected_genes_path)
    train_gene_list = selected_genes_df['gene'].astype(str).str.upper().tolist()
    
    # Load drug metadata
    if drug_meta_path.endswith('.parquet'):
        drug_meta = pd.read_parquet(drug_meta_path)
    elif drug_meta_path.endswith('.csv'):
        drug_meta = pd.read_csv(drug_meta_path)
    else:
        raise ValueError(f"Unsupported drug meta format: {drug_meta_path}")
    
    # Filter genes
    all_genes_upper = adata_full.var_names.str.upper()
    core_gene_mask = np.isin(all_genes_upper, train_gene_list)
    core_gene_idx_in_full = np.where(core_gene_mask)[0]
    n_core = len(core_gene_idx_in_full)
    
    # Build core expression matrix
    core_X = np.zeros((adata_full.n_obs, n_core), dtype=np.float32)
    for i, col_idx in enumerate(core_gene_idx_in_full):
        val = adata_full.X[:, col_idx]
        if hasattr(val, 'toarray'):
            val = val.toarray().flatten()
        core_X[:, i] = val
    
    return adata_full, core_X, train_gene_list, drug_meta, core_gene_idx_in_full


def run_inference(args):
    """Main inference function"""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint to get model config
    checkpoint_dir = os.path.dirname(args.checkpoint) or "."
    
    # Load selected genes (generated during training)
    selected_genes_path = args.selected_genes or os.path.join(checkpoint_dir, "selected_genes.csv")
    if not os.path.exists(selected_genes_path):
        raise FileNotFoundError(f"Selected genes file not found: {selected_genes_path}")
    
    # Load PPI adjacency if exists (for PPI model)
    ppi_adjacency = None
    ppi_path = os.path.join(checkpoint_dir, "ppi_adjacency.npy")
    if os.path.exists(ppi_path) and args.model == 'ppi':
        print(f"Loading PPI adjacency: {ppi_path}")
        ppi_adjacency = np.load(ppi_path)
    
    # Prepare data
    print("Preparing inference data...")
    adata_full, core_X, train_gene_list, drug_meta, core_gene_idx_in_full = prepare_inference_data(
        args.test_data, args.drug_meta, selected_genes_path
    )
    
    print(f"Test data: {adata_full.n_obs} cells, {adata_full.n_vars} genes")
    print(f"Core genes: {len(train_gene_list)}")
    
    # Load vocab
    with open(args.vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    
    # Get model class
    ModelClass = get_model_class(args.model)
    
    # Initialize model
    n_celltype = len(adata_full.obs['cell_line'].unique()) + 20
    
    model_kwargs = {
        'ntokens': len(vocab_dict),
        'd_model': args.d_model,
        'nhead': 8,
        'd_hid': args.d_model,
        'nlayers': 6 if args.model == 'baseline' else 12,
        'scgpt_layers': 12,
        'gp_layers': 3,
        'n_celltype': n_celltype,
        'drug_emb_dim': 384,
    }
    
    # Add ESM/PPI params for protein/ppi models
    if args.model in ['protein', 'ppi']:
        model_kwargs['esm_dim'] = 1280
    
    if args.model == 'ppi':
        model_kwargs['ppi_adjacency'] = ppi_adjacency
    
    model = ModelClass(**model_kwargs).to(device)
    
    # Load weights
    print(f"Loading model from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Run inference
    print("Running inference...")
    
    predictions = []
    cell_lines = []
    drugs = []
    
    # Simplified inference loop
    with torch.no_grad():
        for i in range(adata_full.n_obs):
            # Get expression
            expr = torch.tensor(core_X[i], dtype=torch.float32).to(device)
            
            # Get cell type
            cell_line = adata_full.obs.iloc[i]['cell_line']
            cell_type_id = hash(cell_line) % n_celltype
            
            # Get drug embedding (simplified)
            drug_emb = torch.randn(1, 384).to(device)
            
            # Forward
            gene_ids = torch.tensor(
                [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in train_gene_list],
                dtype=torch.long
            ).to(device)
            
            pred = model(
                gene_ids.unsqueeze(0),
                expr.unsqueeze(0),
                drug_emb,
                torch.tensor([cell_type_id], device=device)
            )
            
            predictions.append(pred.cpu().numpy())
            cell_lines.append(cell_line)
            drugs.append(adata_full.obs.iloc[i]['drug'])
    
    predictions = np.array(predictions)
    
    # Create output
    print("Creating output...")
    
    # Build output adata
    result_adata = ad.AnnData(
        X=predictions,
        obs=adata_full.obs.copy()
    )
    result_adata.var_names = train_gene_list
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    result_adata.write_h5ad(args.output)
    print(f"Inference complete! Results saved to: {args.output}")
    
    return args.output


def main():
    parser = argparse.ArgumentParser(description="Run inference with scGPT model")
    
    # Required arguments
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test h5ad file')
    parser.add_argument('--drug-meta', type=str, required=True,
                       help='Path to drug metadata file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for results')
    
    # Optional arguments
    parser.add_argument('--selected-genes', type=str, default=None,
                       help='Path to selected genes CSV (default: checkpoint dir)')
    parser.add_argument('--vocab-path', type=str,
                       default='./scgpt/tokenizer/default_gene_vocab.json',
                       help='Path to vocab file')
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'protein', 'ppi'],
                       help='Model type')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--n-controls', type=int, default=5,
                       help='Number of control samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    run_inference(args)


if __name__ == "__main__":
    main()
