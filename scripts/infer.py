"""
Unified Inference Script for scGPT Perturbation Experiments

Usage:
    # Inference only
    python scripts/infer.py --checkpoint checkpoints1/model.pt --output infer/result.h5ad
    
    # Inference with evaluation
    python scripts/infer.py --checkpoint checkpoints1/model.pt --output infer/result.h5ad --evaluate
"""

import os
import sys

# Set environment variables BEFORE importing torch
os.environ["HF_SKIP_CHECK_TORCH_LOAD_SAFE"] = "True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Apply torchtext patch BEFORE importing torch
from types import ModuleType
try:
    import torch
    
    # 1. Mock torchtext._extension to prevent loading .so library
    mock_extension = ModuleType("torchtext._extension")
    mock_extension._init_extension = lambda: None
    sys.modules["torchtext._extension"] = mock_extension
    
    # 2. Intercept torch.ops.load_library
    orig_load_library = torch.ops.load_library
    def mocked_load(path):
        if "libtorchtext" in path:
            return
        return orig_load_library(path)
    torch.ops.load_library = mocked_load
    
    # 3. Construct mock Vocab class and module
    class MockVocab:
        def __init__(self, vocab):
            self.vocab = vocab
            self.itos = list(vocab.keys()) if isinstance(vocab, dict) else []
        def __len__(self): 
            return len(self.itos)
    
    # Create mock torchtext.vocab module
    mt_vocab = ModuleType("torchtext.vocab")
    mt_vocab.Vocab = MockVocab
    sys.modules["torchtext.vocab"] = mt_vocab
    
    # Create mock torchtext top-level module
    mt_root = ModuleType("torchtext")
    mt_root.vocab = mt_vocab
    sys.modules["torchtext"] = mt_root
    
    print("--- torchtext patch applied successfully ---")
    
except Exception as e:
    print(f"--- torchtext patch failed: {e} ---")

import argparse
import json
import numpy as np

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
from src.models import (
    scGPTBaseline, 
    scGPTWithVirtualProtein, 
    scGPTWithVirtualProteinAndPPI,
    scGPTWithTargetBias,
    scGPTWithMetadata
)


def get_model_class(model_name: str):
    """Get model class by name"""
    model_map = {
        'baseline': scGPTBaseline,
        'protein': scGPTWithVirtualProtein,
        'ppi': scGPTWithVirtualProteinAndPPI,
        'target_bias': scGPTWithTargetBias,
        'metaselection': scGPTWithMetadata,
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


def run_evaluation(infer_path: str, test_path: str, core_genes_path: str, output_dir: str):
    """Run evaluation after inference"""
    from evaluate import evaluate_model
    
    print("\n" + "="*60)
    print("Running Evaluation...")
    print("="*60)
    
    evaluate_model(
        infer_path=infer_path,
        test_path=test_path,
        core_genes_path=core_genes_path,
        output_dir=output_dir
    )


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
    
    # Add target_bias params
    if args.model == 'target_bias':
        # Load gene_ids for target_bias model
        gene_ids_tensor = torch.tensor(
            [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in train_gene_list],
            dtype=torch.long
        )
        model_kwargs['gene_ids'] = gene_ids_tensor
        model_kwargs['target_bias_value'] = args.target_bias_value
    
    # Add metadata_dim for metaselection
    if args.model == 'metaselection':
        model_kwargs['metadata_dim'] = 4
    
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
    
    # Build drug to target mapping for target_bias model
    drug_to_target_gene_ids = {}
    if args.model == 'target_bias' and args.drug_meta:
        for _, row in drug_meta.iterrows():
            drug = row['drug']
            targets = str(row['targets']).split(',') if pd.notna(row.get('targets', '')) else []
            t_ids = []
            for t in targets:
                t_upper = t.upper().strip()
                if t_upper in vocab_dict:
                    t_ids.append(vocab_dict[t_upper])
            if len(t_ids) > 0:
                drug_to_target_gene_ids[drug] = t_ids
    
    # Simplified inference loop
    with torch.no_grad():
        for i in range(adata_full.n_obs):
            # Get expression
            expr = torch.tensor(core_X[i], dtype=torch.float32).to(device)
            
            # Get cell type
            cell_line = adata_full.obs.iloc[i]['cell_line']
            cell_type_id = hash(cell_line) % n_celltype
            
            # Get drug embedding (for now use zeros or random - should load precomputed)
            drug_name = adata_full.obs.iloc[i]['drug']
            drug_emb = torch.randn(1, 384).to(device)
            
            # Forward
            gene_ids = torch.tensor(
                [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in train_gene_list],
                dtype=torch.long
            ).to(device)
            
            if args.model == 'target_bias':
                target_gene_ids = drug_to_target_gene_ids.get(drug_name, [])
                if target_gene_ids:
                    target_tensor = torch.tensor([target_gene_ids], dtype=torch.long).to(device)
                else:
                    target_tensor = torch.tensor([[-1]], dtype=torch.long).to(device)
                
                pred = model(
                    c_gene=expr.unsqueeze(0),
                    drug_emb=drug_emb,
                    cell_type_id=torch.tensor([cell_type_id], device=device),
                    target_gene_ids=target_tensor
                )
            else:
                pred = model(
                    gene_ids.unsqueeze(0),
                    expr.unsqueeze(0),
                    drug_emb,
                    torch.tensor([cell_type_id], device=device)
                )
            
            predictions.append(pred.cpu().numpy())
            cell_lines.append(cell_line)
            drugs.append(drug_name)
    
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
    
    # Run evaluation if requested
    if args.evaluate:
        # Extract checkpoint name for output directory
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
        eval_output_dir = f"./evaluation/{checkpoint_name}"
        run_evaluation(
            infer_path=args.output,
            test_path=args.test_data,
            core_genes_path=selected_genes_path,
            output_dir=eval_output_dir
        )
    
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
                       choices=['baseline', 'protein', 'ppi', 'target_bias', 'metaselection'],
                       help='Model type')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--target-bias-value', type=float, default=5.0,
                       help='Target bias value for target_bias model')
    parser.add_argument('--n-controls', type=int, default=5,
                       help='Number of control samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # Evaluation options
    parser.add_argument('--evaluate', '-e', action='store_true',
                       help='Run evaluation after inference')
    
    args = parser.parse_args()
    
    run_inference(args)


if __name__ == "__main__":
    main()
