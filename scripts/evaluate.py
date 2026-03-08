"""
Evaluation Module for scGPT Perturbation Experiments

Based on previous/evaluate2.py evaluation logic

Metrics:
- delta_pearson: Pearson correlation of perturbation delta (predicted - control)
- total_pearson: Pearson correlation of total expression
- rmse: Root Mean Square Error
- top_100_overlap: Overlap of top-100 changed genes
- direction_acc: Direction accuracy of gene changes
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Optional
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"--- Random seed set to: {seed} ---")


def evaluate_model(
    infer_path: str,
    test_path: str,
    core_genes_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    control_drug: str = "DMSO_TF",
    top_k: int = 100
) -> pd.DataFrame:
    """
    Evaluate model predictions against ground truth
    
    Args:
        infer_path: Path to predicted results (h5ad)
        test_path: Path to ground truth test data (h5ad)
        core_genes_path: Path to selected genes CSV (optional)
        output_dir: Directory to save results
        control_drug: Name of control/drug condition
        top_k: Number of top changed genes to evaluate
        
    Returns:
        DataFrame with evaluation results
    """
    print(f"Loading data...")
    print(f"  Inference: {infer_path}")
    print(f"  Test: {test_path}")
    
    # Load data
    adata_infer = sc.read_h5ad(infer_path)
    adata_test = sc.read_h5ad(test_path)
    
    print(f"Inference shape: {adata_infer.shape}")
    print(f"Test shape: {adata_test.shape}")
    
    # 1. Gene alignment
    print("Aligning genes...")
    # Check if gene names match (need same length first)
    if len(adata_infer.var_names) == len(adata_test.var_names) and all(adata_infer.var_names == adata_test.var_names):
        print("  Gene names match perfectly, no alignment needed")
    else:
        # Find common genes
        common_genes = adata_infer.var_names[np.isin(adata_infer.var_names, adata_test.var_names)]
        print(f"  Inference genes: {len(adata_infer.var_names)}")
        print(f"  Test genes: {len(adata_test.var_names)}")
        print(f"  Common genes: {len(common_genes)}")
        
        if len(common_genes) == 0:
            raise ValueError("No common genes found between inference and test data!")
        
        adata_infer = adata_infer[:, common_genes].copy()
        adata_test = adata_test[:, common_genes].copy()
    
    # 2. Filter to core genes if provided
    if core_genes_path and os.path.exists(core_genes_path):
        print(f"Filtering to core genes from: {core_genes_path}")
        selected_genes_df = pd.read_csv(core_genes_path)
        core_genes = [g.upper() for g in selected_genes_df['gene'].tolist()]
        
        # Get genes that exist in both
        test_genes_upper = adata_test.var_names.str.upper()
        core_genes = [g for g in core_genes if g in test_genes_upper.values]
        
        mask_test = test_genes_upper.isin(core_genes)
        mask_infer = adata_infer.var_names.str.upper().isin(core_genes)
        
        adata_test = adata_test[:, mask_test].copy()
        adata_infer = adata_infer[:, mask_infer].copy()
        print(f"Evaluating on {len(core_genes)} core genes")
    
    results = []
    cell_lines = adata_test.obs['cell_line'].unique()
    gene_names = adata_test.var_names.values
    
    print(f"\nEvaluating {len(cell_lines)} cell lines...")
    
    for cl in tqdm(cell_lines, desc="Evaluating"):
        test_cl = adata_test[adata_test.obs['cell_line'] == cl]
        infer_cl = adata_infer[adata_infer.obs['cell_line'] == cl]
        
        def get_mean(adata_obj, drug_name):
            """Get mean expression for a specific drug"""
            subset = adata_obj[adata_obj.obs['drug'] == drug_name]
            if subset.n_obs == 0:
                return None
            # Handle sparse matrix
            X = subset.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            return np.mean(X, axis=0).ravel()
        
        # Get control (DMSO) expression from test data
        ctrl_test_mean = get_mean(test_cl, control_drug)
        
        # If no DMSO in test data, try to get it from inference results
        if ctrl_test_mean is None:
            ctrl_infer_mean = get_mean(infer_cl, control_drug)
            if ctrl_infer_mean is not None:
                print(f"  Using predicted DMSO for cell line {cl}")
                ctrl_test_mean = ctrl_infer_mean
            else:
                # Last resort: use mean of all predictions for this cell line as control
                all_pred = infer_cl.X
                if hasattr(all_pred, 'toarray'):
                    all_pred = all_pred.toarray()
                if all_pred.shape[0] > 0:
                    ctrl_test_mean = np.mean(all_pred, axis=0).ravel()
                    print(f"  Using mean expression as control for cell line {cl}")
                else:
                    print(f"Warning: No predictions for cell line {cl}")
                    continue
        
        if ctrl_test_mean is None:
            print(f"Warning: No control data for cell line {cl}")
            continue
        
        # Get drugs (excluding control)
        drugs = [d for d in test_cl.obs['drug'].unique() if d != control_drug]
        
        for drug in drugs:
            real_drug_mean = get_mean(test_cl, drug)
            pred_drug_mean = get_mean(infer_cl, drug)
            
            if real_drug_mean is None or pred_drug_mean is None:
                continue
            
            # Calculate delta (perturbation effect)
            delta_real = real_drug_mean - ctrl_test_mean
            delta_pred = pred_drug_mean - ctrl_test_mean
            
            # Core metrics
            # 1. Delta Pearson correlation
            p_corr, p_value = pearsonr(np.nan_to_num(delta_real), np.nan_to_num(delta_pred))
            
            # 2. Total expression Pearson correlation
            total_p_corr, _ = pearsonr(np.nan_to_num(real_drug_mean), np.nan_to_num(pred_drug_mean))
            
            # 3. RMSE
            rmse = np.sqrt(mean_squared_error(real_drug_mean, pred_drug_mean))
            
            # 4. Top-K gene overlap
            top_k_real_idx = np.argsort(np.abs(delta_real))[-top_k:]
            top_k_pred_idx = np.argsort(np.abs(delta_pred))[-top_k:]
            
            overlap = len(set(top_k_real_idx) & set(top_k_pred_idx)) / top_k
            
            # 5. Direction accuracy
            # Check if predicted direction matches real direction for top-K genes
            match_direction = np.sign(delta_real[top_k_real_idx]) == np.sign(delta_pred[top_k_real_idx])
            direction_acc = np.mean(match_direction)
            
            # 6. Top-10 gene names (for analysis)
            top_10_real_genes = gene_names[np.argsort(np.abs(delta_real))[-10:][::-1]].tolist()
            top_10_pred_genes = gene_names[np.argsort(np.abs(delta_pred))[-10:][::-1]].tolist()
            
            results.append({
                'cell_line': cl,
                'drug': drug,
                'delta_pearson': p_corr,
                'delta_p_value': p_value,
                'total_pearson': total_p_corr,
                'rmse': rmse,
                f'top_{top_k}_overlap': overlap,
                'direction_acc': direction_acc,
                'top_10_real_genes': '|'.join(top_10_real_genes),
                'top_10_pred_genes': '|'.join(top_10_pred_genes)
            })
    
    # Create results DataFrame
    res_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    if len(res_df) > 0:
        summary = res_df.drop(columns=['top_10_real_genes', 'top_10_pred_genes']).groupby('cell_line').mean(numeric_only=True)
        
        # Overall summary
        overall = {
            'delta_pearson': res_df['delta_pearson'].mean(),
            'total_pearson': res_df['total_pearson'].mean(),
            'rmse': res_df['rmse'].mean(),
            f'top_{top_k}_overlap': res_df[f'top_{top_k}_overlap'].mean(),
            'direction_acc': res_df['direction_acc'].mean()
        }
    else:
        summary = pd.DataFrame()
        overall = {}
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detail results (per cell_line x drug)
        res_df.to_csv(f"{output_dir}/detail.csv", index=False)
        
        # Save summary results (per cell_line)
        if len(summary) > 0:
            summary.to_csv(f"{output_dir}/summary.csv")
        
        # Save overall metrics
        with open(f"{output_dir}/overall.txt", 'w') as f:
            for k, v in overall.items():
                f.write(f"{k}: {v:.4f}\n")
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - detail.csv: Detailed results per cell_line x drug")
        print(f"  - summary.csv: Summary results per cell_line")
        print(f"  - overall.txt: Overall metrics")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    if overall:
        print(f"\nOverall Metrics:")
        print(f"  Delta Pearson:     {overall['delta_pearson']:.4f}")
        print(f"  Total Pearson:    {overall['total_pearson']:.4f}")
        print(f"  RMSE:             {overall['rmse']:.4f}")
        print(f"  Top-{top_k} Overlap:     {overall[f'top_{top_k}_overlap']:.4f}")
        print(f"  Direction Acc:    {overall['direction_acc']:.4f}")
    
    if len(summary) > 0:
        print(f"\nPer Cell Line:")
        print(summary.round(4).to_string())
    
    print("\n" + "="*60)
    
    return res_df


def evaluate_by_celltype(
    infer_path: str,
    test_path: str,
    output_dir: str,
    control_drug: str = "DMSO_TF"
) -> pd.DataFrame:
    """
    Simplified evaluation grouped by cell type only
    """
    return evaluate_model(
        infer_path=infer_path,
        test_path=test_path,
        output_dir=output_dir,
        control_drug=control_drug
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate scGPT perturbation predictions")
    
    # Required arguments
    parser.add_argument('--infer-path', '-i', type=str, required=True,
                       help='Path to predicted results (h5ad)')
    parser.add_argument('--test-path', '-t', type=str, required=True,
                       help='Path to ground truth test data (h5ad)')
    
    # Optional arguments
    parser.add_argument('--core-genes', '-c', type=str, default=None,
                       help='Path to selected genes CSV')
    parser.add_argument('--output-dir', '-o', type=str, default='./evaluation',
                       help='Output directory for results')
    parser.add_argument('--control-drug', type=str, default='DMSO_TF',
                       help='Control drug name (default: DMSO_TF)')
    parser.add_argument('--top-k', type=int, default=100,
                       help='Top-K genes for overlap evaluation (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Run evaluation
    evaluate_model(
        infer_path=args.infer_path,
        test_path=args.test_path,
        core_genes_path=args.core_genes,
        output_dir=args.output_dir,
        control_drug=args.control_drug,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
