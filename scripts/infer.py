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
from tqdm import tqdm
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
    
    # ========== Apply log1p transformation if needed ==========
    X_max = adata_full.X.max() if hasattr(adata_full.X, 'max') else adata_full.X.toarray().max()
    if X_max > 30:
        print(f"Applying log1p transformation to test data (max value: {X_max:.2f})...")
        sc.pp.log1p(adata_full)
    else:
        print(f"Test data appears to be already log1p transformed (max: {X_max:.2f}), skipping log1p")
    
    # Load selected genes (order matters - this is the order used during training)
    selected_genes_df = pd.read_csv(selected_genes_path)
    train_gene_list = selected_genes_df['gene'].astype(str).str.upper().tolist()
    
    # Get test data gene names (upper case for matching)
    test_genes_upper = adata_full.var_names.str.upper()
    
    # Build mapping from test gene -> index in test data
    test_gene_to_idx = {g: i for i, g in enumerate(test_genes_upper)}
    
    # For each gene in train list, find its index in test data (or -1 if not found)
    gene_idx_mapping = []
    missing_genes = []
    for gene in train_gene_list:
        if gene in test_gene_to_idx:
            gene_idx_mapping.append(test_gene_to_idx[gene])
        else:
            gene_idx_mapping.append(-1)  # Not found in test data
            missing_genes.append(gene)
    
    n_core = len(train_gene_list)
    
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found in test data, will use zeros")
        print(f"  Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"  Missing genes: {missing_genes}")
    
    # Build core expression matrix for all samples (using train gene order)
    core_X = np.zeros((adata_full.n_obs, n_core), dtype=np.float32)
    for i, test_idx in enumerate(gene_idx_mapping):
        if test_idx >= 0:
            val = adata_full.X[:, test_idx]
            if hasattr(val, 'toarray'):
                val = val.toarray().flatten()
            core_X[:, i] = val
        # else: keep zeros (already initialized)
    
    # actual_gene_list is the same as train_gene_list (we keep all genes, missing ones will be zeros)
    actual_gene_list = train_gene_list
    
    print(f"Selected genes from CSV: {len(train_gene_list)}")
    print(f"Genes found in test data: {n_core - len(missing_genes)} / {n_core}")
    
    # Load drug metadata
    if drug_meta_path.endswith('.parquet'):
        drug_meta = pd.read_parquet(drug_meta_path)
    elif drug_meta_path.endswith('.csv'):
        drug_meta = pd.read_csv(drug_meta_path)
    else:
        raise ValueError(f"Unsupported drug meta format: {drug_meta_path}")
    
    # Build core expression matrix for all samples
    core_X = np.zeros((adata_full.n_obs, n_core), dtype=np.float32)
    for i, col_idx in enumerate(core_gene_idx_in_full):
        val = adata_full.X[:, col_idx]
        if hasattr(val, 'toarray'):
            val = val.toarray().flatten()
        core_X[:, i] = val
    
    # ========== Build DMSO control pool ==========
    control_drug_name = "DMSO_TF"
    control_pool = {}
    
    for cl in adata_full.obs['cell_line'].unique():
        ctrl_mask = (adata_full.obs['cell_line'] == cl) & (adata_full.obs['drug'] == control_drug_name)
        ctrl_indices = np.where(ctrl_mask)[0]
        if len(ctrl_indices) > 0:
            # Store indices and their expressions
            ctrl_expressions = core_X[ctrl_indices]
            control_pool[cl] = {
                'indices': ctrl_indices,
                'expressions': ctrl_expressions
            }
    
    print(f"Built control pool for {len(control_pool)} cell lines")
    print(f"  Cell lines with DMSO: {list(control_pool.keys())[:10]}...")
    
    return adata_full, core_X, train_gene_list, actual_gene_list, drug_meta, gene_idx_mapping, control_pool


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


def run_inference(args, config=None):
    """Main inference function"""
    config = config or {}  # Handle case where config is not passed
    
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
    adata_full, core_X, train_gene_list, actual_gene_list, drug_meta, gene_idx_mapping, control_pool = prepare_inference_data(
        args.test_data, args.drug_meta, selected_genes_path
    )
    
    print(f"Test data: {adata_full.n_obs} cells, {adata_full.n_vars} genes")
    print(f"Core genes (from CSV): {len(train_gene_list)}")
    print(f"Core genes (in test data): {len(actual_gene_list)}")
    
    # Load vocab
    with open(args.vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    
    # Get model class
    ModelClass = get_model_class(args.model)
    
    # Initialize model
    n_celltype = 67
    
    # Build model kwargs
    # Note: ChemBERTa produces 384-dim embeddings
    model_kwargs = {
        'ntokens': len(vocab_dict),
        'd_model': args.d_model,
        'nhead': 8,
        'd_hid': args.d_model,
        'n_celltype': n_celltype,
        'drug_emb_dim': 384,  # ChemBERTa-77M-MLM output dimension
    }
    
    # Add model-specific parameters
    if args.model in ['protein', 'ppi']:
        model_kwargs['scgpt_layers'] = 12
        model_kwargs['gp_layers'] = 3
    elif args.model in ['baseline', 'metaselection', 'target_bias']:
        model_kwargs['nlayers'] = 12
    else:
        model_kwargs['nlayers'] = 12
    
    # Load ESM embeddings for protein/ppi models
    esm_embeddings = None
    esm_path = None
    
    if args.model in ['protein', 'ppi']:
        # Try to load ESM embeddings from config or checkpoint directory
        esm_path = getattr(args, 'esm_path', None)
        if not esm_path:
            esm_path = config.get('esm_path', None)
        
        if not esm_path:
            # Try checkpoint directory
            checkpoint_dir = os.path.dirname(args.checkpoint) or "."
            esm_path = os.path.join(checkpoint_dir, "protein_esm_embeddings.pt")
        
        if os.path.exists(esm_path):
            print(f"Loading ESM embeddings from: {esm_path}")
            esm_matrix = torch.load(esm_path, map_location='cpu')
            esm_dim = esm_matrix.shape[1]
            print(f"ESM embedding dimension: {esm_dim}")
            
            # Update model_kwargs with correct esm_dim
            model_kwargs['esm_dim'] = esm_dim
            
            # Build ESM embeddings for ALL genes (use zeros for missing genes)
            esm_embeddings = torch.zeros(len(actual_gene_list), esm_dim, dtype=torch.float32)
            valid_genes = []
            for i, g in enumerate(actual_gene_list):
                if g in vocab_dict:
                    vocab_idx = vocab_dict[g]
                    if vocab_idx < esm_matrix.shape[0]:
                        esm_embeddings[i] = esm_matrix[vocab_idx]
                        valid_genes.append(g)
            
            esm_embeddings = esm_embeddings.to(device)
            print(f"ESM embeddings loaded: shape {esm_embeddings.shape}, {len(valid_genes)} genes with embeddings")
        else:
            print(f"Warning: ESM embeddings not found at {esm_path}")
            esm_embeddings = None
    
    # Add PPI adjacency for ppi model
    if args.model == 'ppi':
        model_kwargs['ppi_adjacency'] = ppi_adjacency
    
    # Add target_bias params
    if args.model == 'target_bias':
        # Load gene_ids for target_bias model (using actual_gene_list)
        gene_ids_tensor = torch.tensor(
            [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in actual_gene_list],
            dtype=torch.long
        )
        model_kwargs['gene_ids'] = gene_ids_tensor
        model_kwargs['target_bias_value'] = args.target_bias_value
    
    # Add metadata_dim for metaselection
    if args.model == 'metaselection':
        model_kwargs['metadata_dim'] = 4  # Default: tscp_count, pcnt_mito, S_score, G2M_score
    
    # Check if we need to load metadata columns for metaselection
    use_metadata = args.model == 'metaselection'
    metadata_cols = ['tscp_count', 'pcnt_mito', 'S_score', 'G2M_score'] if use_metadata else None
    
    # Load cell_line vocabulary from training (for consistent ID mapping)
    # Try both possible filenames (early vocab is saved at training start)
    cell_line_vocab_path = os.path.join(checkpoint_dir, "cell_line_vocab_early.pkl")
    if not os.path.exists(cell_line_vocab_path):
        cell_line_vocab_path = os.path.join(checkpoint_dir, "cell_line_vocab.pkl")
    
    if os.path.exists(cell_line_vocab_path):
        import pickle
        with open(cell_line_vocab_path, 'rb') as f:
            cell_line_vocab = pickle.load(f)
        print(f"Loaded cell line vocabulary from: {cell_line_vocab_path}")
        print(f"  Vocabulary size: {len(cell_line_vocab)}")
    else:
        print(f"Warning: Cell line vocabulary not found at {cell_line_vocab_path}")
        print(f"  Will use data-driven mapping (may cause issues with zero-shot cell lines)")
        cell_line_vocab = list(adata_full.obs['cell_line'].unique())
    
    # Build cell_line to ID mapping from vocabulary (ensures consistency with training)
    cell_line_to_id = {cl: idx for idx, cl in enumerate(cell_line_vocab)}
    
    # Identify zero-shot cell lines (not in training vocabulary)
    data_cell_lines = set(adata_full.obs['cell_line'].unique())
    vocab_cell_lines = set(cell_line_vocab)
    zero_shot_cell_lines = data_cell_lines - vocab_cell_lines
    in_vocab_cell_lines = data_cell_lines & vocab_cell_lines
    
    print(f"Cell line mapping:")
    print(f"  - Test data cell lines: {len(data_cell_lines)}")
    print(f"  - In vocabulary (seen in training): {len(in_vocab_cell_lines)} - {in_vocab_cell_lines}")
    print(f"  - Zero-shot (not in training): {len(zero_shot_cell_lines)} - {zero_shot_cell_lines}")
    
    # Initialize model first (before using model properties)
    model = ModelClass(**model_kwargs).to(device)
    
    # Get cell type embedding layer from model
    cell_emb_layer = model.cell_emb if hasattr(model, 'cell_emb') else None
    
    # Load weights
    print(f"Loading model from: {args.checkpoint}")
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load with weights_only=False: {e}")
        # Try alternative loading
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    
    def get_cell_type_embedding(cell_line, cell_emb_layer, cell_line_to_id):
        """
        Get cell type embedding for a given cell line.
        For zero-shot cell lines, use average of known cell line embeddings.
        
        Args:
            cell_line: cell line name
            cell_emb_layer: the embedding layer from model
            cell_line_to_id: mapping from cell line name to ID
            
        Returns:
            cell_type_id: tensor of shape (1,) for model input
        """
        if cell_line in cell_line_to_id:
            # Known cell line - use direct mapping
            cell_type_id = cell_line_to_id[cell_line]
            return torch.tensor([cell_type_id], device=device), "known"
        else:
            # Zero-shot cell line - use average of known embeddings
            if cell_emb_layer is None:
                print(f"Warning: No cell_emb_layer found, using 0 for zero-shot cell line: {cell_line}")
                return torch.tensor([0], device=device), "zero_shot_fallback"
            
            # Get all known cell line IDs
            known_ids = list(cell_line_to_id.values())
            # Get embeddings for all known cell lines
            known_embeddings = cell_emb_layer.weight[known_ids]  # shape: (n_known, d_model)
            # Average
            avg_embedding = known_embeddings.mean(dim=0)
            
            # Find an unused ID or use a special ID
            max_id = max(known_ids) if known_ids else 0
            zero_shot_id = max_id + 1
            
            # Copy average embedding to a new position in the embedding layer
            with torch.no_grad():
                if zero_shot_id < cell_emb_layer.weight.shape[0]:
                    cell_emb_layer.weight[zero_shot_id] = avg_embedding
                    return torch.tensor([zero_shot_id], device=device), "zero_shot_avg"
                else:
                    # If embedding table is full, just use the average and return a dummy ID
                    # The model will use the average but the ID itself doesn't matter
                    print(f"Warning: Cell type embedding table full, using known cell line for: {cell_line}")
                    # Use first known cell line as fallback
                    return torch.tensor([known_ids[0]], device=device), "zero_shot_fallback"
    
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
    # Precompute cell line to ID mapping (same as training)
    cell_line_to_id = {cl: idx for idx, cl in enumerate(adata_full.obs['cell_line'].unique())}
    
    # Precompute drug embeddings (same as training)
    print("Precomputing drug embeddings for inference...")
    drug_embeddings = {}
    
    # Check for precomputed drug embeddings in checkpoint directory
    drug_emb_path = os.path.join(checkpoint_dir, "drug_embeddings.pkl")
    if os.path.exists(drug_emb_path):
        import pickle
        with open(drug_emb_path, 'rb') as f:
            drug_embeddings_raw = pickle.load(f)
        # Convert numpy arrays back to tensors
        drug_embeddings = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
                         for k, v in drug_embeddings_raw.items()}
        print(f"Loaded {len(drug_embeddings)} precomputed drug embeddings")
        print(f"  Drug vocab: {list(drug_embeddings.keys())[:10]}...")
    else:
        # Generate drug embeddings using ChemBERTa
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            drug_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", use_safetensors=True).to(device)
            drug_model.eval()
            
            # Get unique drugs
            unique_drugs = adata_full.obs['drug'].unique()
            print(f"Generating ChemBERTa embeddings for {len(unique_drugs)} drugs...")
            print(f"  Test data drugs: {list(unique_drugs)[:10]}...")
            
            with torch.no_grad():
                for drug_name in unique_drugs:
                    # Get SMILES for this drug
                    drug_row = drug_meta[drug_meta['drug'] == drug_name]
                    smiles = None
                    if len(drug_row) > 0 and 'canonical_smiles' in drug_row.columns:
                        smiles = drug_row.iloc[0].get('canonical_smiles', None)
                    
                    if not smiles or not isinstance(smiles, str):
                        # Use zero embedding for control or missing SMILES
                        drug_embeddings[drug_name] = torch.zeros(384, device=device)
                    else:
                        inputs = tokenizer(smiles, return_tensors="pt", padding=True, 
                                         truncation=True, max_length=128).to(device)
                        outputs = drug_model(**inputs)
                        # Use CLS token
                        emb = outputs.last_hidden_state[0, 0, :]
                        drug_embeddings[drug_name] = emb.cpu()
            
            del drug_model
            torch.cuda.empty_cache()
            print(f"Generated {len(drug_embeddings)} drug embeddings")
            
        except Exception as e:
            print(f"Warning: Failed to generate drug embeddings: {e}")
            print("Using zero embeddings instead")
            for drug_name in adata_full.obs['drug'].unique():
                drug_embeddings[drug_name] = torch.zeros(384)
    
    # Ensure all drug embeddings are on the same device for batch processing
    # If no embeddings available, create zero embeddings for all unique drugs
    if not drug_embeddings:
        unique_drugs = adata_full.obs['drug'].unique()
        print(f"No drug embeddings found, creating zero embeddings for {len(unique_drugs)} drugs")
        drug_embeddings = {drug: torch.zeros(384) for drug in unique_drugs}
    
    drug_embeddings_on_device = {k: v.to(device) for k, v in drug_embeddings.items()}
    
    # Flag to track if ChemBERTa model is loaded for on-the-fly embedding generation
    drug_model_loaded = [False]  # Use list to allow modification in nested scope
    drug_tokenizer = [None]
    drug_model = [None]
    
    with torch.no_grad():
        # ========== Batch processing for faster inference ==========
        batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size else 64
        
        # Pre-compute gene_ids once (same for all samples - using actual_gene_list)
        gene_ids = torch.tensor(
            [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in actual_gene_list],
            dtype=torch.long
        ).to(device)
        
        # Number of DMSO samples to draw for each (cell_line, drug) pair
        n_dmso_samples = 100
        print(f"Using {n_dmso_samples} DMSO samples per (cell_line, drug) pair")
        
        # Pre-compute all unique (cell_line, drug) pairs
        unique_pairs = adata_full.obs[['cell_line', 'drug']].drop_duplicates().values
        print(f"Total unique (cell_line, drug) pairs: {len(unique_pairs)}")
        
        # Create a mapping from (cell_line, drug) to predicted expression (averaged over n_dmso_samples)
        pair_to_pred = {}
        
        for pair_idx, (cl, dr) in enumerate(tqdm(unique_pairs, desc="Processing pairs")):
            # Get all samples for this (cell_line, drug) pair
            mask = (adata_full.obs['cell_line'] == cl) & (adata_full.obs['drug'] == dr)
            sample_indices = np.where(mask)[0]
            
            # Get cell type id for this cell line
            ct_id, _ = get_cell_type_embedding(cl, cell_emb_layer, cell_line_to_id)
            cell_type_id = ct_id.item()
            
            # Get drug embedding
            if dr in drug_embeddings_on_device:
                drug_emb = drug_embeddings_on_device[dr]
            else:
                # Generate embedding on-the-fly if missing
                if dr in drug_meta['drug'].values:
                    drug_row = drug_meta[drug_meta['drug'] == dr]
                    smiles = None
                    if len(drug_row) > 0 and 'canonical_smiles' in drug_row.columns:
                        smiles = drug_row.iloc[0].get('canonical_smiles', None)
                    
                    if smiles and isinstance(smiles, str):
                        if not drug_model_loaded[0]:
                            from transformers import AutoTokenizer, AutoModel
                            drug_tokenizer[0] = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
                            drug_model[0] = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", use_safetensors=True).to(device)
                            drug_model[0].eval()
                            drug_model_loaded[0] = True
                        
                        with torch.no_grad():
                            inputs = drug_tokenizer[0](smiles, return_tensors="pt", padding=True, 
                                             truncation=True, max_length=128).to(device)
                            outputs = drug_model[0](**inputs)
                            emb = outputs.last_hidden_state[0, 0, :]
                    else:
                        emb = torch.zeros(384, device=device)
                else:
                    emb = torch.zeros(384, device=device)
                drug_emb = emb
            
            drug_emb = drug_emb.unsqueeze(0)
            cell_type_id_tensor = torch.tensor([cell_type_id], dtype=torch.long).to(device)
            
            # Get target gene ids if using target_bias model
            target_gene_ids = None
            if args.model == 'target_bias' and dr != "DMSO_TF":
                target_gene_ids = drug_to_target_gene_ids.get(dr, [])
                if target_gene_ids:
                    target_tensor = torch.tensor([target_gene_ids], dtype=torch.long).to(device)
                else:
                    target_tensor = torch.tensor([[-1]], dtype=torch.long).to(device)
            
            # For DMSO samples: directly output the DMSO expression (no perturbation)
            # For drug samples: input DMSO, predict perturbation
            if dr == "DMSO_TF":
                # For DMSO, just use the actual expressions
                if cl in control_pool:
                    ctrl_expressions = control_pool[cl]['expressions']
                    pair_to_pred[(cl, dr)] = np.mean(ctrl_expressions, axis=0)
                else:
                    pair_to_pred[(cl, dr)] = np.zeros(core_X.shape[1], dtype=np.float32)
            else:
                # For drug samples, sample DMSO from control pool and predict
                if cl not in control_pool:
                    # No control available - use zero
                    pair_to_pred[(cl, dr)] = np.zeros(core_X.shape[1], dtype=np.float32)
                    continue
                
                ctrl_expressions = control_pool[cl]['expressions']
                n_ctrl = len(ctrl_expressions)
                
                # Run predictions with multiple DMSO samples
                all_preds = []
                for dmso_idx in range(n_dmso_samples):
                    # Randomly sample a DMSO
                    ctrl_idx = np.random.randint(0, n_ctrl)
                    c_gene_expr = torch.tensor(ctrl_expressions[ctrl_idx], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        if args.model == 'target_bias':
                            pred = model(
                                c_gene=c_gene_expr,
                                drug_emb=drug_emb,
                                cell_type_id=cell_type_id_tensor,
                                target_gene_ids=target_tensor
                            )
                            pred_delta = pred.cpu().numpy().squeeze()
                            pred_expr = c_gene_expr.cpu().numpy().squeeze() + pred_delta
                        elif args.model in ['protein', 'ppi']:
                            batch_gene_ids = gene_ids.unsqueeze(0)
                            pred_delta = model(
                                batch_gene_ids,
                                c_gene_expr,
                                drug_emb,
                                cell_type_id_tensor,
                                esm_embeddings
                            ).cpu().numpy().squeeze()
                            pred_expr = c_gene_expr.cpu().numpy().squeeze() + pred_delta
                        elif args.model == 'metaselection' and metadata_cols:
                            batch_gene_ids = gene_ids.unsqueeze(0)
                            # Get metadata for this cell line
                            cl_meta = adata_full.obs.iloc[sample_indices[0]][metadata_cols].values.astype(np.float32)
                            batch_meta = torch.tensor(cl_meta).unsqueeze(0).to(device)
                            pred_delta = model(
                                batch_gene_ids,
                                c_gene_expr,
                                drug_emb,
                                cell_type_id_tensor,
                                batch_meta
                            ).cpu().numpy().squeeze()
                            pred_expr = c_gene_expr.cpu().numpy().squeeze() + pred_delta
                        else:
                            # baseline
                            batch_gene_ids = gene_ids.unsqueeze(0)
                            pred_delta = model(
                                batch_gene_ids,
                                c_gene_expr,
                                drug_emb,
                                cell_type_id_tensor
                            ).cpu().numpy().squeeze()
                            pred_expr = c_gene_expr.cpu().numpy().squeeze() + pred_delta
                        
                        all_preds.append(pred_expr)
            
            # Average over all DMSO samples
            pair_to_pred[(cl, dr)] = np.mean(all_preds, axis=0)
        
        # Build predictions array based on pair_to_pred mapping
        predictions = []
        for idx in range(adata_full.n_obs):
            cl = adata_full.obs.iloc[idx]['cell_line']
            dr = adata_full.obs.iloc[idx]['drug']
            predictions.append(pair_to_pred[(cl, dr)])
        
        predictions = np.array(predictions)
        
    # Create output
    print("Creating output...")
    
    # Build output adata
    result_adata = ad.AnnData(
        X=predictions,
        obs=adata_full.obs.copy()
    )
    result_adata.var_names = actual_gene_list
    
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
    import yaml
    
    parser = argparse.ArgumentParser(description="Run inference with scGPT model")
    
    # Config file - THIS IS THE PREFERRED WAY
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to config YAML file (recommended - all other args will be loaded from config)')
    
    # Alternative: specify experiment type directly
    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='Experiment type: baseline, protein, ppi, target_bias, metaselection')
    
    # Required arguments (can be overridden by config)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test h5ad file')
    parser.add_argument('--drug-meta', type=str, default=None,
                       help='Path to drug metadata parquet file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for results')
    
    # Optional arguments
    parser.add_argument('--selected-genes', type=str, default=None,
                       help='Path to selected genes CSV (default: checkpoint dir)')
    parser.add_argument('--vocab-path', type=str,
                       default='./scgpt/tokenizer/default_gene_vocab.json',
                       help='Path to vocab file')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference (default: 64)')
    parser.add_argument('--target-bias-value', type=float, default=5.0,
                       help='Target bias value for target_bias model')
    parser.add_argument('--n-controls', type=int, default=5,
                       help='Number of control samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--esm-path', type=str, default=None,
                       help='Path to ESM protein embeddings')

    # Evaluation options
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after inference')
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        # Load from config file
        config_path = args.config
        # If config is just experiment name, look for it in configs/ directory
        if not os.path.exists(args.config):
            possible_paths = [
                args.config,
                os.path.join('./configs', f'{args.config}.yaml'),
                os.path.join('./configs', f'{args.config}.yml'),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    config_path = p
                    break
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from: {config_path}")
            print(f"Config: {config}")
        else:
            print(f"Warning: Config file not found: {args.config}")
    
    # Override args with config values
    # Priority: command line args > config file > defaults
    
    # Experiment type
    if 'experiment_type' in config:
        args.experiment = config['experiment_type']
    elif args.experiment is None:
        args.experiment = 'baseline'
    
    # Model type (same as experiment_type for most cases)
    args.model = args.experiment
    
    # Paths from config
    if args.checkpoint is None:
        args.checkpoint = config.get('checkpoint_path', config.get('checkpoint', None))
    if args.test_data is None:
        args.test_data = config.get('test_data_path', config.get('test_data', None))
    if args.drug_meta is None:
        args.drug_meta = config.get('drug_meta_path', config.get('drug_meta', None))
    if args.output is None:
        args.output = config.get('output_path', config.get('infer_output', f'./infer/{args.experiment}_result.h5ad'))
    if args.selected_genes is None:
        args.selected_genes = config.get('selected_genes_path', None)
    if args.vocab_path is None:
        args.vocab_path = config.get('vocab_path', './scgpt/tokenizer/default_gene_vocab.json')
    
    # Model params from config
    if args.d_model is None or args.d_model == 512:
        args.d_model = config.get('d_model', 512)
    if 'target_bias_value' in config:
        args.target_bias_value = config['target_bias_value']
    if 'esm_path' in config:
        args.esm_path = config['esm_path']
    
    # Validate required arguments
    missing_args = []
    if args.checkpoint is None:
        missing_args.append('--checkpoint')
    if args.test_data is None:
        missing_args.append('--test-data')
    if args.drug_meta is None:
        missing_args.append('--drug-meta')
    if args.output is None:
        missing_args.append('--output')
    
    if missing_args:
        print(f"\nError: Missing required arguments: {', '.join(missing_args)}")
        print(f"Please provide them via command line or config file.")
        print(f"\nExample config file should contain:")
        print(f"  experiment_type: baseline")
        print(f"  checkpoint_path: /path/to/model.pt")
        print(f"  test_data_path: /path/to/test.h5ad")
        print(f"  drug_meta_path: /path/to/drug_meta.parquet")
        print(f"  output_path: ./infer/result.h5ad")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Running inference with:")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test data: {args.test_data}")
    print(f"  Drug meta: {args.drug_meta}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")
    
    run_inference(args, config)


if __name__ == "__main__":
    main()
