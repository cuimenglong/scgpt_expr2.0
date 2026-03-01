"""
Unified Training Script for scGPT Perturbation Experiments

Supports: baseline, protein, ppi, target_bias, metaselection

Usage:
    # Single experiment
    python scripts/train.py --config config/experiments/baseline.yaml --checkpoint 1
    
    # Or with experiment name and custom paths
    python scripts/train.py --experiment baseline --checkpoint 1 --data-path /path/to/train.h5ad
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
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import scanpy as sc
import pandas as pd

# Import src modules
from src.models import (
    scGPTBaseline, 
    scGPTWithVirtualProtein, 
    scGPTWithVirtualProteinAndPPI,
    scGPTWithTargetBias,
    scGPTWithMetadata
)
from src.data import H5ADPerturbationDataset, GeneProcessorFactory


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_class(model_name: str):
    """Get model class by name"""
    model_map = {
        'baseline': scGPTBaseline,
        'scgptbaseline': scGPTBaseline,
        'protein': scGPTWithVirtualProtein,
        'scgptwithvirtualprotein': scGPTWithVirtualProtein,
        'ppi': scGPTWithVirtualProteinAndPPI,
        'scgptwithvirtualproteinandppi': scGPTWithVirtualProteinAndPPI,
        'target_bias': scGPTWithTargetBias,
        'scgptwithtargetbias': scGPTWithTargetBias,
        'metaselection': scGPTWithMetadata,
        'scgptwithmetadata': scGPTWithMetadata,
    }
    
    model_name_lower = model_name.lower()
    if model_name_lower not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name_lower]


def prepare_drug_targets(drug_meta_path: str, adata, vocab_dict: dict) -> dict:
    """
    Prepare drug to target gene IDs mapping
    Same logic as previous/1_target_bias/train_dp.py
    """
    drug_to_target_gene_ids = {}
    
    if not drug_meta_path or not os.path.exists(drug_meta_path):
        print("Warning: drug metadata not found, no target bias will be applied")
        return drug_to_target_gene_ids
    
    drug_meta = pd.read_parquet(drug_meta_path)
    actual_drugs = set(adata.obs['drug'].unique())
    
    total_targets_found = 0
    drugs_with_targets = 0
    
    print("\n--- Matching drug targets ---")
    for _, row in drug_meta.iterrows():
        drug = row['drug']
        if drug not in actual_drugs:
            continue
            
        targets = str(row['targets']).split(',') if pd.notna(row['targets']) else []
        t_ids = []
        found_names = []
        for t in targets:
            t_upper = t.upper().strip()
            if t_upper in vocab_dict:
                t_ids.append(vocab_dict[t_upper])
                found_names.append(t_upper)
        
        if len(t_ids) > 0:
            drug_to_target_gene_ids[drug] = t_ids
            total_targets_found += len(t_ids)
            drugs_with_targets += 1
            if drugs_with_targets <= 5:
                print(f"Drug: {drug} | Matched targets: {found_names}")
    
    print(f"Stats: {len(actual_drugs)} drugs in data, {drugs_with_targets} matched with targets, total target points: {total_targets_found}")
    print("--- Matching done ---\n")
    
    return drug_to_target_gene_ids


def my_collate(batch):
    """
    Custom collate_fn for target_bias experiment with variable length target gene lists
    """
    c_genes = torch.stack([item['c_gene'] for item in batch])
    p_genes = torch.stack([item['p_gene'] for item in batch])
    cell_type_ids = torch.tensor([item['cell_type_id'] for item in batch], dtype=torch.long)
    drug_embs = torch.stack([item['drug_emb'] for item in batch])
    
    # For target_bias: target_gene_ids
    if 'target_nodes' in batch[0] and batch[0]['target_nodes'] is not None:
        target_ids_list = [item['target_nodes'] for item in batch]
        return c_genes, p_genes, cell_type_ids, drug_embs, target_ids_list
    
    # For baseline/protein/ppi: no target_ids
    return c_genes, p_genes, cell_type_ids, drug_embs


def train(args):
    """Main training function"""
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Build config from args
        config = {
            'experiment_type': args.experiment,
            'model': args.model or args.experiment,
            'n_hvg': args.n_hvg,
            'esm_path': args.esm_path,
            'use_ppi': args.use_ppi,
            'ppi_tsv_path': args.ppi_tsv_path,
            'ppi_cache_dir': args.ppi_cache_dir,
            'data_path': args.data_path,
            'vocab_path': args.vocab_path,
            'pretrained_path': args.pretrained_path,
            'save_dir': args.save_dir,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'd_model': args.d_model,
            'drug_meta_path': args.drug_meta_path,
            'target_bias_value': args.target_bias_value,
            'metadata_cols': args.metadata_cols,
        }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(42)
    
    # Get paths from config
    experiment_type = config.get('experiment_type', args.experiment)
    model_name = config.get('model', experiment_type)
    adata_path = config.get('data_path', args.data_path)
    vocab_path = config.get('vocab_path', args.vocab_path)
    pretrained_path = config.get('pretrained_path', args.pretrained_path)
    drug_meta_path = config.get('drug_meta_path', args.drug_meta_path)
    save_dir = config.get('save_dir', args.save_dir)
    
    # Override with command line args if provided
    batch_size = args.batch_size or config.get('batch_size', 64)
    lr = args.lr or config.get('lr', 1e-4)
    epochs = args.epochs or config.get('epochs', 30)
    d_model = args.d_model or config.get('d_model', 512)
    target_bias_value = args.target_bias_value or config.get('target_bias_value', 5.0)
    metadata_cols = args.metadata_cols or config.get('metadata_cols', ['tscp_count', 'pcnt_mito', 'S_score', 'G2M_score'])
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Experiment type: {experiment_type}")
    print(f"Model: {model_name}")
    print(f"Data: {adata_path}")
    print(f"Save dir: {save_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"d_model: {d_model}")
    print(f"PPI Prior: {'启用' if config.get('use_ppi', True) else '禁用'}")
    print(f"{'='*60}\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Step 1: Load and process data ---
    print("Loading data...")
    adata = sc.read_h5ad(adata_path)
    
    # For protein/ppi models, convert Ensembl IDs to gene symbols using gene_purifier
    if experiment_type.lower() in ['protein', 'ppi']:
        try:
            from src.utils.gene_purifier import convert_with_gseapy
            print("Converting Ensembl IDs to gene symbols...")
            adata = convert_with_gseapy(adata)
        except ImportError as e:
            print(f"Warning: gene_purifier not available: {e}")
            print("Using uppercase gene names instead...")
            adata.var_names = [g.upper() for g in adata.var_names]
    else:
        # For other experiments, just convert to uppercase
        adata.var_names = [g.upper() for g in adata.var_names]
    
    # Get gene processor
    gene_processor = GeneProcessorFactory.get_processor(experiment_type)
    
    # Process genes
    gene_config = {
        'n_hvg': config.get('n_hvg', 2000),
        'esm_path': config.get('esm_path', ''),
        'use_ppi': config.get('use_ppi', True),
        'ppi_tsv_path': config.get('ppi_tsv_path', None),
        'ppi_cache_dir': config.get('ppi_cache_dir', './data')
    }
    selected_genes = gene_processor.process(adata, gene_config)
    
    # Filter adata to selected genes
    adata = adata[:, selected_genes].copy()
    
    # Save gene list
    gene_list_path = os.path.join(save_dir, "selected_genes.csv")
    gene_processor.save_gene_list(selected_genes, gene_list_path)
    
    # Get PPI data if using PPI model
    ppi_adjacency = None
    if experiment_type.lower() == 'ppi':
        if hasattr(gene_processor, 'get_ppi_data'):
            ppi_stats, ppi_adjacency = gene_processor.get_ppi_data()
            if ppi_stats:
                with open(os.path.join(save_dir, "ppi_statistics.json"), 'w') as f:
                    json.dump(ppi_stats, f, indent=2)
            if ppi_adjacency is not None:
                np.save(os.path.join(save_dir, "ppi_adjacency.npy"), ppi_adjacency)
    
    # --- Step 2: Prepare data for training ---
    print("Preparing dataset...")
    
    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    
    gene_ids = torch.tensor(
        [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in selected_genes],
        dtype=torch.long
    ).to(device)
    
    # Load ESM embeddings for protein/ppi models (using generate_protein_embeddings from previous code)
    esm_embeddings = None
    if experiment_type.lower() in ['protein', 'ppi']:
        # Import the ESM embedding generation function
        from src.utils.esm_embeddings import generate_protein_embeddings, load_esm_embeddings
        
        esm_path = config.get('esm_path', '')
        
        # Create hvg_vocab_dict for selected genes
        hvg_vocab_dict = {g: vocab_dict[g] for g in selected_genes if g in vocab_dict}
        
        if not esm_path:
            esm_path = os.path.join(save_dir, "protein_esm_embeddings.pt")
        
        # Generate or load ESM embeddings
        if not os.path.exists(esm_path):
            print(f"ESM embeddings file not found at {esm_path}")
            print("Generating ESM embeddings using ESM-2 model (this may take a while)...")
            esm_matrix = generate_protein_embeddings(
                hvg_vocab_dict=hvg_vocab_dict,
                save_path=esm_path,
                device=device,
                full_vocab_size=len(vocab_dict)
            )
        else:
            print(f"Loading ESM embeddings from: {esm_path}")
            esm_matrix = torch.load(esm_path, map_location='cpu')
        
        # Extract embeddings for selected genes [N_genes, 1280]
        gene_ids_for_esm = torch.tensor(
            [vocab_dict[g] for g in selected_genes],
            dtype=torch.long
        )
        esm_embeddings = esm_matrix[gene_ids_for_esm].to(device)
        print(f"ESM embeddings prepared: shape {esm_embeddings.shape}")
    
    # Prepare drug targets for target_bias experiment
    drug_to_target_gene_ids = {}
    use_collate_fn = False
    
    if experiment_type.lower() == 'target_bias':
        drug_to_target_gene_ids = prepare_drug_targets(drug_meta_path, adata, vocab_dict)
        use_collate_fn = True
    
    # Check if using metadata (metaselection)
    use_knn_matching = experiment_type.lower() == 'metaselection'
    
    # Create dataset with appropriate settings
    dataset = H5ADPerturbationDataset(
        adata, 
        drug_to_target_nodes=drug_to_target_gene_ids, 
        device=device,
        use_knn_matching=use_knn_matching,
        metadata_cols=metadata_cols if use_knn_matching else None
    )
    
    # Use custom collate_fn for target_bias
    if use_collate_fn:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples, {len(train_loader)} batches")
    
    # --- Step 3: Initialize model ---
    print("Initializing model...")
    
    # Get model class
    ModelClass = get_model_class(model_name)
    
    # Get number of cell types
    n_celltype = len(adata.obs['cell_line'].unique()) + 20
    
    # Build model kwargs
    model_kwargs = {
        'ntokens': len(vocab_dict),
        'd_model': d_model,
        'nhead': 8,
        'd_hid': d_model,
        'nlayers': 12,
        'scgpt_layers': 12,
        'gp_layers': 3,
        'n_celltype': n_celltype,
        'drug_emb_dim': 384,
    }
    
    # Add ESM path if needed
    if model_name in ['protein', 'ppi']:
        model_kwargs['esm_dim'] = 1280
    
    # Add PPI adjacency if using PPI model
    if model_name in ['ppi', 'scgptwithvirtualproteinandppi']:
        model_kwargs['ppi_adjacency'] = ppi_adjacency
    
    # Add target_bias specific params
    if experiment_type.lower() == 'target_bias':
        model_kwargs['gene_ids'] = gene_ids
        model_kwargs['target_bias_value'] = target_bias_value
    
    # Add metadata_dim for metaselection
    if experiment_type.lower() == 'metaselection':
        model_kwargs['metadata_dim'] = len(metadata_cols)
    
    # Initialize model
    model = ModelClass(**model_kwargs).to(device)
    
    # Load pretrained weights if available (using previous code's detailed mapping logic)
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        
        # Handle scGPT checkpoint format (may be wrapped in "model_state_dict" or "model")
        checkpoint = torch.load(pretrained_path, map_location=device)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        
        model_dict = model.state_dict()
        matched_dict = {}
        
        # Define prefix mapping (same as previous/3_protein/train_dp.py)
        # For protein/ppi models: transformer_encoder -> scgpt_transformer (12 layers)
        # For baseline/metaselection/target_bias: transformer_encoder -> transformer_encoder (6 layers)
        is_protein_model = experiment_type.lower() in ['protein', 'ppi']
        transformer_target = "scgpt_transformer" if is_protein_model else "transformer_encoder"
        
        mapping = {
            "gene_encoder": "gene_encoder",
            "value_encoder": "value_encoder",
            "transformer_encoder": transformer_target,
            "encoder": transformer_target
        }
        
        print("Starting weight matching...")
        for k, v in pretrained_dict.items():
            new_k = None
            
            # Try prefix replacement
            for pre_old, pre_new in mapping.items():
                if k.startswith(pre_old + "."):
                    new_k = k.replace(pre_old, pre_new)
                    break
            
            # If no match, keep original key
            if new_k is None:
                new_k = k
            
            # Check if exists in model and shape matches
            if new_k in model_dict:
                if v.shape == model_dict[new_k].shape:
                    matched_dict[new_k] = v
                else:
                    print(f"  [Skip] Shape mismatch: {k} ({list(v.shape)}) -> {new_k} ({list(model_dict[new_k].shape)})")
            else:
                # Handle DataParallel compatibility (module.xxx)
                dp_k = "module." + new_k
                if dp_k in model_dict and v.shape == model_dict[dp_k].shape:
                    matched_dict[dp_k] = v
        
        # Update model weights
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
        
        # Detailed report
        scgpt_backbone_count = len([k for k in matched_dict.keys() if "scgpt_transformer" in k or "transformer_encoder" in k])
        gene_enc_count = len([k for k in matched_dict.keys() if "gene_encoder" in k])
        
        print("\n" + "="*50)
        print(f"Weight Loading Report:")
        print(f"  - Total matched tensors: {len(matched_dict)}")
        print(f"  - Transformer backbone layers: {scgpt_backbone_count}")
        print(f"  - Gene encoder layers: {gene_enc_count}")
        
        # Check new modules remain randomly initialized
        new_modules = ["gp_transformer", "drug_projector", "gene_to_protein_mlp", "esm_projector", "decoder", "meta_encoder"]
        print(f"  - New modules (should remain random):")
        for mod in new_modules:
            mod_params = [k for k in model_dict.keys() if k.startswith(mod)]
            is_loaded = any(k in matched_dict for k in mod_params)
            status = "⚠️ Unexpected" if is_loaded else "✅ Random"
            print(f"    * {mod}: {status}")
        print("="*50 + "\n")
    
    # --- Step 4: Training loop ---
    print(f"\nStarting training for {epochs} epochs...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    best_loss = float('inf')
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # Handle different batch formats
            if use_collate_fn:
                # target_bias: c_genes, p_genes, cell_type_ids, drug_embs, target_ids_list
                c_gene, p_gene, cell_type_id, drug_emb, target_ids_list = batch
                c_gene = c_gene.to(device)
                p_gene = p_gene.to(device)
                cell_type_id = cell_type_id.to(device)
                drug_emb = drug_emb.to(device)
                
                # Process target_ids
                max_targets = max([len(t) for t in target_ids_list]) if len(target_ids_list) > 0 else 0
                if max_targets > 0:
                    target_genes = torch.full((len(target_ids_list), max_targets), -1, dtype=torch.long).to(device)
                    for i, t_list in enumerate(target_ids_list):
                        if len(t_list) > 0:
                            if isinstance(t_list, torch.Tensor):
                                target_genes[i, :len(t_list)] = t_list
                            else:
                                target_genes[i, :len(t_list)] = torch.tensor(t_list, dtype=torch.long)
                else:
                    target_genes = torch.full((len(target_ids_list), 1), -1, dtype=torch.long).to(device)
                
                # Forward pass
                pred_delta = model(
                    c_gene=c_gene,
                    drug_emb=drug_emb,
                    cell_type_id=cell_type_id,
                    target_gene_ids=target_genes
                )
            else:
                # baseline/protein/ppi/metaselection
                c_gene = batch['c_gene'].to(device)
                p_gene = batch['p_gene'].to(device)
                cell_type_id = batch['cell_type_id'].to(device)
                drug_emb = batch['drug_emb'].to(device)
                
                # Forward pass based on model type
                if experiment_type.lower() == 'metaselection':
                    metadata = batch.get('metadata')
                    if metadata is not None:
                        metadata = metadata.to(device)
                    
                    pred_delta = model(
                        gene_ids.unsqueeze(0).expand(c_gene.shape[0], -1),
                        c_gene,
                        drug_emb,
                        cell_type_id,
                        metadata
                    )
                elif experiment_type.lower() in ['protein', 'ppi']:
                    # Load ESM embeddings from file (same as used in previous code)
                    if 'esm_embeddings' in locals():
                        # ESM embeddings already loaded outside the loop
                        esm_batch = esm_embeddings.unsqueeze(0).expand(c_gene.shape[0], -1, -1)
                    else:
                        # Fallback: generate random embeddings (should not happen with proper config)
                        print("Warning: ESM embeddings not loaded, using random embeddings")
                        esm_batch = torch.randn(c_gene.shape[0], len(selected_genes), 1280).to(device)
                    pred_delta = model(
                        gene_ids.unsqueeze(0).expand(c_gene.shape[0], -1),
                        c_gene,
                        drug_emb,
                        cell_type_id,
                        esm_batch
                    )
                else:
                    # baseline
                    pred_delta = model(
                        gene_ids.unsqueeze(0).expand(c_gene.shape[0], -1),
                        c_gene,
                        drug_emb,
                        cell_type_id
                    )
            
            # Calculate loss
            true_delta = p_gene - c_gene
            loss = criterion(pred_delta, true_delta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")
        
        # Save best and last models
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(save_model.state_dict(), best_path)
            print(f"--> Best model updated (Loss: {best_loss:.6f})")
        
        last_path = os.path.join(save_dir, "last_model.pt")
        save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(save_model.state_dict(), last_path)
    
    print(f"\nTraining complete! Best Loss: {best_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    
    # Save training config
    config_info = {
        'experiment_type': experiment_type,
        'model': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'd_model': d_model,
        'num_genes': len(selected_genes),
        'best_loss': best_loss
    }
    with open(os.path.join(save_dir, "train_config.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Train scGPT perturbation model")
    
    # Config file
    parser.add_argument('--config', '-c', type=str, help='Path to config YAML file')
    
    # Experiment name (alternative to config)
    parser.add_argument('--experiment', '-e', type=str, default='baseline',
                       help='Experiment type: baseline, protein, ppi, target_bias, metaselection')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Model name (if different from experiment)')
    
    # Data paths
    parser.add_argument('--data-path', type=str, 
                       default='/home/dataset-assist-0/cuimenglong/workspace/data/tahoe/experiment/processed/train.h5ad',
                       help='Path to training data')
    parser.add_argument('--vocab-path', type=str,
                       default='./scgpt/tokenizer/default_gene_vocab.json',
                       help='Path to vocab file')
    parser.add_argument('--pretrained-path', type=str,
                       default='/home/dataset-assist-0/cuimenglong/workspace/data/scgpt/best_model.pt',
                       help='Path to pretrained model')
    parser.add_argument('--drug-meta-path', type=str,
                       default='/home/dataset-assist-0/cuimenglong/workspace/data/tahoe/drug_metadata.parquet',
                       help='Path to drug metadata for target_bias')
    
    # Model settings
    parser.add_argument('--n-hvg', type=int, default=2000,
                       help='Number of highly variable genes')
    parser.add_argument('--esm-path', type=str, default='',
                       help='Path to ESM protein embeddings')
    parser.add_argument('--use-ppi', action='store_true', default=True,
                       help='Use PPI network (for PPI model)')
    parser.add_argument('--ppi-tsv-path', type=str, default=None,
                       help='Path to local PPI TSV file (e.g., omnipath_ppi.tsv)')
    parser.add_argument('--ppi-cache-dir', type=str, default='./data',
                       help='Directory to cache PPI data')
    parser.add_argument('--target-bias-value', type=float, default=5.0,
                       help='Attention bias value for target genes')
    parser.add_argument('--metadata-cols', nargs='+', 
                       default=['tscp_count', 'pcnt_mito', 'S_score', 'G2M_score'],
                       help='Metadata columns for metaselection')
    
    # Training settings
    parser.add_argument('--checkpoint', '-n', type=int, required=True,
                       help='Checkpoint number for saving')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--d-model', type=int, default=None,
                       help='Model dimension')
    
    # Save directory
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Save directory (default: ./checkpoints{n})')
    
    args = parser.parse_args()
    
    # Set save_dir if not provided
    if args.save_dir is None:
        args.save_dir = f"./checkpoints{args.checkpoint}"
    
    # Run training
    train(args)


if __name__ == "__main__":
    main()
