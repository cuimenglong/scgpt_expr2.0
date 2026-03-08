"""
Resume Training Script for scGPT Perturbation Experiments

Loads a checkpoint and continues training from where it left off.

Usage:
    python scripts/resume_train.py --checkpoint checkpoints1/model.pt --epochs 50
"""

import os
import sys
import json
import numpy as np
import pandas as pd

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
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import src modules
from src.models import (
    scGPTBaseline, 
    scGPTWithVirtualProtein, 
    scGPTWithVirtualProteinAndPPI,
    scGPTWithTargetBias,
    scGPTWithMetadata
)
from src.data import H5ADPerturbationDataset, GeneProcessorFactory


def load_checkpoint_info(checkpoint_dir):
    """Load training info from checkpoint directory"""
    
    # Try to load epoch_loss.csv to find current epoch
    loss_csv_path = os.path.join(checkpoint_dir, "epoch_loss.csv")
    start_epoch = 0
    if os.path.exists(loss_csv_path):
        loss_df = pd.read_csv(loss_csv_path)
        start_epoch = len(loss_df)
        print(f"Found existing training history: {start_epoch} epochs completed")
        if start_epoch > 0:
            print(f"  Previous losses: {loss_df['loss'].tolist()}")
    
    # Load config if available (ONLY exists after FULL training)
    config_path = os.path.join(checkpoint_dir, "train_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config: {config}")
    else:
        print("Note: train_config.json not found (only saved after full training)")
        print("      You may need to provide --model, --d-model, etc. manually")
    
    return start_epoch, config


def resume_training(args):
    """Resume training from a checkpoint"""
    
    checkpoint_dir = os.path.dirname(args.checkpoint) or "."
    checkpoint_name = os.path.basename(args.checkpoint)
    
    # Load checkpoint info
    start_epoch, config = load_checkpoint_info(checkpoint_dir)
    
    # Determine model name
    # 1. Try from command line
    model_name = args.model
    # 2. Try from config (if training completed)
    if not model_name and config.get('model'):
        model_name = config.get('model')
    # 3. Try from directory name
    if not model_name:
        if 'baseline' in checkpoint_dir.lower():
            model_name = 'baseline'
        elif 'ppi' in checkpoint_dir.lower():
            model_name = 'ppi'
        elif 'protein' in checkpoint_dir.lower():
            model_name = 'protein'
        elif 'target_bias' in checkpoint_dir.lower():
            model_name = 'target_bias'
        elif 'metaselection' in checkpoint_dir.lower():
            model_name = 'metaselection'
    
    if not model_name:
        raise ValueError("Cannot determine model type. Please provide --model argument.")
    
    print(f"Resuming training for model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Determine total epochs
    total_epochs = args.epochs or config.get('epochs', 10)
    additional_epochs = total_epochs - start_epoch
    
    if additional_epochs <= 0:
        print(f"Error: Already trained for {start_epoch} epochs, which is >= {total_epochs}")
        print("Use --epochs to specify a higher number")
        return
    
    print(f"Will train for {additional_epochs} more epochs (epoch {start_epoch+1} to {total_epochs})")
    
    # Get parameters - prioritize args, then config
    # These are REQUIRED if config doesn't exist
    data_path = args.data_path or config.get('data_path', None)
    if not data_path:
        raise ValueError("data_path not specified. Please provide --data-path argument.")
    
    drug_meta_path = args.drug_meta or config.get('drug_meta_path', None)
    batch_size = args.batch_size or config.get('batch_size', 32)
    lr = args.lr or config.get('lr', 1e-4)
    d_model = args.d_model or config.get('d_model', 512)
    
    # Load selected genes
    selected_genes_path = os.path.join(checkpoint_dir, "selected_genes.csv")
    if not os.path.exists(selected_genes_path):
        raise FileNotFoundError(f"Selected genes not found: {selected_genes_path}")
    
    selected_genes_df = pd.read_csv(selected_genes_path)
    selected_genes = selected_genes_df['gene'].astype(str).str.upper().tolist()
    print(f"Selected genes: {len(selected_genes)}")
    
    # Load vocab
    vocab_path = args.vocab_path or os.path.join(checkpoint_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = "./scgpt/tokenizer/default_gene_vocab.json"
    
    with open(vocab_path, 'r') as f:
        vocab_dict = json.load(f)
    
    # ========== Setup device ==========
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== Load data ==========
    print("Loading data...")
    adata = sc.read_h5ad(data_path)
    adata.var_names = [g.upper() for g in adata.var_names]
    
    # Apply log1p if needed
    X_max = adata.X.max() if hasattr(adata.X, 'max') else adata.X.toarray().max()
    if X_max > 30:
        print(f"Applying log1p transformation (max value: {X_max:.2f})...")
        sc.pp.log1p(adata)
    
    # ========== Load drug metadata ==========
    drug_meta = None
    drug_to_target_gene_ids = {}
    if drug_meta_path:
        if drug_meta_path.endswith('.parquet'):
            drug_meta = pd.read_parquet(drug_meta_path)
        elif drug_meta_path.endswith('.csv'):
            drug_meta = pd.read_csv(drug_meta_path)
        
        # Build drug to target mapping
        for _, row in drug_meta.iterrows():
            drug = row['drug']
            targets = str(row.get('targets', '')).split(',') if pd.notna(row.get('targets', '')) else []
            t_ids = []
            for t in targets:
                t_upper = t.upper().strip()
                if t_upper in vocab_dict:
                    t_ids.append(vocab_dict[t_upper])
            if len(t_ids) > 0:
                drug_to_target_gene_ids[drug] = t_ids
    
    # ========== Create dataset ==========
    # First try to load saved drug embeddings from checkpoint directory
    drug_emb_save_path = os.path.join(checkpoint_dir, "drug_embeddings.pkl")
    precomputed_drug_embeddings = None
    
    if os.path.exists(drug_emb_save_path):
        print(f"Loading precomputed drug embeddings from: {drug_emb_save_path}")
        import pickle
        with open(drug_emb_save_path, 'rb') as f:
            drug_emb_dict = pickle.load(f)
        # Convert numpy back to tensors
        precomputed_drug_embeddings = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
                                         for k, v in drug_emb_dict.items()}
        print(f"Loaded {len(precomputed_drug_embeddings)} drug embeddings")
    
    # Create dataset (use precomputed embeddings if available)
    print("Creating dataset...")
    dataset = H5ADPerturbationDataset(
        adata, 
        drug_to_target_nodes=drug_to_target_gene_ids, 
        device=device,
        compute_drug_embeddings=precomputed_drug_embeddings is None,  # Only compute if not loaded
        precomputed_drug_embeddings=precomputed_drug_embeddings,  # Pass loaded embeddings
        apply_log1p=False  # Already applied above
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    # ========== Build model ==========
    print(f"Building model: {model_name}")
    ModelClass = {
        'baseline': scGPTBaseline,
        'protein': scGPTWithVirtualProtein,
        'ppi': scGPTWithVirtualProteinAndPPI,
        'target_bias': scGPTWithTargetBias,
        'metaselection': scGPTWithMetadata,
    }[model_name]
    
    # Load cell_line vocabulary for consistency
    cell_line_vocab_path = os.path.join(checkpoint_dir, "cell_line_vocab.pkl")
    if not os.path.exists(cell_line_vocab_path):
        cell_line_vocab_path = os.path.join(checkpoint_dir, "cell_line_vocab_early.pkl")
    
    if os.path.exists(cell_line_vocab_path):
        import pickle
        with open(cell_line_vocab_path, 'rb') as f:
            cell_line_vocab = pickle.load(f)
        n_celltype = len(cell_line_vocab)
        print(f"Loaded cell_line vocabulary: {n_celltype} cell lines")
    else:
        # Fallback to current data
        n_celltype = len(adata.obs['cell_line'].unique())
        print(f"Warning: cell_line_vocab not found, using data: {n_celltype} cell lines")
    
    model_kwargs = {
        'ntokens': len(vocab_dict),
        'd_model': d_model,
        'nhead': 8,
        'd_hid': d_model,
        'nlayers': 12,
        'n_celltype': n_celltype,
        'drug_emb_dim': 384,
    }
    
    # Load ESM embeddings for protein/ppi models
    esm_embeddings = None
    if model_name in ['protein', 'ppi']:
        esm_path = os.path.join(checkpoint_dir, "protein_esm_embeddings.pt")
        if os.path.exists(esm_path):
            print(f"Loading ESM embeddings from: {esm_path}")
            esm_matrix = torch.load(esm_path, map_location='cpu')
            esm_dim = esm_matrix.shape[1]
            model_kwargs['esm_dim'] = esm_dim
            
            valid_genes = [g for g in selected_genes if g in vocab_dict]
            gene_ids_for_esm = torch.tensor(
                [vocab_dict[g] for g in valid_genes],
                dtype=torch.long
            )
            esm_embeddings = esm_matrix[gene_ids_for_esm].to(device)
            print(f"ESM embeddings loaded: shape {esm_embeddings.shape}")
    
    # Load PPI adjacency for ppi model
    if model_name == 'ppi':
        ppi_path = os.path.join(checkpoint_dir, "ppi_adjacency.npy")
        if os.path.exists(ppi_path):
            ppi_adjacency = np.load(ppi_path)
            model_kwargs['ppi_adjacency'] = ppi_adjacency
    
    model = ModelClass(**model_kwargs).to(device)
    
    # Build model first (handle DataParallel BEFORE loading weights)
    # This ensures proper key matching between checkpoint and model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        dp_prefix = "module."
    else:
        dp_prefix = ""
    
    # Load checkpoint weights
    print(f"Loading model weights from: {args.checkpoint}")
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # Handle DataParallel checkpoint (with "module." prefix) - adapt to current model
        new_state_dict = {}
        for k, v in state_dict.items():
            # If checkpoint has "module." prefix but model doesn't (or vice versa)
            if k.startswith('module.') and not dp_prefix:
                new_key = k[7:]  # Remove "module." prefix
            elif not k.startswith('module.') and dp_prefix:
                new_key = "module." + k  # Add "module." prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        # Try loading with strict=False to allow missing keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys (will use random init): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys (ignored): {unexpected_keys[:5]}...")
        
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== Setup optimizer ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scaler = GradScaler()
    
    # ========== Training loop ==========
    print(f"\nResuming training from epoch {start_epoch + 1} to {total_epochs}...")
    
    model.train()
    best_loss = float('inf')
    
    # Load existing loss history
    loss_csv_path = os.path.join(checkpoint_dir, "epoch_loss.csv")
    if os.path.exists(loss_csv_path):
        epoch_losses = pd.read_csv(loss_csv_path).to_dict('records')
        best_loss = min(epoch_losses[-1]['loss'], best_loss)
    else:
        epoch_losses = []
    
    for epoch in range(start_epoch, total_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch in pbar:
            c_gene = batch['control'].to(device)
            p_gene = batch['perturbed'].to(device)
            cell_type_id = batch['cell_type_id'].to(device)
            drug_emb = batch['drug_emb'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                if model_name == 'target_bias':
                    target_genes = torch.full((c_gene.shape[0], 1), -1, dtype=torch.long).to(device)
                    pred_delta = model(
                        c_gene=c_gene,
                        drug_emb=drug_emb,
                        cell_type_id=cell_type_id,
                        target_gene_ids=target_genes
                    )
                elif model_name in ['protein', 'ppi']:
                    gene_ids = torch.tensor(
                        [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in selected_genes],
                        dtype=torch.long
                    ).to(device)
                    gene_ids = gene_ids.unsqueeze(0).expand(c_gene.shape[0], -1)
                    pred_delta = model(
                        gene_ids,
                        c_gene,
                        drug_emb,
                        cell_type_id,
                        esm_embeddings
                    )
                else:
                    gene_ids = torch.tensor(
                        [vocab_dict.get(g, vocab_dict.get("<unk>", 0)) for g in selected_genes],
                        dtype=torch.long
                    ).to(device)
                    gene_ids = gene_ids.unsqueeze(0).expand(c_gene.shape[0], -1)
                    pred_delta = model(
                        gene_ids,
                        c_gene,
                        drug_emb,
                        cell_type_id
                    )
                
                true_delta = p_gene - c_gene
                loss = criterion(pred_delta, true_delta)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{total_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Record epoch-loss
        epoch_losses.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Save best and last models
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(save_model.state_dict(), best_path)
            print(f"--> Best model updated (Loss: {best_loss:.6f})")
        
        last_path = os.path.join(checkpoint_dir, "last_model.pt")
        save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(save_model.state_dict(), last_path)
        
        # Save epoch loss
        loss_df = pd.DataFrame(epoch_losses)
        loss_df.to_csv(loss_csv_path, index=False)
    
    print(f"\nTraining complete! Best Loss: {best_loss:.6f}")
    print(f"Total epochs trained: {total_epochs}")
    
    return checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (last_model.pt or best_model.pt)')
    
    # Optional - will try to load from config if not provided
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to training h5ad file')
    parser.add_argument('--drug-meta', type=str, default=None,
                       help='Path to drug metadata')
    parser.add_argument('--vocab-path', type=str, default=None,
                       help='Path to vocab file')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=None,
                       help='Total epochs to train (will continue from current)')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--d-model', type=int, default=None,
                       help='Model dimension')
    parser.add_argument('--model', type=str, default=None,
                       help='Model type: baseline, protein, ppi, target_bias, metaselection')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    resume_training(args)


if __name__ == "__main__":
    main()
