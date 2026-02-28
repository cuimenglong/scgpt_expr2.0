"""
ESM Protein Embedding Generator
Uses ESM-2 model to extract protein semantic features
Based on previous/3_protein/get_protein_embedding.py
"""

import os
import torch
import json
import requests
import time
from typing import Dict, List
from tqdm import tqdm
from io import StringIO
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel


def get_uniprot_sequences_batch(gene_list, taxon_id="9606"):
    """
    Batch fetch protein sequences from UniProt for given gene names
    taxon_id: 9606 is Human
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    gene_to_seq = {}
    
    # Query 50 genes at a time to avoid overly long URLs
    batch_size = 50
    print(f"Fetching protein sequences for {len(gene_list)} genes from UniProt...")
    
    for i in range(0, len(gene_list), batch_size):
        batch_genes = gene_list[i:i+batch_size]
        # Query: Human + Swiss-Prot + gene name list
        query = f"model_organism:{taxon_id} AND reviewed:true AND (" + \
                " OR ".join([f"gene_exact:{g}" for g in batch_genes]) + ")"
        
        params = {
            "query": query,
            "format": "fasta",
            "size": 500
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                fasta_data = response.text
                for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
                    desc = record.description
                    for g in batch_genes:
                        if f"GN={g} " in desc or desc.endswith(f"GN={g}"):
                            gene_to_seq[g] = str(record.seq)
                            break
            else:
                print(f"Batch {i//batch_size} request failed, status code: {response.status_code}")
        except Exception as e:
            print(f"Batch {i//batch_size} error: {e}")
        
        time.sleep(0.2)
    
    return gene_to_seq


def generate_protein_embeddings(
    hvg_vocab_dict: Dict[str, int], 
    save_path: str, 
    device: torch.device, 
    full_vocab_size: int,
    model_name: str = "facebook/esm2_t33_650M_UR50D"
) -> torch.Tensor:
    """
    Generate ESM-2 protein embeddings for selected genes
    
    Args:
        hvg_vocab_dict: Dictionary mapping gene names to vocab IDs
        save_path: Path to save the embedding matrix
        device: Device to run the model on
        full_vocab_size: Size of full vocabulary
        model_name: ESM model name
        
    Returns:
        ESM embedding matrix [full_vocab_size, 1280]
    """
    # 1. Load ESM2 model and tokenizer
    print(f"Loading ESM model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_esm = EsmModel.from_pretrained(model_name).to(device).eval()
    
    # 2. Get sequences
    gene_list = list(hvg_vocab_dict.keys())
    gene_to_seq = get_uniprot_sequences_batch(gene_list)
    print(f"Successfully retrieved {len(gene_to_seq)} protein sequences")
    
    # 3. Initialize full matrix [full_vocab_size, 1280]
    esm_matrix = torch.zeros((full_vocab_size, 1280))
    
    # 4. Extract features
    print("Extracting ESM feature vectors...")
    for gene_name, gene_id in tqdm(hvg_vocab_dict.items()):
        seq = gene_to_seq.get(gene_name)
        if not seq:
            continue  # If sequence not found, keep as zeros
        
        # Tokenize
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_esm(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
            # Mean Pooling: average excluding CLS and EOS tokens
            seq_rep = last_hidden_states[0, 1:-1, :].mean(dim=0)
            
            # Fill into matrix at corresponding index
            esm_matrix[gene_id] = seq_rep.cpu()
    
    # 5. Save result
    torch.save(esm_matrix, save_path)
    print(f"ESM embedding matrix saved to: {save_path}")
    return esm_matrix


def load_esm_embeddings(
    esm_path: str, 
    selected_genes: List[str], 
    vocab_dict: Dict[str, int],
    device: torch.device
) -> torch.Tensor:
    """
    Load ESM embeddings from file and extract for selected genes
    
    Args:
        esm_path: Path to ESM embedding file
        selected_genes: List of selected gene names
        vocab_dict: Vocabulary dictionary
        device: Device to load embeddings to
        
    Returns:
        ESM embeddings for selected genes [len(selected_genes), 1280]
    """
    if not os.path.exists(esm_path):
        raise FileNotFoundError(f"ESM embedding file not found: {esm_path}")
    
    print(f"Loading ESM embeddings from: {esm_path}")
    esm_matrix = torch.load(esm_path, map_location='cpu')
    
    # Extract embeddings for selected genes based on their vocab IDs
    gene_ids = [vocab_dict[g] for g in selected_genes]
    selected_esm = esm_matrix[gene_ids].to(device)
    
    print(f"ESM embeddings loaded: shape {selected_esm.shape}")
    return selected_esm
