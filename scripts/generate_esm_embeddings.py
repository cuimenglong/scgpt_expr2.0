"""
Generate ESM-2 protein embeddings for selected genes
"""
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
import json
import os
import time
from tqdm import tqdm
from io import StringIO
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel

# Set HF-Mirror as the HuggingFace endpoint
HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_MIRROR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable faster download
print(f"Using HuggingFace mirror: {HF_MIRROR}")


def get_uniprot_sequences_batch(gene_list, taxon_id="9606"):
    """
    通过 UniProt REST API 批量获取基因对应的蛋白质序列
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    gene_to_seq = {}
    
    batch_size = 50
    print(f"Fetching protein sequences for {len(gene_list)} genes from UniProt...")
    
    for i in range(0, len(gene_list), batch_size):
        batch_genes = gene_list[i:i+batch_size]
        query = f"model_organism:{taxon_id} AND reviewed:true AND (" + \
                " OR ".join([f"gene_exact:{g}" for g in batch_genes]) + ")"
        
        params = {
            "query": query,
            "format": "fasta",
            "size": 500
        }
        
        try:
            response = session.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                fasta_data = response.text
                for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
                    desc = record.description
                    for g in batch_genes:
                        if f"GN={g} " in desc or desc.endswith(f"GN={g}"):
                            gene_to_seq[g] = str(record.seq)
                            break
            else:
                print(f"Batch {i//batch_size} failed, status: {response.status_code}")
        except Exception as e:
            print(f"Batch {i//batch_size} error: {e}")
        
        time.sleep(0.2)
    
    return gene_to_seq


def generate_protein_embeddings(hvg_gene_list, save_path, device, model_name="facebook/esm2_t33_650M_UR50D"):
    """
    使用 ESM-2 提取蛋白质特征向量
    """
    # 1. Load ESM2 model and tokenizer
    print(f"Loading ESM model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_esm = EsmModel.from_pretrained(model_name).to(device).eval()
    
    # 2. Get protein sequences
    gene_to_seq = get_uniprot_sequences_batch(hvg_gene_list)
    print(f"Successfully retrieved {len(gene_to_seq)} protein sequences")
    
    # 3. Initialize embedding matrix
    embedding_dim = model_esm.config.hidden_size  # 1280 for ESM2-650M
    esm_matrix = torch.zeros((len(hvg_gene_list), embedding_dim))
    
    # 4. Extract features
    print("Extracting ESM embeddings...")
    gene_to_idx = {g: i for i, g in enumerate(hvg_gene_list)}
    
    for gene_name in tqdm(hvg_gene_list):
        seq = gene_to_seq.get(gene_name)
        if not seq:
            continue
            
        # Tokenize
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1022)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_esm(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling (exclude CLS and EOS tokens)
            seq_rep = last_hidden_states[0, 1:-1, :].mean(dim=0)
            
            # Fill in matrix
            esm_matrix[gene_to_idx[gene_name]] = seq_rep.cpu()
    
    # 5. Save result
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(esm_matrix, save_path)
    print(f"ESM embeddings saved to: {save_path}")
    
    # Save gene mapping
    gene_mapping = {g: i for i, g in enumerate(hvg_gene_list)}
    mapping_path = save_path.replace('.pt', '_genes.json')
    with open(mapping_path, 'w') as f:
        json.dump(gene_mapping, f)
    print(f"Gene mapping saved to: {mapping_path}")
    
    return esm_matrix


def main():
    parser = argparse.ArgumentParser(description="Generate ESM protein embeddings")
    parser.add_argument('--genes', '-g', type=str, required=True,
                       help='Path to selected genes JSON file or comma-separated gene list')
    parser.add_argument('--output', '-o', type=str, default='./protein_esm_embeddings.pt',
                       help='Output path for ESM embeddings')
    parser.add_argument('--model', type=str, default='facebook/esm2_t33_650M_UR50D',
                       help='ESM model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Load gene list
    if os.path.isfile(args.genes):
        with open(args.genes, 'r') as f:
            if args.genes.endswith('.json'):
                gene_list = json.load(f)
            else:
                gene_list = [line.strip() for line in f if line.strip()]
        # Handle dict format (gene -> idx)
        if isinstance(gene_list, dict):
            gene_list = list(gene_list.keys())
    else:
        gene_list = args.genes.split(',')
    
    print(f"Generating embeddings for {len(gene_list)} genes...")
    
    generate_protein_embeddings(gene_list, args.output, device, args.model)


if __name__ == "__main__":
    main()
