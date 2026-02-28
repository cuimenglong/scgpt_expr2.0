import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scgpt.model import GeneEncoder, ContinuousValueEncoder, TransformerEncoder
from torch.nn import TransformerEncoderLayer

from .protein import GeneProteinBiasTransformer


class scGPTWithVirtualProteinAndPPI(nn.Module):
    """
    scGPT model with ESM protein embeddings and PPI prior network
    Supports bidirectional attention mask between protein nodes
    """
    def __init__(
        self,
        ntokens: int,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        scgpt_layers: int = 12,
        gp_layers: int = 3,
        dropout: float = 0.1,
        n_celltype: int = 5,
        esm_dim: int = 1280,
        drug_emb_dim: int = 384,
        pos_bias: float = 2.0,
        neg_bias: float = -5.0,
        ppi_adjacency: np.ndarray = None  # PPI adjacency matrix
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_bias = pos_bias
        self.neg_bias = neg_bias
        
        # Register PPI adjacency matrix
        if ppi_adjacency is not None:
            self.register_buffer('ppi_adjacency', torch.from_numpy(ppi_adjacency).float())
        else:
            self.register_buffer('ppi_adjacency', None)
        
        # --- 1. scGPT Base Components ---
        self.gene_encoder = GeneEncoder(ntokens, d_model)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.cell_emb = nn.Embedding(n_celltype, d_model)
        
        # Drug projection layer
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.scgpt_transformer = TransformerEncoder(encoder_layer, num_layers=scgpt_layers)
        
        # --- 2. Gene-Protein Interaction Components ---
        self.gene_to_protein_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.esm_projector = nn.Linear(esm_dim, d_model)
        
        self.gp_transformer = GeneProteinBiasTransformer(
            d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=gp_layers, dropout=dropout
        )
        
        # --- 3. Output Layer ---
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_hid),
            nn.LeakyReLU(),
            nn.Linear(d_hid, 1)
        )

    def create_dynamic_gp_bias(self, n_genes, prot_indices, device):
        """
        Construct asymmetric bias matrix with Drug Token
        Sequence structure: [Drug] (1) + [Genes] (n_genes) + [Proteins] (n_prot)
        
        Now supports PPI prior: add bidirectional positive bias between protein nodes with PPI edges
        """
        n_prot = len(prot_indices)
        total_len = 1 + n_genes + n_prot
        bias = torch.zeros((total_len, total_len), device=device)
        
        gene_start = 1
        prot_start = 1 + n_genes
        
        # 1. Fill G->P and P->G base negative values (excluding Drug token)
        bias[gene_start:prot_start, prot_start:] = self.neg_bias
        bias[prot_start:, gene_start:prot_start] = self.neg_bias
        
        # 2. Fill corresponding positive values (genes with their corresponding protein nodes)
        for i, gene_idx in enumerate(prot_indices):
            actual_gene_idx = gene_start + gene_idx
            actual_prot_idx = prot_start + i
            # Gene -> Prot
            bias[actual_gene_idx, actual_prot_idx] = self.pos_bias
            # Prot -> Gene
            bias[actual_prot_idx, actual_gene_idx] = self.pos_bias
        
        # 3. If PPI adjacency matrix exists, add bidirectional edges between protein nodes
        if self.ppi_adjacency is not None and n_prot > 0:
            for i, gene_idx_i in enumerate(prot_indices):
                for j, gene_idx_j in enumerate(prot_indices):
                    if i == j:
                        continue  # Skip self-loop
                    # Check if there's an edge in the original PPI matrix
                    if gene_idx_i < self.ppi_adjacency.shape[0] and gene_idx_j < self.ppi_adjacency.shape[1]:
                        if self.ppi_adjacency[gene_idx_i, gene_idx_j] > 0:
                            # PPI exists between protein nodes i and j, add bidirectional positive bias
                            actual_prot_i = prot_start + i
                            actual_prot_j = prot_start + j
                            bias[actual_prot_i, actual_prot_j] = self.pos_bias
                            bias[actual_prot_j, actual_prot_i] = self.pos_bias
        
        # Drug Token (Index 0) default bias is 0 with all nodes (Full Attention)
        return bias

    def forward(self, gene_ids, c_gene, drug_emb, cell_type_id, esm_embeddings, src_key_padding_mask=None):
        batch_size, n_genes = gene_ids.shape
        device = gene_ids.device
        
        # --- Step 1: scGPT 12-layer encoding ---
        gene_embs = self.gene_encoder(gene_ids) 
        value_embs = self.value_encoder(c_gene)
        cell_emb = self.cell_emb(cell_type_id).unsqueeze(1)
        x_gene_scgpt = gene_embs + value_embs + cell_emb
        scgpt_out = self.scgpt_transformer(x_gene_scgpt, src_key_padding_mask=src_key_padding_mask)
        
        # --- Step 2: Identify and extract valid protein nodes ---
        with torch.no_grad():
            has_protein = (esm_embeddings[0].abs().sum(dim=-1) > 0)
            prot_indices = torch.where(has_protein)[0].tolist()
            n_prot = len(prot_indices)
            
        prot_all = self.gene_to_protein_mlp(gene_embs) + self.esm_projector(esm_embeddings)
        prot_valid = prot_all[:, prot_indices, :]
        
        # --- Step 3: Drug Token Processing ---
        drug_token = self.drug_projector(drug_emb).unsqueeze(1) # [B, 1, d_model]
        
        # --- Step 4: Concatenate sequence [Drug, Genes, Proteins] ---
        combined_input = torch.cat([drug_token, scgpt_out, prot_valid], dim=1) 
        
        # Construct new Padding Mask
        combined_padding_mask = None
        if src_key_padding_mask is not None:
            drug_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            prot_padding_mask = src_key_padding_mask[:, prot_indices]
            combined_padding_mask = torch.cat([drug_pad, src_key_padding_mask, prot_padding_mask], dim=1)
            
        # Construct dynamic Attention Bias with Drug offset
        attn_bias = self.create_dynamic_gp_bias(n_genes, prot_indices, device)
        
        # --- Step 5: Interaction Transformer ---
        gp_out = self.gp_transformer(
            combined_input, 
            attn_bias=attn_bias, 
            src_key_padding_mask=combined_padding_mask
        )
        
        # --- Step 6: Decode ---
        final_gene_out = gp_out[:, 1 : 1 + n_genes, :]
        out_genes = self.decoder(final_gene_out).squeeze(-1)
        
        return out_genes


# Backward compatibility
class scGPTWithVirtualProteinPPI(scGPTWithVirtualProteinAndPPI):
    """Alias for backward compatibility"""
    pass
