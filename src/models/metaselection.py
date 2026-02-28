import torch
from torch import nn
from typing import Optional

# Import base components from scgpt
from scgpt.model import GeneEncoder, ContinuousValueEncoder, TransformerEncoder
from torch.nn import TransformerEncoderLayer


class scGPTWithMetadata(nn.Module):
    """
    scGPT model with Metadata Selection (Metaselection)
    Adds metadata encoder for cell/drug metadata features (dosage, time, etc.)
    """
    def __init__(
        self,
        ntokens: int,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 6,
        dropout: float = 0.1,
        drug_emb_dim: int = 384,
        n_celltype: int = 5,
        metadata_dim: int = 4  # e.g., dosage, time, cell_line_type, etc.
    ):
        super().__init__()
        self.d_model = d_model
        
        # Gene and expression encoding
        self.gene_encoder = GeneEncoder(ntokens, d_model)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.cell_emb = nn.Embedding(n_celltype, d_model) 
        
        # Metadata encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Drug projection layer
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN as used in scGPT pretraining
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=nlayers
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_hid),
            nn.LeakyReLU(),
            nn.Linear(d_hid, 1)
        )

    def forward(self, gene_ids, c_gene, drug_emb, cell_type_id, meta_data, src_key_padding_mask=None):
        gene_embs = self.gene_encoder(gene_ids) 
        value_embs = self.value_encoder(c_gene)
        
        # Add metadata embedding
        meta_embs = self.meta_encoder(meta_data).unsqueeze(1)
        total_gene_embs = gene_embs + value_embs + meta_embs 
        
        cell_emb = self.cell_emb(cell_type_id).unsqueeze(1)
        total_gene_embs = total_gene_embs + cell_emb
        
        drug_token = self.drug_projector(drug_emb).unsqueeze(1)
        input_embs = torch.cat([drug_token, total_gene_embs], dim=1)
        
        
        # Run Transformer
        output = self.transformer_encoder(
            input_embs, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        gene_output = output[:, 1:, :] 
        pred_delta = self.decoder(gene_output).squeeze(-1)
        
        return pred_delta
