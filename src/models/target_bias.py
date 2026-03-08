import torch
from torch import nn
from typing import Optional

# Import base components from scgpt
from scgpt.model import GeneEncoder, ContinuousValueEncoder, TransformerEncoder
from torch.nn import TransformerEncoderLayer


class scGPTWithTargetBias(nn.Module):
    """
    scGPT model with Target Gene Bias
    Adds attention bias to specific target genes (drug targets)
    """
    def __init__(
        self,
        ntokens: int,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 12,
        dropout: float = 0.1,
        drug_emb_dim: int = 768,  # ChemBERTa-77M-MLM output dimension
        n_celltype: int = 5,
        target_bias_value: float = 5.0,
        gene_ids: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.nhead = nhead
        self.target_bias_value = target_bias_value
        
        # Register static gene_ids
        if gene_ids is not None:
            self.register_buffer("static_gene_ids", gene_ids)
        
        # Encoder
        self.gene_encoder = GeneEncoder(ntokens, d_model)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.cell_emb = nn.Embedding(n_celltype, d_model)
        
        # Drug projection
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_emb_dim, d_model), 
            nn.LayerNorm(d_model),
            nn.ReLU(), 
            nn.Linear(d_model, d_model)
        )
        
        # PyTorch native Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_hid), 
            nn.LeakyReLU(), 
            nn.Linear(d_hid, 1)
        )

    def forward(self, c_gene, drug_emb, cell_type_id, target_gene_ids=None, src_key_padding_mask=None):
        batch_size = c_gene.shape[0]
        
        # Get static gene IDs and expand to batch size
        gene_ids = self.static_gene_ids.unsqueeze(0).expand(batch_size, -1)
        
        seq_len = gene_ids.shape[1]
        full_len = seq_len + 1  # +1 for drug token
        
        # 1. Input Embedding
        g_emb = self.gene_encoder(gene_ids) + self.value_encoder(c_gene)
        g_emb = g_emb + self.cell_emb(cell_type_id).unsqueeze(1)
        
        d_emb = self.drug_projector(drug_emb).unsqueeze(1)
        
        # Concatenate Drug Token
        x = torch.cat([d_emb, g_emb], dim=1)  # [B, L+1, D]
        
        # 2. Build Attention Bias Matrix
        # PyTorch TransformerEncoderLayer uses src_mask with shape [seq, seq]
        # We average the target bias across the batch for simplicity
        attn_bias = torch.zeros(full_len, full_len, device=gene_ids.device)
        
        if target_gene_ids is not None:
            # Collect target bias positions across batch
            for i in range(batch_size):
                valid_tpts = target_gene_ids[i][target_gene_ids[i] != -1]
                if len(valid_tpts) > 0:
                    mask = torch.isin(gene_ids[i], valid_tpts)
                    t_idx = torch.where(mask)[0] + 1  # +1 for drug token
                    if len(t_idx) > 0:
                        # Drug token (index 0) <-> Target genes
                        attn_bias[t_idx, 0] += self.target_bias_value / batch_size
                        attn_bias[0, t_idx] += self.target_bias_value / batch_size
        
        # 3. Transformer Encoder
        # Use src_key_padding_mask for padding, attn_bias for target bias
        output = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask,
            src_mask=attn_bias
        )
        
        # 4. Decode
        return self.decoder(output[:, 1:, :]).squeeze(-1)
