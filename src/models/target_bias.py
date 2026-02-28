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
        nlayers: int = 6,
        dropout: float = 0.1,
        drug_emb_dim: int = 384,
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
        
        # 2. Build Bias Matrix
        bias = torch.zeros(batch_size, full_len, full_len, device=gene_ids.device)
        
        if target_gene_ids is not None:
            for i in range(batch_size):
                valid_tpts = target_gene_ids[i][target_gene_ids[i] != -1]
                if len(valid_tpts) > 0:
                    mask = torch.isin(gene_ids[i], valid_tpts)
                    t_idx = torch.where(mask)[0] + 1
                    if len(t_idx) > 0:
                        bias[i, t_idx, 0] = self.target_bias_value
                        bias[i, 0, t_idx] = self.target_bias_value
                        
        # 3. Padding Mask
        if src_key_padding_mask is not None:
            full_p_mask = torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.bool, device=gene_ids.device), 
                src_key_padding_mask
            ], dim=1)
            p_bias = full_p_mask.unsqueeze(1).expand(-1, full_len, -1)
            bias = bias.masked_fill(p_bias, float("-inf"))
            
        # 4. Transformer Encoder
        bias = bias.repeat_interleave(self.nhead, dim=0)
        output = self.transformer_encoder(x, mask=bias)
        
        # 5. Decode
        return self.decoder(output[:, 1:, :]).squeeze(-1)
