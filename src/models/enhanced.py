"""
Enhanced scGPT model combining:
1. Filtered confidence-weighted PPI attention bias
2. Pathway co-membership attention bias
3. Drug -> Target -> PPI signal flow
4. Population-level drug effect decomposition
5. Virtual protein nodes with ESM embeddings
"""

import torch
from torch import nn
import numpy as np
from scgpt.model import GeneEncoder, ContinuousValueEncoder, TransformerEncoder
from torch.nn import TransformerEncoderLayer

from .protein import GeneProteinBiasTransformer


class scGPTEnhanced(nn.Module):

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
        drug_emb_dim: int = 768,
        # PPI parameters
        pos_bias: float = 2.0,
        neg_bias: float = -5.0,
        ppi_adjacency: np.ndarray = None,       # confidence-weighted (n_genes, n_genes)
        # Pathway parameters
        pathway_adjacency: np.ndarray = None,    # co-pathway adjacency (n_genes, n_genes)
        pathway_bias_scale: float = 1.0,
        # Drug-target parameters
        target_bias_value: float = 3.0,
        target_propagation_decay: float = 0.5,   # decay for PPI neighbor propagation
        # Population decomposition
        use_population_decomposition: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_bias = pos_bias
        self.neg_bias = neg_bias
        self.pathway_bias_scale = pathway_bias_scale
        self.target_bias_value = target_bias_value
        self.target_propagation_decay = target_propagation_decay
        self.use_population_decomposition = use_population_decomposition

        # Register PPI adjacency (continuous weights)
        if ppi_adjacency is not None:
            self.register_buffer('ppi_adjacency', torch.from_numpy(ppi_adjacency).float())
        else:
            self.register_buffer('ppi_adjacency', None)

        # Register pathway adjacency
        if pathway_adjacency is not None:
            self.register_buffer('pathway_adjacency', torch.from_numpy(pathway_adjacency).float())
        else:
            self.register_buffer('pathway_adjacency', None)

        # --- 1. scGPT Base Components ---
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

        # --- 3. Output Layers ---
        if use_population_decomposition:
            # Population decoder: shared drug mechanism across all cell types
            self.population_decoder = nn.Sequential(
                nn.Linear(d_model, d_hid),
                nn.LeakyReLU(),
                nn.Linear(d_hid, 1)
            )
            # Cell-specific residual decoder: gene_out + cell_emb
            self.cell_residual_decoder = nn.Sequential(
                nn.Linear(d_model + d_model, d_hid),
                nn.LeakyReLU(),
                nn.Linear(d_hid, 1)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_hid),
                nn.LeakyReLU(),
                nn.Linear(d_hid, 1)
            )

    def create_enhanced_bias(self, n_genes, prot_indices, target_gene_mask, device):
        """
        Enhanced attention bias matrix combining 5 sources (fully vectorized).

        Sequence: [Drug](1) + [Genes](n_genes) + [Proteins](n_prot)
        """
        n_prot = len(prot_indices)
        total_len = 1 + n_genes + n_prot
        bias = torch.zeros((total_len, total_len), device=device)

        gene_start = 1
        prot_start = 1 + n_genes

        # --- 1. Gene-Protein base bias (vectorized) ---
        if n_prot > 0:
            bias[gene_start:prot_start, prot_start:] = self.neg_bias
            bias[prot_start:, gene_start:prot_start] = self.neg_bias
            # Matched gene<->protein pairs: use advanced indexing
            prot_idx_tensor = torch.tensor(prot_indices, dtype=torch.long, device=device)
            gene_positions = gene_start + prot_idx_tensor
            prot_positions = prot_start + torch.arange(n_prot, device=device)
            bias[gene_positions, prot_positions] = self.pos_bias
            bias[prot_positions, gene_positions] = self.pos_bias

        # --- 2. Confidence-weighted PPI between protein nodes ---
        if self.ppi_adjacency is not None and n_prot > 0:
            ppi_sub = self.ppi_adjacency[prot_idx_tensor][:, prot_idx_tensor]
            bias[prot_start:prot_start + n_prot, prot_start:prot_start + n_prot] += ppi_sub * self.pos_bias

        # --- 3. Pathway co-membership bias between genes ---
        if self.pathway_adjacency is not None:
            bias[gene_start:prot_start, gene_start:prot_start] += (
                self.pathway_adjacency * self.pathway_bias_scale
            )

        # --- 4 & 5. Drug-Target bias + Target-PPI propagation (vectorized) ---
        if target_gene_mask is not None:
            t_mask = target_gene_mask.any(dim=0) if target_gene_mask.dim() == 2 else target_gene_mask
            target_indices = torch.where(t_mask)[0]

            if len(target_indices) > 0:
                target_positions = gene_start + target_indices
                # Drug <-> Target genes: positive bias
                bias[0, target_positions] += self.target_bias_value
                bias[target_positions, 0] += self.target_bias_value

                # Target-PPI propagation: vectorized over all targets at once
                if self.ppi_adjacency is not None:
                    # Sum PPI weights from all target genes to all other genes
                    # shape: (n_targets, n_genes) -> max over targets -> (n_genes,)
                    target_ppi_weights = self.ppi_adjacency[target_indices]  # (n_targets, n_genes)
                    propagated = target_ppi_weights.max(dim=0).values  # (n_genes,)
                    propagated = propagated * (self.target_bias_value * self.target_propagation_decay)
                    # Zero out target genes themselves to avoid double-counting
                    propagated[target_indices] = 0.0
                    neighbor_mask = propagated > 0
                    neighbor_positions = gene_start + torch.where(neighbor_mask)[0]
                    neighbor_values = propagated[neighbor_mask]
                    bias[0, neighbor_positions] += neighbor_values
                    bias[neighbor_positions, 0] += neighbor_values

        return bias

    def forward(self, gene_ids, c_gene, drug_emb, cell_type_id, esm_embeddings,
                target_gene_mask=None, src_key_padding_mask=None):
        batch_size, n_genes = gene_ids.shape
        device = gene_ids.device

        # --- Step 1: scGPT encoding ---
        gene_embs = self.gene_encoder(gene_ids)
        value_embs = self.value_encoder(c_gene)
        cell_emb_vec = self.cell_emb(cell_type_id).unsqueeze(1)
        x = gene_embs + value_embs + cell_emb_vec
        scgpt_out = self.scgpt_transformer(x, src_key_padding_mask=src_key_padding_mask)

        # --- Step 2: Protein nodes ---
        with torch.no_grad():
            has_protein = (esm_embeddings[0].abs().sum(dim=-1) > 0)
            prot_indices = torch.where(has_protein)[0].tolist()
            n_prot = len(prot_indices)

        prot_all = self.gene_to_protein_mlp(gene_embs) + self.esm_projector(esm_embeddings)
        prot_valid = prot_all[:, prot_indices, :]

        # --- Step 3: Drug token ---
        drug_token = self.drug_projector(drug_emb).unsqueeze(1)

        # --- Step 4: Concatenate [Drug, Genes, Proteins] ---
        combined = torch.cat([drug_token, scgpt_out, prot_valid], dim=1)

        # Padding mask
        combined_padding_mask = None
        if src_key_padding_mask is not None:
            drug_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            prot_pad = src_key_padding_mask[:, prot_indices]
            combined_padding_mask = torch.cat([drug_pad, src_key_padding_mask, prot_pad], dim=1)

        # --- Step 5: Enhanced attention bias ---
        attn_bias = self.create_enhanced_bias(n_genes, prot_indices, target_gene_mask, device)

        # --- Step 6: GP transformer ---
        gp_out = self.gp_transformer(
            combined, attn_bias=attn_bias, src_key_padding_mask=combined_padding_mask
        )

        # --- Step 7: Decode ---
        gene_out = gp_out[:, 1:1 + n_genes, :]

        if self.use_population_decomposition:
            pop_delta = self.population_decoder(gene_out).squeeze(-1)
            cell_emb_expanded = cell_emb_vec.expand(-1, n_genes, -1)
            cell_input = torch.cat([gene_out, cell_emb_expanded], dim=-1)
            cell_residual = self.cell_residual_decoder(cell_input).squeeze(-1)
            pred_delta = pop_delta + cell_residual
        else:
            pred_delta = self.decoder(gene_out).squeeze(-1)

        return pred_delta
