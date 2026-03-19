"""
Improved TSE Module with better patch/token similarity computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer, l2norm


class ImprovedTexualEmbeddingLayer(TexualEmbeddingLayer):
    """
    Improved TSE for text with cross-attention for better patch/token similarity
    """
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3, use_cross_attention=True):
        super().__init__(input_dim, embed_dim, ratio)
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            # Cross-attention for better token-to-token similarity
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            # Layer norm for stability
            self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, features, text, atten):
        # Get base features from parent class
        base_features = super().forward(features, text, atten)
        
        if self.use_cross_attention and len(base_features.shape) == 2:
            # Apply cross-attention for better similarity computation
            # Expand to sequence for attention: [batch, 1, embed_dim]
            base_features_seq = base_features.unsqueeze(1)
            
            # Self-attention for refinement
            attn_output, _ = self.cross_attn(
                base_features_seq, base_features_seq, base_features_seq
            )
            
            # Residual connection and layer norm
            attn_output = self.layer_norm(attn_output + base_features_seq)
            base_features = attn_output.squeeze(1)
        
        return base_features


class ImprovedVisualEmbeddingLayer(VisualEmbeddingLayer):
    """
    Improved TSE for vision with better patch similarity computation
    """
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3, use_patch_similarity=True):
        super().__init__(input_dim, embed_dim, ratio)
        self.use_patch_similarity = use_patch_similarity
        
        if use_patch_similarity:
            # Patch similarity enhancement
            self.patch_sim_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
    
    def forward(self, base_features, atten):
        # Get base features from parent class
        base_features = super().forward(base_features, atten)
        
        if self.use_patch_similarity:
            # Enhance patch similarity representation
            base_features = self.patch_sim_proj(base_features) + base_features
        
        return base_features


class CrossModalSimilarity(nn.Module):
    """
    Compute improved similarity between image patches and text tokens
    """
    def __init__(self, embed_dim=1024, temperature=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Projection layers for cross-modal alignment
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.txt_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, img_features, txt_features):
        """
        Args:
            img_features: [batch, num_patches, embed_dim]
            txt_features: [batch, num_tokens, embed_dim]
        Returns:
            similarity_matrix: [batch, num_tokens, num_patches]
        """
        # Project to common space
        img_proj = F.normalize(self.img_proj(img_features), p=2, dim=-1)
        txt_proj = F.normalize(self.txt_proj(txt_features), p=2, dim=-1)
        
        # Compute cross-modal similarity
        # [batch, num_tokens, embed_dim] @ [batch, embed_dim, num_patches]
        similarity = torch.bmm(txt_proj, img_proj.transpose(1, 2)) / self.temperature
        
        return similarity

