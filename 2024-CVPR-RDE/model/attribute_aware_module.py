"""
Attribute-Aware Module for better handling of attribute-level noise
Based on the paper's suggestion: "need mechanisms that more effectively capture fine-grained local features"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeAwareModule(nn.Module):
    """
    Module to better capture fine-grained local features (colors, attributes)
    for improved robustness to attribute-level noise
    """
    def __init__(self, embed_dim, num_attributes=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attributes = num_attributes
        
        # Attribute-specific projection layers
        self.attribute_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for _ in range(num_attributes)
        ])
        
        # Attention mechanism for attribute weighting
        self.attribute_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_attributes),
            nn.Softmax(dim=-1)
        )
        
        # Local alignment enhancement
        self.local_aligner = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, features, attention_weights=None):
        """
        Args:
            features: [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
            attention_weights: Optional attention weights for attribute selection
        Returns:
            enhanced_features: Enhanced features with better local attribute awareness
        """
        if len(features.shape) == 2:
            # Global features: [batch_size, embed_dim]
            batch_size, embed_dim = features.shape
            seq_len = 1
            features = features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        else:
            batch_size, seq_len, embed_dim = features.shape
        
        # Compute attribute attention weights
        if attention_weights is None:
            # Use mean pooling for attention computation
            pooled = features.mean(dim=1)  # [batch_size, embed_dim]
            attn_weights = self.attribute_attention(pooled)  # [batch_size, num_attributes]
        else:
            attn_weights = attention_weights
        
        # Apply attribute-specific projections
        attribute_features = []
        for i, proj in enumerate(self.attribute_projections):
            attr_feat = proj(features)  # [batch_size, seq_len, embed_dim]
            # Weight by attention
            weight = attn_weights[:, i:i+1].unsqueeze(-1)  # [batch_size, 1, 1]
            attribute_features.append(attr_feat * weight)
        
        # Combine attribute features
        combined = sum(attribute_features)  # [batch_size, seq_len, embed_dim]
        
        # Apply local alignment enhancement
        enhanced = self.local_aligner(combined)
        
        # Residual connection
        enhanced = enhanced + features
        
        # Return to original shape
        if seq_len == 1:
            enhanced = enhanced.squeeze(1)  # [batch_size, embed_dim]
        
        return enhanced, attn_weights


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion of BGE and TSE based on noise rate
    At high noise rates, weight BGE more; at low noise rates, use both equally
    """
    def __init__(self, noise_rate=0.0):
        super().__init__()
        self.noise_rate = noise_rate
        
    def forward(self, sims_bge, sims_tse):
        """
        Args:
            sims_bge: BGE similarity matrix
            sims_tse: TSE similarity matrix
        Returns:
            fused_sims: Adaptively fused similarity
        """
        # Adaptive weighting based on noise rate
        # At noise_rate=0.0: equal weighting (0.5, 0.5)
        # At noise_rate=0.7: more weight to BGE (0.7, 0.3)
        # Linear interpolation
        alpha = 0.5 + 0.3 * self.noise_rate  # ranges from 0.5 to 0.71
        beta = 1.0 - alpha
        
        # Normalize similarities first
        sims_bge_norm = F.normalize(sims_bge, p=2, dim=1)
        sims_tse_norm = F.normalize(sims_tse, p=2, dim=1)
        
        # Adaptive fusion
        fused = alpha * sims_bge_norm + beta * sims_tse_norm
        
        return fused
    
    def set_noise_rate(self, noise_rate):
        """Update noise rate for adaptive weighting"""
        self.noise_rate = noise_rate

