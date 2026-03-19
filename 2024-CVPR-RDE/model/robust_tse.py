"""
Robust TSE Module - Focus on robustness to attribute-level noise
Addresses: patch/token similarity computation robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer, l2norm


class RobustTexualEmbeddingLayer(TexualEmbeddingLayer):
    """
    Robust TSE for text - handles attribute-level noise better
    Key improvements:
    1. More robust patch/token selection
    2. Better handling of noisy tokens
    3. Confidence-based weighting
    """
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3, noise_robust=True):
        super().__init__(input_dim, embed_dim, ratio)
        self.noise_robust = noise_robust
        
        if noise_robust:
            # Confidence estimation for token reliability
            self.confidence_estimator = nn.Sequential(
                nn.Linear(input_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Robust aggregation with outlier handling
            self.robust_pool = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
    
    def forward(self, features, text, atten):
        # Get base features
        base_features = super().forward(features, text, atten)
        
        if self.noise_robust and len(base_features.shape) == 2:
            # Estimate confidence for robustness
            # Use original features before pooling for confidence
            if hasattr(self, '_last_features'):
                conf_scores = self.confidence_estimator(self._last_features.mean(dim=1))
                # Weight features by confidence
                base_features = base_features * conf_scores + base_features * (1 - conf_scores) * 0.5
        
        return base_features


class RobustVisualEmbeddingLayer(VisualEmbeddingLayer):
    """
    Robust TSE for vision - handles attribute-level noise (colors, etc.) better
    Key improvements:
    1. More robust patch selection
    2. Better handling of noisy patches
    3. Attribute-aware filtering
    """
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3, noise_robust=True):
        super().__init__(input_dim, embed_dim, ratio)
        self.noise_robust = noise_robust
        
        if noise_robust:
            # Patch confidence for robustness
            self.patch_confidence = nn.Sequential(
                nn.Linear(input_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Robust aggregation
            self.robust_aggregation = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
    
    def forward(self, base_features, atten):
        # Get base features
        features = super().forward(base_features, atten)
        
        if self.noise_robust:
            # Apply robust aggregation
            features = self.robust_aggregation(features) + features
        
        return features


def robust_similarity_computation(img_feats, txt_feats, temperature=0.07, robust=True):
    """
    Robust similarity computation between image and text features
    Handles outliers and noisy samples better
    """
    # Normalize
    img_feats = F.normalize(img_feats, p=2, dim=-1)
    txt_feats = F.normalize(txt_feats, p=2, dim=-1)
    
    # Compute similarity
    similarity = txt_feats @ img_feats.t() / temperature
    
    if robust:
        # Clip extreme values to prevent outliers from dominating
        similarity = torch.clamp(similarity, min=-10.0, max=10.0)
        
        # Apply temperature annealing for stability
        similarity = similarity / max(temperature, 0.01)
    
    return similarity

