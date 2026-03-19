"""
Enhanced TSE Module with improved patch/token information
Key idea: Enhance patch/token features BEFORE similarity computation
This should improve performance, especially for attribute-level noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer, l2norm


class PatchTokenEnhancer(nn.Module):
    """
    Enhances patch/token information before TSE processing
    Improves: attribute-level feature representation, local alignment
    """
    def __init__(self, input_dim=768, embed_dim=1024, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Multi-layer feature enhancement
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else embed_dim
            layers.extend([
                nn.Linear(in_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.enhancement_layers = nn.Sequential(*layers)
        
        # Attribute-aware enhancement (for colors, attributes)
        self.attribute_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Cross-modal alignment enhancement
        self.alignment_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, features, attention_weights=None):
        """
        Enhance patch/token features
        
        Args:
            features: [batch, seq_len, input_dim] or [batch, input_dim]
            attention_weights: Optional attention for weighting
        Returns:
            enhanced_features: [batch, seq_len, embed_dim] or [batch, embed_dim]
        """
        original_shape = features.shape
        is_2d = len(features.shape) == 2
        
        if is_2d:
            features = features.unsqueeze(1)  # [batch, 1, input_dim]
        
        # Feature enhancement
        enhanced = self.enhancement_layers(features)
        
        # Attribute-aware enhancement (important for colors, attributes)
        enhanced = self.attribute_enhancer(enhanced) + enhanced
        
        # Cross-modal alignment projection
        enhanced = self.alignment_proj(enhanced) + enhanced
        
        # Apply attention weighting if provided
        if attention_weights is not None:
            if len(attention_weights.shape) == 2:
                attention_weights = attention_weights.unsqueeze(-1)
            enhanced = enhanced * attention_weights
        
        if is_2d:
            enhanced = enhanced.squeeze(1)
        
        return enhanced


class EnhancedTexualEmbeddingLayer(TexualEmbeddingLayer):
    """
    Enhanced TSE for text with improved token information
    """
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3, enhance_features=True):
        super().__init__(input_dim, embed_dim, ratio)
        self.enhance_features = enhance_features
        
        if enhance_features:
            # Enhance token features before processing
            self.token_enhancer = PatchTokenEnhancer(
                input_dim=input_dim,
                embed_dim=input_dim,  # Keep same dim for compatibility
                num_layers=2
            )
    
    def forward(self, features, text, atten):
        # ENHANCEMENT: Improve token information BEFORE TSE processing
        if self.enhance_features:
            # Enhance all token features first
            features = self.token_enhancer(features, attention_weights=None)
        
        # Then apply original TSE logic
        return super().forward(features, text, atten)


class EnhancedVisualEmbeddingLayer(VisualEmbeddingLayer):
    """
    Enhanced TSE for vision with improved patch information
    """
    def __init__(self, input_dim=768, embed_dim=1024, ratio=0.3, enhance_features=True):
        super().__init__(input_dim, embed_dim, ratio)
        self.enhance_features = enhance_features
        
        if enhance_features:
            # Enhance patch features before processing
            self.patch_enhancer = PatchTokenEnhancer(
                input_dim=input_dim,
                embed_dim=input_dim,  # Keep same dim for compatibility
                num_layers=2
            )
    
    def forward(self, base_features, atten):
        # ENHANCEMENT: Improve patch information BEFORE TSE processing
        if self.enhance_features:
            # Enhance all patch features first
            base_features = self.patch_enhancer(base_features, attention_weights=atten)
        
        # Then apply original TSE logic
        return super().forward(base_features, atten)


class ImprovedSimilarityComputation(nn.Module):
    """
    Improved similarity computation with enhanced patch/token information
    """
    def __init__(self, embed_dim=1024, temperature=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Similarity enhancement
        self.sim_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, img_patches, txt_tokens):
        """
        Compute improved similarity with enhanced features
        
        Args:
            img_patches: [batch, num_patches, embed_dim]
            txt_tokens: [batch, num_tokens, embed_dim]
        Returns:
            similarity: [batch, num_tokens, num_patches]
        """
        # Enhance features before similarity
        img_enhanced = self.sim_enhancer(img_patches)
        txt_enhanced = self.sim_enhancer(txt_tokens)
        
        # Normalize
        img_enhanced = F.normalize(img_enhanced, p=2, dim=-1)
        txt_enhanced = F.normalize(txt_enhanced, p=2, dim=-1)
        
        # Compute similarity
        similarity = torch.bmm(txt_enhanced, img_enhanced.transpose(1, 2)) / self.temperature
        
        return similarity

