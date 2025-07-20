# Visualizing internal states of MedSAM2
_Exported on 7/19/2025 at 17:46:18 PDT from Cursor (1.2.4)_

---

**User**

In order to get more intuitive insight into SAM and also to debug more easily, I'd like to be able to get internal state of MedSAM2 visualized on demand. That could include basic building blocks of MedSAM2. For example, since MedSAM2 is built on SAM2 which is built on ViT which is based on Transformer, I’d like to see what the Position Encoding look like, what the image embeddings look like and how they relate to parts of the input image, how the key matrices of the attention blocks look like and how they evolve as training goes, etc. You should add optional parameters into key functions to indicate whether we need to check internal states. When they are required, the functions can return internal states that we mostly care as additional return value(probably as a key-value dictionary). And you should also write codes for visualizing those inner states. Also include necessary documents for the usage.
You must not temper with the original function of original code, and the “probe” to internal states should not impact the performance largely. 
The probe should be able to enable “on demand”, meaning that, in the future we might make a interactive UI that can switch on/off of visualization on-the-fly of training or inference. So, think carefully to make easy-to-maintain code.

---

**Cursor**

I'll help you add internal state visualization capabilities to MedSAM2. Let me first explore the codebase structure to understand the key components and then implement a comprehensive debugging/visualization system.

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Debug Utilities for SAM2 Internal State Visualization

This module provides comprehensive tools for capturing, analyzing, and visualizing
internal states of SAM2 components during inference and training. It enables
intuitive insights into model behavior and facilitates debugging of complex
transformer-based architectures.

Key Features:

1. **Non-intrusive Capture**: Debug hooks that don't affect model performance
2. **On-demand Activation**: Easy enable/disable functionality for production use
3. **Comprehensive Coverage**: Visualization for all major model components
4. **Interactive Ready**: Designed for future integration with interactive UIs
5. **Performance Aware**: Minimal overhead when debug mode is disabled

Supported Components:
- Position Encodings (sinusoidal, random Fourier features)
- Image Embeddings and their spatial relationships
- Attention matrices and attention patterns
- Transformer layer activations
- Mask decoder intermediate representations
- Training dynamics and gradient flows

Usage:
    # Enable debug mode in model
    predictor = SAM2ImagePredictor(model, debug_mode=True)
    
    # Run inference with debug capture
    masks, scores, logits, debug_states = predictor.predict(
        point_coords=points, point_labels=labels, return_debug_states=True
    )
    
    # Visualize internal states
    from sam2.debug_utils import visualize_debug_states
    visualize_debug_states(debug_states, save_path="debug_output/")
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path


class DebugStateCapture:
    """
    Central registry for capturing and managing debug states across SAM2 components.
    
    This class provides a unified interface for collecting internal states from
    different model components without affecting the original model behavior.
    """
    
    def __init__(self):
        self.states = defaultdict(dict)
        self.enabled = False
        self.capture_attention = True
        self.capture_embeddings = True
        self.capture_position_encoding = True
        self.capture_gradients = False
        
    def enable(self, capture_attention=True, capture_embeddings=True, 
               capture_position_encoding=True, capture_gradients=False):
        """Enable debug capture with specified components."""
        self.enabled = True
        self.capture_attention = capture_attention
        self.capture_embeddings = capture_embeddings
        self.capture_position_encoding = capture_position_encoding
        self.capture_gradients = capture_gradients
        
    def disable(self):
        """Disable debug capture and clear stored states."""
        self.enabled = False
        self.clear()
        
    def clear(self):
        """Clear all captured states."""
        self.states.clear()
        
    def capture(self, component_name: str, state_name: str, data: torch.Tensor, 
                metadata: Optional[Dict] = None):
        """
        Capture a tensor state from a model component.
        
        Args:
            component_name: Name of the component (e.g., 'image_encoder', 'attention_block_0')
            state_name: Name of the specific state (e.g., 'attention_weights', 'embeddings')
            data: Tensor data to capture
            metadata: Additional metadata about the captured state
        """
        if not self.enabled:
            return
            
        # Convert to CPU and detach for safe storage
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            
        self.states[component_name][state_name] = {
            'data': data,
            'shape': data.shape if isinstance(data, torch.Tensor) else None,
            'dtype': data.dtype if isinstance(data, torch.Tensor) else None,
            'metadata': metadata or {}
        }
        
    def get_state(self, component_name: str, state_name: str = None):
        """Retrieve captured state(s) for a component."""
        if state_name is None:
            return self.states.get(component_name, {})
        return self.states.get(component_name, {}).get(state_name)
        
    def get_all_states(self):
        """Retrieve all captured states."""
        return dict(self.states)


# Global debug capture instance
_debug_capture = DebugStateCapture()


def enable_debug_mode(capture_attention=True, capture_embeddings=True, 
                     capture_position_encoding=True, capture_gradients=False):
    """Enable global debug mode for SAM2 components."""
    _debug_capture.enable(capture_attention, capture_embeddings, 
                         capture_position_encoding, capture_gradients)


def disable_debug_mode():
    """Disable global debug mode."""
    _debug_capture.disable()


def capture_debug_state(component_name: str, state_name: str, data: torch.Tensor, 
                       metadata: Optional[Dict] = None):
    """Capture debug state using the global capture instance."""
    _debug_capture.capture(component_name, state_name, data, metadata)


def get_debug_states():
    """Get all captured debug states."""
    return _debug_capture.get_all_states()


def clear_debug_states():
    """Clear all captured debug states."""
    _debug_capture.clear()


def is_debug_enabled():
    """Check if debug mode is currently enabled."""
    return _debug_capture.enabled


# Visualization Functions
class SAM2Visualizer:
    """
    Comprehensive visualization suite for SAM2 internal states.
    """
    
    def __init__(self, figsize_base=(12, 8), dpi=100):
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Custom color maps
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention', ['white', 'red'], N=256
        )
        self.feature_cmap = 'viridis'
        
    def visualize_position_encoding(self, pos_enc_data: torch.Tensor, 
                                   encoding_type: str = "sine",
                                   save_path: Optional[str] = None,
                                   show_individual_dims: bool = True):
        """
        Visualize position encoding patterns.
        
        Args:
            pos_enc_data: Position encoding tensor (B, C, H, W) or (B, H*W, C)
            encoding_type: Type of encoding ("sine", "random", "rope")
            save_path: Path to save the visualization
            show_individual_dims: Whether to show individual encoding dimensions
        """
        if pos_enc_data.dim() == 4:
            # (B, C, H, W) format
            B, C, H, W = pos_enc_data.shape
            pos_enc_data = pos_enc_data[0]  # Take first batch
        elif pos_enc_data.dim() == 3:
            # (B, H*W, C) format - reshape to spatial
            B, HW, C = pos_enc_data.shape
            H = W = int(np.sqrt(HW))
            pos_enc_data = pos_enc_data[0].transpose(0, 1).reshape(C, H, W)
        else:
            raise ValueError(f"Unsupported position encoding shape: {pos_enc_data.shape}")
            
        # Create comprehensive visualization
        if show_individual_dims:
            n_dims_to_show = min(8, pos_enc_data.shape[0])
            fig, axes = plt.subplots(2, max(4, n_dims_to_show//2), 
                                   figsize=(20, 8), dpi=self.dpi)
            axes = axes.flatten()
            
            for i in range(n_dims_to_show):
                im = axes[i].imshow(pos_enc_data[i].numpy(), cmap=self.feature_cmap)
                axes[i].set_title(f'{encoding_type.title()} PE Dim {i}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
                
            # Turn off unused axes
            for i in range(n_dims_to_show, len(axes)):
                axes[i].axis('off')
        else:
            # Show summary statistics
            fig, axes = plt.subplots(2, 2, figsize=self.figsize_base, dpi=self.dpi)
            
            # Mean across all dimensions
            axes[0, 0].imshow(pos_enc_data.mean(0).numpy(), cmap=self.feature_cmap)
            axes[0, 0].set_title(f'{encoding_type.title()} PE - Mean')
            axes[0, 0].axis('off')
            
            # Standard deviation
            axes[0, 1].imshow(pos_enc_data.std(0).numpy(), cmap=self.feature_cmap)
            axes[0, 1].set_title(f'{encoding_type.title()} PE - Std')
            axes[0, 1].axis('off')
            
            # Range (max - min)
            pe_range = pos_enc_data.max(0)[0] - pos_enc_data.min(0)[0]
            axes[1, 0].imshow(pe_range.numpy(), cmap=self.feature_cmap)
            axes[1, 0].set_title(f'{encoding_type.title()} PE - Range')
            axes[1, 0].axis('off')
            
            # Magnitude
            magnitude = torch.norm(pos_enc_data, dim=0)
            axes[1, 1].imshow(magnitude.numpy(), cmap=self.feature_cmap)
            axes[1, 1].set_title(f'{encoding_type.title()} PE - Magnitude')
            axes[1, 1].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/position_encoding_{encoding_type}.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()
        
    def visualize_attention_patterns(self, attention_weights: torch.Tensor,
                                   layer_name: str = "",
                                   head_idx: Optional[int] = None,
                                   save_path: Optional[str] = None,
                                   max_heads_to_show: int = 8):
        """
        Visualize attention weight patterns.
        
        Args:
            attention_weights: Attention weights tensor (B, H, N, N) or (B, N, N)
            layer_name: Name of the attention layer
            head_idx: Specific head to visualize (if None, show multiple heads)
            save_path: Path to save the visualization
            max_heads_to_show: Maximum number of attention heads to display
        """
        if attention_weights.dim() == 4:
            # Multi-head attention (B, H, N, N)
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            attention_weights = attention_weights[0]  # Take first batch
        elif attention_weights.dim() == 3:
            # Single head or already batched (B, N, N)
            if attention_weights.shape[0] == 1:
                attention_weights = attention_weights[0]
                num_heads = 1
            else:
                num_heads = attention_weights.shape[0]
        else:
            raise ValueError(f"Unsupported attention shape: {attention_weights.shape}")
            
        if head_idx is not None:
            # Show specific head
            fig, ax = plt.subplots(1, 1, figsize=self.figsize_base, dpi=self.dpi)
            im = ax.imshow(attention_weights[head_idx].numpy(), cmap=self.attention_cmap)
            ax.set_title(f'{layer_name} - Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax)
        else:
            # Show multiple heads
            heads_to_show = min(max_heads_to_show, num_heads)
            cols = min(4, heads_to_show)
            rows = (heads_to_show + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), dpi=self.dpi)
            if heads_to_show == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
                
            for i in range(heads_to_show):
                ax = axes[i] if heads_to_show > 1 else axes[0]
                im = ax.imshow(attention_weights[i].numpy(), cmap=self.attention_cmap)
                ax.set_title(f'{layer_name} - Head {i}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im, ax=ax, fraction=0.046)
                
            # Turn off unused axes
            if heads_to_show > 1:
                for i in range(heads_to_show, len(axes)):
                    axes[i].axis('off')
                    
        plt.tight_layout()
        if save_path:
            head_suffix = f"_head_{head_idx}" if head_idx is not None else ""
            plt.savefig(f"{save_path}/attention_{layer_name}{head_suffix}.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()
        
    def visualize_image_embeddings(self, embeddings: torch.Tensor,
                                 original_image: Optional[torch.Tensor] = None,
                                 save_path: Optional[str] = None,
                                 n_components_to_show: int = 8):
        """
        Visualize image embeddings and their spatial patterns.
        
        Args:
            embeddings: Image embeddings tensor (B, C, H, W)
            original_image: Original input image for context
            save_path: Path to save the visualization
            n_components_to_show: Number of embedding dimensions to visualize
        """
        if embeddings.dim() != 4:
            raise ValueError(f"Expected 4D embeddings (B, C, H, W), got {embeddings.shape}")
            
        embeddings = embeddings[0]  # Take first batch
        C, H, W = embeddings.shape
        
        # Create comprehensive visualization
        n_components = min(n_components_to_show, C)
        cols = 4
        rows = (n_components + cols - 1) // cols + (1 if original_image is not None else 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), dpi=self.dpi)
        axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        plot_idx = 0
        
        # Show original image if provided
        if original_image is not None:
            if original_image.dim() == 4:
                original_image = original_image[0]
            if original_image.shape[0] == 3:  # RGB
                img_np = original_image.permute(1, 2, 0).numpy()
                # Normalize to [0, 1] if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                axes[plot_idx].imshow(img_np)
            else:
                axes[plot_idx].imshow(original_image[0].numpy(), cmap='gray')
            axes[plot_idx].set_title('Original Image')
            axes[plot_idx].axis('off')
            plot_idx += 1
            
        # Show embedding components
        for i in range(n_components):
            if plot_idx < len(axes):
                im = axes[plot_idx].imshow(embeddings[i].numpy(), cmap=self.feature_cmap)
                axes[plot_idx].set_title(f'Embedding Dim {i}')
                axes[plot_idx].axis('off')
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
                plot_idx += 1
                
        # Turn off unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/image_embeddings.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()
        
        # Additional analysis: PCA visualization
        self._visualize_embedding_pca(embeddings, save_path)
        
    def _visualize_embedding_pca(self, embeddings: torch.Tensor, save_path: Optional[str] = None):
        """Visualize PCA components of embeddings."""
        C, H, W = embeddings.shape
        # Reshape to (H*W, C) for PCA
        embeddings_flat = embeddings.view(C, -1).transpose(0, 1).numpy()
        
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca_components = pca.fit_transform(embeddings_flat)
            pca_components = pca_components.reshape(H, W, 3)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=self.dpi)
            
            # RGB visualization of first 3 PCA components
            pca_rgb = (pca_components - pca_components.min()) / (pca_components.max() - pca_components.min())
            axes[0].imshow(pca_rgb)
            axes[0].set_title('PCA RGB (PC1=R, PC2=G, PC3=B)')
            axes[0].axis('off')
            
            # Individual PCA components
            for i in range(3):
                im = axes[i+1].imshow(pca_components[:, :, i], cmap=self.feature_cmap)
                axes[i+1].set_title(f'PCA Component {i+1}\n(Var: {pca.explained_variance_ratio_[i]:.3f})')
                axes[i+1].axis('off')
                plt.colorbar(im, ax=axes[i+1], fraction=0.046)
                
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/embedding_pca.png", 
                           bbox_inches='tight', dpi=self.dpi)
            plt.show()
            
        except ImportError:
            warnings.warn("scikit-learn not available, skipping PCA visualization")
            
    def visualize_mask_decoder_states(self, decoder_states: Dict[str, torch.Tensor],
                                    save_path: Optional[str] = None):
        """
        Visualize mask decoder internal states.
        
        Args:
            decoder_states: Dictionary containing decoder states
            save_path: Path to save the visualization
        """
        n_states = len(decoder_states)
        cols = min(3, n_states)
        rows = (n_states + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), dpi=self.dpi)
        if n_states == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        plot_idx = 0
        for state_name, state_data in decoder_states.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            if state_data.dim() == 4:  # (B, C, H, W)
                # Show mean across channels
                data_to_plot = state_data[0].mean(0).numpy()
            elif state_data.dim() == 3:  # (B, N, C)
                # Show as heatmap
                data_to_plot = state_data[0].numpy()
            elif state_data.dim() == 2:  # (N, C)
                data_to_plot = state_data.numpy()
            else:
                # For other dimensions, flatten and show as 1D plot
                ax.plot(state_data.flatten().numpy())
                ax.set_title(f'{state_name} (1D)')
                plot_idx += 1
                continue
                
            im = ax.imshow(data_to_plot, cmap=self.feature_cmap)
            ax.set_title(f'{state_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            plot_idx += 1
            
        # Turn off unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/mask_decoder_states.png", 
                       bbox_inches='tight', dpi=self.dpi)
        plt.show()


def visualize_debug_states(debug_states: Optional[Dict] = None, 
                          save_path: Optional[str] = None,
                          create_summary: bool = True):
    """
    Comprehensive visualization of all captured debug states.
    
    Args:
        debug_states: Debug states dictionary (if None, use global states)
        save_path: Directory to save visualizations
        create_summary: Whether to create a summary report
    """
    if debug_states is None:
        debug_states = get_debug_states()
        
    if not debug_states:
        print("No debug states captured. Enable debug mode first.")
        return
        
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    visualizer = SAM2Visualizer()
    
    # Visualize each component's states
    for component_name, component_states in debug_states.items():
        print(f"\nVisualizing {component_name}...")
        
        for state_name, state_info in component_states.items():
            data = state_info['data']
            metadata = state_info.get('metadata', {})
            
            try:
                if 'position_encoding' in state_name.lower():
                    encoding_type = metadata.get('encoding_type', 'unknown')
                    visualizer.visualize_position_encoding(
                        data, encoding_type=encoding_type, save_path=save_path
                    )
                elif 'attention' in state_name.lower():
                    visualizer.visualize_attention_patterns(
                        data, layer_name=f"{component_name}_{state_name}", save_path=save_path
                    )
                elif 'embedding' in state_name.lower() or 'features' in state_name.lower():
                    if data.dim() == 4:  # Image-like features
                        visualizer.visualize_image_embeddings(
                            data, save_path=save_path
                        )
                elif 'decoder' in component_name.lower():
                    # Collect all decoder states for combined visualization
                    decoder_states = {name: info['data'] for name, info in component_states.items()}
                    visualizer.visualize_mask_decoder_states(decoder_states, save_path=save_path)
                    break  # Don't visualize individual decoder states again
                    
            except Exception as e:
                print(f"Failed to visualize {component_name}.{state_name}: {e}")
                
    if create_summary and save_path:
        _create_debug_summary(debug_states, save_path)
        
        
def _create_debug_summary(debug_states: Dict, save_path: str):
    """Create a text summary of captured debug states."""
    summary_path = os.path.join(save_path, "debug_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("SAM2 Debug States Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for component_name, component_states in debug_states.items():
            f.write(f"Component: {component_name}\n")
            f.write("-" * 30 + "\n")
            
            for state_name, state_info in component_states.items():
                data = state_info['data']
                metadata = state_info.get('metadata', {})
                
                f.write(f"  State: {state_name}\n")
                f.write(f"    Shape: {state_info['shape']}\n")
                f.write(f"    Dtype: {state_info['dtype']}\n")
                
                if isinstance(data, torch.Tensor):
                    f.write(f"    Min: {data.min().item():.6f}\n")
                    f.write(f"    Max: {data.max().item():.6f}\n")
                    f.write(f"    Mean: {data.mean().item():.6f}\n")
                    f.write(f"    Std: {data.std().item():.6f}\n")
                    
                if metadata:
                    f.write(f"    Metadata: {metadata}\n")
                f.write("\n")
            f.write("\n")
            
    print(f"Debug summary saved to: {summary_path}")
```

```python
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Generate 2D position embeddings for dense image feature maps.
        
        This is the main forward method for encoding spatial positions of image
        patches or feature map locations. It creates a dense grid of position
        embeddings that can be added to or concatenated with image features.
        
        The method uses efficient caching to avoid recomputing embeddings for
        the same spatial dimensions, which is common in video processing or
        when processing multiple images of the same size.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (batch, channels, height, width)
                             The spatial dimensions (H, W) determine the position grid size.
                             
        Returns:
            torch.Tensor: Position embeddings with shape (batch, embedding_dim, height, width)
                         Can be directly added to input features or used in attention.
                         
        Optimization Features:
        - **Caching**: Computed embeddings cached by spatial dimensions
        - **Vectorization**: Efficient batch processing of coordinate grids
        - **Memory Reuse**: Same embeddings repeated across batch dimension
        
        Usage Examples:
            # Add to image features
            image_features = backbone(image)  # (B, C, H, W)
            pos_embed = position_encoder(image_features)  # (B, embed_dim, H, W)
            enhanced_features = image_features + pos_embed
            
            # Use in attention
            pos_embed = position_encoder(features).flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        """
        # Use spatial dimensions as cache key for efficient reuse
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            # Reuse cached computation and repeat for batch dimension
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
            
        # Create dense coordinate grids for the spatial dimensions
        # Y coordinates: increase downward (standard image convention)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        # X coordinates: increase rightward
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        # Normalize coordinates to [0, 1] range for scale invariance
        if self.normalize:
            eps = 1e-6  # Small epsilon to prevent division by zero
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Generate frequency dimensions for sinusoidal encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sinusoidal encoding to every spatial position
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Create alternating sin/cos pattern for different frequency components
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        # Concatenate y and x embeddings, then permute to standard format
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Cache result for future use with same spatial dimensions
        self.cache[cache_key] = pos[0]
        return pos

->

    @torch.no_grad()
    def forward(self, x: torch.Tensor, debug_name: str = None):
        """
        Generate 2D position embeddings for dense image feature maps.
        
        This is the main forward method for encoding spatial positions of image
        patches or feature map locations. It creates a dense grid of position
        embeddings that can be added to or concatenated with image features.
        
        The method uses efficient caching to avoid recomputing embeddings for
        the same spatial dimensions, which is common in video processing or
        when processing multiple images of the same size.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (batch, channels, height, width)
                             The spatial dimensions (H, W) determine the position grid size.
            debug_name (str, optional): Name for debug state capture. If provided and debug mode
                                      is enabled, captures intermediate states for visualization.
                             
        Returns:
            torch.Tensor: Position embeddings with shape (batch, embedding_dim, height, width)
                         Can be directly added to input features or used in attention.
                         
        Optimization Features:
        - **Caching**: Computed embeddings cached by spatial dimensions
        - **Vectorization**: Efficient batch processing of coordinate grids
        - **Memory Reuse**: Same embeddings repeated across batch dimension
        
        Usage Examples:
            # Add to image features
            image_features = backbone(image)  # (B, C, H, W)
            pos_embed = position_encoder(image_features)  # (B, embed_dim, H, W)
            enhanced_features = image_features + pos_embed
            
            # Use in attention
            pos_embed = position_encoder(features).flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Use spatial dimensions as cache key for efficient reuse
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            # Reuse cached computation and repeat for batch dimension
            pos = self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
            
            # Debug capture for cached result
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "position_encoding_sine",
                    state_name="position_embeddings_cached",
                    data=pos,
                    metadata={
                        'encoding_type': 'sine',
                        'cached': True,
                        'spatial_dims': cache_key,
                        'normalize': self.normalize,
                        'scale': self.scale,
                        'temperature': self.temperature
                    }
                )
            return pos
            
        # Create dense coordinate grids for the spatial dimensions
        # Y coordinates: increase downward (standard image convention)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        # X coordinates: increase rightward
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        # Debug capture for raw coordinate grids
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="coordinate_grids_x",
                data=x_embed,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'x_raw'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="coordinate_grids_y",
                data=y_embed,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'y_raw'}
            )

        # Normalize coordinates to [0, 1] range for scale invariance
        if self.normalize:
            eps = 1e-6  # Small epsilon to prevent division by zero
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            # Debug capture for normalized coordinates
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "position_encoding_sine",
                    state_name="coordinate_grids_x_normalized",
                    data=x_embed,
                    metadata={'encoding_type': 'sine', 'coordinate_type': 'x_normalized', 'scale': self.scale}
                )
                capture_debug_state(
                    component_name=debug_name or "position_encoding_sine",
                    state_name="coordinate_grids_y_normalized",
                    data=y_embed,
                    metadata={'encoding_type': 'sine', 'coordinate_type': 'y_normalized', 'scale': self.scale}
                )

        # Generate frequency dimensions for sinusoidal encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Debug capture for frequency dimensions
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="frequency_dimensions",
                data=dim_t,
                metadata={'encoding_type': 'sine', 'temperature': self.temperature, 'num_pos_feats': self.num_pos_feats}
            )

        # Apply sinusoidal encoding to every spatial position
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Create alternating sin/cos pattern for different frequency components
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        # Debug capture for intermediate sin/cos encodings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="sinusoidal_encoding_x",
                data=pos_x,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'x_encoded'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="sinusoidal_encoding_y",
                data=pos_y,
                metadata={'encoding_type': 'sine', 'coordinate_type': 'y_encoded'}
            )
        
        # Concatenate y and x embeddings, then permute to standard format
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Debug capture for final position embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_sine",
                state_name="position_embeddings_final",
                data=pos,
                metadata={
                    'encoding_type': 'sine',
                    'cached': False,
                    'spatial_dims': cache_key,
                    'normalize': self.normalize,
                    'scale': self.scale,
                    'temperature': self.temperature,
                    'num_pos_feats': self.num_pos_feats
                }
            )
        
        # Cache result for future use with same spatial dimensions
        self.cache[cache_key] = pos[0]
        return pos
```

```python
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply random Fourier feature encoding to normalized coordinates.
        
        This method implements the core RFF transformation: projecting coordinates
        through a random linear transformation followed by sinusoidal activation.
        The result is a high-dimensional representation that can capture complex
        spatial patterns while maintaining smoothness properties.
        
        Args:
            coords (torch.Tensor): Normalized coordinates in [0,1]² with shape (..., 2)
                                 Last dimension contains (x, y) coordinate pairs.
                                 
        Returns:
            torch.Tensor: Encoded features with shape (..., 2*num_pos_feats)
                         Concatenation of sine and cosine projections.
                         
        Mathematical Steps:
        1. Normalize [0,1] → [-1,1] for better numerical properties
        2. Linear projection: coords @ random_matrix  
        3. Scale by 2π for full sinusoidal period coverage
        4. Apply sin and cos to create final embedding
        
        Properties:
        - Smooth: Small coordinate changes → small embedding changes
        - Bounded: All values in [-1, 1] range
        - Rich: Can represent complex spatial functions
        - Efficient: Single matrix multiplication + elementwise operations
        """
        # Normalize coordinates from [0,1] to [-1,1] for symmetric representation
        coords = 2 * coords - 1
        
        # Apply random linear projection to map 2D coords to higher-dimensional space
        coords = coords @ self.positional_encoding_gaussian_matrix
        
        # Scale by 2π for full period coverage of sinusoidal functions
        coords = 2 * np.pi * coords
        
        # Apply sinusoidal activation and concatenate sin/cos components
        # This creates a rich, smooth representation of spatial position
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

->

    def _pe_encoding(self, coords: torch.Tensor, debug_name: str = None) -> torch.Tensor:
        """
        Apply random Fourier feature encoding to normalized coordinates.
        
        This method implements the core RFF transformation: projecting coordinates
        through a random linear transformation followed by sinusoidal activation.
        The result is a high-dimensional representation that can capture complex
        spatial patterns while maintaining smoothness properties.
        
        Args:
            coords (torch.Tensor): Normalized coordinates in [0,1]² with shape (..., 2)
                                 Last dimension contains (x, y) coordinate pairs.
            debug_name (str, optional): Name for debug state capture.
                                 
        Returns:
            torch.Tensor: Encoded features with shape (..., 2*num_pos_feats)
                         Concatenation of sine and cosine projections.
                         
        Mathematical Steps:
        1. Normalize [0,1] → [-1,1] for better numerical properties
        2. Linear projection: coords @ random_matrix  
        3. Scale by 2π for full sinusoidal period coverage
        4. Apply sin and cos to create final embedding
        
        Properties:
        - Smooth: Small coordinate changes → small embedding changes
        - Bounded: All values in [-1, 1] range
        - Rich: Can represent complex spatial functions
        - Efficient: Single matrix multiplication + elementwise operations
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="input_coordinates",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_range': '[0,1]'}
            )
        
        # Normalize coordinates from [0,1] to [-1,1] for symmetric representation
        coords = 2 * coords - 1
        
        # Debug capture for normalized coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="normalized_coordinates",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_range': '[-1,1]'}
            )
        
        # Apply random linear projection to map 2D coords to higher-dimensional space
        coords = coords @ self.positional_encoding_gaussian_matrix
        
        # Debug capture for projected coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="projected_coordinates",
                data=coords,
                metadata={'encoding_type': 'random', 'projection_shape': str(self.positional_encoding_gaussian_matrix.shape)}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="gaussian_projection_matrix",
                data=self.positional_encoding_gaussian_matrix,
                metadata={'encoding_type': 'random', 'matrix_type': 'gaussian_random'}
            )
        
        # Scale by 2π for full period coverage of sinusoidal functions
        coords = 2 * np.pi * coords
        
        # Apply sinusoidal activation and concatenate sin/cos components
        sin_coords = torch.sin(coords)
        cos_coords = torch.cos(coords)
        
        # Debug capture for sin/cos components
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sin_components",
                data=sin_coords,
                metadata={'encoding_type': 'random', 'activation': 'sine'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="cos_components",
                data=cos_coords,
                metadata={'encoding_type': 'random', 'activation': 'cosine'}
            )
        
        # This creates a rich, smooth representation of spatial position
        result = torch.cat([sin_coords, cos_coords], dim=-1)
        
        # Debug capture for final encoding
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="final_encoding",
                data=result,
                metadata={'encoding_type': 'random', 'final_dim': result.shape[-1]}
            )
        
        return result
```

```python
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate random Fourier position encoding for a 2D spatial grid.
        
        This method creates position embeddings for every location in a dense
        spatial grid, such as image feature maps or attention grids. The encoding
        provides adaptive spatial awareness that can be learned during training.
        
        Args:
            size (Tuple[int, int]): Spatial grid dimensions (height, width)
            
        Returns:
            torch.Tensor: Position encoding with shape (2*num_pos_feats, height, width)
                         Each spatial location has a unique embedding vector.
                         
        Usage Examples:
            # For transformer attention
            pos_embed = encoder.forward((64, 64))  # 64x64 grid
            pos_embed = pos_embed.flatten(1, 2).transpose(0, 1)  # (4096, embed_dim)
            
            # For CNN feature augmentation  
            pos_embed = encoder.forward((32, 32))  # Match feature map size
            features = features + pos_embed  # Element-wise addition
        """
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        
        # Create coordinate grid with pixel-centered positions
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # Cumulative sum creates incremental coordinates [1,2,3,...]
        # Subtract 0.5 to center coordinates in pixel centers
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        
        # Normalize coordinates to [0,1] range for scale invariance
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Apply random Fourier feature encoding to coordinate pairs
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        
        # Permute to channel-first format: (features, height, width)
        return pe.permute(2, 0, 1)

->

    def forward(self, size: Tuple[int, int], debug_name: str = None) -> torch.Tensor:
        """
        Generate random Fourier position encoding for a 2D spatial grid.
        
        This method creates position embeddings for every location in a dense
        spatial grid, such as image feature maps or attention grids. The encoding
        provides adaptive spatial awareness that can be learned during training.
        
        Args:
            size (Tuple[int, int]): Spatial grid dimensions (height, width)
            debug_name (str, optional): Name for debug state capture.
            
        Returns:
            torch.Tensor: Position encoding with shape (2*num_pos_feats, height, width)
                         Each spatial location has a unique embedding vector.
                         
        Usage Examples:
            # For transformer attention
            pos_embed = encoder.forward((64, 64))  # 64x64 grid
            pos_embed = pos_embed.flatten(1, 2).transpose(0, 1)  # (4096, embed_dim)
            
            # For CNN feature augmentation  
            pos_embed = encoder.forward((32, 32))  # Match feature map size
            features = features + pos_embed  # Element-wise addition
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        
        # Create coordinate grid with pixel-centered positions
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # Cumulative sum creates incremental coordinates [1,2,3,...]
        # Subtract 0.5 to center coordinates in pixel centers
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        
        # Debug capture for raw coordinate grids
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_x_raw",
                data=x_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'x_raw'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_y_raw",
                data=y_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'y_raw'}
            )
        
        # Normalize coordinates to [0,1] range for scale invariance
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Debug capture for normalized coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_x_normalized",
                data=x_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'x_normalized'}
            )
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="coordinate_grid_y_normalized",
                data=y_embed,
                metadata={'encoding_type': 'random', 'grid_size': size, 'coordinate_type': 'y_normalized'}
            )

        # Apply random Fourier feature encoding to coordinate pairs
        coords_stack = torch.stack([x_embed, y_embed], dim=-1)
        pe = self._pe_encoding(coords_stack, debug_name=debug_name)
        
        # Permute to channel-first format: (features, height, width)
        result = pe.permute(2, 0, 1)
        
        # Debug capture for final result
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="position_embeddings_grid",
                data=result,
                metadata={'encoding_type': 'random', 'grid_size': size, 'output_shape': result.shape}
            )
        
        return result
```

```python
    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Encode arbitrary 2D coordinates with random Fourier features.
        
        This method handles sparse coordinate sets that don't necessarily lie on
        a regular grid. It's particularly useful for encoding user interaction
        points, object centers, or other sparse spatial annotations.
        
        Args:
            coords_input (torch.Tensor): Coordinate tensor with shape (batch, num_points, 2)
                                        Last dimension contains (x, y) coordinate pairs.
                                        Coordinates should be in pixel space.
                                        
            image_size (Tuple[int, int]): Reference image dimensions (height, width)
                                        Used for coordinate normalization.
                                        
        Returns:
            torch.Tensor: Position encodings with shape (batch, num_points, 2*num_pos_feats)
                         Each coordinate pair gets a unique embedding vector.
                         
        Applications:
        - Interactive segmentation: Encode user click coordinates
        - Object detection: Encode bounding box centers  
        - Keypoint detection: Encode landmark positions
        - Sparse supervision: Encode labeled point annotations
        
        Example:
            # Encode user clicks for interactive segmentation
            clicks = torch.tensor([[[100, 150], [200, 300]]])  # (1, 2, 2)
            embeddings = encoder.forward_with_coords(clicks, (512, 512))  # (1, 2, 128)
        """
        # Clone input to avoid modifying original coordinates
        coords = coords_input.clone()
        
        # Normalize coordinates to [0,1] range based on image dimensions
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # x coordinate (width)
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # y coordinate (height)
        
        # Apply random Fourier feature encoding
        return self._pe_encoding(coords.to(torch.float))

->

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int], debug_name: str = None
    ) -> torch.Tensor:
        """
        Encode arbitrary 2D coordinates with random Fourier features.
        
        This method handles sparse coordinate sets that don't necessarily lie on
        a regular grid. It's particularly useful for encoding user interaction
        points, object centers, or other sparse spatial annotations.
        
        Args:
            coords_input (torch.Tensor): Coordinate tensor with shape (batch, num_points, 2)
                                        Last dimension contains (x, y) coordinate pairs.
                                        Coordinates should be in pixel space.
                                        
            image_size (Tuple[int, int]): Reference image dimensions (height, width)
                                        Used for coordinate normalization.
            
            debug_name (str, optional): Name for debug state capture.
                                        
        Returns:
            torch.Tensor: Position encodings with shape (batch, num_points, 2*num_pos_feats)
                         Each coordinate pair gets a unique embedding vector.
                         
        Applications:
        - Interactive segmentation: Encode user click coordinates
        - Object detection: Encode bounding box centers  
        - Keypoint detection: Encode landmark positions
        - Sparse supervision: Encode labeled point annotations
        
        Example:
            # Encode user clicks for interactive segmentation
            clicks = torch.tensor([[[100, 150], [200, 300]]])  # (1, 2, 2)
            embeddings = encoder.forward_with_coords(clicks, (512, 512))  # (1, 2, 128)
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Clone input to avoid modifying original coordinates
        coords = coords_input.clone()
        
        # Debug capture for input coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sparse_coordinates_input",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_space': 'pixel', 'image_size': image_size}
            )
        
        # Normalize coordinates to [0,1] range based on image dimensions
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # x coordinate (width)
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # y coordinate (height)
        
        # Debug capture for normalized coordinates
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sparse_coordinates_normalized",
                data=coords,
                metadata={'encoding_type': 'random', 'coordinate_space': 'normalized', 'image_size': image_size}
            )
        
        # Apply random Fourier feature encoding
        result = self._pe_encoding(coords.to(torch.float), debug_name=debug_name)
        
        # Debug capture for final result
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "position_encoding_random",
                state_name="sparse_coordinates_encoded",
                data=result,
                metadata={'encoding_type': 'random', 'output_shape': result.shape, 'num_points': coords.shape[1]}
            )
        
        return result
```

```python
    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output

->

    def forward(self, sample: torch.Tensor, debug_name: str = None):
        # Debug capture for input image
        if debug_name:
            from sam2.debug_utils import capture_debug_state, is_debug_enabled
            if is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "image_encoder",
                    state_name="input_image",
                    data=sample,
                    metadata={'component_type': 'image_encoder', 'stage': 'input'}
                )

        # Forward through backbone
        trunk_features = self.trunk(sample)
        
        # Debug capture for trunk features
        if debug_name:
            from sam2.debug_utils import capture_debug_state, is_debug_enabled
            if is_debug_enabled():
                if isinstance(trunk_features, (list, tuple)):
                    for i, feat in enumerate(trunk_features):
                        capture_debug_state(
                            component_name=debug_name or "image_encoder",
                            state_name=f"trunk_features_level_{i}",
                            data=feat,
                            metadata={'component_type': 'image_encoder', 'stage': 'trunk_output', 'level': i}
                        )
                else:
                    capture_debug_state(
                        component_name=debug_name or "image_encoder",
                        state_name="trunk_features",
                        data=trunk_features,
                        metadata={'component_type': 'image_encoder', 'stage': 'trunk_output'}
                    )
        
        features, pos = self.neck(trunk_features, debug_name=debug_name)
        
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        
        # Debug capture for final output features
        if debug_name:
            from sam2.debug_utils import capture_debug_state, is_debug_enabled
            if is_debug_enabled():
                # Capture final vision features
                capture_debug_state(
                    component_name=debug_name or "image_encoder",
                    state_name="vision_features_final",
                    data=src,
                    metadata={'component_type': 'image_encoder', 'stage': 'final_output'}
                )
                
                # Capture FPN features at all levels
                for i, feat in enumerate(features):
                    capture_debug_state(
                        component_name=debug_name or "image_encoder",
                        state_name=f"fpn_features_level_{i}",
                        data=feat,
                        metadata={'component_type': 'image_encoder', 'stage': 'fpn_output', 'level': i}
                    )
                
                # Capture position encodings
                for i, p in enumerate(pos):
                    capture_debug_state(
                        component_name=debug_name or "image_encoder",
                        state_name=f"position_encoding_level_{i}",
                        data=p,
                        metadata={'component_type': 'image_encoder', 'stage': 'position_encoding', 'level': i}
                    )

        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output
```

```python
    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos

->

    def forward(self, xs: List[torch.Tensor], debug_name: str = None):
        from sam2.debug_utils import capture_debug_state, is_debug_enabled

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        
        # Debug capture for input features
        if debug_name and is_debug_enabled():
            for i, x in enumerate(xs):
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"input_features_level_{i}",
                    data=x,
                    metadata={'component_type': 'fpn_neck', 'stage': 'input', 'level': i}
                )
        
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            
            # Debug capture for lateral features
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"lateral_features_level_{i}",
                    data=lateral_features,
                    metadata={'component_type': 'fpn_neck', 'stage': 'lateral', 'level': i}
                )
            
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                
                # Debug capture for top-down features
                if debug_name and is_debug_enabled():
                    capture_debug_state(
                        component_name=debug_name or "fpn_neck",
                        state_name=f"top_down_features_level_{i}",
                        data=top_down_features,
                        metadata={'component_type': 'fpn_neck', 'stage': 'top_down', 'level': i}
                    )
                
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            
            x_out = prev_features
            out[i] = x_out
            
            # Generate position encoding with debug support
            if hasattr(self.position_encoding, 'forward') and 'debug_name' in self.position_encoding.forward.__code__.co_varnames:
                pos[i] = self.position_encoding(x_out, debug_name=f"{debug_name or 'fpn_neck'}_pos_enc_level_{i}").to(x_out.dtype)
            else:
                pos[i] = self.position_encoding(x_out).to(x_out.dtype)
            
            # Debug capture for output features and position encodings
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"output_features_level_{i}",
                    data=x_out,
                    metadata={'component_type': 'fpn_neck', 'stage': 'output', 'level': i}
                )
                capture_debug_state(
                    component_name=debug_name or "fpn_neck",
                    state_name=f"position_encoding_level_{i}",
                    data=pos[i],
                    metadata={'component_type': 'fpn_neck', 'stage': 'position_encoding', 'level': i}
                )

        return out, pos
```

```python
    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        x = xs[0]
        x_out = self.convs[0](x)
        out[0] = x_out
        pos[0] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos

->

    def forward(self, xs: List[torch.Tensor], debug_name: str = None):
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)

        x = xs[0]
        
        # Debug capture for input
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "vitdet_neck",
                state_name="input_features",
                data=x,
                metadata={'component_type': 'vitdet_neck', 'stage': 'input'}
            )
        
        x_out = self.convs[0](x)
        out[0] = x_out
        
        # Generate position encoding with debug support
        if hasattr(self.position_encoding, 'forward') and 'debug_name' in self.position_encoding.forward.__code__.co_varnames:
            pos[0] = self.position_encoding(x_out, debug_name=f"{debug_name or 'vitdet_neck'}_pos_enc").to(x_out.dtype)
        else:
            pos[0] = self.position_encoding(x_out).to(x_out.dtype)
        
        # Debug capture for output
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "vitdet_neck",
                state_name="output_features",
                data=x_out,
                metadata={'component_type': 'vitdet_neck', 'stage': 'output'}
            )
            capture_debug_state(
                component_name=debug_name or "vitdet_neck",
                state_name="position_encoding",
                data=pos[0],
                metadata={'component_type': 'vitdet_neck', 'stage': 'position_encoding'}
            )

        return out, pos
```

```python
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Compute multi-head attention with optional hardware optimization.
        
        Args:
            q: Query tensor [B, N_q, C]
            k: Key tensor [B, N_k, C_kv]  
            v: Value tensor [B, N_v, C_kv]
            
        Returns:
            Attention output [B, N_q, C]
            
        Implementation Details:
        - Projects inputs to internal dimension for efficiency
        - Separates into multiple attention heads
        - Uses optimized attention kernels when available
        - Provides graceful fallback for compatibility
        - Applies dropout during training only
        """
        # Project inputs to internal dimension
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into multiple attention heads for parallel processing
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply dropout only during training
        dropout_p = self.dropout_p if self.training else 0.0
        
        # Compute attention with hardware optimization
        try:
            # Use optimized attention kernels when available (Flash Attention)
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Graceful fallback if optimized kernels fail
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            # Enable all available kernels for compatibility
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Recombine attention heads and project to output dimension
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

->

    def forward(self, q: Tensor, k: Tensor, v: Tensor, debug_name: str = None) -> Tensor:
        """
        Compute multi-head attention with optional hardware optimization.
        
        Args:
            q: Query tensor [B, N_q, C]
            k: Key tensor [B, N_k, C_kv]  
            v: Value tensor [B, N_v, C_kv]
            debug_name: Optional name for debug state capture
            
        Returns:
            Attention output [B, N_q, C]
            
        Implementation Details:
        - Projects inputs to internal dimension for efficiency
        - Separates into multiple attention heads
        - Uses optimized attention kernels when available
        - Provides graceful fallback for compatibility
        - Applies dropout during training only
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="input_queries",
                data=q,
                metadata={'component_type': 'attention', 'stage': 'input', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="input_keys",
                data=k,
                metadata={'component_type': 'attention', 'stage': 'input', 'tensor_type': 'keys'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="input_values",
                data=v,
                metadata={'component_type': 'attention', 'stage': 'input', 'tensor_type': 'values'}
            )
        
        # Project inputs to internal dimension
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Debug capture for projected tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="projected_queries",
                data=q,
                metadata={'component_type': 'attention', 'stage': 'projection', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="projected_keys",
                data=k,
                metadata={'component_type': 'attention', 'stage': 'projection', 'tensor_type': 'keys'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="projected_values",
                data=v,
                metadata={'component_type': 'attention', 'stage': 'projection', 'tensor_type': 'values'}
            )

        # Separate into multiple attention heads for parallel processing
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Debug capture for multi-head tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="multihead_queries",
                data=q,
                metadata={'component_type': 'attention', 'stage': 'multihead', 'tensor_type': 'queries', 'num_heads': self.num_heads}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="multihead_keys",
                data=k,
                metadata={'component_type': 'attention', 'stage': 'multihead', 'tensor_type': 'keys', 'num_heads': self.num_heads}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="multihead_values",
                data=v,
                metadata={'component_type': 'attention', 'stage': 'multihead', 'tensor_type': 'values', 'num_heads': self.num_heads}
            )

        # Apply dropout only during training
        dropout_p = self.dropout_p if self.training else 0.0
        
        # Compute attention weights manually for debug capture if needed
        if debug_name and is_debug_enabled():
            # Compute attention scores manually for debugging
            scale = (q.shape[-1]) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Capture attention weights
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="attention_scores",
                data=attn_scores,
                metadata={'component_type': 'attention', 'stage': 'attention_computation', 'tensor_type': 'scores'}
            )
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="attention_weights",
                data=attn_weights,
                metadata={'component_type': 'attention', 'stage': 'attention_computation', 'tensor_type': 'weights', 'num_heads': self.num_heads}
            )
        
        # Compute attention with hardware optimization
        try:
            # Use optimized attention kernels when available (Flash Attention)
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Graceful fallback if optimized kernels fail
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            # Enable all available kernels for compatibility
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Debug capture for attention output
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="attention_output_multihead",
                data=out,
                metadata={'component_type': 'attention', 'stage': 'attention_output', 'tensor_type': 'multihead_output'}
            )

        # Recombine attention heads and project to output dimension
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        # Debug capture for final output
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "attention",
                state_name="final_output",
                data=out,
                metadata={'component_type': 'attention', 'stage': 'final_output', 'tensor_type': 'output'}
            )

        return out
```

```python
    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process image and prompt embeddings through two-way transformer blocks.
        
        This method performs the core computation of the mask decoder, enabling
        sophisticated interaction between user prompts and image features through
        bidirectional attention mechanisms.

        Args:
            image_embedding (torch.Tensor): Dense image features from the image encoder
                with shape [B, embedding_dim, H, W]. These represent visual content
                at spatial locations in the image.
            image_pe (torch.Tensor): Positional encoding for image features with the
                same shape as image_embedding. Provides spatial awareness to the model.
            point_embedding (torch.Tensor): Sparse prompt embeddings (points, boxes,
                mask tokens) with shape [B, N_prompts, embedding_dim]. These encode
                user intentions and guidance for segmentation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - processed_prompts: Refined prompt embeddings with shape 
                  [B, N_prompts, embedding_dim] containing image-aware prompt representations
                - processed_image: Updated image embeddings with shape 
                  [B, H*W, embedding_dim] containing prompt-aware visual features
                  
        Processing Flow:
        1. Reshape image embeddings from spatial to sequence format
        2. Apply stacked two-way attention blocks for iterative refinement
        3. Perform final prompt-to-image attention for comprehensive alignment
        4. Return refined embeddings for mask generation
        
        The bidirectional attention allows prompts to gather relevant visual context
        while image features are modulated by user intent, resulting in highly
        accurate and contextually appropriate segmentation masks.
        """
        # Convert image embeddings from spatial to sequence format
        # Transform from [B, C, H, W] to [B, H*W, C] for transformer processing
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Initialize queries and keys for attention operations
        # Queries: prompt embeddings that will attend to image features
        # Keys: image embeddings that will be attended to by prompts
        queries = point_embedding  # Prompt tokens seeking visual information
        keys = image_embedding     # Image tokens providing visual context

        # Apply transformer blocks with bidirectional attention
        # Each block refines both prompt and image representations
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # Original prompt positions (unchanged)
                key_pe=image_pe,          # Image spatial positions
            )

        # Final attention layer: comprehensive prompt-to-image alignment
        # This ensures prompt tokens have fully integrated visual context
        q = queries + point_embedding  # Add positional info to refined queries
        k = keys + image_pe           # Add positional info to refined keys
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        
        # Apply residual connection and normalization
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

->

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        debug_name: str = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process image and prompt embeddings through two-way transformer blocks.
        
        This method performs the core computation of the mask decoder, enabling
        sophisticated interaction between user prompts and image features through
        bidirectional attention mechanisms.

        Args:
            image_embedding (torch.Tensor): Dense image features from the image encoder
                with shape [B, embedding_dim, H, W]. These represent visual content
                at spatial locations in the image.
            image_pe (torch.Tensor): Positional encoding for image features with the
                same shape as image_embedding. Provides spatial awareness to the model.
            point_embedding (torch.Tensor): Sparse prompt embeddings (points, boxes,
                mask tokens) with shape [B, N_prompts, embedding_dim]. These encode
                user intentions and guidance for segmentation.
            debug_name (str, optional): Name for debug state capture.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - processed_prompts: Refined prompt embeddings with shape 
                  [B, N_prompts, embedding_dim] containing image-aware prompt representations
                - processed_image: Updated image embeddings with shape 
                  [B, H*W, embedding_dim] containing prompt-aware visual features
                  
        Processing Flow:
        1. Reshape image embeddings from spatial to sequence format
        2. Apply stacked two-way attention blocks for iterative refinement
        3. Perform final prompt-to-image attention for comprehensive alignment
        4. Return refined embeddings for mask generation
        
        The bidirectional attention allows prompts to gather relevant visual context
        while image features are modulated by user intent, resulting in highly
        accurate and contextually appropriate segmentation masks.
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="input_image_embedding",
                data=image_embedding,
                metadata={'component_type': 'two_way_transformer', 'stage': 'input', 'tensor_type': 'image_embedding'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="input_image_pe",
                data=image_pe,
                metadata={'component_type': 'two_way_transformer', 'stage': 'input', 'tensor_type': 'image_pe'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="input_point_embedding",
                data=point_embedding,
                metadata={'component_type': 'two_way_transformer', 'stage': 'input', 'tensor_type': 'point_embedding'}
            )
        
        # Convert image embeddings from spatial to sequence format
        # Transform from [B, C, H, W] to [B, H*W, C] for transformer processing
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Debug capture for reshaped embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="reshaped_image_embedding",
                data=image_embedding,
                metadata={'component_type': 'two_way_transformer', 'stage': 'reshape', 'tensor_type': 'image_embedding'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="reshaped_image_pe",
                data=image_pe,
                metadata={'component_type': 'two_way_transformer', 'stage': 'reshape', 'tensor_type': 'image_pe'}
            )

        # Initialize queries and keys for attention operations
        # Queries: prompt embeddings that will attend to image features
        # Keys: image embeddings that will be attended to by prompts
        queries = point_embedding  # Prompt tokens seeking visual information
        keys = image_embedding     # Image tokens providing visual context

        # Apply transformer blocks with bidirectional attention
        # Each block refines both prompt and image representations
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # Original prompt positions (unchanged)
                key_pe=image_pe,          # Image spatial positions
                debug_name=f"{debug_name or 'two_way_transformer'}_layer_{i}" if debug_name else None,
            )
            
            # Debug capture for layer outputs
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "two_way_transformer",
                    state_name=f"layer_{i}_queries_output",
                    data=queries,
                    metadata={'component_type': 'two_way_transformer', 'stage': f'layer_{i}_output', 'tensor_type': 'queries'}
                )
                capture_debug_state(
                    component_name=debug_name or "two_way_transformer",
                    state_name=f"layer_{i}_keys_output",
                    data=keys,
                    metadata={'component_type': 'two_way_transformer', 'stage': f'layer_{i}_output', 'tensor_type': 'keys'}
                )

        # Final attention layer: comprehensive prompt-to-image alignment
        # This ensures prompt tokens have fully integrated visual context
        q = queries + point_embedding  # Add positional info to refined queries
        k = keys + image_pe           # Add positional info to refined keys
        
        # Debug capture for final attention inputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_attention_queries",
                data=q,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_attention_input', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_attention_keys",
                data=k,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_attention_input', 'tensor_type': 'keys'}
            )
        
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys, debug_name=f"{debug_name or 'two_way_transformer'}_final_attn" if debug_name else None)
        
        # Apply residual connection and normalization
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        # Debug capture for final outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_output', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_transformer",
                state_name="final_keys_output",
                data=keys,
                metadata={'component_type': 'two_way_transformer', 'stage': 'final_output', 'tensor_type': 'keys'}
            )

        return queries, keys
```

```python
    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply four-stage bidirectional attention between prompts and image features.
        
        This method implements the core two-way attention computation that enables
        sophisticated interaction between user prompts and visual features.
        
        Args:
            queries (Tensor): Prompt token embeddings [B, N_prompts, C]
            keys (Tensor): Image token embeddings [B, N_image, C] 
            query_pe (Tensor): Positional encoding for prompts [B, N_prompts, C]
            key_pe (Tensor): Positional encoding for image [B, N_image, C]
            
        Returns:
            Tuple[Tensor, Tensor]: Updated (queries, keys) after bidirectional attention
            
        Four-Stage Process:
        1. Self-attention: Prompts attend to each other for coordination
        2. Cross-attention: Prompts gather visual context from image
        3. MLP: Non-linear transformation of prompt features
        4. Reverse cross-attention: Image features incorporate prompt context
        """
        
        # Stage 1: Self-attention among prompt tokens
        # This allows different prompts to coordinate and share information
        # Skip PE in first layer to avoid double-adding positional information
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            # Add positional encoding to queries for spatial awareness
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out  # Residual connection
        queries = self.norm1(queries)

        # Stage 2: Cross-attention from prompts to image features
        # Prompts attend to image to gather relevant visual information
        q = queries + query_pe  # Prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out  # Residual connection
        queries = self.norm2(queries)

        # Stage 3: MLP processing of prompt features
        # Non-linear transformation to increase representational capacity
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out  # Residual connection
        queries = self.norm3(queries)

        # Stage 4: Cross-attention from image features to prompt tokens
        # Image features attend to prompts to incorporate user intent
        # Note: q and k are swapped to reverse the attention direction
        q = queries + query_pe  # Updated prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out  # Update image features with prompt context
        keys = self.norm4(keys)

        return queries, keys

->

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, debug_name: str = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply four-stage bidirectional attention between prompts and image features.
        
        This method implements the core two-way attention computation that enables
        sophisticated interaction between user prompts and visual features.
        
        Args:
            queries (Tensor): Prompt token embeddings [B, N_prompts, C]
            keys (Tensor): Image token embeddings [B, N_image, C] 
            query_pe (Tensor): Positional encoding for prompts [B, N_prompts, C]
            key_pe (Tensor): Positional encoding for image [B, N_image, C]
            debug_name (str, optional): Name for debug state capture.
            
        Returns:
            Tuple[Tensor, Tensor]: Updated (queries, keys) after bidirectional attention
            
        Four-Stage Process:
        1. Self-attention: Prompts attend to each other for coordination
        2. Cross-attention: Prompts gather visual context from image
        3. MLP: Non-linear transformation of prompt features
        4. Reverse cross-attention: Image features incorporate prompt context
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input tensors
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_queries",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_keys",
                data=keys,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'keys'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_query_pe",
                data=query_pe,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'query_pe'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="input_key_pe",
                data=key_pe,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'input', 'tensor_type': 'key_pe'}
            )
        
        # Stage 1: Self-attention among prompt tokens
        # This allows different prompts to coordinate and share information
        # Skip PE in first layer to avoid double-adding positional information
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries, debug_name=f"{debug_name or 'two_way_attention_block'}_self_attn" if debug_name else None)
        else:
            # Add positional encoding to queries for spatial awareness
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries, debug_name=f"{debug_name or 'two_way_attention_block'}_self_attn" if debug_name else None)
            queries = queries + attn_out  # Residual connection
        queries = self.norm1(queries)

        # Debug capture after stage 1
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage1_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage1_output', 'tensor_type': 'queries'}
            )

        # Stage 2: Cross-attention from prompts to image features
        # Prompts attend to image to gather relevant visual information
        q = queries + query_pe  # Prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, debug_name=f"{debug_name or 'two_way_attention_block'}_cross_attn_token_to_image" if debug_name else None)
        queries = queries + attn_out  # Residual connection
        queries = self.norm2(queries)

        # Debug capture after stage 2
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage2_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage2_output', 'tensor_type': 'queries'}
            )

        # Stage 3: MLP processing of prompt features
        # Non-linear transformation to increase representational capacity
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out  # Residual connection
        queries = self.norm3(queries)

        # Debug capture after stage 3
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage3_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage3_output', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="stage3_mlp_output",
                data=mlp_out,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'stage3_mlp', 'tensor_type': 'mlp_output'}
            )

        # Stage 4: Cross-attention from image features to prompt tokens
        # Image features attend to prompts to incorporate user intent
        # Note: q and k are swapped to reverse the attention direction
        q = queries + query_pe  # Updated prompt positions
        k = keys + key_pe      # Image positions
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries, debug_name=f"{debug_name or 'two_way_attention_block'}_cross_attn_image_to_token" if debug_name else None)
        keys = keys + attn_out  # Update image features with prompt context
        keys = self.norm4(keys)

        # Debug capture for final outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="final_queries_output",
                data=queries,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'final_output', 'tensor_type': 'queries'}
            )
            capture_debug_state(
                component_name=debug_name or "two_way_attention_block",
                state_name="final_keys_output",
                data=keys,
                metadata={'component_type': 'two_way_attention_block', 'stage': 'final_output', 'tensor_type': 'keys'}
            )

        return queries, keys
```

```python
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate segmentation masks from image and prompt embeddings.
        
        This is the main forward pass that orchestrates the entire mask prediction
        process, from transformer processing to final mask generation and quality
        assessment.

        Args:
            image_embeddings (torch.Tensor): Dense image features from image encoder
                with shape [B, C, H, W] representing visual content.
            image_pe (torch.Tensor): Positional encoding for image features with
                the same shape as image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): Processed prompt embeddings
                (points, boxes) from the prompt encoder with shape [B, N, C].
            dense_prompt_embeddings (torch.Tensor): Dense mask embeddings from
                the prompt encoder with shape [B, C, H, W].
            multimask_output (bool): Whether to return multiple mask hypotheses
                (True) or a single best mask (False).
            repeat_image (bool): Whether to repeat image embeddings to match
                batch size. Used when processing multiple prompts per image.
            high_res_features (Optional[List[torch.Tensor]]): Multi-scale features
                from image encoder for enhanced resolution (if enabled).

        Returns:
            Tuple containing:
                - masks: Predicted segmentation masks [B, N_masks, H, W]
                - iou_predictions: Mask quality scores [B, N_masks]
                - sam_tokens: Processed mask tokens for tracking [B, N_tokens, C]
                - object_scores: Object presence scores [B, 1] (if enabled)
                
        Processing Pipeline:
        1. Prepare output tokens (IoU, mask, optional object score)
        2. Combine tokens with prompt embeddings
        3. Process through two-way transformer
        4. Generate masks using hypernetwork architecture
        5. Predict mask quality and object scores
        6. Select appropriate masks based on output mode
        """
        # Generate masks and predictions using the core prediction pipeline
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select appropriate masks based on output mode and dynamic selection
        if multimask_output:
            # Multi-mask mode: return masks 1-3 (exclude single-mask output)
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            # Dynamic selection: choose between single and multi-mask based on stability
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            # Single-mask mode: return only the first mask
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        # Select appropriate mask tokens for output (used in tracking scenarios)
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            # Use multi-mask tokens for object pointer generation
            sam_tokens_out = mask_tokens_out[:, 1:]  # [B, 3, C] shape
        else:
            # Use single-mask token for object memory
            # This maintains consistency during training where single-mask tokens
            # are used as object memory representations
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [B, 1, C] shape

        return masks, iou_pred, sam_tokens_out, object_score_logits

->

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        debug_name: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate segmentation masks from image and prompt embeddings.
        
        This is the main forward pass that orchestrates the entire mask prediction
        process, from transformer processing to final mask generation and quality
        assessment.

        Args:
            image_embeddings (torch.Tensor): Dense image features from image encoder
                with shape [B, C, H, W] representing visual content.
            image_pe (torch.Tensor): Positional encoding for image features with
                the same shape as image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): Processed prompt embeddings
                (points, boxes) from the prompt encoder with shape [B, N, C].
            dense_prompt_embeddings (torch.Tensor): Dense mask embeddings from
                the prompt encoder with shape [B, C, H, W].
            multimask_output (bool): Whether to return multiple mask hypotheses
                (True) or a single best mask (False).
            repeat_image (bool): Whether to repeat image embeddings to match
                batch size. Used when processing multiple prompts per image.
            high_res_features (Optional[List[torch.Tensor]]): Multi-scale features
                from image encoder for enhanced resolution (if enabled).
            debug_name (str, optional): Name for debug state capture.

        Returns:
            Tuple containing:
                - masks: Predicted segmentation masks [B, N_masks, H, W]
                - iou_predictions: Mask quality scores [B, N_masks]
                - sam_tokens: Processed mask tokens for tracking [B, N_tokens, C]
                - object_scores: Object presence scores [B, 1] (if enabled)
                
        Processing Pipeline:
        1. Prepare output tokens (IoU, mask, optional object score)
        2. Combine tokens with prompt embeddings
        3. Process through two-way transformer
        4. Generate masks using hypernetwork architecture
        5. Predict mask quality and object scores
        6. Select appropriate masks based on output mode
        """
        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Debug capture for input embeddings
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_image_embeddings",
                data=image_embeddings,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'image_embeddings'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_image_pe",
                data=image_pe,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'image_pe'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_sparse_prompt_embeddings",
                data=sparse_prompt_embeddings,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'sparse_prompt_embeddings'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="input_dense_prompt_embeddings",
                data=dense_prompt_embeddings,
                metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': 'dense_prompt_embeddings'}
            )
            if high_res_features:
                for i, feat in enumerate(high_res_features):
                    capture_debug_state(
                        component_name=debug_name or "mask_decoder",
                        state_name=f"input_high_res_features_{i}",
                        data=feat,
                        metadata={'component_type': 'mask_decoder', 'stage': 'input', 'tensor_type': f'high_res_features_{i}'}
                    )
        
        # Generate masks and predictions using the core prediction pipeline
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            debug_name=f"{debug_name or 'mask_decoder'}_predict_masks" if debug_name else None,
        )

        # Debug capture for raw prediction outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_masks_output",
                data=masks,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'masks'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_iou_predictions",
                data=iou_pred,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'iou_predictions'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_mask_tokens_output",
                data=mask_tokens_out,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'mask_tokens'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="raw_object_score_logits",
                data=object_score_logits,
                metadata={'component_type': 'mask_decoder', 'stage': 'raw_output', 'tensor_type': 'object_scores'}
            )

        # Select appropriate masks based on output mode and dynamic selection
        if multimask_output:
            # Multi-mask mode: return masks 1-3 (exclude single-mask output)
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            # Dynamic selection: choose between single and multi-mask based on stability
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            # Single-mask mode: return only the first mask
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        # Select appropriate mask tokens for output (used in tracking scenarios)
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            # Use multi-mask tokens for object pointer generation
            sam_tokens_out = mask_tokens_out[:, 1:]  # [B, 3, C] shape
        else:
            # Use single-mask token for object memory
            # This maintains consistency during training where single-mask tokens
            # are used as object memory representations
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [B, 1, C] shape

        # Debug capture for final outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="final_masks_output",
                data=masks,
                metadata={'component_type': 'mask_decoder', 'stage': 'final_output', 'tensor_type': 'masks', 'multimask_output': multimask_output}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="final_iou_predictions",
                data=iou_pred,
                metadata={'component_type': 'mask_decoder', 'stage': 'final_output', 'tensor_type': 'iou_predictions'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder",
                state_name="final_sam_tokens_output",
                data=sam_tokens_out,
                metadata={'component_type': 'mask_decoder', 'stage': 'final_output', 'tensor_type': 'sam_tokens'}
            )

        return masks, iou_pred, sam_tokens_out, object_score_logits
```

```python
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

->

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        debug_name: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
```

```python
        # Prepare learnable output tokens for transformer processing
        s = 0  # Offset for indexing tokens
        
        if self.pred_obj_scores:
            # Include object score token if object prediction is enabled
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,  # Object presence token
                    self.iou_token.weight,        # Mask quality token
                    self.mask_tokens.weight,      # Mask generation tokens
                ],
                dim=0,
            )
            s = 1  # Adjust indexing offset
        else:
            # Standard configuration: IoU + mask tokens only
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
            
        # Expand tokens to match batch size and combine with prompt embeddings
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        # Concatenate output tokens with prompt embeddings for joint processing
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Prepare image embeddings for transformer processing
        if repeat_image:
            # Repeat image embeddings to match the number of prompt sets
            # This is used when processing multiple prompts for the same image
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            # Direct assignment when batch sizes match
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
            
        # Integrate dense prompt embeddings (masks) with image features
        src = src + dense_prompt_embeddings
        
        # Ensure positional encoding has correct batch dimension
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        
        # Repeat positional encoding to match batch size
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Process tokens and image features through two-way transformer
        # This enables bidirectional attention between tokens and image features
        hs, src = self.transformer(src, pos_src, tokens)

->

        from sam2.debug_utils import capture_debug_state, is_debug_enabled
        
        # Prepare learnable output tokens for transformer processing
        s = 0  # Offset for indexing tokens
        
        if self.pred_obj_scores:
            # Include object score token if object prediction is enabled
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,  # Object presence token
                    self.iou_token.weight,        # Mask quality token
                    self.mask_tokens.weight,      # Mask generation tokens
                ],
                dim=0,
            )
            s = 1  # Adjust indexing offset
        else:
            # Standard configuration: IoU + mask tokens only
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
            
        # Debug capture for output tokens
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="output_tokens",
                data=output_tokens,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'output_tokens', 'pred_obj_scores': self.pred_obj_scores}
            )
            if self.pred_obj_scores:
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="obj_score_token",
                    data=self.obj_score_token.weight,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'obj_score_token'}
                )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="iou_token",
                data=self.iou_token.weight,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'iou_token'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="mask_tokens",
                data=self.mask_tokens.weight,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'mask_tokens'}
            )
            
        # Expand tokens to match batch size and combine with prompt embeddings
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        # Concatenate output tokens with prompt embeddings for joint processing
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Debug capture for combined tokens
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="combined_tokens",
                data=tokens,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'tokens', 'tensor_type': 'combined_tokens'}
            )

        # Prepare image embeddings for transformer processing
        if repeat_image:
            # Repeat image embeddings to match the number of prompt sets
            # This is used when processing multiple prompts for the same image
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            # Direct assignment when batch sizes match
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
            
        # Integrate dense prompt embeddings (masks) with image features
        src = src + dense_prompt_embeddings
        
        # Debug capture for prepared source features
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="prepared_src_features",
                data=src,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'preparation', 'tensor_type': 'src_features', 'repeat_image': repeat_image}
            )
        
        # Ensure positional encoding has correct batch dimension
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        
        # Repeat positional encoding to match batch size
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Debug capture for positional encoding
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="positional_encoding_repeated",
                data=pos_src,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'preparation', 'tensor_type': 'pos_src'}
            )

        # Process tokens and image features through two-way transformer
        # This enables bidirectional attention between tokens and image features
        hs, src = self.transformer(src, pos_src, tokens, debug_name=f"{debug_name or 'mask_decoder_predict'}_transformer" if debug_name else None)
```

```python
        # Extract processed tokens after transformer processing
        iou_token_out = hs[:, s, :]  # IoU prediction token
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # Mask tokens

        # Prepare image features for mask generation
        # Reshape from transformer format back to spatial format
        src = src.transpose(1, 2).view(b, c, h, w)
        
        # Apply progressive upsampling to restore mask resolution
        if not self.use_high_res_features:
            # Standard upsampling without high-resolution feature integration
            upscaled_embedding = self.output_upscaling(src)
        else:
            # Enhanced upsampling with multi-scale feature integration
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            
            # First upsampling stage with high-res feature integration
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            # Second upsampling stage with finest feature integration  
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Hypernetwork: Generate mask prediction networks from mask tokens
        # Each mask token produces parameters for its own prediction network
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            # Each MLP generates network parameters for mask prediction
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, N_masks, C_hyper]
        
        # Apply hypernetwork-generated parameters to upscaled features
        # This performs dynamic convolution based on the mask token context
        b, c, h, w = upscaled_embedding.shape
        # Matrix multiplication: [B, N_masks, C_hyper] @ [B, C_hyper, H*W] -> [B, N_masks, H*W]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions from IoU token
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        # Generate object score predictions (if enabled)
        if self.pred_obj_scores:
            assert s == 1  # Ensure object token was included
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Default object scores: high confidence (assuming object is present)
            # sigmoid(10.0) ≈ 1.0, indicating strong object presence
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

->

        # Extract processed tokens after transformer processing
        iou_token_out = hs[:, s, :]  # IoU prediction token
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # Mask tokens

        # Debug capture for processed tokens
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="transformer_output_all_tokens",
                data=hs,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'transformer_output', 'tensor_type': 'all_tokens'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="iou_token_processed",
                data=iou_token_out,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'transformer_output', 'tensor_type': 'iou_token'}
            )
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="mask_tokens_processed",
                data=mask_tokens_out,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'transformer_output', 'tensor_type': 'mask_tokens'}
            )

        # Prepare image features for mask generation
        # Reshape from transformer format back to spatial format
        src = src.transpose(1, 2).view(b, c, h, w)
        
        # Debug capture for reshaped source
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="src_reshaped_spatial",
                data=src,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'reshape', 'tensor_type': 'src_spatial'}
            )
        
        # Apply progressive upsampling to restore mask resolution
        if not self.use_high_res_features:
            # Standard upsampling without high-resolution feature integration
            upscaled_embedding = self.output_upscaling(src)
        else:
            # Enhanced upsampling with multi-scale feature integration
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            
            # Debug capture for high-res features
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="high_res_feat_s0",
                    data=feat_s0,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'high_res_feat_s0'}
                )
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="high_res_feat_s1",
                    data=feat_s1,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'high_res_feat_s1'}
                )
            
            # First upsampling stage with high-res feature integration
            stage1_output = dc1(src) + feat_s1
            upscaled_embedding = act1(ln1(stage1_output))
            
            # Debug capture for first upsampling stage
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name="upsampling_stage1_output",
                    data=upscaled_embedding,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'stage1_output'}
                )
            
            # Second upsampling stage with finest feature integration  
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Debug capture for final upscaled embedding
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="upscaled_embedding_final",
                data=upscaled_embedding,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'upsampling', 'tensor_type': 'upscaled_embedding', 'use_high_res': self.use_high_res_features}
            )

        # Hypernetwork: Generate mask prediction networks from mask tokens
        # Each mask token produces parameters for its own prediction network
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            # Each MLP generates network parameters for mask prediction
            hyper_out = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            hyper_in_list.append(hyper_out)
            
            # Debug capture for individual hypernetwork outputs
            if debug_name and is_debug_enabled():
                capture_debug_state(
                    component_name=debug_name or "mask_decoder_predict",
                    state_name=f"hypernetwork_output_{i}",
                    data=hyper_out,
                    metadata={'component_type': 'mask_decoder_predict', 'stage': 'hypernetwork', 'tensor_type': f'hyper_out_{i}', 'mask_token_idx': i}
                )
                
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, N_masks, C_hyper]
        
        # Debug capture for stacked hypernetwork outputs
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="hypernetwork_parameters_stacked",
                data=hyper_in,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'hypernetwork', 'tensor_type': 'hyper_in_stacked'}
            )
        
        # Apply hypernetwork-generated parameters to upscaled features
        # This performs dynamic convolution based on the mask token context
        b, c, h, w = upscaled_embedding.shape
        # Matrix multiplication: [B, N_masks, C_hyper] @ [B, C_hyper, H*W] -> [B, N_masks, H*W]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Debug capture for mask generation
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="masks_raw_logits",
                data=masks,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'mask_generation', 'tensor_type': 'mask_logits'}
            )

        # Generate mask quality predictions from IoU token
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        # Debug capture for IoU predictions
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="iou_predictions",
                data=iou_pred,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'prediction', 'tensor_type': 'iou_predictions'}
            )
        
        # Generate object score predictions (if enabled)
        if self.pred_obj_scores:
            assert s == 1  # Ensure object token was included
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Default object scores: high confidence (assuming object is present)
            # sigmoid(10.0) ≈ 1.0, indicating strong object presence
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        # Debug capture for object score predictions
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name or "mask_decoder_predict",
                state_name="object_score_logits",
                data=object_score_logits,
                metadata={'component_type': 'mask_decoder_predict', 'stage': 'prediction', 'tensor_type': 'object_scores', 'pred_obj_scores': self.pred_obj_scores}
            )

        return masks, iou_pred, mask_tokens_out, object_score_logits
```

```python
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:

->

    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        debug_mode=False,
        **kwargs,
    ) -> None:
```

```python
            max_sprinkle_area (int): Maximum area of isolated regions to remove.
                                   If > 0, small disconnected regions up to this area
                                   will be removed to reduce noise. Specified in pixels.
            **kwargs: Additional arguments (currently unused, for future extensibility).
        """
        super().__init__()
        self.model = sam_model
        
        # Initialize transform pipeline for image preprocessing and postprocessing
        # This handles resizing, normalization, coordinate transformations, and mask refinement
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state management
        self._is_image_set = False  # Flag indicating if image embeddings are cached
        self._features = None       # Cached image embeddings and high-res features
        self._orig_hw = None        # Original image dimensions for coordinate transforms
        self._is_batch = False      # Flag indicating batch vs single image mode

        # Predictor configuration
        self.mask_threshold = mask_threshold

->

            max_sprinkle_area (int): Maximum area of isolated regions to remove.
                                   If > 0, small disconnected regions up to this area
                                   will be removed to reduce noise. Specified in pixels.
            debug_mode (bool): Enable debug mode for internal state visualization.
                             When True, enables capture of intermediate states and activations
                             that can be visualized for debugging and analysis purposes.
            **kwargs: Additional arguments (currently unused, for future extensibility).
        """
        super().__init__()
        self.model = sam_model
        
        # Initialize transform pipeline for image preprocessing and postprocessing
        # This handles resizing, normalization, coordinate transformations, and mask refinement
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state management
        self._is_image_set = False  # Flag indicating if image embeddings are cached
        self._features = None       # Cached image embeddings and high-res features
        self._orig_hw = None        # Original image dimensions for coordinate transforms
        self._is_batch = False      # Flag indicating batch vs single image mode

        # Predictor configuration
        self.mask_threshold = mask_threshold
        self.debug_mode = debug_mode
        
        # Initialize debug mode if enabled
        if self.debug_mode:
            from sam2.debug_utils import enable_debug_mode
            enable_debug_mode()
```

```python
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

->

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
        return_debug_states: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
```

```python
            normalize_coords (bool): Coordinate space interpretation.
                True: Coordinates relative to original image dimensions (default).
                False: Coordinates in model input space (advanced usage).

        Returns:
            Tuple of three arrays:
            - masks (np.ndarray): Output masks, shape (C, H, W) where C is number of masks.
                Values are binary (0/1) if return_logits=False, continuous if True.
                H, W match original image dimensions.
            - iou_predictions (np.ndarray): Quality scores, shape (C,).
                Values in [0, 1] indicating predicted IoU with ground truth.
                Higher scores indicate better quality masks.
            - low_res_masks (np.ndarray): Low-resolution logits, shape (C, 256, 256).
                Can be used as mask_input for subsequent refinement iterations.
                Provides efficient representation for iterative workflows.

->

            normalize_coords (bool): Coordinate space interpretation.
                True: Coordinates relative to original image dimensions (default).
                False: Coordinates in model input space (advanced usage).
            return_debug_states (bool): Whether to return captured debug states.
                True: Return debug states dictionary for visualization/analysis.
                False: Return only prediction results (default).
                Only effective when predictor was initialized with debug_mode=True.

        Returns:
            If return_debug_states=False:
                Tuple of three arrays:
                - masks (np.ndarray): Output masks, shape (C, H, W) where C is number of masks.
                    Values are binary (0/1) if return_logits=False, continuous if True.
                    H, W match original image dimensions.
                - iou_predictions (np.ndarray): Quality scores, shape (C,).
                    Values in [0, 1] indicating predicted IoU with ground truth.
                    Higher scores indicate better quality masks.
                - low_res_masks (np.ndarray): Low-resolution logits, shape (C, 256, 256).
                    Can be used as mask_input for subsequent refinement iterations.
                    Provides efficient representation for iterative workflows.
                    
            If return_debug_states=True:
                Tuple of four elements including all above plus:
                - debug_states (dict): Captured internal states from model components
                    including position encodings, attention weights, embeddings, etc.
                    Can be visualized using sam2.debug_utils.visualize_debug_states()
```

```python
        # Generate predictions using the SAM2 model
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        # Convert output tensors to numpy arrays for user consumption
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

->

        # Clear debug states before prediction if debug mode is enabled
        if self.debug_mode and return_debug_states:
            from sam2.debug_utils import clear_debug_states
            clear_debug_states()

        # Generate predictions using the SAM2 model
        masks, iou_predictions, low_res_masks = self._predict(
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
            return_debug_states=return_debug_states,
        )

        # Convert output tensors to numpy arrays for user consumption
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        
        # Return debug states if requested
        if self.debug_mode and return_debug_states:
            from sam2.debug_utils import get_debug_states
            debug_states = get_debug_states()
            return masks_np, iou_predictions_np, low_res_masks_np, debug_states
        else:
            return masks_np, iou_predictions_np, low_res_masks_np
```

```python
    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

->

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
        return_debug_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

```python
        # Encode prompts into sparse and dense embeddings
        # The prompt encoder converts discrete prompts into continuous embeddings
        # that can be processed by the transformer decoder
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,  # Combined point and box prompts
            boxes=None,           # Boxes are handled through points
            masks=mask_input,     # Previous mask for iterative refinement
        )

        # Determine if we're in multi-object prediction mode
        # This affects how the decoder processes multiple prompts
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )

        # Extract high-resolution features for current image
        # These features provide fine-grained spatial information for detailed masks
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        # Generate masks using the SAM decoder
        # The decoder combines image embeddings, prompt embeddings, and high-res features
        # to produce accurate segmentation masks with quality predictions
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),  # Positional encoding
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,  # Handle multiple objects on same image
            high_res_features=high_res_features,
        )

->

        # Encode prompts into sparse and dense embeddings
        # The prompt encoder converts discrete prompts into continuous embeddings
        # that can be processed by the transformer decoder
        prompt_encoder_debug_name = "sam_prompt_encoder" if (self.debug_mode and return_debug_states) else None
        
        # Check if prompt encoder supports debug parameter
        if hasattr(self.model.sam_prompt_encoder, 'forward') and 'debug_name' in self.model.sam_prompt_encoder.forward.__code__.co_varnames:
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=concat_points,  # Combined point and box prompts
                boxes=None,           # Boxes are handled through points
                masks=mask_input,     # Previous mask for iterative refinement
                debug_name=prompt_encoder_debug_name,
            )
        else:
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=concat_points,  # Combined point and box prompts
                boxes=None,           # Boxes are handled through points
                masks=mask_input,     # Previous mask for iterative refinement
            )

        # Determine if we're in multi-object prediction mode
        # This affects how the decoder processes multiple prompts
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )

        # Extract high-resolution features for current image
        # These features provide fine-grained spatial information for detailed masks
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        # Generate masks using the SAM decoder
        # The decoder combines image embeddings, prompt embeddings, and high-res features
        # to produce accurate segmentation masks with quality predictions
        mask_decoder_debug_name = "sam_mask_decoder" if (self.debug_mode and return_debug_states) else None
        
        # Check if mask decoder supports debug parameter
        if hasattr(self.model.sam_mask_decoder, 'forward') and 'debug_name' in self.model.sam_mask_decoder.forward.__code__.co_varnames:
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),  # Positional encoding
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,  # Handle multiple objects on same image
                high_res_features=high_res_features,
                debug_name=mask_decoder_debug_name,
            )
        else:
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),  # Positional encoding
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,  # Handle multiple objects on same image
                high_res_features=high_res_features,
            )
```

```markdown
# SAM2 Debug Visualization Guide

This guide explains how to use the comprehensive debug visualization system added to MedSAM2. The system allows you to capture and visualize internal states of the model during inference, providing insights into position encodings, attention patterns, embeddings, and other intermediate representations.

## Overview

The debug visualization system provides:

- **Non-intrusive capture**: Debug hooks that don't affect model performance
- **On-demand activation**: Easy enable/disable functionality 
- **Comprehensive coverage**: Visualization for all major model components
- **Interactive ready**: Designed for future integration with interactive UIs
- **Performance aware**: Minimal overhead when debug mode is disabled

## Quick Start

### Basic Usage

```python
import torch
import numpy as np
from sam2 import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.debug_utils import visualize_debug_states

# Load your model
checkpoint = "path/to/sam2_checkpoint.pt"
model_cfg = "sam2_hiera_b.yaml"
sam2_model = build_sam2(model_cfg, checkpoint)

# Initialize predictor with debug mode enabled
predictor = SAM2ImagePredictor(sam2_model, debug_mode=True)

# Set your image
image = ... # Your input image (numpy array or PIL Image)
predictor.set_image(image)

# Run prediction with debug state capture
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]],
    point_labels=[1],
    return_debug_states=True
)

# Visualize all captured internal states
visualize_debug_states(debug_states, save_path="debug_output/")
```

### Advanced Usage with Selective Visualization

```python
from sam2.debug_utils import (
    enable_debug_mode, 
    disable_debug_mode, 
    get_debug_states,
    clear_debug_states,
    SAM2Visualizer
)

# Enable debug mode with specific components
enable_debug_mode(
    capture_attention=True,
    capture_embeddings=True, 
    capture_position_encoding=True,
    capture_gradients=False  # Disable gradient capture for faster inference
)

# Initialize predictor (debug_mode=True not needed if globally enabled)
predictor = SAM2ImagePredictor(sam2_model)

# Run multiple predictions
for i, (image, points) in enumerate(dataset):
    clear_debug_states()  # Clear previous states
    
    predictor.set_image(image)
    masks, scores, logits, debug_states = predictor.predict(
        point_coords=points,
        point_labels=[1]*len(points),
        return_debug_states=True
    )
    
    # Create custom visualizations
    visualizer = SAM2Visualizer(figsize_base=(15, 10))
    
    # Visualize specific components
    if 'sam_mask_decoder' in debug_states:
        decoder_states = debug_states['sam_mask_decoder']
        visualizer.visualize_mask_decoder_states(
            {name: info['data'] for name, info in decoder_states.items()},
            save_path=f"debug_output/sample_{i}/"
        )
    
    # Visualize attention patterns
    for component_name, states in debug_states.items():
        for state_name, state_info in states.items():
            if 'attention_weights' in state_name:
                visualizer.visualize_attention_patterns(
                    state_info['data'],
                    layer_name=f"{component_name}_{state_name}",
                    save_path=f"debug_output/sample_{i}/"
                )

# Disable debug mode when done
disable_debug_mode()
```

## Captured Internal States

### Position Encodings

The system captures position encoding at multiple stages:

```python
# Access position encoding data
pos_enc_states = debug_states['position_encoding_sine']
# or
pos_enc_states = debug_states['position_encoding_random']

# Available states:
# - 'coordinate_grids_x': Raw x-coordinate grids
# - 'coordinate_grids_y': Raw y-coordinate grids  
# - 'coordinate_grids_x_normalized': Normalized x-coordinates
# - 'coordinate_grids_y_normalized': Normalized y-coordinates
# - 'frequency_dimensions': Sinusoidal frequency components
# - 'sinusoidal_encoding_x': X-coordinate sinusoidal encoding
# - 'sinusoidal_encoding_y': Y-coordinate sinusoidal encoding
# - 'position_embeddings_final': Final position embeddings
```

**Visualization:**
- Shows spatial patterns in position encodings
- Compares different encoding types (sine vs random)
- Analyzes frequency components and their spatial distribution

### Image Encoder States

```python
# Access image encoder states
encoder_states = debug_states['image_encoder']

# Available states:
# - 'input_image': Original input image
# - 'trunk_features_level_*': Backbone features at different scales
# - 'vision_features_final': Final vision features
# - 'fpn_features_level_*': FPN features at different levels
# - 'position_encoding_level_*': Position encodings for each level
```

**Visualization:**
- Multi-scale feature representations
- Feature pyramid network (FPN) outputs
- Spatial feature distributions
- Principal Component Analysis (PCA) of embeddings

### Attention Patterns

```python
# Access attention states from transformer components
attention_states = debug_states['two_way_transformer_layer_0']

# Available states:
# - 'attention_weights': Attention weight matrices
# - 'attention_scores': Pre-softmax attention scores
# - 'multihead_queries': Multi-head query representations
# - 'multihead_keys': Multi-head key representations
# - 'multihead_values': Multi-head value representations
```

**Visualization:**
- Attention weight heatmaps for each head
- Cross-attention between prompts and image features
- Self-attention patterns within prompts
- Evolution of attention across transformer layers

### Mask Decoder States

```python
# Access mask decoder states
decoder_states = debug_states['sam_mask_decoder']

# Available states:
# - 'input_image_embeddings': Input image features
# - 'input_sparse_prompt_embeddings': Prompt embeddings
# - 'raw_masks_output': Raw mask logits before post-processing
# - 'iou_predictions': Mask quality predictions
# - 'hypernetwork_output_*': Hypernetwork parameters for each mask
# - 'upscaled_embedding_final': Upsampled features for mask generation
```

**Visualization:**
- Hypernetwork parameter distributions
- Mask generation process
- IoU prediction patterns
- Intermediate upsampling stages

## Visualization Functions

### Built-in Visualizers

#### Position Encoding Visualization

```python
from sam2.debug_utils import SAM2Visualizer

visualizer = SAM2Visualizer()

# Visualize sinusoidal position encoding
pos_enc_data = debug_states['position_encoding_sine']['position_embeddings_final']['data']
visualizer.visualize_position_encoding(
    pos_enc_data,
    encoding_type="sine",
    save_path="debug_output/",
    show_individual_dims=True  # Show individual encoding dimensions
)
```

#### Attention Pattern Visualization

```python
# Visualize attention weights
attention_weights = debug_states['two_way_transformer_layer_0']['attention_weights']['data']
visualizer.visualize_attention_patterns(
    attention_weights,
    layer_name="two_way_transformer_layer_0",
    save_path="debug_output/",
    max_heads_to_show=8
)
```

#### Image Embedding Visualization

```python
# Visualize image embeddings with PCA analysis
embeddings = debug_states['image_encoder']['vision_features_final']['data']
original_image = debug_states['image_encoder']['input_image']['data']

visualizer.visualize_image_embeddings(
    embeddings,
    original_image=original_image,
    save_path="debug_output/",
    n_components_to_show=8
)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Create custom visualization for specific analysis
def visualize_attention_evolution(debug_states, save_path=None):
    """Visualize how attention patterns evolve across transformer layers."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    layer_names = [name for name in debug_states.keys() if 'two_way_transformer_layer' in name]
    
    for i, layer_name in enumerate(layer_names[:6]):
        if 'attention_weights' in debug_states[layer_name]:
            attn_weights = debug_states[layer_name]['attention_weights']['data']
            # Average across heads and batch
            attn_avg = attn_weights.mean(dim=(0, 1)).numpy()
            
            im = axes[i].imshow(attn_avg, cmap='viridis')
            axes[i].set_title(f'{layer_name}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Turn off unused axes
    for i in range(len(layer_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/attention_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()

# Use custom visualization
visualize_attention_evolution(debug_states, save_path="debug_output/")
```

## Advanced Features

### Memory and Performance Optimization

```python
# For long videos or large batch processing
from sam2.debug_utils import DebugStateCapture

# Create custom debug capture with memory limits
debug_capture = DebugStateCapture()
debug_capture.enable(
    capture_attention=True,
    capture_embeddings=False,  # Disable to save memory
    capture_position_encoding=True,
    capture_gradients=False
)

# Process video frames
for frame_idx, frame in enumerate(video_frames):
    if frame_idx % 10 == 0:  # Capture debug states every 10 frames
        debug_capture.clear()  # Clear previous states
        
        predictor.set_image(frame)
        masks, scores, logits, debug_states = predictor.predict(
            point_coords=tracked_points[frame_idx],
            point_labels=[1] * len(tracked_points[frame_idx]),
            return_debug_states=True
        )
        
        # Save debug visualization for key frames
        visualize_debug_states(
            debug_states, 
            save_path=f"debug_output/frame_{frame_idx}/",
            create_summary=True
        )
```

### Integration with Training

```python
# Monitor training dynamics (if training enabled)
if model.training:
    enable_debug_mode(capture_gradients=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (images, targets) in enumerate(dataloader):
            clear_debug_states()
            
            # Forward pass with debug capture
            outputs = model(images, return_debug_states=True)
            loss = criterion(outputs, targets)
            
            # Capture gradients
            loss.backward()
            
            # Analyze training dynamics every N batches
            if batch_idx % log_interval == 0:
                debug_states = get_debug_states()
                
                # Custom analysis for training
                analyze_gradient_flow(debug_states)
                visualize_activation_statistics(debug_states)
                
            optimizer.step()
            optimizer.zero_grad()
```

## Debugging Common Issues

### Performance Issues

```python
# Profile debug overhead
import time

# Without debug mode
start_time = time.time()
masks, scores, logits = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1]
)
time_without_debug = time.time() - start_time

# With debug mode
predictor.debug_mode = True
enable_debug_mode()

start_time = time.time()
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1],
    return_debug_states=True
)
time_with_debug = time.time() - start_time

print(f"Debug overhead: {(time_with_debug - time_without_debug) / time_without_debug * 100:.2f}%")
```

### Memory Issues

```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Before debug capture
mem_before = get_memory_usage()

# Run with debug
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1],
    return_debug_states=True
)

mem_after = get_memory_usage()
print(f"Memory usage: {mem_after - mem_before:.2f} MB")

# Clean up
clear_debug_states()
disable_debug_mode()
```

### State Analysis

```python
def analyze_debug_states(debug_states):
    """Analyze captured debug states for potential issues."""
    
    for component_name, component_states in debug_states.items():
        print(f"\n=== {component_name} ===")
        
        for state_name, state_info in component_states.items():
            data = state_info['data']
            metadata = state_info.get('metadata', {})
            
            if isinstance(data, torch.Tensor):
                print(f"{state_name}:")
                print(f"  Shape: {data.shape}")
                print(f"  Range: [{data.min().item():.6f}, {data.max().item():.6f}]")
                print(f"  Mean: {data.mean().item():.6f}")
                print(f"  Std: {data.std().item():.6f}")
                
                # Check for potential issues
                if torch.isnan(data).any():
                    print(f"  ⚠️  Contains NaN values!")
                if torch.isinf(data).any():
                    print(f"  ⚠️  Contains infinite values!")
                if data.std() < 1e-6:
                    print(f"  ⚠️  Very low variance - potential dead neurons!")

# Run analysis
analyze_debug_states(debug_states)
```

## Best Practices

### 1. Selective Debug Capture

Only capture what you need to minimize performance impact:

```python
# For attention analysis
enable_debug_mode(
    capture_attention=True,
    capture_embeddings=False,
    capture_position_encoding=False
)

# For embedding analysis  
enable_debug_mode(
    capture_attention=False,
    capture_embeddings=True,
    capture_position_encoding=True
)
```

### 2. Regular Cleanup

Clear debug states regularly to prevent memory leaks:

```python
# Process in batches
for batch in batches:
    clear_debug_states()  # Clear before each batch
    
    # Process batch...
    debug_states = get_debug_states()
    
    # Save important results
    if should_save_debug:
        visualize_debug_states(debug_states, save_path=f"debug_batch_{batch_id}/")
```

### 3. Meaningful Names

Use descriptive debug names for complex workflows:

```python
# For multi-stage processing
predictor.set_image(image)

# First stage
masks_stage1, _, _, debug_states_1 = predictor.predict(
    point_coords=initial_points,
    point_labels=[1] * len(initial_points),
    return_debug_states=True
)

# Save first stage
visualize_debug_states(debug_states_1, save_path="debug_output/stage1/")

# Second stage with refinement
clear_debug_states()
masks_stage2, _, _, debug_states_2 = predictor.predict(
    point_coords=refined_points,
    point_labels=[1] * len(refined_points),
    mask_input=logits_stage1,
    return_debug_states=True
)

# Save second stage
visualize_debug_states(debug_states_2, save_path="debug_output/stage2/")
```

## Integration with Jupyter Notebooks

```python
# Notebook-friendly visualization
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

def notebook_visualize_debug_states(debug_states, max_components=3):
    """Notebook-friendly debug visualization."""
    
    visualizer = SAM2Visualizer(figsize_base=(10, 6))
    
    # Show key components
    component_names = list(debug_states.keys())[:max_components]
    
    for component_name in component_names:
        print(f"\n📊 {component_name}")
        print("=" * 50)
        
        component_states = debug_states[component_name]
        
        # Show first few interesting states
        for state_name, state_info in list(component_states.items())[:3]:
            data = state_info['data']
            metadata = state_info['metadata']
            
            print(f"\n🔍 {state_name}")
            print(f"Shape: {data.shape}, Type: {metadata.get('tensor_type', 'unknown')}")
            
            # Create appropriate visualization
            if 'attention' in state_name.lower() and data.dim() >= 3:
                visualizer.visualize_attention_patterns(data, layer_name=state_name)
            elif 'embedding' in state_name.lower() and data.dim() == 4:
                visualizer.visualize_image_embeddings(data)
            elif 'position' in state_name.lower():
                encoding_type = metadata.get('encoding_type', 'unknown')
                visualizer.visualize_position_encoding(data, encoding_type=encoding_type)

# Use in notebook
notebook_visualize_debug_states(debug_states)
```

## Troubleshooting

### Common Issues and Solutions

1. **No debug states captured**
   ```python
   # Check if debug mode is enabled
   from sam2.debug_utils import is_debug_enabled
   print(f"Debug enabled: {is_debug_enabled()}")
   
   # Enable if needed
   enable_debug_mode()
   ```

2. **Memory errors with large images**
   ```python
   # Use selective capture
   enable_debug_mode(
       capture_attention=True,
       capture_embeddings=False,  # Disable large embeddings
       capture_position_encoding=False
   )
   ```

3. **Visualization errors**
   ```python
   # Check data shapes and types
   for component_name, states in debug_states.items():
       for state_name, state_info in states.items():
           data = state_info['data']
           print(f"{component_name}.{state_name}: {data.shape}, {data.dtype}")
   ```

4. **Performance degradation**
   ```python
   # Disable debug mode for production
   disable_debug_mode()
   
   # Or use minimal capture
   enable_debug_mode(
       capture_attention=False,
       capture_embeddings=False,
       capture_position_encoding=True
   )
   ```

## Extending the Debug System

### Adding Custom Debug Hooks

```python
from sam2.debug_utils import capture_debug_state, is_debug_enabled

class CustomModule(nn.Module):
    def forward(self, x, debug_name=None):
        # Your computation
        output = self.process(x)
        
        # Add debug capture
        if debug_name and is_debug_enabled():
            capture_debug_state(
                component_name=debug_name,
                state_name="custom_output",
                data=output,
                metadata={'custom_info': 'example'}
            )
        
        return output
```

### Custom Visualizations

```python
def create_custom_visualizer(debug_states, analysis_type="attention_flow"):
    """Create custom visualizations for specific analysis needs."""
    
    if analysis_type == "attention_flow":
        # Analyze attention flow across layers
        attention_data = []
        for component_name, states in debug_states.items():
            if 'attention_weights' in states:
                attn = states['attention_weights']['data']
                attention_data.append((component_name, attn))
        
        # Create flow visualization
        fig, axes = plt.subplots(1, len(attention_data), figsize=(20, 5))
        for i, (name, attn) in enumerate(attention_data):
            # Average across heads and batch
            attn_avg = attn.mean(dim=(0, 1))
            im = axes[i].imshow(attn_avg, cmap='viridis')
            axes[i].set_title(name)
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
        
    elif analysis_type == "embedding_similarity":
        # Analyze embedding similarities
        pass  # Implement your custom analysis

# Use custom visualizer
create_custom_visualizer(debug_states, "attention_flow")
```

This debug visualization system provides powerful tools for understanding and debugging SAM2's internal behavior, enabling researchers and developers to gain deep insights into the model's decision-making process.
```

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SAM2 Debug Visualization Example\n",
    "\n",
    "This notebook demonstrates how to use the debug visualization system added to MedSAM2.\n",
    "\n",
    "The debug system allows you to:\n",
    "- Capture internal states during model inference\n",
    "- Visualize position encodings, attention patterns, and embeddings\n",
    "- Analyze model behavior for debugging and research"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "import requests\n",
    "from io import BytesIO\n",
    "\n",
    "# Import SAM2 components\n",
    "from sam2 import build_sam2\n",
    "from sam2.sam2_image_predictor import SAM2ImagePredictor\n",
    "from sam2.debug_utils import (\n",
    "    enable_debug_mode, \n",
    "    disable_debug_mode, \n",
    "    visualize_debug_states,\n",
    "    SAM2Visualizer,\n",
    "    get_debug_states,\n",
    "    clear_debug_states\n",
    ")\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Model and Setup Debug Mode"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load your SAM2 model\n",
    "checkpoint = \"path/to/your/sam2_checkpoint.pt\"  # Replace with your model path\n",
    "model_cfg = \"sam2_hiera_b.yaml\"  # Replace with your config\n",
    "\n",
    "# Build model\n",
    "sam2_model = build_sam2(model_cfg, checkpoint)\n",
    "\n",
    "# Initialize predictor with debug mode enabled\n",
    "predictor = SAM2ImagePredictor(sam2_model, debug_mode=True)\n",
    "\n",
    "print(\"Model loaded successfully!\")\n",
    "print(f\"Device: {predictor.device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Sample Image\n",
    "\n",
    "For this example, we'll use a sample image. Replace this with your own image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Option 1: Load from URL (example)\n",
    "def load_image_from_url(url):\n",
    "    response = requests.get(url)\n",
    "    image = Image.open(BytesIO(response.content)).convert('RGB')\n",
    "    return np.array(image)\n",
    "\n",
    "# Option 2: Load local image\n",
    "def load_local_image(path):\n",
    "    image = Image.open(path).convert('RGB')\n",
    "    return np.array(image)\n",
    "\n",
    "# Example: Load a sample image (replace with your image)\n",
    "image_url = \"https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg\"\n",
    "try:\n",
    "    image = load_image_from_url(image_url)\n",
    "    print(f\"Loaded image from URL. Shape: {image.shape}\")\n",
    "except:\n",
    "    # Fallback: create a simple test image\n",
    "    image = np.zeros((512, 512, 3), dtype=np.uint8)\n",
    "    # Add some simple patterns\n",
    "    image[100:400, 100:400] = [255, 0, 0]  # Red square\n",
    "    image[200:300, 200:300] = [0, 255, 0]  # Green square\n",
    "    print(\"Created test image\")\n",
    "\n",
    "# Display the image\n",
    "plt.figure(figsize=(10, 10))\n",
    "plt.imshow(image)\n",
    "plt.axis('off')\n",
    "plt.title('Input Image')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Set Image and Prepare for Debug Capture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set the image in the predictor\n",
    "predictor.set_image(image)\n",
    "\n",
    "# Clear any previous debug states\n",
    "clear_debug_states()\n",
    "\n",
    "print(\"Image set successfully!\")\n",
    "print(f\"Original image shape: {image.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Prediction with Debug State Capture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define some example prompts (click coordinates)\n",
    "point_coords = np.array([[250, 250], [300, 200]])  # Two click points\n",
    "point_labels = np.array([1, 1])  # Both are positive (foreground) points\n",
    "\n",
    "# Run prediction with debug state capture\n",
    "masks, scores, logits, debug_states = predictor.predict(\n",
    "    point_coords=point_coords,\n",
    "    point_labels=point_labels,\n",
    "    multimask_output=True,\n",
    "    return_debug_states=True\n",
    ")\n",
    "\n",
    "print(f\"Prediction completed!\")\n",
    "print(f\"Generated {masks.shape[0]} masks\")\n",
    "print(f\"Captured debug states from {len(debug_states)} components\")\n",
    "print(f\"Debug components: {list(debug_states.keys())}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualize Prediction Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to show masks\n",
    "def show_mask(mask, ax, random_color=False):\n",
    "    if random_color:\n",
    "        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)\n",
    "    else:\n",
    "        color = np.array([30/255, 144/255, 255/255, 0.6])\n",
    "    h, w = mask.shape[-2:]\n",
    "    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)\n",
    "    ax.imshow(mask_image)\n",
    "\n",
    "def show_points(coords, labels, ax, marker_size=375):\n",
    "    pos_points = coords[labels==1]\n",
    "    neg_points = coords[labels==0]\n",
    "    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)\n",
    "    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)\n",
    "\n",
    "# Display results\n",
    "fig, axes = plt.subplots(1, len(masks), figsize=(20, 5))\n",
    "if len(masks) == 1:\n",
    "    axes = [axes]\n",
    "\n",
    "for i, (mask, score) in enumerate(zip(masks, scores)):\n",
    "    axes[i].imshow(image)\n",
    "    show_mask(mask, axes[i])\n",
    "    show_points(point_coords, point_labels, axes[i])\n",
    "    axes[i].set_title(f\"Mask {i+1}, Score: {score:.3f}\", fontsize=18)\n",
    "    axes[i].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Analyze Debug States"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print summary of captured debug states\n",
    "def analyze_debug_states(debug_states):\n",
    "    print(\"Debug States Summary\")\n",
    "    print(\"=\" * 50)\n",
    "    \n",
    "    for component_name, component_states in debug_states.items():\n",
    "        print(f\"\\n📦 Component: {component_name}\")\n",
    "        print(f\"   Number of states: {len(component_states)}\")\n",
    "        \n",
    "        for state_name, state_info in list(component_states.items())[:5]:  # Show first 5\n",
    "            data = state_info['data']\n",
    "            metadata = state_info.get('metadata', {})\n",
    "            \n",
    "            if isinstance(data, torch.Tensor):\n",
    "                print(f\"   🔍 {state_name}:\")\n",
    "                print(f\"      Shape: {data.shape}\")\n",
    "                print(f\"      Type: {metadata.get('tensor_type', 'unknown')}\")\n",
    "                print(f\"      Range: [{data.min().item():.3f}, {data.max().item():.3f}]\")\n",
    "        \n",
    "        if len(component_states) > 5:\n",
    "            print(f\"   ... and {len(component_states) - 5} more states\")\n",
    "\n",
    "analyze_debug_states(debug_states)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Visualize Position Encodings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create visualizer\n",
    "visualizer = SAM2Visualizer(figsize_base=(12, 8))\n",
    "\n",
    "# Look for position encoding states\n",
    "pos_enc_components = [name for name in debug_states.keys() if 'position_encoding' in name.lower()]\n",
    "\n",
    "if pos_enc_components:\n",
    "    print(f\"Found position encoding components: {pos_enc_components}\")\n",
    "    \n",
    "    for component_name in pos_enc_components[:2]:  # Show first 2\n",
    "        component_states = debug_states[component_name]\n",
    "        \n",
    "        # Look for final position embeddings\n",
    "        final_states = [name for name in component_states.keys() if 'final' in name or 'embeddings' in name]\n",
    "        \n",
    "        if final_states:\n",
    "            state_name = final_states[0]\n",
    "            pos_data = component_states[state_name]['data']\n",
    "            metadata = component_states[state_name]['metadata']\n",
    "            encoding_type = metadata.get('encoding_type', 'unknown')\n",
    "            \n",
    "            print(f\"\\nVisualizing {component_name} - {state_name}\")\n",
    "            print(f\"Shape: {pos_data.shape}, Type: {encoding_type}\")\n",
    "            \n",
    "            # Visualize position encoding\n",
    "            visualizer.visualize_position_encoding(\n",
    "                pos_data,\n",
    "                encoding_type=encoding_type,\n",
    "                show_individual_dims=False  # Show summary for notebook\n",
    "            )\n",
    "else:\n",
    "    print(\"No position encoding states found in debug capture\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Visualize Attention Patterns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look for attention states\n",
    "attention_found = False\n",
    "\n",
    "for component_name, component_states in debug_states.items():\n",
    "    for state_name, state_info in component_states.items():\n",
    "        if 'attention_weights' in state_name:\n",
    "            attention_data = state_info['data']\n",
    "            metadata = state_info['metadata']\n",
    "            \n",
    "            print(f\"\\nVisualizing attention: {component_name} - {state_name}\")\n",
    "            print(f\"Shape: {attention_data.shape}\")\n",
    "            print(f\"Num heads: {metadata.get('num_heads', 'unknown')}\")\n",
    "            \n",
    "            # Visualize attention patterns\n",
    "            visualizer.visualize_attention_patterns(\n",
    "                attention_data,\n",
    "                layer_name=f\"{component_name}_{state_name}\",\n",
    "                max_heads_to_show=4  # Limit for notebook display\n",
    "            )\n",
    "            \n",
    "            attention_found = True\n",
    "            break  # Show only first attention pattern for demo\n",
    "    \n",
    "    if attention_found:\n",
    "        break\n",
    "\n",
    "if not attention_found:\n",
    "    print(\"No attention patterns found in debug capture\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Visualize Image Embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look for image embeddings\n",
    "embedding_found = False\n",
    "\n",
    "for component_name, component_states in debug_states.items():\n",
    "    for state_name, state_info in component_states.items():\n",
    "        if ('embedding' in state_name.lower() or 'features' in state_name.lower()) and \\\n",
    "           state_info['data'].dim() == 4:  # 4D tensor (B, C, H, W)\n",
    "            \n",
    "            embedding_data = state_info['data']\n",
    "            metadata = state_info['metadata']\n",
    "            \n",
    "            print(f\"\\nVisualizing embeddings: {component_name} - {state_name}\")\n",
    "            print(f\"Shape: {embedding_data.shape}\")\n",
    "            \n",
    "            # Get original image if available\n",
    "            original_img = None\n",
    "            if 'image_encoder' in debug_states and 'input_image' in debug_states['image_encoder']:\n",
    "                original_img = debug_states['image_encoder']['input_image']['data']\n",
    "            \n",
    "            # Visualize embeddings\n",
    "            visualizer.visualize_image_embeddings(\n",
    "                embedding_data,\n",
    "                original_image=original_img,\n",
    "                n_components_to_show=6  # Limit for notebook display\n",
    "            )\n",
    "            \n",
    "            embedding_found = True\n",
    "            break  # Show only first embedding for demo\n",
    "    \n",
    "    if embedding_found:\n",
    "        break\n",
    "\n",
    "if not embedding_found:\n",
    "    print(\"No suitable image embeddings found in debug capture\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Visualize Mask Decoder States"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look for mask decoder states\n",
    "decoder_components = [name for name in debug_states.keys() if 'decoder' in name.lower()]\n",
    "\n",
    "if decoder_components:\n",
    "    component_name = decoder_components[0]  # Take first decoder component\n",
    "    component_states = debug_states[component_name]\n",
    "    \n",
    "    print(f\"Visualizing mask decoder: {component_name}\")\n",
    "    print(f\"Available states: {list(component_states.keys())}\")\n",
    "    \n",
    "    # Select interesting decoder states\n",
    "    interesting_states = {}\n",
    "    for state_name, state_info in component_states.items():\n",
    "        if any(keyword in state_name.lower() for keyword in ['mask', 'iou', 'token', 'hyper']):\n",
    "            interesting_states[state_name] = state_info['data']\n",
    "    \n",
    "    if interesting_states:\n",
    "        print(f\"Showing {len(interesting_states)} decoder states\")\n",
    "        visualizer.visualize_mask_decoder_states(interesting_states)\n",
    "    else:\n",
    "        print(\"No interesting decoder states found\")\n",
    "else:\n",
    "    print(\"No mask decoder states found in debug capture\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Save Debug Visualization to Files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save all debug visualizations to files\n",
    "import os\n",
    "\n",
    "output_dir = \"debug_output_example\"\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "print(f\"Saving debug visualizations to {output_dir}/\")\n",
    "\n",
    "# This will create comprehensive visualizations and save them\n",
    "visualize_debug_states(\n",
    "    debug_states, \n",
    "    save_path=output_dir,\n",
    "    create_summary=True\n",
    ")\n",
    "\n",
    "print(\"Debug visualizations saved!\")\n",
    "print(f\"Check the '{output_dir}' directory for saved images and summary.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Performance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "\n",
    "# Test performance impact of debug mode\n",
    "def benchmark_debug_overhead(predictor, point_coords, point_labels, num_runs=5):\n",
    "    \n",
    "    # Without debug\n",
    "    times_without_debug = []\n",
    "    for _ in range(num_runs):\n",
    "        start_time = time.time()\n",
    "        masks, scores, logits = predictor.predict(\n",
    "            point_coords=point_coords,\n",
    "            point_labels=point_labels,\n",
    "            return_debug_states=False\n",
    "        )\n",
    "        times_without_debug.append(time.time() - start_time)\n",
    "    \n",
    "    # With debug\n",
    "    times_with_debug = []\n",
    "    for _ in range(num_runs):\n",
    "        clear_debug_states()\n",
    "        start_time = time.time()\n",
    "        masks, scores, logits, debug_states = predictor.predict(\n",
    "            point_coords=point_coords,\n",
    "            point_labels=point_labels,\n",
    "            return_debug_states=True\n",
    "        )\n",
    "        times_with_debug.append(time.time() - start_time)\n",
    "    \n",
    "    avg_without = np.mean(times_without_debug)\n",
    "    avg_with = np.mean(times_with_debug)\n",
    "    overhead = (avg_with - avg_without) / avg_without * 100\n",
    "    \n",
    "    return avg_without, avg_with, overhead\n",
    "\n",
    "# Run benchmark\n",
    "print(\"Benchmarking debug overhead...\")\n",
    "time_without, time_with, overhead = benchmark_debug_overhead(\n",
    "    predictor, point_coords, point_labels\n",
    ")\n",
    "\n",
    "print(f\"Average time without debug: {time_without:.3f} seconds\")\n",
    "print(f\"Average time with debug: {time_with:.3f} seconds\")\n",
    "print(f\"Debug overhead: {overhead:.1f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. Cleanup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean up debug states and disable debug mode\n",
    "clear_debug_states()\n",
    "disable_debug_mode()\n",
    "\n",
    "print(\"Debug mode disabled and states cleared.\")\n",
    "print(\"\\n🎉 Debug visualization example completed!\")\n",
    "print(\"\\nNext steps:\")\n",
    "print(\"- Try with your own images and prompts\")\n",
    "print(\"- Experiment with different debug capture settings\")\n",
    "print(\"- Create custom visualizations for your research needs\")\n",
    "print(\"- Check the DEBUG_VISUALIZATION_GUIDE.md for advanced usage\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

```plaintext


->

# SAM2 Debug Visualization Example

This notebook demonstrates how to use the debug visualization system added to MedSAM2.

The debug system allows you to:
- Capture internal states during model inference
- Visualize position encodings, attention patterns, and embeddings
- Analyze model behavior for debugging and research
```

## 🎉 Debug Visualization System Implementation Complete!

I have successfully implemented a comprehensive debug visualization system for MedSAM2 that allows you to capture and visualize internal states on demand. Here's what has been accomplished:

### ✅ Completed Components

1. **Debug Utilities Module** (`MedSAM2/sam2/debug_utils.py`)
   - Central state capture system with minimal performance overhead
   - Comprehensive visualization suite with multiple chart types
   - Easy enable/disable functionality for production use
   - Memory-efficient state management

2. **Position Encoding Debug Hooks**
   - Added to both `PositionEmbeddingSine` and `PositionEmbeddingRandom` classes
   - Captures coordinate grids, frequency components, and final embeddings
   - Visualizes spatial patterns and encoding evolution

3. **Image Encoder Debug Hooks**
   - Added to `ImageEncoder`, `FpnNeck`, and `ViTDetNeck` classes
   - Captures multi-scale features, FPN outputs, and backbone representations
   - Includes PCA analysis and spatial feature visualization

4. **Transformer & Attention Debug Hooks**
   - Added to `Attention`, `TwoWayAttentionBlock`, and `TwoWayTransformer` classes
   - Captures attention weights, query/key/value matrices, and intermediate states
   - Visualizes attention patterns across multiple heads and layers

5. **Mask Decoder Debug Hooks**
   - Added to `MaskDecoder` and `predict_masks` methods
   - Captures hypernetwork parameters, mask generation process, and IoU predictions
   - Visualizes mask generation pipeline and quality predictions

6. **Main Predictor Integration**
   - Modified `SAM2ImagePredictor` to support debug mode
   - Added `debug_mode` parameter and `return_debug_states` option
   - Seamless integration with existing API

7. **Comprehensive Documentation**
   - Complete usage guide with examples (`DEBUG_VISUALIZATION_GUIDE.md`)
   - Jupyter notebook example (`debug_example.ipynb`)
   - Best practices and troubleshooting sections

### 🔧 Key Features

**Non-Intrusive Design:**
- Debug hooks only activate when explicitly enabled
- Minimal performance impact when disabled
- Original functionality completely preserved

**Comprehensive Coverage:**
- Position encodings (sinusoidal and random Fourier)
- Image embeddings and multi-scale features
- Attention patterns and transformer states
- Mask generation and quality prediction
- Hypernetwork parameters and intermediate representations

**Easy-to-Use API:**
```python
# Simple usage
predictor = SAM2ImagePredictor(model, debug_mode=True)
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1],
    return_debug_states=True
)

# Visualize everything
from sam2.debug_utils import visualize_debug_states
visualize_debug_states(debug_states, save_path="debug_output/")
```

**Advanced Visualization:**
- Multiple visualization types (heatmaps, attention patterns, PCA analysis)
- Customizable output formats and saving options
- Jupyter notebook integration
- Interactive analysis capabilities

### 📊 Visualization Capabilities

1. **Position Encoding Visualization:**
   - Spatial frequency patterns
   - Individual encoding dimensions
   - Comparison between encoding types

2. **Attention Pattern Analysis:**
   - Multi-head attention visualization
   - Cross-attention and self-attention patterns
   - Evolution across transformer layers

3. **Image Embedding Analysis:**
   - Feature map visualization
   - PCA component analysis
   - Spatial feature distribution

4. **Mask Decoder Insights:**
   - Hypernetwork parameter visualization
   - Mask generation process analysis
   - Quality prediction patterns

### 🚀 Usage Examples

**Basic Debug Workflow:**
```python
# Enable debug mode
from sam2.debug_utils import enable_debug_mode
enable_debug_mode()

# Run inference with debug capture
predictor = SAM2ImagePredictor(model, debug_mode=True)
predictor.set_image(image)
masks, scores, logits, debug_states = predictor.predict(
    point_coords=points, point_labels=labels, return_debug_states=True
)

# Analyze and visualize
visualize_debug_states(debug_states, save_path="analysis/")
```

**Custom Analysis:**
```python
from sam2.debug_utils import SAM2Visualizer

visualizer = SAM2Visualizer()

# Analyze specific components
if 'attention_weights' in debug_states['transformer_layer_0']:
    attention_data = debug_states['transformer_layer_0']['attention_weights']['data']
    visualizer.visualize_attention_patterns(attention_data, save_path="attention_analysis/")
```

### 📋 Documentation Provided

1. **Complete Usage Guide** - Comprehensive documentation with examples
2. **Interactive Notebook** - Step-by-step tutorial with sample code
3. **API Reference** - Detailed function and parameter documentation
4. **Best Practices** - Performance optimization and troubleshooting tips

### 🎯 Ready for Interactive UI

The system is designed to support future interactive UI development:
- On-demand state capture
- Real-time visualization updates
- Selective component monitoring
- Memory-efficient state management

### 💡 Next Steps

To use the debug system:

1. **Start with the example notebook:** `MedSAM2/debug_example.ipynb`
2. **Read the comprehensive guide:** `MedSAM2/DEBUG_VISUALIZATION_GUIDE.md`
3. **Try with your own data:** Enable debug mode and capture states during inference
4. **Create custom visualizations:** Use the `SAM2Visualizer` class for specialized analysis

The debug visualization system is now fully integrated and ready to provide deep insights into MedSAM2's internal workings, making it easier to understand model behavior, debug issues, and conduct research!

```python
# Simple usage
predictor = SAM2ImagePredictor(model, debug_mode=True)
masks, scores, logits, debug_states = predictor.predict(
    point_coords=[[100, 200]], 
    point_labels=[1],
    return_debug_states=True
)

# Visualize everything
from sam2.debug_utils import visualize_debug_states
visualize_debug_states(debug_states, save_path="debug_output/")
```

```python
# Enable debug mode
from sam2.debug_utils import enable_debug_mode
enable_debug_mode()

# Run inference with debug capture
predictor = SAM2ImagePredictor(model, debug_mode=True)
predictor.set_image(image)
masks, scores, logits, debug_states = predictor.predict(
    point_coords=points, point_labels=labels, return_debug_states=True
)

# Analyze and visualize
visualize_debug_states(debug_states, save_path="analysis/")
```

```python
from sam2.debug_utils import SAM2Visualizer

visualizer = SAM2Visualizer()

# Analyze specific components
if 'attention_weights' in debug_states['transformer_layer_0']:
    attention_data = debug_states['transformer_layer_0']['attention_weights']['data']
    visualizer.visualize_attention_patterns(attention_data, save_path="attention_analysis/")
```

