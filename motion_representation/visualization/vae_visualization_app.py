"""
Web application for visualizing VAE model reconstructions and latent space.
"""

import os
import sys
import gradio as gr
import numpy as np
import torch
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add in2IN library path for InterHuman representation support
# in2IN is located at: /localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/scripts/New_2025/Download/in2IN
# It's one level up from project_root (Salsa-Agent)
in2in_path = project_root.parent / "Download" / "in2IN"
if in2in_path.exists():
    sys.path.insert(0, str(in2in_path))
else:
    # Fallback: try absolute path
    in2in_abs_path = Path("/localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/scripts/New_2025/Download/in2IN")
    if in2in_abs_path.exists():
        sys.path.insert(0, str(in2in_abs_path))

from motion_representation.models import MotionModel
# Backward compatibility
MotionVAE = MotionModel
from motion_representation.data.motion_dataset import create_dataloader
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from visualization.visualization_utils import render_combined_skeletons

# ============================================================================
# InterHuman representation support (in2IN library)
# ============================================================================
# Import in2IN functions for InterHuman visualization
# These are used when representation_type='interhuman'
try:
    from in2in.utils.plot import plot_3d_motion as plot_3d_motion_interhuman
    from in2in.utils.paramUtil import HML_KINEMATIC_CHAIN
    IN2IN_AVAILABLE = True
except ImportError as e:
    IN2IN_AVAILABLE = False
    print(f"Warning: in2IN library not available. InterHuman visualization will not work. Error: {e}")
    print(f"  Tried paths: {in2in_path}, {in2in_abs_path if 'in2in_abs_path' in locals() else 'N/A'}")


class VAEVisualizationApp:
    """Main VAE visualization application."""
    
    def __init__(self):
        self.model: Optional[MotionVAE] = None
        self.dataloader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        self.current_idx = 0
        self.config = None
        self.representation_type = 'humanml3d'  # Default representation type
        self.tsne_indices = None  # Store indices for t-SNE plot points
        self.tsne_coords = None  # Store t-SNE coordinates
        self.heatmap_hist = None  # Store histogram data for heatmap regeneration
        self.heatmap_edges = None  # Store histogram edges
        self.raw_lmdb_env = None  # Store raw LMDB environment for long sequence loading
        self.raw_lmdb_videos = None  # Store list of videos from raw LMDB
        
        # Token cluster exploration state (VQ-VAE only)
        self.code_to_samples = None  # Mapping: code_idx -> list of dataset sample indices
        self.sample_to_code = None  # Mapping: dataset sample_idx -> code_idx
        
        # Default paths
        self.default_lmdb_dir = "dataset_processed_New/lmdb_Salsa_pair/lmdb_train"
        self.default_checkpoint_dir = "motion_representation/checkpoints"
    
    def load_model(self, checkpoint_path: str) -> Tuple[str, bool]:
        """
        Load VAE model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Status message and success flag
        """
        try:
            if not os.path.exists(checkpoint_path):
                return f"Error: Checkpoint not found: {checkpoint_path}", False
            
            # Validate checkpoint file before loading
            file_size = os.path.getsize(checkpoint_path)
            if file_size == 0:
                return f"Error: Checkpoint file is empty: {checkpoint_path}", False
            
            if file_size < 1024:  # Less than 1KB is suspicious
                return f"Error: Checkpoint file is too small ({file_size} bytes). It may be corrupted or incomplete: {checkpoint_path}", False
            
            # Try to validate it's a valid zip file (PyTorch checkpoints are zip archives)
            import zipfile
            try:
                with zipfile.ZipFile(checkpoint_path, 'r') as zip_file:
                    # Check if it's a valid zip file
                    zip_file.testzip()
            except zipfile.BadZipFile:
                return f"Error: Checkpoint file is not a valid PyTorch checkpoint (corrupted zip archive): {checkpoint_path}\n\nPossible causes:\n- File was not fully written (training interrupted)\n- File was corrupted during transfer\n- File is not a PyTorch checkpoint\n\nPlease check the file or try loading a different checkpoint.", False
            except Exception as zip_error:
                # If it's not a zip file at all, that's also a problem
                return f"Error: Checkpoint file validation failed: {str(zip_error)}\n\nFile: {checkpoint_path}", False
            
            # Now try to load the checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except RuntimeError as e:
                if "failed reading zip archive" in str(e) or "central directory" in str(e):
                    return f"Error: Checkpoint file is corrupted or incomplete: {checkpoint_path}\n\nError details: {str(e)}\n\nPossible causes:\n- File was not fully written (training interrupted)\n- File was corrupted during transfer\n- Disk space issue during checkpoint saving\n\nPlease check:\n1. File size: {file_size:,} bytes\n2. Try loading 'best_checkpoint.pth' or another checkpoint file\n3. Check if training completed successfully", False
                else:
                    raise  # Re-raise if it's a different RuntimeError
            self.config = checkpoint['config']
            
            # Detect representation type from config or infer from input_dim
            self.representation_type = self.config.get('representation_type', None)
            if self.representation_type is None:
                # Infer from input_dim
                input_dim = self.config.get('input_dim', 263)
                if input_dim == 263:
                    self.representation_type = 'humanml3d'
                elif input_dim == 262:
                    self.representation_type = 'interhuman'
                elif input_dim == 4:
                    self.representation_type = 'relationship'  # [w, z, x, z] - quaternion components + position
                elif input_dim == 3:
                    # Legacy support for old relationship models (should be migrated to 4D)
                    self.representation_type = 'relationship'
                    print(f"Warning: input_dim=3 detected. Relationship features should use input_dim=4 (quaternion components).")
                else:
                    self.representation_type = 'humanml3d'  # Default fallback
                    print(f"Warning: Unknown input_dim {input_dim}, defaulting to 'humanml3d'")
            
            self.model = MotionModel(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                latent_dim=self.config['latent_dim'],
                seq_len=self.config['seq_len'],
                dropout=self.config['dropout'],
                encoder_type=self.config.get('encoder_type', 'gru'),
                decoder_type=self.config.get('decoder_type', 'gru'),
                num_heads=self.config.get('num_heads', 8),
                ff_size=self.config.get('ff_size', 2048),
                activation=self.config.get('activation', 'gelu'),
                use_vae=self.config.get('use_vae', False),
                use_vqvae=self.config.get('use_vqvae', False),
                nb_code=self.config.get('nb_code', self.config.get('vq_codebook_size', 512)),  # Support both names
                quantizer=self.config.get('quantizer', self.config.get('vq_quantizer', 'ema_reset')),
                vq_mu=self.config.get('vq_mu', self.config.get('vq_ema_mu', 0.99)),
                vq_beta=self.config.get('vq_beta', self.config.get('vq_commitment_cost', 1.0)),
            ).to(self.device)
            
            # Load state dict with flexible handling for VQVAE and other architectures
            state_dict = checkpoint['model_state_dict']
            model_state_dict = self.model.state_dict()
            
            # Filter out keys that don't exist in current model (for backward compatibility)
            filtered_state_dict = {}
            missing_keys = []
            unexpected_keys = []
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        unexpected_keys.append(f"{key} (shape mismatch: {model_state_dict[key].shape} vs {value.shape})")
                else:
                    unexpected_keys.append(key)
            
            for key in model_state_dict:
                if key not in filtered_state_dict:
                    missing_keys.append(key)
            
            self.model.load_state_dict(filtered_state_dict, strict=False)
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint (using defaults): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            
            self.model.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            
            # Build detailed config string
            config_str = "Model Configuration:\n"
            config_str += f"  Encoder: {self.config.get('encoder_type', 'gru')}\n"
            config_str += f"  Decoder: {self.config.get('decoder_type', 'gru')}\n"
            config_str += f"  Use VAE: {self.config.get('use_vae', False)}\n"
            config_str += f"  Use VQ-VAE: {self.config.get('use_vqvae', False)}\n"
            config_str += f"  Representation Type: {self.representation_type}\n"
            config_str += f"  Input Dim: {self.config.get('input_dim', 263)}\n"
            config_str += f"  Hidden Dim: {self.config.get('hidden_dim', 512)}\n"
            config_str += f"  Num Layers: {self.config.get('num_layers', 2)}\n"
            config_str += f"  Latent Dim: {self.config.get('latent_dim', 512)}\n"
            config_str += f"  Seq Len: {self.config.get('seq_len', 20)}\n"
            config_str += f"  Dropout: {self.config.get('dropout', 0.1)}\n"
            if self.config.get('encoder_type') == 'transformer' or self.config.get('decoder_type') == 'transformer':
                config_str += f"  Num Heads: {self.config.get('num_heads', 8)}\n"
                config_str += f"  FF Size: {self.config.get('ff_size', 2048)}\n"
                config_str += f"  Activation: {self.config.get('activation', 'gelu')}\n"
            if self.config.get('use_vqvae', False):
                config_str += f"  VQ Codebook Size: {self.config.get('nb_code', self.config.get('vq_codebook_size', 512))}\n"
                config_str += f"  VQ Quantizer: {self.config.get('quantizer', self.config.get('vq_quantizer', 'ema_reset'))}\n"
                config_str += f"  VQ Mu (EMA): {self.config.get('vq_mu', self.config.get('vq_ema_mu', 0.99))}\n"
                config_str += f"  VQ Beta: {self.config.get('vq_beta', self.config.get('vq_commitment_cost', 1.0))}\n"
            
            status = f"Model loaded successfully!\n\n"
            status += config_str + "\n"
            status += f"Training Info:\n"
            status += f"  Epoch: {epoch}\n"
            status += f"  Loss: {loss:.4f}\n"
            status += f"  Device: {self.device}\n"
            status += f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            
            # Add model architecture
            status += f"\nModel Architecture:\n"
            # Use a simple string representation, limit lines for readability
            arch_str = str(self.model)
            # Limit to first 50 lines to avoid overwhelming output
            arch_lines = arch_str.split('\n')
            if len(arch_lines) > 50:
                arch_str = '\n'.join(arch_lines[:50]) + f"\n... ({len(arch_lines) - 50} more lines)"
            status += arch_str
            
            return status, True
        except Exception as e:
            import traceback
            error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
            return error_msg, False
    
    def load_dataset(self, lmdb_dir: str, is_MDM: bool = True, train_relationship: bool = False) -> Tuple[str, int]:
        """
        Load dataset.
        
        Args:
            lmdb_dir: Path to LMDB directory
            is_MDM: Whether data is in MDM format
            train_relationship: If True, load relationship features (4D) instead of motions (263D)
                                (Deprecated: use representation_type from model instead)
            
        Returns:
            Status message and total samples
        """
        try:
            if not os.path.exists(lmdb_dir):
                return f"Error: LMDB path does not exist: {lmdb_dir}", 0
            
            # Use representation_type from model if available, otherwise infer from train_relationship or model's input_dim
            if self.model is not None and hasattr(self, 'representation_type') and self.representation_type:
                representation_type = self.representation_type
            elif train_relationship:
                representation_type = 'relationship'
            elif self.model is not None and hasattr(self.model, 'input_dim'):
                # Infer from model's input_dim as fallback
                input_dim = self.model.input_dim
                if input_dim == 263:
                    representation_type = 'humanml3d'
                elif input_dim == 262:
                    representation_type = 'interhuman'
                elif input_dim == 4:
                    representation_type = 'relationship'  # [w, z, x, z]
                elif input_dim == 3:
                    # Legacy support for old relationship models (should be migrated to 4D)
                    representation_type = 'relationship'
                    print(f"Warning: Detected input_dim=3. Relationship features now use input_dim=4 (quaternion components).")
                else:
                    representation_type = 'humanml3d'  # Default fallback
                    print(f"Warning: Unknown input_dim {input_dim}, defaulting to 'humanml3d'")
            else:
                representation_type = 'humanml3d'  # Default
            
            # Create args object
            class DataLoaderArgs:
                def __init__(self, is_MDM_val, device_val):
                    self.is_MDM = is_MDM_val
                    self.device = device_val
            
            data_args = DataLoaderArgs(is_MDM, self.device)
            
            self.dataloader = create_dataloader(
                args=data_args,
                lmdb_dir=lmdb_dir,
                window_size=20,
                stride=10,
                batch_size=32,
                shuffle=False,
                num_workers=2,
                use_both_roles=True,
                normalize=True,  # Use normalized data (default)
                representation_type=representation_type,  # Use representation_type from model
            )
            
            total_samples = len(self.dataloader.dataset)
            
            # Verify that the dataloader is using the correct representation_type
            if hasattr(self.dataloader.dataset, 'representation_type'):
                actual_rep_type = self.dataloader.dataset.representation_type
                if actual_rep_type != representation_type:
                    print(f"Warning: Representation type mismatch! Requested: {representation_type}, Actual: {actual_rep_type}")
            
            # Also verify feature_dim matches
            if hasattr(self.dataloader.dataset, 'feature_dim'):
                actual_feature_dim = self.dataloader.dataset.feature_dim
                expected_feature_dim = {'humanml3d': 263, 'interhuman': 262, 'relationship': 4}.get(representation_type, 263)
                if actual_feature_dim != expected_feature_dim:
                    print(f"Warning: Feature dimension mismatch! Expected: {expected_feature_dim} for {representation_type}, Actual: {actual_feature_dim}")
            
            status = f"Dataset loaded successfully!\n"
            status += f"Representation Type: {representation_type}\n"
            if hasattr(self.dataloader.dataset, 'feature_dim'):
                status += f"Feature Dim: {self.dataloader.dataset.feature_dim}\n"
            status += f"Total samples: {total_samples}\n"
            status += f"LMDB: {lmdb_dir}"
            
            return status, total_samples
        except Exception as e:
            import traceback
            error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
            return error_msg, 0
    
    def _denormalize_motion(self, motion: np.ndarray) -> np.ndarray:
        """
        Denormalize motion data if normalization was applied.
        
        Args:
            motion: Normalized motion array (seq_len, 263) or (batch, seq_len, 263)
            
        Returns:
            Denormalized motion array
        """
        if self.dataloader is None or self.dataloader.dataset is None:
            return motion
        
        dataset = self.dataloader.dataset
        if not dataset.normalize or dataset.mean is None or dataset.std is None:
            return motion
        
        # Denormalize: motion * std + mean
        # Handle both (seq_len, 263) and (batch, seq_len, 263) shapes
        if motion.ndim == 2:
            # (seq_len, 263)
            motion_denorm = motion * dataset.std.numpy() + dataset.mean.numpy()
        else:
            # (batch, seq_len, 263)
            motion_denorm = motion * dataset.std.numpy()[None, None, :] + dataset.mean.numpy()[None, None, :]
        
        return motion_denorm
    
    def _extract_keypoints_from_motion(self, motion: np.ndarray) -> np.ndarray:
        """
        Extract 3D keypoints from motion based on representation type.
        
        Args:
            motion: Denormalized motion array
                  - HumanML3D: (seq_len, 263)
                  - InterHuman: (seq_len, 262) where seq_len is typically 19
                  - Relationship: (seq_len, 4) - not supported for 3D visualization
        
        Returns:
            keypoints: (seq_len, 22, 3) numpy array of 3D joint positions
        """
        if self.representation_type == 'humanml3d':
            # Use existing recover_from_ric for HumanML3D
            keypoints = recover_from_ric(
                torch.from_numpy(motion).float().to(self.device),
                22
            ).cpu().numpy()
        elif self.representation_type == 'interhuman':
            # InterHuman: first 66 dims (22*3) contain joint positions
            # motion shape is (seq_len, 262)
            seq_len = motion.shape[0]
            # Extract first 66 dimensions and reshape to (seq_len, 22, 3)
            keypoints = motion[:, :66].reshape(seq_len, 22, 3)
        elif self.representation_type == 'relationship':
            # Relationship features cannot be converted to 3D keypoints
            raise ValueError("Relationship features (4D) cannot be visualized as 3D keypoints. Use time series plots instead.")
        else:
            raise ValueError(f"Unknown representation type: {self.representation_type}")
        
        return keypoints
    
    def _get_kinematic_chain(self):
        """Get the appropriate kinematic chain for visualization based on representation type."""
        if self.representation_type == 'humanml3d':
            return t2m_kinematic_chain
        elif self.representation_type == 'interhuman':
            if not IN2IN_AVAILABLE:
                raise ImportError("in2IN library is required for InterHuman visualization. Please install it.")
            return HML_KINEMATIC_CHAIN
        else:
            raise ValueError(f"Unknown representation type: {self.representation_type}")
    
    def _plot_3d_motion(self, save_path: str, kinematic_chain, keypoints: np.ndarray, 
                       title: str, fps: int = 20, radius: int = 4):
        """
        Plot 3D motion using the appropriate visualization function based on representation type.
        
        Args:
            save_path: Path to save the video
            kinematic_chain: Kinematic chain for skeleton structure
            keypoints: (seq_len, 22, 3) numpy array of 3D joint positions
            title: Title for the visualization
            fps: Frames per second
            radius: Visualization radius
        """
        if self.representation_type == 'humanml3d':
            # Use existing plot_3d_motion for HumanML3D
            plot_3d_motion(
                save_path,
                kinematic_chain,
                keypoints,
                title=title,
                fps=fps,
                radius=radius
            )
        elif self.representation_type == 'interhuman':
            # Use in2IN's plot_3d_motion for InterHuman
            if not IN2IN_AVAILABLE:
                raise ImportError("in2IN library is required for InterHuman visualization. Please install it.")
            # in2IN's plot_3d_motion expects mp_joints as a list
            plot_3d_motion_interhuman(
                save_path=save_path,
                kinematic_tree=kinematic_chain,
                mp_joints=[keypoints],  # List of (seq_len, 22, 3) arrays
                title=title,
                fps=fps,
                radius=radius
            )
        else:
            raise ValueError(f"Unknown representation type: {self.representation_type}")
    
    def visualize_reconstruction(self, idx: int) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize original vs reconstructed motion.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (original_video_path, reconstructed_video_path, info_string)
        """
        if self.model is None:
            return None, None, "Error: Model not loaded. Please load a model first."
        
        if self.dataloader is None:
            return None, None, "Error: Dataset not loaded. Please load dataset first."
        
        try:
            if idx < 0 or idx >= len(self.dataloader.dataset):
                return None, None, f"Error: Index {idx} out of range (0-{len(self.dataloader.dataset)-1})"
            
            # Get sample (already normalized if dataset.normalize=True)
            motion = self.dataloader.dataset[idx]  # (20, 263)
            motion = motion.unsqueeze(0).to(self.device)  # (1, 20, 263)
            
            # Reconstruct - handle different model types
            with torch.no_grad():
                forward_result = self.model(motion)
                
                # VQ-VAE returns: (recon_x, z, commit_loss, perplexity, code_idx)
                # VAE/Vanilla returns: (recon_x, mean, logvar, z)
                if self.model.use_vqvae:
                    recon_motion, z, commit_loss, perplexity, code_idx = forward_result
                    mean = None
                    logvar = None
                else:
                    recon_motion, mean, logvar, z = forward_result
            
            # Convert to numpy
            original = motion[0].cpu().numpy()  # (20, 263) - normalized
            reconstructed = recon_motion[0].cpu().numpy()  # (20, 263) - normalized
            
            # Denormalize before computing metrics and visualization
            original_denorm = self._denormalize_motion(original)  # (seq_len, input_dim)
            reconstructed_denorm = self._denormalize_motion(reconstructed)  # (seq_len, input_dim)
            
            # Compute metrics on denormalized data (more meaningful)
            mse = np.mean((original_denorm - reconstructed_denorm) ** 2)
            mae = np.mean(np.abs(original_denorm - reconstructed_denorm))
            
            # Convert to 3D keypoints (use denormalized data)
            original_keypoints = self._extract_keypoints_from_motion(original_denorm)
            reconstructed_keypoints = self._extract_keypoints_from_motion(reconstructed_denorm)
            
            # Get appropriate kinematic chain
            kinematic_chain = self._get_kinematic_chain()
            
            # Create videos
            original_path = os.path.join(self.temp_dir, f"original_vae_{idx}.mp4")
            reconstructed_path = os.path.join(self.temp_dir, f"reconstructed_vae_{idx}.mp4")
            
            self._plot_3d_motion(
                original_path,
                kinematic_chain,
                original_keypoints,
                title=f"Original (Sample {idx})",
                fps=20,
                radius=4
            )
            
            self._plot_3d_motion(
                reconstructed_path,
                kinematic_chain,
                reconstructed_keypoints,
                title=f"Reconstructed (MSE: {mse:.4f})",
                fps=20,
                radius=4
            )
            
            # Build info string based on model type
            info = f"Sample Index: {idx}\n"
            info += f"MSE: {mse:.6f}\n"
            info += f"MAE: {mae:.6f}\n"
            info += f"Latent z shape: {z.shape}\n"
            
            if self.model.use_vqvae:
                info += f"VQ-VAE Commit Loss: {commit_loss.item():.6f}\n"
                info += f"VQ-VAE Perplexity: {perplexity.item():.4f}\n"
                info += f"VQ Code Index shape: {code_idx.shape}\n"
                info += f"Unique codes used: {len(torch.unique(code_idx))}"
            elif self.model.use_vae:
                info += f"Latent mean norm: {torch.norm(mean).item():.4f}\n"
                info += f"Latent std: {torch.exp(0.5 * logvar).mean().item():.4f}"
            else:
                info += f"Latent norm: {torch.norm(z).item():.4f}\n"
                info += f"Model type: Vanilla Autoencoder"
            
            return original_path, reconstructed_path, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing reconstruction: {str(e)}\n{traceback.format_exc()}"
            return None, None, error_msg
    
    def visualize_latent_space(self, num_samples: int, use_kmeans: bool = False, n_clusters: int = 5) -> Tuple[Optional[dict], Optional[dict], str]:
        """
        Visualize latent space using t-SNE with interactive plotly plot and heatmap.
        
        Args:
            num_samples: Number of samples to encode and visualize
            use_kmeans: Whether to apply k-means clustering and color by clusters (only if not VQ-VAE)
            n_clusters: Number of clusters for k-means
            
        Returns:
            Tuple of (t-SNE_plotly_figure_dict, heatmap_plotly_figure_dict, info_string)
        """
        if self.model is None:
            return None, None, "Error: Model not loaded. Please load a model first."
        
        if self.dataloader is None:
            return None, None, "Error: Dataset not loaded. Please load dataset first."
        
        try:
            import plotly.graph_objects as go
            from sklearn.manifold import TSNE
            from sklearn.cluster import KMeans
            
            total_samples = len(self.dataloader.dataset)
            num_samples = min(num_samples, total_samples)
            
            # Randomly sample indices
            indices = np.random.choice(total_samples, num_samples, replace=False)
            self.tsne_indices = indices  # Store for click handling
            
            info = f"Encoding {num_samples} samples...\n"
            
            # Encode samples
            latents = []
            vq_code_indices = []  # Store VQ code indices for VQ-VAE
            self.model.eval()
            
            with torch.no_grad():
                for idx in tqdm(indices, desc="Encoding samples"):
                    motion = self.dataloader.dataset[idx]  # (20, 263)
                    motion = motion.unsqueeze(0).to(self.device)  # (1, 20, 263)
                    
                    # Encode to latent using the encode() method
                    # VQ-VAE returns: (z, commit_loss, perplexity, code_idx)
                    # VAE/Vanilla returns: (z, mean, logvar)
                    encode_result = self.model.encode(motion)
                    
                    if self.model.use_vqvae:
                        z, commit_loss, perplexity, code_idx = encode_result
                        vq_code_indices.append(code_idx[0].item())  # Store VQ code index
                    else:
                        z, mean, logvar = encode_result
                        vq_code_indices.append(None)
                    
                    latents.append(z[0].cpu().numpy())  # (latent_dim,)
            
            latents = np.array(latents)  # (num_samples, latent_dim)
            vq_code_indices = np.array(vq_code_indices) if any(x is not None for x in vq_code_indices) else None
            
            info += f"Latent shape: {latents.shape}\n"
            
            # Determine coloring: VQ code indices take precedence, then k-means, then single color
            color_labels = None
            color_title = "Color"
            use_vq_colors = False
            
            if self.model.use_vqvae and vq_code_indices is not None:
                # Use VQ code indices for coloring
                color_labels = vq_code_indices
                color_title = "VQ Code Index"
                use_vq_colors = True
                info += f"Using VQ code indices for coloring (unique codes: {len(np.unique(vq_code_indices))})\n"
            elif use_kmeans:
                info += f"Applying k-means clustering (k={n_clusters})...\n"
                n_clusters = min(n_clusters, num_samples)  # Can't have more clusters than samples
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                color_labels = kmeans.fit_predict(latents)
                color_title = "Cluster"
                info += f"Clustering complete. Cluster sizes: {np.bincount(color_labels)}\n"
            
            info += f"Computing t-SNE...\n"
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_samples - 1))
            latents_2d = tsne.fit_transform(latents)
            self.tsne_coords = latents_2d  # Store for click handling
            
            # Create t-SNE plot with colors
            fig_tsne = go.Figure()
            
            if color_labels is not None:
                # Color by VQ code indices or k-means clusters
                fig_tsne.add_trace(go.Scatter(
                    x=latents_2d[:, 0],
                    y=latents_2d[:, 1],
                    mode='markers',
                    name='Samples',
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        color=color_labels,
                        colorscale='viridis' if use_vq_colors else 'viridis',
                        colorbar=dict(title=color_title),
                        line=dict(width=0.5, color='white'),
                        showscale=True
                    ),
                    text=[f'Sample {idx} ({color_title}: {color_labels[i]})' for i, idx in enumerate(indices)],
                    hovertemplate='<b>%{text}</b><br>' +
                                't-SNE 1: %{x:.2f}<br>' +
                                't-SNE 2: %{y:.2f}<extra></extra>',
                    customdata=indices  # Store original indices for click handling
                ))
            else:
                # Single color
                fig_tsne.add_trace(go.Scatter(
                    x=latents_2d[:, 0],
                    y=latents_2d[:, 1],
                    mode='markers',
                    name='Samples',
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        color='blue',
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f'Sample {idx}' for idx in indices],
                    hovertemplate='<b>%{text}</b><br>' +
                                't-SNE 1: %{x:.2f}<br>' +
                                't-SNE 2: %{y:.2f}<extra></extra>',
                    customdata=indices  # Store original indices for click handling
                ))
            
            fig_tsne.update_layout(
                title=f'Interactive Latent Space t-SNE<br>{num_samples} samples' + 
                      (f' | Colored by {color_title}' if color_labels is not None else ''),
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2',
                hovermode='closest',
                width=700,
                height=600,
                template='plotly_white',
                showlegend=color_labels is not None,
                clickmode='event+select'
            )
            
            # Create heatmap (density plot) with Gaussian smoothing
            # Create 2D histogram/density
            x_bins = np.linspace(latents_2d[:, 0].min(), latents_2d[:, 0].max(), 100)
            y_bins = np.linspace(latents_2d[:, 1].min(), latents_2d[:, 1].max(), 100)
            
            # Compute 2D histogram
            H, x_edges, y_edges = np.histogram2d(
                latents_2d[:, 0], latents_2d[:, 1],
                bins=[x_bins, y_bins]
            )
            
            # Store histogram data for regeneration with different sigma
            self.heatmap_hist = H.T
            self.heatmap_edges = (x_edges, y_edges)
            
            # Create initial heatmap with default sigma=2.0
            fig_heatmap = self._create_smoothed_heatmap(H.T, x_edges, y_edges, sigma=2.0, num_samples=num_samples)
            
            # Store indices for click handling
            self.tsne_indices = indices
            
            # Add statistics
            info += f"t-SNE complete!\n"
            info += f"Latent dimension: {latents.shape[1]}\n"
            info += f"Latent mean: {latents.mean(axis=0).mean():.4f}\n"
            info += f"Latent std: {latents.std(axis=0).mean():.4f}\n"
            info += f"Latent range: [{latents.min():.4f}, {latents.max():.4f}]"
            if use_vq_colors:
                unique_codes = len(np.unique(vq_code_indices))
                info += f"\nUnique VQ codes used: {unique_codes}"
            elif use_kmeans:
                info += f"\nNumber of clusters: {n_clusters}"
            
            return fig_tsne, fig_heatmap, info
            
        except ImportError as e:
            missing = str(e)
            if 'plotly' in missing.lower():
                return None, None, "Error: plotly not available. Please install: pip install plotly"
            return None, None, f"Error: sklearn not available. Please install: pip install scikit-learn"
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing latent space: {str(e)}\n{traceback.format_exc()}"
            return None, None, error_msg
    
    def _create_smoothed_heatmap(self, H, x_edges, y_edges, sigma=2.0, num_samples=100):
        """
        Create a smoothed heatmap using Gaussian filter.
        
        Args:
            H: 2D histogram array (already transposed)
            x_edges: X-axis bin edges
            y_edges: Y-axis bin edges
            sigma: Gaussian filter sigma (smoothing parameter)
            num_samples: Number of samples for title
            
        Returns:
            Plotly figure with smoothed heatmap
        """
        import plotly.graph_objects as go
        
        try:
            from scipy.ndimage import gaussian_filter
            # Apply Gaussian smoothing
            H_smooth = gaussian_filter(H, sigma=sigma)
        except ImportError:
            # Fallback if scipy not available - use unsmoothed
            H_smooth = H
        
        # Create heatmap
        fig_heatmap = go.Figure()
        fig_heatmap.add_trace(go.Heatmap(
            z=H_smooth,
            x=x_edges[:-1],
            y=y_edges[:-1],
            colorscale='Hot',
            colorbar=dict(title="Density"),
            hovertemplate='Density: %{z:.2f}<br>' +
                        't-SNE 1: %{x:.2f}<br>' +
                        't-SNE 2: %{y:.2f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title=f'Latent Space Density Heatmap (σ={sigma:.1f})<br>{num_samples} samples',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            width=700,
            height=600,
            template='plotly_white'
        )
        
        return fig_heatmap
    
    def find_nearest_sample(self, click_x: float, click_y: float) -> Optional[int]:
        """
        Find the nearest sample to a clicked point in t-SNE space.
        
        Args:
            click_x: X coordinate of clicked point
            click_y: Y coordinate of clicked point
            
        Returns:
            Sample index or None if no data available
        """
        if self.tsne_coords is None or self.tsne_indices is None:
            return None
        
        # Find nearest point
        distances = np.sqrt((self.tsne_coords[:, 0] - click_x)**2 + 
                           (self.tsne_coords[:, 1] - click_y)**2)
        nearest_idx = np.argmin(distances)
        
        return int(self.tsne_indices[nearest_idx])
    
    def navigate_sample(self, direction: str, current_idx: int, total_samples: int) -> int:
        """
        Navigate to next/previous sample.
        
        Args:
            direction: 'next' or 'prev'
            current_idx: Current sample index
            total_samples: Total number of samples
            
        Returns:
            New sample index
        """
        if direction == 'next':
            new_idx = min(current_idx + 1, total_samples - 1)
        elif direction == 'prev':
            new_idx = max(current_idx - 1, 0)
        else:
            new_idx = current_idx
        
        self.current_idx = new_idx
        return new_idx
    
    def analyze_codebook(self, num_samples: int = 1000) -> Tuple[Optional[dict], str]:
        """
        Comprehensive codebook analysis for debugging collapse.
        
        Args:
            num_samples: Number of samples to use for analysis
            
        Returns:
            Tuple of (plotly_figure_dict, info_string)
        """
        if self.model is None:
            return None, "Error: Model not loaded. Please load a model first."
        
        if not self.model.use_vqvae:
            return None, "Error: Model is not a VQ-VAE. Codebook analysis only available for VQ-VAE models."
        
        if self.dataloader is None:
            return None, "Error: Dataset not loaded. Please load dataset first."
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            from sklearn.manifold import TSNE
            from sklearn.metrics import pairwise_distances
            
            # Get quantizer
            quantizer = self.model.vq_layer.quantizer
            
            # Get codebook - handle different quantizer types
            if hasattr(quantizer, 'codebook'):
                codebook = quantizer.codebook.detach().cpu().numpy()  # (nb_code, code_dim)
            elif hasattr(quantizer, 'embedding'):
                # For Quantizer class (original), it might use 'embedding'
                codebook = quantizer.embedding.weight.detach().cpu().numpy()
            else:
                return None, f"Error: Cannot find codebook in quantizer. Quantizer type: {type(quantizer).__name__}"
            
            nb_code, code_dim = codebook.shape
            
            info = f"Codebook Analysis\n"
            info += f"{'='*60}\n"
            info += f"Codebook size: {nb_code}\n"
            info += f"Code dimension: {code_dim}\n"
            info += f"Quantizer type: {type(quantizer).__name__}\n"
            if hasattr(quantizer, 'init'):
                info += f"Codebook initialized: {quantizer.init}\n"
            else:
                info += f"Codebook initialized: Unknown (quantizer doesn't track init state)\n"
            
            # 1. Codebook diversity metrics
            info += f"\n{'='*60}\n"
            info += f"1. CODEBOOK DIVERSITY METRICS\n"
            info += f"{'='*60}\n"
            
            # Compute pairwise distances
            distances = pairwise_distances(codebook, metric='euclidean')
            # Remove diagonal (self-distances)
            mask = ~np.eye(nb_code, dtype=bool)
            pairwise_dists = distances[mask]
            
            avg_distance = np.mean(pairwise_dists)
            min_distance = np.min(pairwise_dists)
            max_distance = np.max(pairwise_dists)
            std_distance = np.std(pairwise_dists)
            
            info += f"Average pairwise distance: {avg_distance:.4f}\n"
            info += f"Min pairwise distance: {min_distance:.4f}\n"
            info += f"Max pairwise distance: {max_distance:.4f}\n"
            info += f"Std pairwise distance: {std_distance:.4f}\n"
            
            # Check for collapsed codes (very close to each other)
            close_threshold = avg_distance * 0.1  # 10% of average distance
            close_pairs = np.sum(pairwise_dists < close_threshold)
            info += f"\nCollapse indicator: {close_pairs} pairs with distance < {close_threshold:.4f} ({100*close_pairs/len(pairwise_dists):.2f}%)\n"
            
            # Codebook statistics
            codebook_mean = np.mean(codebook, axis=0)
            codebook_std = np.std(codebook, axis=0)
            codebook_norm = np.linalg.norm(codebook, axis=1)
            
            info += f"\nCodebook vector statistics:\n"
            info += f"  Mean norm: {np.mean(codebook_norm):.4f} ± {np.std(codebook_norm):.4f}\n"
            info += f"  Min norm: {np.min(codebook_norm):.4f}\n"
            info += f"  Max norm: {np.max(codebook_norm):.4f}\n"
            info += f"  Mean per-dimension: {np.mean(codebook_mean):.4f} ± {np.mean(codebook_std):.4f}\n"
            
            # 2. Code usage analysis
            info += f"\n{'='*60}\n"
            info += f"2. CODE USAGE ANALYSIS\n"
            info += f"{'='*60}\n"
            
            total_samples = len(self.dataloader.dataset)
            num_samples = min(num_samples, total_samples)
            
            # Sample random indices
            indices = np.random.choice(total_samples, num_samples, replace=False)
            
            # Collect code indices
            code_indices_list = []
            encoder_outputs = []
            self.model.eval()
            
            with torch.no_grad():
                for idx in tqdm(indices[:num_samples], desc="Analyzing code usage"):
                    motion = self.dataloader.dataset[idx]
                    motion = motion.unsqueeze(0).to(self.device)
                    
                    # Encode
                    encoded = self.model.encoder(motion)  # (1, latent_dim)
                    encoder_outputs.append(encoded[0].cpu().numpy())
                    
                    # Quantize
                    code_idx = quantizer.quantize(encoded)
                    code_indices_list.append(code_idx[0].item())
            
            code_indices = np.array(code_indices_list)
            encoder_outputs = np.array(encoder_outputs)  # (num_samples, code_dim)
            
            # Count code usage
            unique_codes, counts = np.unique(code_indices, return_counts=True)
            usage_freq = np.zeros(nb_code)
            usage_freq[unique_codes] = counts
            
            # Compute perplexity
            prob = usage_freq / (np.sum(usage_freq) + 1e-8)
            perplexity = np.exp(-np.sum(prob * np.log(prob + 1e-7)))
            
            info += f"Analyzed {num_samples} samples\n"
            info += f"Unique codes used: {len(unique_codes)} / {nb_code} ({100*len(unique_codes)/nb_code:.2f}%)\n"
            info += f"Perplexity: {perplexity:.2f} (max: {nb_code:.2f})\n"
            info += f"Unused codes: {nb_code - len(unique_codes)}\n"
            
            # Top and bottom used codes
            sorted_indices = np.argsort(usage_freq)[::-1]
            top_k = min(10, len(unique_codes))
            info += f"\nTop {top_k} most used codes:\n"
            for i in range(top_k):
                code_idx = sorted_indices[i]
                info += f"  Code {code_idx}: {usage_freq[code_idx]:.0f} times ({100*usage_freq[code_idx]/num_samples:.2f}%)\n"
            
            # 3. Encoder output vs codebook analysis
            info += f"\n{'='*60}\n"
            info += f"3. ENCODER OUTPUT vs CODEBOOK\n"
            info += f"{'='*60}\n"
            
            # Compute distances from encoder outputs to their assigned codes
            assigned_codes = codebook[code_indices]  # (num_samples, code_dim)
            distances_to_assigned = np.linalg.norm(encoder_outputs - assigned_codes, axis=1)
            
            # Also compute distance to nearest codebook entry (might be different)
            distances_to_codebook = pairwise_distances(encoder_outputs, codebook, metric='euclidean')
            nearest_code_indices = np.argmin(distances_to_codebook, axis=1)
            distances_to_nearest = distances_to_codebook[np.arange(len(encoder_outputs)), nearest_code_indices]
            
            # Check if assigned code is actually the nearest
            matches_nearest = (code_indices == nearest_code_indices).sum()
            
            info += f"Distance to assigned code:\n"
            info += f"  Mean: {np.mean(distances_to_assigned):.4f} ± {np.std(distances_to_assigned):.4f}\n"
            info += f"  Min: {np.min(distances_to_assigned):.4f}\n"
            info += f"  Max: {np.max(distances_to_assigned):.4f}\n"
            
            info += f"\nDistance to nearest codebook entry:\n"
            info += f"  Mean: {np.mean(distances_to_nearest):.4f} ± {np.std(distances_to_nearest):.4f}\n"
            info += f"  Min: {np.min(distances_to_nearest):.4f}\n"
            info += f"  Max: {np.max(distances_to_nearest):.4f}\n"
            
            info += f"\nAssigned code matches nearest: {matches_nearest}/{num_samples} ({100*matches_nearest/num_samples:.2f}%)\n"
            
            # 4. Create visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Codebook t-SNE (colored by usage)',
                    'Code Usage Distribution',
                    'Distance to Assigned Code',
                    'Encoder Output vs Codebook t-SNE'
                ),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # 4a. Codebook t-SNE
            if nb_code > 1:
                tsne_codebook = TSNE(n_components=2, random_state=42, perplexity=min(30, nb_code-1))
                codebook_2d = tsne_codebook.fit_transform(codebook)
                
                # Color by usage frequency
                fig.add_trace(
                    go.Scatter(
                        x=codebook_2d[:, 0],
                        y=codebook_2d[:, 1],
                        mode='markers',
                        name='Codebook',
                        marker=dict(
                            size=8,
                            opacity=0.7,
                            color=usage_freq,
                            colorscale='Viridis',
                            colorbar=dict(title="Usage Count", x=1.15),
                            showscale=True,
                            line=dict(width=0.5, color='white')
                        ),
                        text=[f'Code {i}<br>Usage: {usage_freq[i]:.0f}' for i in range(nb_code)],
                        hovertemplate='<b>%{text}</b><br>' +
                                    't-SNE 1: %{x:.2f}<br>' +
                                    't-SNE 2: %{y:.2f}<extra></extra>',
                    ),
                    row=1, col=1
                )
            
            # 4b. Code usage distribution
            sorted_usage = np.sort(usage_freq)[::-1]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sorted_usage))),
                    y=sorted_usage,
                    name='Usage',
                    marker_color='steelblue',
                    hovertemplate='Code rank: %{x}<br>Usage: %{y:.0f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 4c. Distance to assigned code histogram
            fig.add_trace(
                go.Histogram(
                    x=distances_to_assigned,
                    name='Distance',
                    marker_color='coral',
                    nbinsx=50,
                    hovertemplate='Distance: %{x:.4f}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4d. Encoder outputs vs codebook t-SNE
            if num_samples > 1:
                # Combine encoder outputs and codebook for joint t-SNE
                combined = np.vstack([encoder_outputs, codebook])
                tsne_combined = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
                combined_2d = tsne_combined.fit_transform(combined)
                
                encoder_2d = combined_2d[:num_samples]
                codebook_2d_combined = combined_2d[num_samples:]
                
                # Plot encoder outputs
                fig.add_trace(
                    go.Scatter(
                        x=encoder_2d[:, 0],
                        y=encoder_2d[:, 1],
                        mode='markers',
                        name='Encoder Outputs',
                        marker=dict(
                            size=6,
                            opacity=0.5,
                            color='blue',
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate='Encoder Output<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=2
                )
                
                # Plot codebook (smaller, different color)
                fig.add_trace(
                    go.Scatter(
                        x=codebook_2d_combined[:, 0],
                        y=codebook_2d_combined[:, 1],
                        mode='markers',
                        name='Codebook',
                        marker=dict(
                            size=8,
                            opacity=0.8,
                            color='red',
                            symbol='x',
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate='Codebook Entry<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Codebook Analysis Dashboard',
                height=1000,
                width=1400,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
            fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
            fig.update_xaxes(title_text="Code Rank", row=1, col=2)
            fig.update_yaxes(title_text="Usage Count", row=1, col=2)
            fig.update_xaxes(title_text="Distance", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_xaxes(title_text="t-SNE 1", row=2, col=2)
            fig.update_yaxes(title_text="t-SNE 2", row=2, col=2)
            
            return fig, info
            
        except ImportError as e:
            missing = str(e)
            if 'plotly' in missing.lower():
                return None, "Error: plotly not available. Please install: pip install plotly"
            return None, f"Error: sklearn not available. Please install: pip install scikit-learn"
        except Exception as e:
            import traceback
            error_msg = f"Error analyzing codebook: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
    
    def load_raw_lmdb(self, lmdb_dir: str) -> Tuple[str, int]:
        """
        Load raw LMDB for long sequence visualization.
        Extracts video and clip information.
        
        Args:
            lmdb_dir: Path to raw LMDB directory
            
        Returns:
            Status message and number of videos
        """
        try:
            import lmdb
            import pyarrow
            
            if not os.path.exists(lmdb_dir):
                return f"Error: LMDB path does not exist: {lmdb_dir}", 0
            
            # Open raw LMDB
            self.raw_lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
            
            # Extract all videos and clips
            videos = []
            with self.raw_lmdb_env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        video = pyarrow.deserialize(value)
                        videos.append(video)
                    except Exception as e:
                        print(f"Warning: Could not deserialize video {key}: {e}")
                        continue
            
            self.raw_lmdb_videos = videos
            
            # Count total clips
            total_clips = sum(len(video.get('clips', [])) for video in videos)
            
            status = f"Raw LMDB loaded successfully!\n"
            status += f"Total videos: {len(videos)}\n"
            status += f"Total clips: {total_clips}\n"
            status += f"LMDB: {lmdb_dir}"
            
            return status, len(videos)
        except Exception as e:
            import traceback
            error_msg = f"Error loading raw LMDB: {str(e)}\n{traceback.format_exc()}"
            return error_msg, 0
    
    def get_video_clips_info(self, video_idx: int) -> Tuple[Optional[dict], str]:
        """
        Get information about clips in a video.
        
        Args:
            video_idx: Index of video
            
        Returns:
            Dictionary with clip info and status message
        """
        if self.raw_lmdb_videos is None:
            return None, "Error: Raw LMDB not loaded. Please load raw LMDB first."
        
        if video_idx < 0 or video_idx >= len(self.raw_lmdb_videos):
            return None, f"Error: Video index {video_idx} out of range (0-{len(self.raw_lmdb_videos)-1})"
        
        video = self.raw_lmdb_videos[video_idx]
        clips = video.get('clips', [])
        
        clip_info = []
        for clip_idx, clip in enumerate(clips):
            # Get motion data
            motion_L = clip.get('HML3D_joints_vec_L', None)
            motion_F = clip.get('HML3D_joints_vec_F', None)
            
            seq_len_L = len(motion_L) if motion_L is not None else 0
            seq_len_F = len(motion_F) if motion_F is not None else 0
            
            clip_info.append({
                'clip_idx': clip_idx,
                'seq_len_L': seq_len_L,
                'seq_len_F': seq_len_F,
                'vid': video.get('vid', 'unknown')
            })
        
        info_str = f"Video: {video.get('vid', 'unknown')}\n"
        info_str += f"Number of clips: {len(clips)}\n\n"
        for ci in clip_info:
            info_str += f"Clip {ci['clip_idx']}: Leader={ci['seq_len_L']} frames, Follower={ci['seq_len_F']} frames\n"
        
        return {'clips': clip_info, 'video': video}, info_str
    
    def extract_sequence_from_raw_lmdb(self, video_idx: int, clip_idx: int, role: str, 
                                       start_frame: int, end_frame: int) -> Tuple[Optional[np.ndarray], str]:
        """
        Extract a sequence from raw LMDB.
        
        Args:
            video_idx: Index of video
            clip_idx: Index of clip within video
            role: 'L' for leader or 'F' for follower
            start_frame: Start frame index
            end_frame: End frame index (exclusive)
            
        Returns:
            Motion sequence array (seq_len, 263) and status message
        """
        if self.raw_lmdb_videos is None:
            return None, "Error: Raw LMDB not loaded."
        
        if video_idx < 0 or video_idx >= len(self.raw_lmdb_videos):
            return None, f"Error: Video index out of range."
        
        video = self.raw_lmdb_videos[video_idx]
        clips = video.get('clips', [])
        
        if clip_idx < 0 or clip_idx >= len(clips):
            return None, f"Error: Clip index out of range."
        
        clip = clips[clip_idx]
        
        # Get motion data based on role
        if role == 'L':
            motion = clip.get('HML3D_joints_vec_L', None)
        else:
            motion = clip.get('HML3D_joints_vec_F', None)
        
        if motion is None:
            return None, f"Error: Motion data not found for role {role}."
        
        # Convert to numpy if needed
        if isinstance(motion, torch.Tensor):
            motion = motion.cpu().numpy()
        
        seq_len = len(motion)
        
        # Validate frame range
        if start_frame < 0 or end_frame > seq_len or start_frame >= end_frame:
            return None, f"Error: Invalid frame range. Sequence has {seq_len} frames."
        
        # Extract sequence - make copy and ensure float32
        sequence = motion[start_frame:end_frame].copy().astype(np.float32)  # (seq_len, 263)
        
        # Normalize if dataset normalization stats are available
        # IMPORTANT: Must use same normalization as training
        if self.dataloader is not None and self.dataloader.dataset is not None:
            dataset = self.dataloader.dataset
            if dataset.normalize and dataset.mean is not None and dataset.std is not None:
                # Convert mean/std to numpy if they're tensors, ensure float32
                if isinstance(dataset.mean, torch.Tensor):
                    mean_np = dataset.mean.cpu().numpy().astype(np.float32)
                else:
                    mean_np = np.array(dataset.mean, dtype=np.float32)
                
                if isinstance(dataset.std, torch.Tensor):
                    std_np = dataset.std.cpu().numpy().astype(np.float32)
                else:
                    std_np = np.array(dataset.std, dtype=np.float32)
                
                epsilon = getattr(dataset, 'epsilon', 1e-8)
                
                # Normalize: (sequence - mean) / (std + epsilon)
                sequence = (sequence - mean_np) / (std_np + epsilon)
            else:
                return None, f"Error: Dataset normalization is disabled or stats not available. Please ensure the regular dataset is loaded with normalize=True."
        else:
            return None, f"Error: Dataset not loaded. Please load the regular dataset first to get normalization stats."
        
        return sequence, f"Extracted {len(sequence)} frames (frames {start_frame}-{end_frame-1}), normalized"
    
    def visualize_long_sequence(self, video_idx: int, clip_idx: int, role: str,
                                start_frame: int, end_frame: int) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize long sequence generation using autoregressive inference.
        
        Args:
            video_idx: Index of video
            clip_idx: Index of clip
            role: 'L' for leader or 'F' for follower
            start_frame: Start frame
            end_frame: End frame
            
        Returns:
            Tuple of (original_video_path, generated_video_path, info_string)
        """
        if self.model is None:
            return None, None, "Error: Model not loaded. Please load a model first."
        
        try:
            # Extract original sequence
            original_sequence, extract_msg = self.extract_sequence_from_raw_lmdb(
                video_idx, clip_idx, role, start_frame, end_frame
            )
            
            if original_sequence is None:
                return None, None, extract_msg
            
            seq_len = len(original_sequence)
            window_size = self.model.seq_len  # 20
            
            # Calculate number of windows needed
            num_windows = (seq_len + window_size - 1) // window_size  # Ceiling division
            
            # Extract non-overlapping consecutive windows
            windows = []
            for i in range(num_windows):
                start = i * window_size
                end = min(start + window_size, seq_len)
                window = original_sequence[start:end]
                
                # Pad last window if needed
                if len(window) < window_size:
                    padding = np.zeros((window_size - len(window), window.shape[1]))
                    window = np.vstack([window, padding])
                
                windows.append(window)
            
            windows_array = np.array(windows)  # (num_windows, window_size, 263)
            
            # Convert to tensor
            windows_tensor = torch.from_numpy(windows_array).float().to(self.device)  # (num_windows, 20, 263)
            
            self.model.eval()
            with torch.no_grad():
                # Encode all windows
                encode_results = []
                for i in range(num_windows):
                    window = windows_tensor[i:i+1]  # (1, 20, 263)
                    encode_result = self.model.inference_encode(window)
                    encode_results.append(encode_result)
                
                # Stack latents/tokens
                if self.model.use_vqvae:
                    # For VQ-VAE: (z, code_idx) for each window
                    z_list = [r[0] for r in encode_results]  # List of (1, latent_dim)
                    code_idx_list = [r[1] for r in encode_results]  # List of (1,) or (1, 1)
                    z = torch.cat(z_list, dim=0)  # (num_windows, latent_dim)
                    code_idx = torch.cat(code_idx_list, dim=0) if code_idx_list[0].dim() > 0 else torch.stack(code_idx_list)
                    latents_or_tokens = (z, code_idx)
                else:
                    # For VAE/Vanilla: (z, mean, logvar) for each window
                    z_list = [r[0] for r in encode_results]  # List of (1, latent_dim)
                    mean_list = [r[1] for r in encode_results]  # List of (1, latent_dim)
                    logvar_list = [r[2] for r in encode_results]  # List of (1, latent_dim)
                    z = torch.cat(z_list, dim=0)  # (num_windows, latent_dim)
                    mean = torch.cat(mean_list, dim=0)  # (num_windows, latent_dim)
                    logvar = torch.cat(logvar_list, dim=0)  # (num_windows, latent_dim)
                    latents_or_tokens = (z, mean, logvar)
                
                # Get first frame from ground truth
                # original_sequence[0:1] gives (1, 263) which is correct shape (batch=1, input_dim)
                first_frame = torch.from_numpy(original_sequence[0:1]).float().to(self.device)  # (1, 263)
                # Ensure it's 2D: (batch=1, input_dim)
                if first_frame.dim() == 1:
                    first_frame = first_frame.unsqueeze(0)  # (1, 263)
                elif first_frame.dim() > 2:
                    # If somehow it's 3D or more, flatten to 2D
                    first_frame = first_frame.view(1, -1)  # (1, input_dim)
                elif first_frame.shape[0] > 1:
                    first_frame = first_frame[0:1]  # Take first frame only: (1, input_dim)
                
                # Decode autoregressively
                generated_sequence = self.model.inference_decode_autoregressive(
                    latents_or_tokens, first_frame, num_windows
                )  # (num_windows * window_size, 263)
                
                # Trim to original sequence length
                generated_sequence = generated_sequence[:seq_len]  # (seq_len, 263)
            
            # Convert to numpy - ensure both are float32 and same shape
            original_np = original_sequence.astype(np.float32)  # Already numpy, normalized
            generated_np = generated_sequence.cpu().numpy().astype(np.float32)  # (seq_len, 263), normalized
            
            # Ensure shapes match (trim if needed)
            if original_np.shape != generated_np.shape:
                min_len = min(len(original_np), len(generated_np))
                original_np = original_np[:min_len]
                generated_np = generated_np[:min_len]
            
            # Denormalize - both should be in normalized space, use same denormalization
            original_denorm = self._denormalize_motion(original_np)
            generated_denorm = self._denormalize_motion(generated_np)
            
            # Debug: Check for NaN/Inf
            if np.any(np.isnan(original_denorm)) or np.any(np.isinf(original_denorm)):
                return None, None, f"Error: NaN/Inf detected in denormalized original. Check normalization stats."
            if np.any(np.isnan(generated_denorm)) or np.any(np.isinf(generated_denorm)):
                return None, None, f"Error: NaN/Inf detected in denormalized generated. Check normalization stats."
            
            # Compute metrics
            mse = np.mean((original_denorm - generated_denorm) ** 2)
            mae = np.mean(np.abs(original_denorm - generated_denorm))
            
            # Convert to 3D keypoints
            original_keypoints = self._extract_keypoints_from_motion(original_denorm)
            generated_keypoints = self._extract_keypoints_from_motion(generated_denorm)
            
            # Get appropriate kinematic chain
            kinematic_chain = self._get_kinematic_chain()
            
            # Create videos
            original_path = os.path.join(self.temp_dir, f"long_original_{video_idx}_{clip_idx}_{role}.mp4")
            generated_path = os.path.join(self.temp_dir, f"long_generated_{video_idx}_{clip_idx}_{role}.mp4")
            
            self._plot_3d_motion(
                original_path,
                kinematic_chain,
                original_keypoints,
                title=f"Original Long Sequence ({seq_len} frames)",
                fps=20,
                radius=4
            )
            
            self._plot_3d_motion(
                generated_path,
                kinematic_chain,
                generated_keypoints,
                title=f"Generated Long Sequence (MSE: {mse:.4f})",
                fps=20,
                radius=4
            )
            
            # Build info
            info = f"Long Sequence Generation\n"
            info += f"{'='*60}\n"
            info += f"Video: {video_idx}, Clip: {clip_idx}, Role: {role}\n"
            info += f"Frames: {start_frame}-{end_frame-1} ({seq_len} frames)\n"
            info += f"Windows: {num_windows} x {window_size} frames\n"
            info += f"MSE: {mse:.6f}\n"
            info += f"MAE: {mae:.6f}\n"
            info += f"Model: {'VQ-VAE' if self.model.use_vqvae else 'VAE' if self.model.use_vae else 'Vanilla AE'}"
            
            return original_path, generated_path, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing long sequence: {str(e)}\n{traceback.format_exc()}"
            return None, None, error_msg
    
    def visualize_combined_reconstruction(self, video_idx: int, clip_idx: int,
                                         start_frame: int, end_frame: int) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize both leader and follower together: original vs reconstructed.
        
        Args:
            video_idx: Index of video
            clip_idx: Index of clip
            start_frame: Start frame
            end_frame: End frame
            
        Returns:
            Tuple of (original_combined_video_path, reconstructed_combined_video_path, info_string)
        """
        if self.model is None:
            return None, None, "Error: Model not loaded. Please load a model first."
        
        try:
            # Extract both leader and follower sequences
            leader_original, msg_l = self.extract_sequence_from_raw_lmdb(
                video_idx, clip_idx, 'L', start_frame, end_frame
            )
            follower_original, msg_f = self.extract_sequence_from_raw_lmdb(
                video_idx, clip_idx, 'F', start_frame, end_frame
            )
            
            if leader_original is None:
                return None, None, f"Error extracting leader: {msg_l}"
            if follower_original is None:
                return None, None, f"Error extracting follower: {msg_f}"
            
            # Ensure same length
            min_len = min(len(leader_original), len(follower_original))
            leader_original = leader_original[:min_len]
            follower_original = follower_original[:min_len]
            
            seq_len = min_len
            window_size = self.model.seq_len
            
            # For reconstruction, we'll process in windows and concatenate
            # But for simplicity, if sequence is <= window_size, process as single window
            if seq_len <= window_size:
                # Single window
                leader_window = leader_original.copy()
                follower_window = follower_original.copy()
                
                # Pad if needed
                if len(leader_window) < window_size:
                    padding = np.zeros((window_size - len(leader_window), leader_window.shape[1]))
                    leader_window = np.vstack([leader_window, padding])
                if len(follower_window) < window_size:
                    padding = np.zeros((window_size - len(follower_window), follower_window.shape[1]))
                    follower_window = np.vstack([follower_window, padding])
                
                # Convert to tensors
                leader_tensor = torch.from_numpy(leader_window).float().unsqueeze(0).to(self.device)  # (1, window_size, 263)
                follower_tensor = torch.from_numpy(follower_window).float().unsqueeze(0).to(self.device)  # (1, window_size, 263)
                
                self.model.eval()
                with torch.no_grad():
                    # Reconstruct both
                    leader_recon = self.model(leader_tensor)[0]  # (1, window_size, 263)
                    follower_recon = self.model(follower_tensor)[0]  # (1, window_size, 263)
                
                # Convert to numpy and trim
                leader_recon_np = leader_recon[0, :seq_len].cpu().numpy().astype(np.float32)
                follower_recon_np = follower_recon[0, :seq_len].cpu().numpy().astype(np.float32)
            else:
                # Multiple windows - process each and concatenate
                leader_recon_list = []
                follower_recon_list = []
                
                num_windows = (seq_len + window_size - 1) // window_size
                
                self.model.eval()
                with torch.no_grad():
                    for i in range(num_windows):
                        start = i * window_size
                        end = min(start + window_size, seq_len)
                        
                        leader_win = leader_original[start:end].copy()
                        follower_win = follower_original[start:end].copy()
                        
                        # Pad if needed
                        if len(leader_win) < window_size:
                            padding = np.zeros((window_size - len(leader_win), leader_win.shape[1]))
                            leader_win = np.vstack([leader_win, padding])
                        if len(follower_win) < window_size:
                            padding = np.zeros((window_size - len(follower_win), follower_win.shape[1]))
                            follower_win = np.vstack([follower_win, padding])
                        
                        leader_tensor = torch.from_numpy(leader_win).float().unsqueeze(0).to(self.device)
                        follower_tensor = torch.from_numpy(follower_win).float().unsqueeze(0).to(self.device)
                        
                        leader_recon = self.model(leader_tensor)[0]  # (1, window_size, 263)
                        follower_recon = self.model(follower_tensor)[0]  # (1, window_size, 263)
                        
                        # Trim to actual length
                        actual_len = end - start
                        leader_recon_list.append(leader_recon[0, :actual_len].cpu().numpy())
                        follower_recon_list.append(follower_recon[0, :actual_len].cpu().numpy())
                
                leader_recon_np = np.vstack(leader_recon_list).astype(np.float32)
                follower_recon_np = np.vstack(follower_recon_list).astype(np.float32)
            
            # Denormalize all sequences
            leader_original_denorm = self._denormalize_motion(leader_original.astype(np.float32))
            follower_original_denorm = self._denormalize_motion(follower_original.astype(np.float32))
            leader_recon_denorm = self._denormalize_motion(leader_recon_np)
            follower_recon_denorm = self._denormalize_motion(follower_recon_np)
            
            # Convert to 3D keypoints
            leader_original_kp = self._extract_keypoints_from_motion(leader_original_denorm)
            follower_original_kp = self._extract_keypoints_from_motion(follower_original_denorm)
            leader_recon_kp = self._extract_keypoints_from_motion(leader_recon_denorm)
            follower_recon_kp = self._extract_keypoints_from_motion(follower_recon_denorm)
            
            # Create combined videos
            original_combined_path = os.path.join(self.temp_dir, f"combined_original_{video_idx}_{clip_idx}.mp4")
            recon_combined_path = os.path.join(self.temp_dir, f"combined_recon_{video_idx}_{clip_idx}.mp4")
            
            render_combined_skeletons(
                leader_original_kp,
                follower_original_kp,
                original_combined_path,
                title=f"Original Combined ({seq_len} frames)",
                fps=20
            )
            
            render_combined_skeletons(
                leader_recon_kp,
                follower_recon_kp,
                recon_combined_path,
                title=f"Reconstructed Combined ({seq_len} frames)",
                fps=20
            )
            
            # Compute metrics
            leader_mse = np.mean((leader_original_denorm - leader_recon_denorm) ** 2)
            follower_mse = np.mean((follower_original_denorm - follower_recon_denorm) ** 2)
            leader_mae = np.mean(np.abs(leader_original_denorm - leader_recon_denorm))
            follower_mae = np.mean(np.abs(follower_original_denorm - follower_recon_denorm))
            
            info = f"Combined Reconstruction\n"
            info += f"{'='*60}\n"
            info += f"Video: {video_idx}, Clip: {clip_idx}\n"
            info += f"Frames: {start_frame}-{end_frame-1} ({seq_len} frames)\n"
            info += f"Leader MSE: {leader_mse:.6f}, MAE: {leader_mae:.6f}\n"
            info += f"Follower MSE: {follower_mse:.6f}, MAE: {follower_mae:.6f}\n"
            info += f"Model: {'VQ-VAE' if self.model.use_vqvae else 'VAE' if self.model.use_vae else 'Vanilla AE'}"
            
            return original_combined_path, recon_combined_path, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing combined reconstruction: {str(e)}\n{traceback.format_exc()}"
            return None, None, error_msg
    
    def visualize_relationship_features(self, idx: int) -> Tuple[Optional[object], str]:
        """
        Visualize relationship features as time series plots for each dimension.
        
        Args:
            idx: Sample index from dataset
            
        Returns:
            Tuple of (plotly_figure_object, info_string)
        """
        if self.model is None:
            return None, "Error: Model not loaded. Please load a model first."
        
        if self.dataloader is None:
            return None, "Error: Dataset not loaded. Please load dataset first."
        
        try:
            if idx < 0 or idx >= len(self.dataloader.dataset):
                return None, f"Error: Index {idx} out of range (0-{len(self.dataloader.dataset)-1})"
            
            dataset = self.dataloader.dataset
            # Get relationship features - check if dataset has get_pair_data (InterHuman datasets)
            used_pair_data = False
            rel_mean_np = None
            rel_std_np = None
            epsilon = getattr(dataset, 'epsilon', 1e-8)
            
            if hasattr(dataset, 'get_pair_data') and dataset.representation_type == 'interhuman':
                # For interhuman datasets with use_both_roles=True, map dataset idx to pair idx
                pair_idx = idx // 2 if (dataset.representation_type == 'interhuman' and dataset.use_both_roles) else idx
                pair_data = dataset.get_pair_data(pair_idx)
                features = torch.from_numpy(pair_data['relationship_features'].copy()).float()  # (19, 4)
                used_pair_data = True
                
                # Load relationship normalization stats if needed
                if dataset.normalize:
                    import pickle
                    import os
                    cache_dir = dataset.cache_dir
                    rel_stats_path = os.path.join(cache_dir, 'normalization_stats_relationship.pkl')
                    if os.path.exists(rel_stats_path):
                        with open(rel_stats_path, 'rb') as f:
                            rel_stats = pickle.load(f)
                            rel_mean_np = rel_stats['mean']  # numpy (4,)
                            rel_std_np = rel_stats['std']  # numpy (4,)
                            rel_mean = torch.from_numpy(rel_mean_np).float()
                            rel_std = torch.from_numpy(rel_std_np).float()
                            features = (features - rel_mean) / (rel_std + epsilon)
            else:
                # Direct access (dataset is relationship type)
                features = dataset[idx]  # (19, 4) for relationship features
                if features.shape[1] != 4:
                    return None, f"Error: Expected 4D relationship features, but got shape {features.shape}. Please ensure you loaded the dataset with representation_type='relationship' or 'interhuman'."
            
            features = features.unsqueeze(0).to(self.device)  # (1, 19, 4)
            
            # Reconstruct - handle different model types
            with torch.no_grad():
                forward_result = self.model(features)
                
                # VQ-VAE returns: (recon_x, z, commit_loss, perplexity, code_idx)
                # VAE/Vanilla returns: (recon_x, mean, logvar, z)
                if self.model.use_vqvae:
                    recon_features, z, commit_loss, perplexity, code_idx = forward_result
                    mean = None
                    logvar = None
                else:
                    recon_features, mean, logvar, z = forward_result
            
            # Convert to numpy
            original = features[0].cpu().numpy()  # (19, 4) - normalized [w, z, x, z]
            reconstructed = recon_features[0].cpu().numpy()  # (19, 4) - normalized [w, z, x, z]
            
            # Denormalize if needed
            if used_pair_data and rel_mean_np is not None and rel_std_np is not None:
                # Use relationship stats for denormalization
                original_denorm = original * rel_std_np + rel_mean_np
                reconstructed_denorm = reconstructed * rel_std_np + rel_mean_np
            elif self.dataloader.dataset.normalize and self.dataloader.dataset.mean is not None:
                original_denorm = self._denormalize_motion(original)
                reconstructed_denorm = self._denormalize_motion(reconstructed)
            else:
                original_denorm = original
                reconstructed_denorm = reconstructed
            
            # Convert quaternion components [w, z] to radians for visualization (only for plotting)
            # Extract [w, z] components (first 2 dims) and convert to angle: yaw = arctan2(z, w)
            original_yaw = np.arctan2(original_denorm[:, 1], original_denorm[:, 0])  # (19,)
            reconstructed_yaw = np.arctan2(reconstructed_denorm[:, 1], reconstructed_denorm[:, 0])  # (19,)
            
            # Stack converted features: [yaw_radians, x, z] for visualization
            original_viz = np.stack([original_yaw, original_denorm[:, 2], original_denorm[:, 3]], axis=1)  # (19, 3)
            reconstructed_viz = np.stack([reconstructed_yaw, reconstructed_denorm[:, 2], reconstructed_denorm[:, 3]], axis=1)  # (19, 3)
            
            # Compute metrics for each dimension
            # Relationship features are 4D: [w, z, x, z] - converted to [yaw_rad, x, z] for visualization
            dim_names = [
                "Relative Yaw (rotation difference in radians)",
                "Relative X Position (follower_x - leader_x in shared space)",
                "Relative Z Position (follower_z - leader_z in shared space)"
            ]
            
            dim_short_names = ["Yaw", "X", "Z"]
            
            metrics = []
            for dim in range(3):
                mse = np.mean((original_viz[:, dim] - reconstructed_viz[:, dim]) ** 2)
                mae = np.mean(np.abs(original_viz[:, dim] - reconstructed_viz[:, dim]))
                metrics.append((mse, mae))
            
            # Create time series plots using plotly
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots: 1x3 grid (3 plots horizontally)
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=dim_names,
                horizontal_spacing=0.12
            )
            
            # Time axis (frames 0-18, since InterHuman reduces length by 1)
            seq_len = original_viz.shape[0]
            time_axis = np.arange(seq_len)
            
            # Plot each dimension
            for dim in range(3):
                row = 1
                col = dim + 1
                
                # Original (ground truth) - blue line
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=original_viz[:, dim],
                        mode='lines+markers',
                        name='Original',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4),
                        showlegend=(dim == 0)  # Only show legend for first plot
                    ),
                    row=row, col=col
                )
                
                # Reconstructed - red line
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=reconstructed_viz[:, dim],
                        mode='lines+markers',
                        name='Reconstructed',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=4),
                        showlegend=(dim == 0)  # Only show legend for first plot
                    ),
                    row=row, col=col
                )
                
                # Update axes labels
                fig.update_xaxes(title_text="Frame", row=row, col=col)
                fig.update_yaxes(title_text="Feature Value", row=row, col=col)
                
                # Add metrics as annotation
                mse, mae = metrics[dim]
                fig.add_annotation(
                    text=f"MSE: {mse:.6f}<br>MAE: {mae:.6f}",
                    xref=f"x{dim+1}", yref=f"y{dim+1}",
                    x=0.02, y=0.98,
                    xanchor='left', yanchor='top',
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    row=row, col=col
                )
            
            # Update layout
            fig.update_layout(
                height=400,
                title_text="Relationship Features: Original vs Reconstructed (InterHuman)",
                title_x=0.5,
                title_font=dict(size=16),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Build info string
            info = f"Relationship Features Visualization\n"
            info += f"{'='*60}\n"
            info += f"Sample Index: {idx}\n"
            info += f"Sequence Length: {seq_len} frames (InterHuman reduces window_size by 1)\n"
            info += f"Feature Dimensions: 4 [w, z, x, z] (converted to [yaw_rad, x, z] for visualization)\n"
            info += f"Representation: InterHuman canonical frames\n"
            info += f"Note: Quaternion components [w, z] converted to radians for visualization only\n\n"
            info += f"Error Metrics:\n"
            for dim in range(3):
                mse, mae = metrics[dim]
                info += f"  {dim_short_names[dim]}: MSE={mse:.6f}, MAE={mae:.6f}\n"
            info += f"\nModel: {'VQ-VAE' if self.model.use_vqvae else 'VAE' if self.model.use_vae else 'Vanilla AE'}"
            
            return fig, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing relationship features: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg

    def visualize_combined_reconstruction(
        self, 
        motion_model_path: str,
        relationship_model_path: str,
        idx: int
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize combined reconstruction using both motion and relationship networks.
        
        This method:
        1. Loads two models: motion network (InterHuman) and relationship network
        2. Reconstructs leader and follower canonicalized motions separately
        3. Reconstructs relationship features
        4. Combines them using rigid_transform to move follower into leader's space
        5. Visualizes GT vs reconstructed side-by-side
        
        Args:
            motion_model_path: Path to motion model checkpoint (InterHuman representation)
            relationship_model_path: Path to relationship model checkpoint
            idx: Sample index from dataset
        
        Returns:
            Tuple of (gt_video_path, recon_video_path, info_string)
        """
        try:
            # Check if dataset is loaded and supports pair data
            if self.dataloader is None:
                return None, None, "Error: Dataset not loaded. Please load dataset first."
            
            dataset = self.dataloader.dataset
            if not hasattr(dataset, 'get_pair_data'):
                return None, None, "Error: Dataset does not support get_pair_data. Please use InterHuman or relationship representation_type."
            
            if idx < 0 or idx >= len(dataset):
                return None, None, f"Error: Index {idx} out of range (0-{len(dataset)-1})"
            
            # Get pair data from cache (dictionary format)
            pair_data = dataset.get_pair_data(idx)
            
            # Load motion model
            if not os.path.exists(motion_model_path):
                return None, None, f"Error: Motion model checkpoint not found: {motion_model_path}"
            
            motion_checkpoint = torch.load(motion_model_path, map_location=self.device)
            motion_config = motion_checkpoint['config']
            motion_model = MotionModel(
                input_dim=motion_config['input_dim'],
                hidden_dim=motion_config['hidden_dim'],
                num_layers=motion_config['num_layers'],
                latent_dim=motion_config['latent_dim'],
                seq_len=motion_config['seq_len'],
                dropout=motion_config['dropout'],
                encoder_type=motion_config.get('encoder_type', 'gru'),
                decoder_type=motion_config.get('decoder_type', 'gru'),
                num_heads=motion_config.get('num_heads', 8),
                ff_size=motion_config.get('ff_size', 2048),
                activation=motion_config.get('activation', 'gelu'),
                use_vae=motion_config.get('use_vae', False),
                use_vqvae=motion_config.get('use_vqvae', False),
                nb_code=motion_config.get('nb_code', motion_config.get('vq_codebook_size', 512)),
                quantizer=motion_config.get('quantizer', motion_config.get('vq_quantizer', 'ema_reset')),
                vq_mu=motion_config.get('vq_mu', motion_config.get('vq_ema_mu', 0.99)),
                vq_beta=motion_config.get('vq_beta', motion_config.get('vq_commitment_cost', 1.0)),
            ).to(self.device)
            motion_model.load_state_dict(motion_checkpoint['model_state_dict'], strict=False)
            motion_model.eval()
            
            # Load relationship model
            if not os.path.exists(relationship_model_path):
                return None, None, f"Error: Relationship model checkpoint not found: {relationship_model_path}"
            
            rel_checkpoint = torch.load(relationship_model_path, map_location=self.device)
            rel_config = rel_checkpoint['config']
            relationship_model = MotionModel(
                input_dim=rel_config['input_dim'],
                hidden_dim=rel_config['hidden_dim'],
                num_layers=rel_config['num_layers'],
                latent_dim=rel_config['latent_dim'],
                seq_len=rel_config['seq_len'],
                dropout=rel_config['dropout'],
                encoder_type=rel_config.get('encoder_type', 'gru'),
                decoder_type=rel_config.get('decoder_type', 'gru'),
                num_heads=rel_config.get('num_heads', 8),
                ff_size=rel_config.get('ff_size', 2048),
                activation=rel_config.get('activation', 'gelu'),
                use_vae=rel_config.get('use_vae', False),
                use_vqvae=rel_config.get('use_vqvae', False),
                nb_code=rel_config.get('nb_code', rel_config.get('vq_codebook_size', 512)),
                quantizer=rel_config.get('quantizer', rel_config.get('vq_quantizer', 'ema_reset')),
                vq_mu=rel_config.get('vq_mu', rel_config.get('vq_ema_mu', 0.99)),
                vq_beta=rel_config.get('vq_beta', rel_config.get('vq_commitment_cost', 1.0)),
            ).to(self.device)
            relationship_model.load_state_dict(rel_checkpoint['model_state_dict'], strict=False)
            relationship_model.eval()
            
            # Extract GT data
            gt_leader_motion = pair_data['leader_motion']  # (19, 262) - canonicalized (separate canonical frame)
            gt_follower_motion = pair_data['follower_motion']  # (19, 262) - canonicalized (separate canonical frame, NOT aligned)
            gt_relationship = pair_data['relationship_features']  # (19, 4) - [w, z, x, z] - temporal relative features
            root_quat_init_L = pair_data['root_quat_init_L']  # (4,)
            root_pos_init_L = pair_data['root_pos_init_L']  # (3,)
            root_quat_init_F = pair_data['root_quat_init_F']  # (4,)
            root_pos_init_F = pair_data['root_pos_init_F']  # (3,)
            
            # NOTE: Both motions are stored as separate canonicalized motions (for single-motion network training)
            # We need to apply rigid_transform using frame 0's relationship features to align follower to leader's space
            
            # Load normalization stats for both motion (interhuman) and relationship types
            # The dataset object only has stats for its representation_type, so we load both separately
            import pickle
            cache_dir = dataset.cache_dir
            epsilon = 1e-8
            
            motion_mean = None
            motion_std = None
            rel_mean = None
            rel_std = None
            
            if dataset.normalize:
                # Load InterHuman motion stats (262 dims)
                motion_stats_path = os.path.join(cache_dir, 'normalization_stats_interhuman.pkl')
                if os.path.exists(motion_stats_path):
                    with open(motion_stats_path, 'rb') as f:
                        motion_stats = pickle.load(f)
                        motion_mean = motion_stats['mean']  # numpy array (262,)
                        motion_std = motion_stats['std']  # numpy array (262,)
                
                # Load relationship stats (4 dims)
                rel_stats_path = os.path.join(cache_dir, 'normalization_stats_relationship.pkl')
                if os.path.exists(rel_stats_path):
                    with open(rel_stats_path, 'rb') as f:
                        rel_stats = pickle.load(f)
                        rel_mean = rel_stats['mean']  # numpy array (4,)
                        rel_std = rel_stats['std']  # numpy array (4,)
            
            # Normalize GT data if needed
            if dataset.normalize and motion_mean is not None:
                gt_leader_norm = (gt_leader_motion - motion_mean) / (motion_std + epsilon)
                gt_follower_norm = (gt_follower_motion - motion_mean) / (motion_std + epsilon)
            else:
                gt_leader_norm = gt_leader_motion
                gt_follower_norm = gt_follower_motion
            
            if dataset.normalize and rel_mean is not None:
                gt_relationship_norm = (gt_relationship - rel_mean) / (rel_std + epsilon)
            else:
                gt_relationship_norm = gt_relationship
            
            # Reconstruct leader motion
            with torch.no_grad():
                leader_input = torch.from_numpy(gt_leader_norm).float().unsqueeze(0).to(self.device)  # (1, 19, 262)
                leader_result = motion_model(leader_input)
                if motion_model.use_vqvae:
                    recon_leader_norm, _, _, _, _ = leader_result
                else:
                    recon_leader_norm, _, _, _ = leader_result
                recon_leader_norm = recon_leader_norm[0].cpu().numpy()  # (19, 262)
            
            # Reconstruct follower motion
            with torch.no_grad():
                follower_input = torch.from_numpy(gt_follower_norm).float().unsqueeze(0).to(self.device)  # (1, 19, 262)
                follower_result = motion_model(follower_input)
                if motion_model.use_vqvae:
                    recon_follower_norm, _, _, _, _ = follower_result
                else:
                    recon_follower_norm, _, _, _ = follower_result
                recon_follower_norm = recon_follower_norm[0].cpu().numpy()  # (19, 262)
            
            # Reconstruct relationship features
            with torch.no_grad():
                relationship_input = torch.from_numpy(gt_relationship_norm).float().unsqueeze(0).to(self.device)  # (1, 19, 4)
                relationship_result = relationship_model(relationship_input)
                if relationship_model.use_vqvae:
                    recon_relationship_norm, _, _, _, _ = relationship_result
                else:
                    recon_relationship_norm, _, _, _ = relationship_result
                recon_relationship_norm = recon_relationship_norm[0].cpu().numpy()  # (19, 4)
            
            # Denormalize using the same stats we loaded for normalization
            if dataset.normalize and motion_mean is not None:
                recon_leader = recon_leader_norm * (motion_std + epsilon) + motion_mean
                recon_follower = recon_follower_norm * (motion_std + epsilon) + motion_mean
            else:
                recon_leader = recon_leader_norm
                recon_follower = recon_follower_norm
            
            if dataset.normalize and rel_mean is not None:
                recon_relationship = recon_relationship_norm * (rel_std + epsilon) + rel_mean
            else:
                recon_relationship = recon_relationship_norm
            
            # Convert relationship features to rigid_transform parameters
            # relationship_features are [w, z, x, z] - quaternion components [w, z] + position [x, z]
            # Frame 0's relationship features match the initial transform exactly (from root_quat_init and root_pos_init)
            # 
            # IMPORTANT: We use frame 0's relationship features to compute the rigid_transform
            # This transform aligns the follower (in its own canonical frame) to the leader's canonical frame
            # 
            # Frame 0's [w, z] are stored as [cos(θ/2), sin(θ/2)] where θ is the full yaw angle
            # To recover the half-angle for rigid_transform: angle_half = arctan2(z, w)
            # rigid_transform expects the half-angle (it does cos(angle) and sin(angle) to create [cos(θ/2), 0, sin(θ/2), 0])
            
            # Import in2IN's rigid_transform and quaternion functions
            from in2in.utils.utils import rigid_transform
            from in2in.utils.quaternion import qmul_np, qinv_np, qrot_np
            
            # ====================================================================
            # DEBUG: Compare relative transform computation (Notebook vs Our Pipeline)
            # ====================================================================
            # Notebook approach: Compute directly from root_quat_init and root_pos_init
            # (matching notebook's salsa_pair_to_interhuman lines 1231-1234)
            r_relative_notebook = qmul_np(root_quat_init_F, qinv_np(root_quat_init_L))  # (4,)
            angle_notebook = np.arctan2(r_relative_notebook[2], r_relative_notebook[0])  # scalar - half-angle
            xz_notebook = qrot_np(root_quat_init_L, root_pos_init_F - root_pos_init_L)[[0, 2]]  # (2,)
            relative_notebook = np.array([angle_notebook, xz_notebook[0], xz_notebook[1]])  # (3,) - [angle_half, x, z]
            
            # Our pipeline approach: Extract from relationship_features[0]
            # (matching our extract_interhuman_relationship_features computation)
            gt_rel_w = gt_relationship[0, 0]  # cos(θ/2)
            gt_rel_z = gt_relationship[0, 1]  # sin(θ/2)
            gt_relative_angle_half = np.arctan2(gt_rel_z, gt_rel_w)  # radians - half of yaw angle
            relative_ours = np.array([gt_relative_angle_half, gt_relationship[0, 2], gt_relationship[0, 3]])  # (3,)
            
            # Compare
            angle_diff = abs(angle_notebook - gt_relative_angle_half)
            x_diff = abs(xz_notebook[0] - gt_relationship[0, 2])
            z_diff = abs(xz_notebook[1] - gt_relationship[0, 3])
            
            print("\n" + "="*70)
            print("DEBUG: Relative Transform Comparison (Notebook vs Our Pipeline)")
            print("="*70)
            print(f"Notebook approach (from root_quat_init/root_pos_init):")
            print(f"  r_relative = qmul_np(root_quat_init_F, qinv_np(root_quat_init_L))")
            print(f"  angle = arctan2(r_relative[2], r_relative[0])")
            print(f"  xz = qrot_np(root_quat_init_L, root_pos_init_F - root_pos_init_L)[[0, 2]]")
            print(f"  relative = [angle, xz[0], xz[1]]")
            print(f"  Result: relative = [{angle_notebook:.6f}, {xz_notebook[0]:.6f}, {xz_notebook[1]:.6f}]")
            print(f"  Angle (half): {np.degrees(angle_notebook):.3f} deg")
            print(f"\nOur pipeline (from relationship_features[0]):")
            print(f"  rel_w = relationship_features[0, 0] = {gt_rel_w:.6f}")
            print(f"  rel_z = relationship_features[0, 1] = {gt_rel_z:.6f}")
            print(f"  angle_half = arctan2(rel_z, rel_w)")
            print(f"  relative = [angle_half, relationship_features[0, 2], relationship_features[0, 3]]")
            print(f"  Result: relative = [{gt_relative_angle_half:.6f}, {gt_relationship[0, 2]:.6f}, {gt_relationship[0, 3]:.6f}]")
            print(f"  Angle (half): {np.degrees(gt_relative_angle_half):.3f} deg")
            print(f"\nDifferences:")
            print(f"  Angle difference: {np.degrees(angle_diff):.6f} deg ({angle_diff:.2e} rad)")
            print(f"  X difference: {x_diff:.6e}")
            print(f"  Z difference: {z_diff:.6e}")
            
            threshold = 1e-4
            if angle_diff < threshold and x_diff < threshold and z_diff < threshold:
                print(f"\n✓ MATCH: Both approaches produce identical relative transform!")
                print(f"  This confirms our relationship_features[0] matches the notebook's computation.")
            else:
                print(f"\n✗ MISMATCH: Approaches differ!")
                print(f"  This indicates a discrepancy between notebook and our pipeline.")
                if angle_diff >= threshold:
                    print(f"  ⚠️  Angle difference is significant: {np.degrees(angle_diff):.3f} deg")
                if x_diff >= threshold:
                    print(f"  ⚠️  X difference is significant: {x_diff:.6e}")
                if z_diff >= threshold:
                    print(f"  ⚠️  Z difference is significant: {z_diff:.6e}")
            print("="*70 + "\n")
            
            # For reconstructed: convert frame 0's relationship features to rigid_transform and apply
            rel_w = recon_relationship[0, 0]  # cos(θ/2)
            rel_z = recon_relationship[0, 1]  # sin(θ/2)
            rel_x = recon_relationship[0, 2]
            rel_z_pos = recon_relationship[0, 3]
            
            # Convert [w, z] quaternion components to half-angle
            # [w, z] = [cos(θ/2), sin(θ/2)] → θ/2 = arctan2(z, w)
            relative_angle_half = np.arctan2(rel_z, rel_w)  # radians - half of yaw angle (what rigid_transform expects)
            relative_transform = np.array([relative_angle_half, rel_x, rel_z_pos])  # [angle_half, x, z]
            
            # DEBUG: Compare reconstructed relationship features with GT
            recon_angle_diff = abs(gt_relative_angle_half - relative_angle_half)
            recon_x_diff = abs(gt_relationship[0, 2] - rel_x)
            recon_z_diff = abs(gt_relationship[0, 3] - rel_z_pos)
            
            print("="*70)
            print("DEBUG: Reconstructed vs Ground Truth Relationship Features (Frame 0)")
            print("="*70)
            print(f"Ground Truth:  relative = [{gt_relative_angle_half:.6f}, {gt_relationship[0, 2]:.6f}, {gt_relationship[0, 3]:.6f}]")
            print(f"Reconstructed: relative = [{relative_angle_half:.6f}, {rel_x:.6f}, {rel_z_pos:.6f}]")
            print(f"\nDifferences:")
            print(f"  Angle difference: {np.degrees(recon_angle_diff):.6f} deg ({recon_angle_diff:.2e} rad)")
            print(f"  X difference: {recon_x_diff:.6e}")
            print(f"  Z difference: {recon_z_diff:.6e}")
            
            recon_threshold = 1e-3  # More lenient for reconstruction
            if recon_angle_diff < np.deg2rad(1.0) and recon_x_diff < 0.01 and recon_z_diff < 0.01:
                print(f"\n✓ RECONSTRUCTION GOOD: Reconstructed relationship features are close to GT")
            else:
                print(f"\n⚠️  RECONSTRUCTION DIFFERS: Model may need more training")
                if recon_angle_diff >= np.deg2rad(1.0):
                    print(f"  ⚠️  Angle difference is significant: {np.degrees(recon_angle_diff):.3f} deg")
                if recon_x_diff >= 0.01:
                    print(f"  ⚠️  X difference is significant: {recon_x_diff:.6e}")
                if recon_z_diff >= 0.01:
                    print(f"  ⚠️  Z difference is significant: {recon_z_diff:.6e}")
            print("="*70 + "\n")
            
            # Apply rigid_transform to reconstructed follower (canonicalized, not aligned)
            # This aligns it to leader's canonical frame using frame 0's relationship features
            recon_follower_rel = rigid_transform(relative_transform, recon_follower.copy())  # (19, 262)
            
            # For GT: also apply rigid_transform using frame 0's relationship features
            # Both GT motions are stored as separate canonicalized motions (not aligned)
            gt_rel_w = gt_relationship[0, 0]  # cos(θ/2)
            gt_rel_z = gt_relationship[0, 1]  # sin(θ/2)
            # Convert [w, z] to half-angle: θ/2 = arctan2(z, w)
            # IMPORTANT: [w, z] = [cos(θ/2), sin(θ/2)], so arctan2(z, w) = θ/2 (half-angle, not full angle)
            gt_relative_angle_half = np.arctan2(gt_rel_z, gt_rel_w)  # radians - half of yaw angle (what rigid_transform expects)
            gt_relative_transform = np.array([gt_relative_angle_half, gt_relationship[0, 2], gt_relationship[0, 3]])
            gt_follower_rel = rigid_transform(gt_relative_transform, gt_follower_motion.copy())  # (19, 262)
            
            # Extract joint positions for visualization
            n_joints = 22
            gt_leader_joints = gt_leader_motion[:, :n_joints*3].reshape(-1, n_joints, 3)  # (19, 22, 3)
            gt_follower_joints = gt_follower_rel[:, :n_joints*3].reshape(-1, n_joints, 3)  # (19, 22, 3)
            recon_leader_joints = recon_leader[:, :n_joints*3].reshape(-1, n_joints, 3)  # (19, 22, 3)
            recon_follower_joints = recon_follower_rel[:, :n_joints*3].reshape(-1, n_joints, 3)  # (19, 22, 3)
            
            # Create visualizations using in2IN's plot function
            if not IN2IN_AVAILABLE:
                return None, None, "Error: in2IN library not available. Cannot visualize InterHuman motions."
            
            from in2in.utils.plot import plot_3d_motion
            from in2in.utils.paramUtil import HML_KINEMATIC_CHAIN
            
            # Save GT visualization
            gt_video_path = os.path.join(self.temp_dir, f"combined_gt_{idx}.mp4")
            plot_3d_motion(
                save_path=gt_video_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[gt_leader_joints, gt_follower_joints],
                title="Ground Truth: Leader + Follower",
                figsize=(12, 12),
                fps=30,
                radius=6
            )
            
            # Save reconstructed visualization
            recon_video_path = os.path.join(self.temp_dir, f"combined_recon_{idx}.mp4")
            plot_3d_motion(
                save_path=recon_video_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[recon_leader_joints, recon_follower_joints],
                title="Reconstructed: Leader + Follower",
                figsize=(12, 12),
                fps=30,
                radius=6
            )
            
            # Build info string
            info = f"Combined Motion + Relationship Reconstruction\n"
            info += f"{'='*60}\n"
            info += f"Sample Index: {idx}\n"
            info += f"Sequence Length: 19 frames (window_size=20, InterHuman reduces by 1)\n\n"
            info += f"Motion Model: {os.path.basename(motion_model_path)}\n"
            info += f"Relationship Model: {os.path.basename(relationship_model_path)}\n\n"
            info += f"Reconstruction Pipeline:\n"
            info += f"  1. Reconstructed leader motion (canonicalized) from motion network\n"
            info += f"  2. Reconstructed follower motion (canonicalized) from motion network\n"
            info += f"  3. Reconstructed relationship features from relationship network\n"
            info += f"  4. Converted relationship[0] [w, z, x, z] to rigid_transform [angle, x, z]\n"
            info += f"  5. Applied rigid_transform to reconstructed follower\n"
            info += f"  6. Visualized GT vs reconstructed side-by-side\n"
            
            return gt_video_path, recon_video_path, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error in combined reconstruction: {str(e)}\n{traceback.format_exc()}"
            return None, None, error_msg

    def quantize_samples(self, num_samples: int = 1000) -> Tuple[str, list]:
        """
        Quantize a specified number of samples (encode -> nearest code index) and store mapping for exploration.
        This is used by the "Token Cluster Exploration" section.
        
        Returns:
            (status_message, available_code_indices_sorted)
        """
        if self.model is None:
            return "Error: Model not loaded. Please load a model first.", []
        
        if not getattr(self.model, "use_vqvae", False):
            return "Error: Model is not a VQ-VAE. This feature only works with VQ-VAE models.", []
        
        if self.dataloader is None:
            return "Error: Dataset not loaded. Please load dataset first.", []
        
        try:
            total_samples = len(self.dataloader.dataset)
            num_samples = int(num_samples) if num_samples is not None else 1000
            num_samples = max(1, min(num_samples, total_samples))
            
            indices = np.random.choice(total_samples, num_samples, replace=False)
            
            self.code_to_samples = {}
            self.sample_to_code = {}
            
            self.model.eval()
            with torch.no_grad():
                for idx in tqdm(indices, desc="Quantizing samples"):
                    sample = self.dataloader.dataset[int(idx)]  # (20, input_dim)
                    sample = sample.unsqueeze(0).to(self.device)  # (1, 20, input_dim)
                    
                    encoded = self.model.encoder(sample)  # (1, latent_dim)
                    code_idx = self.model.vq_layer.quantize(encoded)  # (1,)
                    code_val = int(code_idx.item())
                    
                    self.sample_to_code[int(idx)] = code_val
                    self.code_to_samples.setdefault(code_val, []).append(int(idx))
            
            available_codes = sorted(self.code_to_samples.keys())
            codebook_size = getattr(self.model.vq_layer, "nb_code", None)
            if codebook_size is None:
                codebook_size = getattr(self.model.vq_layer, "n_e", 0)  # fallback for older quantizer variants
            
            code_counts = {c: len(self.code_to_samples[c]) for c in available_codes}
            
            status = f"Quantized {num_samples} samples.\n"
            status += f"Found {len(available_codes)} unique code indices.\n"
            if codebook_size:
                status += f"Codebook size: {codebook_size}\n"
                status += f"Coverage: {len(available_codes)}/{codebook_size} codes used ({100*len(available_codes)/max(1, codebook_size):.1f}%)\n\n"
            status += "Top 10 most used codes:\n"
            for code, count in sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                status += f"  Code {code}: {count} samples\n"
            
            return status, available_codes
        except Exception as e:
            import traceback
            return f"Error quantizing samples: {str(e)}\n{traceback.format_exc()}", []

    def get_samples_for_code(self, code_idx: int) -> Tuple[list, str]:
        """
        Get list of dataset sample indices for a given code index (after quantize_samples()).
        """
        if self.code_to_samples is None:
            return [], "Error: No samples quantized yet. Please click 'Quantize Samples' first."
        
        code_idx = int(code_idx)
        if code_idx not in self.code_to_samples:
            return [], f"Error: Code index {code_idx} not found in the quantized set."
        
        samples = self.code_to_samples[code_idx]
        info = f"Code {code_idx}: {len(samples)} samples\n"
        info += f"Sample indices (first 20): {samples[:20]}{'...' if len(samples) > 20 else ''}"
        return samples, info


def create_interface():
    """Create the Gradio interface."""
    app = VAEVisualizationApp()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px;
    }
    h1 {
        font-size: 28px !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
    }
    h2 {
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-top: 30px !important;
        margin-bottom: 15px !important;
        color: #2c3e50 !important;
    }
    h3 {
        font-size: 20px !important;
        font-weight: 500 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    hr {
        border: 2px solid #bdc3c7 !important;
        margin: 25px 0 !important;
    }
    """
    
    with gr.Blocks(title="VAE Motion Representation Visualization") as demo:
        gr.HTML(f"<style>{custom_css}</style>", visible=False)
        gr.Markdown("# 🎭 VAE Motion Representation Visualization")
        gr.Markdown("Visualize VAE reconstructions and explore the latent space of motion representations.")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📦 Model & Dataset Loading")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                checkpoint_path = gr.Textbox(
                    label="Checkpoint Path",
                    value="motion_representation/checkpoints/best_checkpoint.pth",
                    placeholder="Path to VAE checkpoint (e.g., motion_representation/checkpoints_Relationship_VQVAE_GRU/best_checkpoint.pth)"
                )
                load_model_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False, lines=3)
                
                lmdb_dir = gr.Textbox(
                    label="LMDB Directory",
                    value="dataset_processed_New/lmdb_Salsa_pair/lmdb_train",
                    placeholder="Path to LMDB directory"
                )
                is_MDM = gr.Checkbox(label="Is MDM Format", value=True)
                train_relationship = gr.Checkbox(
                    label="Load Relationship Features (3D InterHuman)",
                    value=False,
                    info="Check if loading relationship features (3D: [yaw, x, z]) instead of motion data"
                )
                load_dataset_btn = gr.Button("Load Dataset", variant="primary")
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False, lines=2)
            
            with gr.Column(scale=1):
                total_samples = gr.Number(label="Total Samples", value=0, interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔄 Reconstruction Visualization")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                with gr.Row():
                    prev_btn = gr.Button("◀ Previous", size="sm")
                    sample_idx = gr.Number(
                        label="Sample Index",
                        value=0,
                        minimum=0,
                        maximum=0,
                        step=1,
                        precision=0
                    )
                    next_btn = gr.Button("Next ▶", size="sm")
                    go_btn = gr.Button("Go", size="sm")
                
                visualize_recon_btn = gr.Button("Visualize Reconstruction", variant="primary")
                
                with gr.Row():
                    original_video = gr.Video(label="Original Motion", scale=1)
                    reconstructed_video = gr.Video(label="Reconstructed Motion", scale=1)
                
                recon_info = gr.Textbox(label="Reconstruction Info", interactive=False, lines=6)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🎨 Latent Space Visualization (t-SNE)")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                with gr.Row():
                    num_samples_tsne = gr.Number(
                        label="Number of Samples",
                        value=100,
                        minimum=10,
                        maximum=10000,
                        step=10,
                        precision=0
                    )
                    use_kmeans = gr.Checkbox(
                        label="Use K-Means Clustering",
                        value=False
                    )
                    n_clusters = gr.Number(
                        label="Number of Clusters",
                        value=5,
                        minimum=2,
                        maximum=50,
                        step=1,
                        precision=0,
                        visible=False
                    )
                
                def toggle_clusters(use_kmeans_val):
                    return gr.update(visible=use_kmeans_val)
                
                use_kmeans.change(
                    fn=toggle_clusters,
                    inputs=[use_kmeans],
                    outputs=[n_clusters]
                )
                
                visualize_latent_btn = gr.Button("Generate Latent Space Visualization", variant="primary")
                
                # Side-by-side layout: t-SNE plot and heatmap
                with gr.Row():
                    with gr.Column(scale=1):
                        latent_plot = gr.Plot(label="Interactive Latent Space t-SNE Visualization")
                    with gr.Column(scale=1):
                        latent_heatmap = gr.Plot(label="Latent Space Density Heatmap")
                        heatmap_sigma = gr.Slider(
                            label="Smoothing (σ)",
                            minimum=1.0,
                            maximum=4.0,
                            value=2.0,
                            step=0.1,
                            info="Adjust Gaussian smoothing for heatmap"
                        )
                
                        latent_info = gr.Textbox(label="Latent Space Info", interactive=False, lines=8)
                        gr.Markdown("**💡 Tip:** Click on any point in the t-SNE plot to visualize that sample!<br>Or manually enter the sample index (shown in hover tooltip) below.")
                    
                # Sample visualization in next row
                with gr.Row():
                    with gr.Column(scale=1):
                        clicked_sample_idx = gr.Number(
                            label="Selected Sample Index (click plot or enter manually)",
                            value=0,
                            minimum=0,
                            step=1,
                            precision=0,
                            elem_id="tsne_clicked_sample_idx"  # Add unique ID for JavaScript
                        )
                        visualize_clicked_btn = gr.Button("Visualize Selected Sample", variant="primary")
                    with gr.Column(scale=1):
                        with gr.Row():
                            clicked_original_video = gr.Video(label="Original Motion", scale=1)
                            clicked_reconstructed_video = gr.Video(label="Reconstructed Motion", scale=1)
                        clicked_recon_info = gr.Textbox(label="Reconstruction Info", interactive=False, lines=6)
                
                # Add JavaScript for click handling (hidden but will execute)
                # Use a div with style to hide it but still execute the script
                plot_js_component = gr.HTML(value="", visible=False)
        
        # Event handlers
        def on_load_model(checkpoint_path_val):
            status, success = app.load_model(checkpoint_path_val)
            return status
        
        def on_load_dataset(lmdb_dir_val, is_MDM_val, train_relationship_val):
            status, total = app.load_dataset(lmdb_dir_val, is_MDM_val, train_relationship_val)
            max_val = max(0, int(total) - 1) if total > 0 else 0
            return status, total, gr.update(maximum=max_val, value=0)
        
        def on_navigate(direction, current_idx_val, total_samples_val):
            try:
                current = int(current_idx_val) if current_idx_val is not None else 0
                total = int(total_samples_val) if total_samples_val is not None else 0
                new_idx = app.navigate_sample(direction, current, total)
                return new_idx
            except Exception as e:
                return 0
        
        def on_visualize_reconstruction(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                original, reconstructed, info = app.visualize_reconstruction(idx)
                return original, reconstructed, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        def on_visualize_latent(num_samples_val, use_kmeans_val, n_clusters_val):
            try:
                num_samples = int(num_samples_val) if num_samples_val is not None else 100
                use_kmeans = bool(use_kmeans_val) if use_kmeans_val is not None else False
                n_clusters = int(n_clusters_val) if n_clusters_val is not None else 5
                plot_fig, heatmap_fig, info = app.visualize_latent_space(num_samples, use_kmeans, n_clusters)
                return plot_fig, heatmap_fig, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        def on_visualize_clicked_sample(idx_val):
            """Visualize a sample by its index (from plot click or manual input)."""
            try:
                idx = int(idx_val) if idx_val is not None else 0
                original, reconstructed, info = app.visualize_reconstruction(idx)
                return original, reconstructed, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
                
                if point_idx is None or point_idx >= len(app.tsne_indices):
                    return None, None, "Error: Invalid point index. Please click on a point in the plot."
                
                # Get the sample index from our stored indices
                sample_idx = int(app.tsne_indices[point_idx])
                
                # Visualize the sample
                original, reconstructed, info = app.visualize_reconstruction(sample_idx)
                info = f"Clicked Sample Index: {sample_idx}\n\n" + info
                return original, reconstructed, info
            except Exception as e:
                import traceback
                error_msg = f"Error handling plot click: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        def on_go(idx_val):
            return on_visualize_reconstruction(idx_val)
        
        # Bind events
        load_model_btn.click(
            fn=on_load_model,
            inputs=[checkpoint_path],
            outputs=[model_status]
        )
        
        load_dataset_btn.click(
            fn=on_load_dataset,
            inputs=[lmdb_dir, is_MDM, train_relationship],
            outputs=[dataset_status, total_samples, sample_idx]
        )
        
        prev_btn.click(
            fn=lambda idx, total: on_navigate('prev', idx, total),
            inputs=[sample_idx, total_samples],
            outputs=[sample_idx]
        )
        
        next_btn.click(
            fn=lambda idx, total: on_navigate('next', idx, total),
            inputs=[sample_idx, total_samples],
            outputs=[sample_idx]
        )
        
        go_btn.click(
            fn=on_go,
            inputs=[sample_idx],
            outputs=[original_video, reconstructed_video, recon_info]
        )
        
        visualize_recon_btn.click(
            fn=on_visualize_reconstruction,
            inputs=[sample_idx],
            outputs=[original_video, reconstructed_video, recon_info]
        )
        
        # Auto-visualize when index changes
        sample_idx.change(
            fn=on_visualize_reconstruction,
            inputs=[sample_idx],
            outputs=[original_video, reconstructed_video, recon_info]
        )
        
        # Button to visualize clicked sample
        visualize_clicked_btn.click(
            fn=on_visualize_clicked_sample,
            inputs=[clicked_sample_idx],
            outputs=[clicked_original_video, clicked_reconstructed_video, clicked_recon_info]
        )
        
        # Auto-visualize when clicked_sample_idx changes (from plot click or manual input)
        clicked_sample_idx.change(
            fn=on_visualize_clicked_sample,
            inputs=[clicked_sample_idx],
            outputs=[clicked_original_video, clicked_reconstructed_video, clicked_recon_info]
        )
        
        # Add JavaScript to capture Plotly clicks and update the input field
        # Update the JavaScript component when plot is generated
        def on_update_heatmap_sigma(sigma_val):
            """Update heatmap with new sigma value."""
            try:
                if app.heatmap_hist is None or app.heatmap_edges is None:
                    return None
                sigma = float(sigma_val) if sigma_val is not None else 2.0
                x_edges, y_edges = app.heatmap_edges
                # Get num_samples from stored data (approximate from histogram size)
                num_samples = int(app.heatmap_hist.sum()) if hasattr(app.heatmap_hist, 'sum') else 100
                heatmap_fig = app._create_smoothed_heatmap(
                    app.heatmap_hist, x_edges, y_edges, sigma=sigma, num_samples=num_samples
                )
                return heatmap_fig
            except Exception as e:
                import traceback
                error_msg = f"Error updating heatmap: {str(e)}\n{traceback.format_exc()}"
                return None
        
        def on_visualize_latent_with_js(num_samples_val, use_kmeans_val, n_clusters_val):
            """Visualize latent space and return plot with JavaScript."""
            try:
                num_samples = int(num_samples_val) if num_samples_val is not None else 100
                use_kmeans = bool(use_kmeans_val) if use_kmeans_val is not None else False
                n_clusters = int(n_clusters_val) if n_clusters_val is not None else 5
                plot_fig, heatmap_fig, info = app.visualize_latent_space(num_samples, use_kmeans, n_clusters)
                
                # Generate JavaScript for click handling - more robust version
                # Use regular string (not f-string) since we're not using any Python variables
                plot_click_js = """
                <div style="display:none;"><script>
                (function() {{
                    var handlerSetup = false;
                    
                    function setupClickHandler() {{
                        if (handlerSetup) return;
                        
                        // Wait for plot to be fully rendered
                        setTimeout(function() {{
                            // Find the plotly div
                            var plotDivs = document.querySelectorAll('.plotly');
                            
                            plotDivs.forEach(function(plotDiv) {{
                                if (plotDiv._fullLayout && !plotDiv._clickHandlerAttached) {{
                                    plotDiv._clickHandlerAttached = true;
                                    
                                    // Handle point selection
                                    function handleSelection(data) {{
                                        if (!data?.points?.[0]?.customdata) return;
                                        var idx = data.points[0].customdata;
                                        var input = document.getElementById('tsne_clicked_sample_idx') ||
                                                   Array.from(document.querySelectorAll('input[type="number"]')).find(function(inp) {{
                                                       var label = (inp.closest('.form, form') || inp.parentElement)?.querySelector('label');
                                                       return label && label.textContent.includes('Selected Sample Index');
                                                   }});
                                        if (input) {{
                                            input.value = idx;
                                            input.dispatchEvent(new Event('change', {{bubbles:true}}));
                                        }}
                                    }}
                                    
                                    // Attach handlers for both click and selection
                                    plotDiv.on('plotly_click', handleSelection);
                                    plotDiv.on('plotly_selected', handleSelection);
                                    
                                    // Old handler (keeping for compatibility)
                                    plotDiv.on('plotly_click', function(data) {{
                                        console.log('Plot clicked!', data);
                                        
                                        if (data && data.points && data.points.length > 0) {{
                                            var point = data.points[0];
                                            var customdata = point.customdata;
                                            
                                            console.log('Custom data:', customdata);
                                            
                                            if (customdata !== undefined && customdata !== null) {{
                                                // Find the input field by searching for the label text
                                                var allInputs = document.querySelectorAll('input[type="number"]');
                                                var targetInput = null;
                                                
                                                for (var i = 0; i < allInputs.length; i++) {{
                                                    var input = allInputs[i];
                                                    // Search up the DOM tree for label
                                                    var element = input;
                                                    for (var j = 0; j < 10 && element; j++) {{
                                                        var labels = element.querySelectorAll ? element.querySelectorAll('label') : [];
                                                        if (labels.length === 0 && element.parentElement) {{
                                                            labels = element.parentElement.querySelectorAll ? element.parentElement.querySelectorAll('label') : [];
                                                        }
                                                        
                                                        for (var k = 0; k < labels.length; k++) {{
                                                            var labelText = labels[k].textContent || labels[k].innerText || '';
                                                            if (labelText.includes('Selected Sample Index')) {{
                                                                targetInput = input;
                                                                break;
                                                            }}
                                                        }}
                                                        
                                                        if (targetInput) break;
                                                        element = element.parentElement;
                                                    }}
                                                    
                                                    if (targetInput) break;
                                                }}
                                                
                                                // Try to find by ID first (more reliable)
                                                var inputById = document.getElementById('tsne_clicked_sample_idx');
                                                if (!inputById) {{
                                                    // Fallback to searching by label
                                                    if (targetInput) {{
                                                        console.log('Found input by label, setting value to:', customdata);
                                                        inputById = targetInput;
                                                    }}
                                                }}
                                                
                                                if (inputById) {{
                                                    console.log('Found input, setting value to:', customdata);
                                                    inputById.value = customdata;
                                                    
                                                    // Trigger multiple events to ensure Gradio picks it up
                                                    var events = ['input', 'change', 'blur'];
                                                    events.forEach(function(eventType) {{
                                                        var evt = new Event(eventType, {{ bubbles: true, cancelable: true }});
                                                        inputById.dispatchEvent(evt);
                                                    }});
                                                    
                                                    // Also try setting it via Gradio's internal mechanism
                                                    if (inputById.oninput) inputById.oninput();
                                                    if (inputById.onchange) inputById.onchange();
                                                    
                                                    // Force update by focusing and blurring
                                                    inputById.focus();
                                                    setTimeout(function() {{ inputById.blur(); }}, 100);
                                                }} else {{
                                                    console.error('Could not find target input field with ID tsne_clicked_sample_idx');
                                                }}
                                            }}
                                        }}
                                    }});
                                    
                                    // Also handle selection event (fires when point is selected/faded)
                                    plotDiv.on('plotly_selected', function(data) {{
                                        if (data && data.points && data.points.length > 0) {{
                                            var point = data.points[0];
                                            var customdata = point.customdata;
                                            if (customdata !== undefined && customdata !== null) {{
                                                var input = document.getElementById('tsne_clicked_sample_idx');
                                                if (!input) {{
                                                    var allInputs = document.querySelectorAll('input[type="number"]');
                                                    for (var i = 0; i < allInputs.length; i++) {{
                                                        var inp = allInputs[i];
                                                        var parent = inp.closest('.form, form') || inp.parentElement;
                                                        var labels = parent ? parent.querySelectorAll('label') : [];
                                                        for (var j = 0; j < labels.length; j++) {{
                                                            if ((labels[j].textContent || '').includes('Selected Sample Index')) {{
                                                                input = inp;
                                                                break;
                                                            }}
                                                        }}
                                                        if (input) break;
                                                    }}
                                                }}
                                                if (input) {{
                                                    input.value = customdata;
                                                    input.dispatchEvent(new Event('change', {{bubbles:true}}));
                                                }}
                                            }}
                                        }}
                                    }});
                                    
                                    handlerSetup = true;
                                }}
                            }});
                        }}, 1000);
                    }}
                    
                    // Setup immediately and also on DOM ready
                    setupClickHandler();
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', setupClickHandler);
                    }}
                    
                    // Re-setup when new content is added
                    var observer = new MutationObserver(function() {{
                        handlerSetup = false;
                        setupClickHandler();
                    }});
                    observer.observe(document.body, {{ childList: true, subtree: true }});
                }})();
                </script></div>
                """
                return plot_fig, heatmap_fig, info, plot_click_js
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg, ""
        
        # Update the click handler to use the new function
        visualize_latent_btn.click(
            fn=on_visualize_latent_with_js,
            inputs=[num_samples_tsne, use_kmeans, n_clusters],
            outputs=[latent_plot, latent_heatmap, latent_info, plot_js_component]
        )
        
        # Update heatmap when sigma changes
        heatmap_sigma.change(
            fn=on_update_heatmap_sigma,
            inputs=[heatmap_sigma],
            outputs=[latent_heatmap]
        )
        
        # Codebook Debugging Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔍 Codebook Debugging (VQ-VAE Only)")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                gr.Markdown("Analyze codebook state, usage, and diversity to debug collapse issues.")
                
                with gr.Row():
                    num_samples_codebook = gr.Number(
                        label="Number of Samples for Analysis",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                        precision=0
                    )
                    analyze_codebook_btn = gr.Button("Analyze Codebook", variant="primary")
                
                codebook_plot = gr.Plot(label="Codebook Analysis Dashboard")
                codebook_info = gr.Textbox(
                    label="Codebook Analysis Results",
                    interactive=False,
                    lines=25,
                    max_lines=30
                )
        
        def on_analyze_codebook(num_samples_val):
            """Analyze codebook state."""
            try:
                num_samples = int(num_samples_val) if num_samples_val is not None else 1000
                plot_fig, info = app.analyze_codebook(num_samples)
                return plot_fig, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, error_msg
        
        analyze_codebook_btn.click(
            fn=on_analyze_codebook,
            inputs=[num_samples_codebook],
            outputs=[codebook_plot, codebook_info]
        )

        # Token Cluster Exploration Section (under codebook analysis)
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔍 Token Cluster Exploration (VQ-VAE Only)")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                gr.Markdown(
                    "Quantize a batch of samples, then pick a **token/code index** and browse samples assigned to it. "
                    "This helps inspect how well the VQ-VAE codes cluster motions."
                )
                
                with gr.Row():
                    num_samples_quantize = gr.Number(
                        label="Number of Samples to Quantize",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                        precision=0,
                    )
                    quantize_btn = gr.Button("Quantize Samples", variant="primary")
                
                quantize_status = gr.Textbox(label="Quantization Status", interactive=False, lines=8)
                
                with gr.Row():
                    code_idx_dropdown = gr.Dropdown(
                        label="Select Token/Code Index",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Dropdown shows only codes observed in the quantized subset"
                    )
                    refresh_codes_btn = gr.Button("🔄 Refresh", size="sm")
                
                cluster_info = gr.Textbox(label="Cluster Information", interactive=False, lines=4)
                
                with gr.Row():
                    cluster_sample_pos = gr.Number(
                        label="Sample Position in Cluster",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                        info="Index within this cluster list (0 = first sample in cluster)"
                    )
                    cluster_prev_btn = gr.Button("◀ Previous", size="sm")
                    cluster_next_btn = gr.Button("Next ▶", size="sm")
                
                visualize_cluster_btn = gr.Button("Visualize Sample from Cluster", variant="primary")
                
                with gr.Row():
                    cluster_original_video = gr.Video(label="Original Motion", scale=1)
                    cluster_reconstructed_video = gr.Video(label="Reconstructed Motion", scale=1)
                
                cluster_recon_info = gr.Textbox(label="Reconstruction Info", interactive=False, lines=8)
        
        def _parse_code_from_choice(code_choice: str) -> Optional[int]:
            if code_choice is None:
                return None
            s = str(code_choice).strip()
            if not s:
                return None
            # Expected: "Code {idx} ({n} samples)"
            try:
                parts = s.split()
                return int(parts[1])
            except Exception:
                try:
                    return int(s)
                except Exception:
                    return None
        
        def on_quantize_samples(num_samples_val):
            try:
                n = int(num_samples_val) if num_samples_val is not None else 1000
                status, available_codes = app.quantize_samples(n)
                
                if available_codes:
                    choices = [f"Code {c} ({len(app.code_to_samples[c])} samples)" for c in available_codes]
                    first_val = choices[0]
                else:
                    choices = []
                    first_val = None
                
                # Reset cluster selection state
                return (
                    status,
                    gr.update(choices=choices, value=first_val),
                    "",
                    gr.update(value=0, minimum=0, maximum=0),
                )
            except Exception as e:
                import traceback
                return (
                    f"Error: {str(e)}\n{traceback.format_exc()}",
                    gr.update(choices=[], value=None),
                    "",
                    gr.update(value=0, minimum=0, maximum=0),
                )
        
        def on_code_selected(code_choice):
            try:
                code_idx = _parse_code_from_choice(code_choice)
                if code_idx is None:
                    return "", gr.update(value=0, minimum=0, maximum=0)
                
                samples, info = app.get_samples_for_code(code_idx)
                max_pos = max(0, len(samples) - 1)
                return info, gr.update(value=0, minimum=0, maximum=max_pos)
            except Exception as e:
                import traceback
                return f"Error: {str(e)}\n{traceback.format_exc()}", gr.update(value=0, minimum=0, maximum=0)
        
        def on_visualize_cluster_sample(code_choice, cluster_pos_val):
            try:
                code_idx = _parse_code_from_choice(code_choice)
                if code_idx is None:
                    return None, None, "Error: Please select a code index first."
                
                samples, _ = app.get_samples_for_code(code_idx)
                if not samples:
                    return None, None, "Error: No samples found for this code."
                
                pos = int(cluster_pos_val) if cluster_pos_val is not None else 0
                pos = max(0, min(pos, len(samples) - 1))
                sample_idx_val = samples[pos]
                
                original, reconstructed, info = app.visualize_reconstruction(sample_idx_val)
                header = f"Cluster sample {pos}/{len(samples)-1}\nDataset sample index: {sample_idx_val}\nCode: {code_idx}\n\n"
                return original, reconstructed, header + info
            except Exception as e:
                import traceback
                return None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        def on_cluster_navigate(direction, code_choice, cluster_pos_val):
            try:
                code_idx = _parse_code_from_choice(code_choice)
                if code_idx is None:
                    return gr.update(value=0), None, None, "Error: Please select a code index first."
                
                samples, _ = app.get_samples_for_code(code_idx)
                if not samples:
                    return gr.update(value=0), None, None, "Error: No samples found for this code."
                
                pos = int(cluster_pos_val) if cluster_pos_val is not None else 0
                if direction == "prev":
                    pos = max(0, pos - 1)
                else:
                    pos = min(len(samples) - 1, pos + 1)
                
                sample_idx_val = samples[pos]
                original, reconstructed, info = app.visualize_reconstruction(sample_idx_val)
                header = f"Cluster sample {pos}/{len(samples)-1}\nDataset sample index: {sample_idx_val}\nCode: {code_idx}\n\n"
                return gr.update(value=pos), original, reconstructed, header + info
            except Exception as e:
                import traceback
                return gr.update(value=0), None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        quantize_btn.click(
            fn=on_quantize_samples,
            inputs=[num_samples_quantize],
            outputs=[quantize_status, code_idx_dropdown, cluster_info, cluster_sample_pos],
        )
        
        refresh_codes_btn.click(
            fn=on_quantize_samples,
            inputs=[num_samples_quantize],
            outputs=[quantize_status, code_idx_dropdown, cluster_info, cluster_sample_pos],
        )
        
        code_idx_dropdown.change(
            fn=on_code_selected,
            inputs=[code_idx_dropdown],
            outputs=[cluster_info, cluster_sample_pos],
        )
        
        visualize_cluster_btn.click(
            fn=on_visualize_cluster_sample,
            inputs=[code_idx_dropdown, cluster_sample_pos],
            outputs=[cluster_original_video, cluster_reconstructed_video, cluster_recon_info],
        )
        
        cluster_prev_btn.click(
            fn=lambda c, p: on_cluster_navigate("prev", c, p),
            inputs=[code_idx_dropdown, cluster_sample_pos],
            outputs=[cluster_sample_pos, cluster_original_video, cluster_reconstructed_video, cluster_recon_info],
        )
        
        cluster_next_btn.click(
            fn=lambda c, p: on_cluster_navigate("next", c, p),
            inputs=[code_idx_dropdown, cluster_sample_pos],
            outputs=[cluster_sample_pos, cluster_original_video, cluster_reconstructed_video, cluster_recon_info],
        )
        
        # Long Sequence Visualization Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🎬 Long Sequence Generation")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                gr.Markdown("Generate long sequences using autoregressive inference. Load raw LMDB to select videos and clips.")
                
                # Raw LMDB loading
                raw_lmdb_dir = gr.Textbox(
                    label="Raw LMDB Directory",
                    value="dataset_processed_New/lmdb_Salsa_pair/lmdb_train",
                    placeholder="Path to raw LMDB directory (not cache)"
                )
                load_raw_lmdb_btn = gr.Button("Load Raw LMDB", variant="primary")
                raw_lmdb_status = gr.Textbox(label="Raw LMDB Status", interactive=False, lines=3)
                
                # Video and clip selection
                with gr.Row():
                    video_idx_input = gr.Number(
                        label="Video Index",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                        info="Index of video in LMDB"
                    )
                    get_clips_btn = gr.Button("Get Clips Info", variant="secondary")
                
                clips_info = gr.Textbox(
                    label="Clips Information",
                    interactive=False,
                    lines=8
                )
                
                # Clip and sequence selection
                with gr.Row():
                    clip_idx_input = gr.Number(
                        label="Clip Index",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0
                    )
                    role_select = gr.Radio(
                        label="Role",
                        choices=["L", "F"],
                        value="L",
                        info="L = Leader, F = Follower"
                    )
                
                with gr.Row():
                    start_frame_input = gr.Number(
                        label="Start Frame",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0
                    )
                    end_frame_input = gr.Number(
                        label="End Frame",
                        value=100,
                        minimum=20,
                        step=1,
                        precision=0,
                        info="Must be multiple of 20 for best results"
                    )
                
                generate_long_btn = gr.Button("Generate Long Sequence", variant="primary")
                
                # Output videos
                with gr.Row():
                    long_original_video = gr.Video(label="Original Long Sequence", scale=1)
                    long_generated_video = gr.Video(label="Generated Long Sequence", scale=1)
                
                long_sequence_info = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    lines=10
                )
        
        # Event handlers for long sequence
        def on_load_raw_lmdb(lmdb_dir_val):
            status, num_videos = app.load_raw_lmdb(lmdb_dir_val)
            max_video = max(0, int(num_videos) - 1) if num_videos > 0 else 0
            return status, gr.update(maximum=max_video, value=0)
        
        def on_get_clips_info(video_idx_val):
            try:
                video_idx = int(video_idx_val) if video_idx_val is not None else 0
                clip_info_dict, info_str = app.get_video_clips_info(video_idx)
                if clip_info_dict is None:
                    return info_str, gr.update(maximum=0, value=0)
                
                num_clips = len(clip_info_dict['clips'])
                max_clip = max(0, num_clips - 1) if num_clips > 0 else 0
                return info_str, gr.update(maximum=max_clip, value=0)
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return error_msg, gr.update(maximum=0, value=0)
        
        def on_generate_long_sequence(video_idx_val, clip_idx_val, role_val, start_frame_val, end_frame_val):
            try:
                video_idx = int(video_idx_val) if video_idx_val is not None else 0
                clip_idx = int(clip_idx_val) if clip_idx_val is not None else 0
                role = str(role_val) if role_val is not None else "L"
                start_frame = int(start_frame_val) if start_frame_val is not None else 0
                end_frame = int(end_frame_val) if end_frame_val is not None else 100
                
                original, generated, info = app.visualize_long_sequence(
                    video_idx, clip_idx, role, start_frame, end_frame
                )
                return original, generated, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        load_raw_lmdb_btn.click(
            fn=on_load_raw_lmdb,
            inputs=[raw_lmdb_dir],
            outputs=[raw_lmdb_status, video_idx_input]
        )
        
        get_clips_btn.click(
            fn=on_get_clips_info,
            inputs=[video_idx_input],
            outputs=[clips_info, clip_idx_input]
        )
        
        generate_long_btn.click(
            fn=on_generate_long_sequence,
            inputs=[video_idx_input, clip_idx_input, role_select, start_frame_input, end_frame_input],
            outputs=[long_original_video, long_generated_video, long_sequence_info]
        )
        
        # Combined Reconstruction Section (reuses same inputs)
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 👥 Combined Reconstruction (Both Dancers)")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                gr.Markdown("Visualize both leader and follower dancing together: original vs reconstructed. Uses the same video/clip/frame selection above.")
                
                with gr.Row():
                    combined_original_video = gr.Video(label="Original Combined (Both Dancers)", scale=1)
                    combined_recon_video = gr.Video(label="Reconstructed Combined (Both Dancers)", scale=1)
                
                combined_recon_info = gr.Textbox(
                    label="Reconstruction Info",
                    interactive=False,
                    lines=8
                )
                
                generate_combined_btn = gr.Button("Generate Combined Reconstruction", variant="primary")
        
        def on_generate_combined(video_idx_val, clip_idx_val, start_frame_val, end_frame_val):
            try:
                video_idx = int(video_idx_val) if video_idx_val is not None else 0
                clip_idx = int(clip_idx_val) if clip_idx_val is not None else 0
                start_frame = int(start_frame_val) if start_frame_val is not None else 0
                end_frame = int(end_frame_val) if end_frame_val is not None else 100
                
                original, reconstructed, info = app.visualize_combined_reconstruction(
                    video_idx, clip_idx, start_frame, end_frame
                )
                return original, reconstructed, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        generate_combined_btn.click(
            fn=on_generate_combined,
            inputs=[video_idx_input, clip_idx_input, start_frame_input, end_frame_input],
            outputs=[combined_original_video, combined_recon_video, combined_recon_info]
        )
        
        # Relationship Features Visualization Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔗 Relationship Features Visualization")
                gr.Markdown("<hr style='border: 2px solid #666; margin: 20px 0;'>")
                gr.Markdown("Visualize 3D relationship features ([yaw, x, z] from InterHuman canonical frames) as time series plots. Each plot shows original vs reconstructed for one dimension.")
                
                with gr.Row():
                    rel_sample_idx = gr.Number(
                        label="Sample Index",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                        info="Index from the relationship features dataset"
                    )
                    rel_prev_btn = gr.Button("◀ Previous", size="sm")
                    rel_next_btn = gr.Button("Next ▶", size="sm")
                
                rel_plot = gr.Plot(label="Relationship Features Time Series")
                rel_info = gr.Textbox(
                    label="Visualization Info",
                    interactive=False,
                    lines=10
                )
                
                visualize_rel_btn = gr.Button("Visualize Relationship Features", variant="primary")
        
        def on_visualize_relationship(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                plot_fig, info = app.visualize_relationship_features(idx)
                return plot_fig, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, error_msg
        
        def on_rel_prev(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                new_idx = max(0, idx - 1)
                plot_fig, info = app.visualize_relationship_features(new_idx)
                return new_idx, plot_fig, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return idx_val, None, error_msg
        
        def on_rel_next(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                max_idx = len(app.dataloader.dataset) - 1 if app.dataloader else 0
                new_idx = min(max_idx, idx + 1)
                plot_fig, info = app.visualize_relationship_features(new_idx)
                return new_idx, plot_fig, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return idx_val, None, error_msg
        
        visualize_rel_btn.click(
            fn=on_visualize_relationship,
            inputs=[rel_sample_idx],
            outputs=[rel_plot, rel_info]
        )
        
        rel_prev_btn.click(
            fn=on_rel_prev,
            inputs=[rel_sample_idx],
            outputs=[rel_sample_idx, rel_plot, rel_info]
        )
        
        rel_next_btn.click(
            fn=on_rel_next,
            inputs=[rel_sample_idx],
            outputs=[rel_sample_idx, rel_plot, rel_info]
        )
        
        # ========================================================================
        # Combined Motion + Relationship Reconstruction Section
        # ========================================================================
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔄 Combined Motion + Relationship Reconstruction")
                gr.Markdown("<hr style='border: 3px solid #333; margin: 20px 0;'>")
                gr.Markdown(
                    "**Reconstruct pairs using both motion and relationship networks.**\n\n"
                    "This section loads two models:\n"
                    "- **Motion Network**: Reconstructs canonicalized InterHuman motions (leader and follower separately)\n"
                    "- **Relationship Network**: Reconstructs relationship features [w, z, x, z]\n\n"
                    "The pipeline:\n"
                    "1. Reconstruct leader motion (canonicalized) from motion network\n"
                    "2. Reconstruct follower motion (canonicalized) from motion network\n"
                    "3. Reconstruct relationship features from relationship network\n"
                    "4. Convert relationship[0] to rigid_transform parameters [angle, x, z]\n"
                    "5. Apply rigid_transform to reconstructed follower to move it into leader's space\n"
                    "6. Visualize GT vs reconstructed side-by-side"
                )
                
                with gr.Row():
                    with gr.Column():
                        motion_model_path = gr.Textbox(
                            label="Motion Model Checkpoint Path",
                            placeholder="path/to/motion_model.ckpt",
                            info="Path to InterHuman motion model checkpoint"
                        )
                        relationship_model_path = gr.Textbox(
                            label="Relationship Model Checkpoint Path",
                            placeholder="path/to/relationship_model.ckpt",
                            info="Path to relationship features model checkpoint"
                        )
                
                with gr.Row():
                    combined_sample_idx = gr.Number(
                        label="Sample Index",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                        info="Index from the dataset (must be InterHuman or relationship representation_type)"
                    )
                    combined_prev_btn = gr.Button("◀ Previous", size="sm")
                    combined_next_btn = gr.Button("Next ▶", size="sm")
                
                with gr.Row():
                    combined_gt_video = gr.Video(label="Ground Truth: Leader + Follower")
                    combined_recon_video = gr.Video(label="Reconstructed: Leader + Follower")
                
                combined_info = gr.Textbox(
                    label="Visualization Info",
                    interactive=False,
                    lines=15
                )
                
                visualize_combined_btn = gr.Button("Visualize Combined Reconstruction", variant="primary")
        
        def on_visualize_combined(motion_path, rel_path, idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                gt_video, recon_video, info = app.visualize_combined_reconstruction(
                    motion_path, rel_path, idx
                )
                return gt_video, recon_video, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        def on_combined_prev(motion_path, rel_path, idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                new_idx = max(0, idx - 1)
                gt_video, recon_video, info = app.visualize_combined_reconstruction(
                    motion_path, rel_path, new_idx
                )
                return new_idx, gt_video, recon_video, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return idx_val, None, None, error_msg
        
        def on_combined_next(motion_path, rel_path, idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                max_idx = len(app.dataloader.dataset) - 1 if app.dataloader else 0
                new_idx = min(max_idx, idx + 1)
                gt_video, recon_video, info = app.visualize_combined_reconstruction(
                    motion_path, rel_path, new_idx
                )
                return new_idx, gt_video, recon_video, info
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return idx_val, None, None, error_msg
        
        visualize_combined_btn.click(
            fn=on_visualize_combined,
            inputs=[motion_model_path, relationship_model_path, combined_sample_idx],
            outputs=[combined_gt_video, combined_recon_video, combined_info]
        )
        
        combined_prev_btn.click(
            fn=on_combined_prev,
            inputs=[motion_model_path, relationship_model_path, combined_sample_idx],
            outputs=[combined_sample_idx, combined_gt_video, combined_recon_video, combined_info]
        )
        
        combined_next_btn.click(
            fn=on_combined_next,
            inputs=[motion_model_path, relationship_model_path, combined_sample_idx],
            outputs=[combined_sample_idx, combined_gt_video, combined_recon_video, combined_info]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7863)

