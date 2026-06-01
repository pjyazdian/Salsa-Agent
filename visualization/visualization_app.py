"""Web application for visualizing LMDB dataset samples.""" 
import os
import sys
import gradio as gr
import numpy as np
import torch
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.visualization_utils import (
    LMDBLoader,
    render_skeleton_from_keypoints,
    render_skeleton_from_ric,
    render_combined_skeletons,
    create_video_with_audio,
    render_vqvae_reconstruction_comparison,
    decode_motion_from_vq_tokens,
    extract_original_video_clip,
    DEBUG,
    debug_print
)


class VisualizationApp:
    """Main visualization application."""
    
    def __init__(self):
        self.loader: Optional[LMDBLoader] = None
        self.current_idx = 0
        self.temp_dir = tempfile.mkdtemp()
        self.is_MDM = False
        self.motion_tokenizer = None
        self.audio_tokenizer = None
        self.parent_dir = Path(__file__).parent.parent
    
    def load_lmdb(self, lmdb_path: str, is_MDM: bool) -> Tuple[str, int, int]:
        """
        Load an LMDB file.
        
        Args:
            lmdb_path: Path to LMDB directory (will auto-detect cache if available)
            is_MDM: Whether data is in MDM format
            
        Returns:
            Status message, current index, and total samples
        """
        try:
            if not os.path.exists(lmdb_path):
                return f"Error: LMDB path does not exist: {lmdb_path}", 0, 0
            
            if self.loader:
                self.loader.close()
            
            self.loader = LMDBLoader(lmdb_path, is_MDM=is_MDM)
            self.is_MDM = is_MDM
            self.current_idx = 0
            
            total_samples = len(self.loader)
            actual_path = self.loader.lmdb_dir
            status_msg = f"Loaded LMDB with {total_samples} samples from: {actual_path}"
            if actual_path != lmdb_path:
                status_msg += f"\n(Using cached version, original: {lmdb_path})"
            return status_msg, 0, total_samples
        except Exception as e:
            import traceback
            error_msg = f"Error loading LMDB: {str(e)}\n{traceback.format_exc()}"
            debug_print(error_msg)
            return error_msg, 0, 0
    
    def get_sample_info(self, idx: int) -> Tuple[str, dict]:
        """
        Get sample information.
        
        Args:
            idx: Sample index
            
        Returns:
            Info string and metadata dictionary
        """
        if self.loader is None:
            return "No LMDB loaded", {}
        
        try:
            if idx < 0 or idx >= len(self.loader):
                return f"Index {idx} out of range (0-{len(self.loader)-1})", {}
            
            sample = self.loader.get_sample(idx)
            aux_info = sample['aux_info']
            
            info_lines = [
                f"Sample Index: {idx}",
                f"Video ID: {aux_info.get('vid', 'N/A')}",
                f"Start Frame: {aux_info.get('start_frame_no', 'N/A')}",
                f"End Frame: {aux_info.get('end_frame_no', 'N/A')}",
                f"Start Time: {aux_info.get('start_time', 'N/A'):.2f}s" if isinstance(aux_info.get('start_time'), (int, float)) else f"Start Time: {aux_info.get('start_time', 'N/A')}",
                f"End Time: {aux_info.get('end_time', 'N/A'):.2f}s" if isinstance(aux_info.get('end_time'), (int, float)) else f"End Time: {aux_info.get('end_time', 'N/A')}",
                f"Duration: {aux_info.get('end_time', 0) - aux_info.get('start_time', 0):.2f}s" if isinstance(aux_info.get('start_time'), (int, float)) and isinstance(aux_info.get('end_time'), (int, float)) else "Duration: N/A",
            ]
            
            if sample.get('ms_desc_L'):
                info_lines.append(f"Leader Motion Script: {sample['ms_desc_L']}")
            if sample.get('ms_des_F'):
                info_lines.append(f"Follower Motion Script: {sample['ms_des_F']}")
            
            return "\n".join(info_lines), aux_info
        except Exception as e:
            return f"Error loading sample: {str(e)}", {}
    
    def visualize_sample(
        self,
        idx: int,
        show_leader: bool,
        show_follower: bool,
        show_combined: bool = False
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
        """
        Visualize a sample.
        
        Args:
            idx: Sample index
            show_leader: Whether to show leader skeleton
            show_follower: Whether to show follower skeleton
            
        Returns:
            Leader video path, follower video path, and info string
        """
        if self.loader is None:
            return None, None, "No LMDB loaded. Please load an LMDB file first."
        
        try:
            debug_print(f"visualize_sample called: idx={idx}, show_leader={show_leader}, show_follower={show_follower}")
            
            if idx < 0:
                debug_print(f"Negative index: {idx}")
                return None, None, f"Index {idx} is negative. Please use a valid index."
            
            # Try to get the sample - if it fails, try a few nearby indices
            sample = None
            actual_idx = idx
            max_attempts = 5
            debug_print(f"Attempting to load sample, starting at idx={idx}, max_attempts={max_attempts}")
            
            for attempt in range(max_attempts):
                try:
                    if actual_idx >= len(self.loader):
                        debug_print(f"Index {actual_idx} >= len(loader)={len(self.loader)}")
                        return None, None, f"Index {idx} out of range (max: {len(self.loader)-1})"
                    
                    debug_print(f"Attempt {attempt + 1}/{max_attempts}: trying to get sample at idx={actual_idx}")
                    sample = self.loader.get_sample(actual_idx)
                    debug_print(f"Successfully loaded sample at idx={actual_idx}")
                    break
                except (KeyError, ValueError, IndexError) as e:
                    debug_print(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                    if attempt < max_attempts - 1:
                        # Try next index
                        actual_idx += 1
                        debug_print(f"Trying next index: {actual_idx}")
                        continue
                    else:
                        debug_print(f"All attempts failed")
                        return None, None, f"Could not load sample at index {idx} (tried up to {actual_idx}): {str(e)}"
            
            if sample is None:
                debug_print("Sample is None after all attempts")
                return None, None, f"Failed to load sample at index {idx}"
            
            debug_print(f"Sample loaded successfully, keys: {list(sample.keys())}")
            
            aux_info = sample['aux_info']
            vid_id = aux_info.get('vid', f'sample_{actual_idx}')
            
            # Update info if we used a different index
            info_note = ""
            if actual_idx != idx:
                info_note = f"Note: Requested index {idx}, but loaded index {actual_idx} (previous indices were invalid)\n"
            
            leader_video = None
            follower_video = None
            combined_video = None
            info_lines = []
            error_messages = []
            
            # Store keypoints for combined rendering
            leader_keypoints_for_combined = None
            follower_keypoints_for_combined = None
            
            # Render leader skeleton
            if show_leader:
                try:
                    debug_print("Rendering leader skeleton...")
                    leader_keypoints = sample['poses_keypoints3d_L']
                    leader_keypoints_for_combined = leader_keypoints.copy()
                    debug_print(f"Leader keypoints shape: {leader_keypoints.shape}, type: {type(leader_keypoints)}")
                    leader_video_path = os.path.join(self.temp_dir, f"leader_{actual_idx}.mp4")
                    title = f"Leader: {vid_id}"
                    debug_print(f"Rendering leader skeleton to {leader_video_path}...")
                    render_skeleton_from_keypoints(
                        leader_keypoints,
                        leader_video_path,
                        title=title,
                        fps=20,
                        radius=4,
                        figsize=(6, 6),  # Smaller figure size for reduced file size
                        dpi=100  # Lower DPI for smaller files
                    )
                    if os.path.exists(leader_video_path):
                        leader_video = leader_video_path
                        file_size = os.path.getsize(leader_video_path)
                        info_lines.append(f"Leader: {leader_keypoints.shape[0]} frames, {leader_keypoints.shape[1]} joints")
                        debug_print(f"Leader video created successfully: {leader_video_path}, size: {file_size} bytes")
                    else:
                        error_msg = f"Leader video file was not created at {leader_video_path}"
                        error_messages.append(error_msg)
                        debug_print(error_msg)
                except Exception as e:
                    import traceback
                    error_msg = f"Error rendering leader: {str(e)}\n{traceback.format_exc()}"
                    error_messages.append(error_msg)
                    debug_print(error_msg)
            
            # Render follower skeleton
            if show_follower:
                try:
                    debug_print("Rendering follower skeleton...")
                    follower_keypoints = sample['poses_keypoints3d_F']
                    follower_keypoints_for_combined = follower_keypoints.copy()
                    debug_print(f"Follower keypoints shape: {follower_keypoints.shape}, type: {type(follower_keypoints)}")
                    follower_video_path = os.path.join(self.temp_dir, f"follower_{actual_idx}.mp4")
                    title = f"Follower: {vid_id}"
                    debug_print(f"Rendering follower skeleton to {follower_video_path}...")
                    render_skeleton_from_keypoints(
                        follower_keypoints,
                        follower_video_path,
                        title=title,
                        fps=20,
                        radius=4,
                        figsize=(6, 6),  # Smaller figure size for reduced file size
                        dpi=100  # Lower DPI for smaller files
                    )
                    if os.path.exists(follower_video_path):
                        follower_video = follower_video_path
                        file_size = os.path.getsize(follower_video_path)
                        info_lines.append(f"Follower: {follower_keypoints.shape[0]} frames, {follower_keypoints.shape[1]} joints")
                        debug_print(f"Follower video created successfully: {follower_video_path}, size: {file_size} bytes")
                    else:
                        error_msg = f"Follower video file was not created at {follower_video_path}"
                        error_messages.append(error_msg)
                        debug_print(error_msg)
                except Exception as e:
                    import traceback
                    error_msg = f"Error rendering follower: {str(e)}\n{traceback.format_exc()}"
                    error_messages.append(error_msg)
                    debug_print(error_msg)
            
            # Get sample info (use actual_idx)
            info_str, _ = self.get_sample_info(actual_idx)
            if info_note:
                info_str = info_note + info_str
            if info_lines:
                info_str += "\n\n" + "\n".join(info_lines)
            # Extract original video if available
            original_video_path = None
            if show_combined and 'aux_info' in sample:
                aux_info = sample['aux_info']
                vid_name = aux_info.get('vid', '')
                start_time = aux_info.get('start_time', 0)
                end_time = aux_info.get('end_time', 0)
                
                if vid_name and start_time < end_time:
                    try:
                        debug_print(f"Extracting original video: {vid_name}, {start_time}s to {end_time}s")
                        original_video_path = extract_original_video_clip(
                            vid_name,
                            start_time,
                            end_time,
                            dataset_root="/localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/Dataset",
                            output_path=os.path.join(self.temp_dir, f"original_{actual_idx}.mp4")
                        )
                        if original_video_path:
                            debug_print(f"Original video extracted: {original_video_path}")
                    except Exception as e:
                        import traceback
                        debug_print(f"Error extracting original video: {str(e)}\n{traceback.format_exc()}")
            
            # Render combined animation if requested and both keypoints are available
            if show_combined and leader_keypoints_for_combined is not None and follower_keypoints_for_combined is not None:
                try:
                    debug_print("Rendering combined animation...")
                    combined_video_path = os.path.join(self.temp_dir, f"combined_{actual_idx}.mp4")
                    title = f"Together: {vid_id}"
                    render_combined_skeletons(
                        leader_keypoints_for_combined,
                        follower_keypoints_for_combined,
                        combined_video_path,
                        title=title,
                        fps=20,
                        figsize=(8, 6),
                        dpi=100
                    )
                    if os.path.exists(combined_video_path):
                        # Try to add audio if available - prefer raw audio over tokens
                        audio_path = None
                        if 'audio_raw' in sample and sample['audio_raw'] is not None:
                            try:
                                debug_print("Using raw audio from sample...")
                                from visualization.visualization_utils import save_audio_waveform, create_video_with_audio
                                
                                # Get raw audio
                                audio_raw = sample['audio_raw']
                                
                                # Convert to numpy if needed
                                import torch
                                if isinstance(audio_raw, torch.Tensor):
                                    audio_raw = audio_raw.detach().cpu().numpy()
                                
                                # Handle shape: might be (channels, samples) or (samples,)
                                if len(audio_raw.shape) > 1:
                                    # If stereo/multi-channel, convert to mono
                                    audio_raw = audio_raw.mean(axis=0) if audio_raw.shape[0] > 1 else audio_raw.squeeze()
                                else:
                                    audio_raw = audio_raw.squeeze()
                                
                                # Save audio
                                audio_path = os.path.join(self.temp_dir, f"audio_{actual_idx}.wav")
                                save_audio_waveform(audio_raw, audio_path, sample_rate=24000)
                                debug_print(f"Raw audio saved to {audio_path}, shape: {audio_raw.shape}")
                                
                                # Combine video with audio
                                combined_video_with_audio = os.path.join(self.temp_dir, f"combined_with_audio_{actual_idx}.mp4")
                                create_video_with_audio(combined_video_path, combined_video_with_audio, audio_path)
                                debug_print(f"Audio combination completed, checking output: {combined_video_with_audio}")
                                
                                # Verify the output file exists and has content
                                if os.path.exists(combined_video_with_audio):
                                    file_size = os.path.getsize(combined_video_with_audio)
                                    debug_print(f"Output file exists, size: {file_size} bytes")
                                else:
                                    debug_print("ERROR: Output file was not created!")
                                
                                if os.path.exists(combined_video_with_audio):
                                    combined_video = combined_video_with_audio
                                    debug_print(f"Combined video with raw audio created: {combined_video_with_audio}")
                                else:
                                    combined_video = combined_video_path
                                    debug_print("Audio combination failed, using video without audio")
                            except Exception as e:
                                import traceback
                                debug_print(f"Error using raw audio: {str(e)}\n{traceback.format_exc()}")
                                combined_video = combined_video_path
                        elif 'audio_tokens' in sample and sample['audio_tokens'] is not None:
                            try:
                                debug_print("Decoding audio tokens...")
                                from visualization.visualization_utils import decode_audio_tokens, save_audio_waveform, create_video_with_audio
                                
                                # Try to get wavtokenizer if available
                                wavtokenizer = None
                                try:
                                    import sys
                                    from pathlib import Path
                                    parent_dir = Path(__file__).parent.parent
                                    sys.path.insert(0, str(parent_dir))
                                    
                                    from utils.salsa_utils.libs.WavTokenizer.decoder.pretrained import WavTokenizer
                                    
                                    # Try to initialize WavTokenizer (may fail if paths are wrong)
                                    # Try relative path first
                                    WavTokenizer_relativeroot = parent_dir / 'utils' / 'salsa_utils' / 'libs' / 'WavTokenizer'
                                    config_path = WavTokenizer_relativeroot / 'configs' / 'wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
                                    model_path = WavTokenizer_relativeroot / 'results' / 'train' / 'wavtokenizer_large_unify_600_24k.ckpt'
                                    
                                    if config_path.exists() and model_path.exists():
                                        wavtokenizer = WavTokenizer.from_pretrained0802(str(config_path), str(model_path))
                                        wavtokenizer = wavtokenizer.to('cpu')
                                        wavtokenizer.eval()
                                        debug_print("WavTokenizer loaded successfully")
                                    else:
                                        debug_print(f"WavTokenizer files not found: config={config_path}, model={model_path}")
                                except Exception as e:
                                    debug_print(f"Could not load WavTokenizer: {e}")
                                    import traceback
                                    debug_print(traceback.format_exc())
                                
                                # Decode audio
                                audio_tokens = sample['audio_tokens']
                                audio_waveform = decode_audio_tokens(audio_tokens, wavtokenizer, device='cpu')
                                
                                if audio_waveform is not None:
                                    audio_path = os.path.join(self.temp_dir, f"audio_{actual_idx}.wav")
                                    save_audio_waveform(audio_waveform, audio_path, sample_rate=24000)
                                    debug_print(f"Audio saved to {audio_path}")
                                    
                                    # Combine video with audio
                                    combined_video_with_audio = os.path.join(self.temp_dir, f"combined_with_audio_{actual_idx}.mp4")
                                    create_video_with_audio(combined_video_path, combined_video_with_audio, audio_path)
                                    debug_print(f"Audio combination completed, checking output: {combined_video_with_audio}")
                                    
                                    # Verify the output file exists and has content
                                    if os.path.exists(combined_video_with_audio):
                                        file_size = os.path.getsize(combined_video_with_audio)
                                        debug_print(f"Output file exists, size: {file_size} bytes")
                                    else:
                                        debug_print("ERROR: Output file was not created!")
                                    
                                    if os.path.exists(combined_video_with_audio):
                                        combined_video = combined_video_with_audio
                                        debug_print(f"Combined video with audio created: {combined_video_with_audio}")
                                    else:
                                        combined_video = combined_video_path
                                        debug_print("Audio combination failed, using video without audio")
                                else:
                                    combined_video = combined_video_path
                                    debug_print("Audio decoding failed, using video without audio")
                            except Exception as e:
                                import traceback
                                debug_print(f"Error adding audio: {str(e)}\n{traceback.format_exc()}")
                                combined_video = combined_video_path
                        else:
                            combined_video = combined_video_path
                            debug_print("No audio tokens in sample")
                        
                        file_size = os.path.getsize(combined_video)
                        info_lines.append(f"Combined: {file_size} bytes")
                        if audio_path and os.path.exists(audio_path):
                            info_lines.append(f"Audio: Added successfully")
                        debug_print(f"Combined video created: {combined_video}, size: {file_size} bytes")
                except Exception as e:
                    import traceback
                    error_msg = f"Error rendering combined: {str(e)}\n{traceback.format_exc()}"
                    error_messages.append(error_msg)
                    debug_print(error_msg)
            
            if error_messages:
                info_str += "\n\nERRORS:\n" + "\n".join(error_messages)
            
            debug_print(f"Returning: leader_video={leader_video}, follower_video={follower_video}, combined_video={combined_video}, original_video={original_video_path}")
            return leader_video, follower_video, combined_video, original_video_path if original_video_path else None, info_str
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing sample: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, None, None, None, error_msg
    
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
    
    def _get_motion_tokenizer(self):
        """Lazy load Motion_tokenizer."""
        if self.motion_tokenizer is None:
            try:
                # Fix import path for WavTokenizer
                # The decoder/pretrained.py imports 'from decoder.feature_extractors'
                # which means it expects 'decoder' to be a top-level package
                # So we need to add the WavTokenizer parent directory to sys.path
                wavtokenizer_base = self.parent_dir / 'utils' / 'salsa_utils' / 'libs' / 'WavTokenizer'
                
                # Add WavTokenizer directory to path so 'decoder' package can be found
                if str(wavtokenizer_base) not in sys.path:
                    sys.path.insert(0, str(wavtokenizer_base))
                
                from options.option_llm import get_args_parser
                from utils.salsa_utils.salsa_dataloader import Motion_tokenizer
                
                # Create args
                args = get_args_parser()
                args.device = 'cpu'  # Use CPU for visualization
                args.is_MDM = self.is_MDM
                args.parent_dir = str(self.parent_dir)
                
                debug_print("Initializing Motion_tokenizer...")
                self.motion_tokenizer = Motion_tokenizer(args)
                debug_print("Motion_tokenizer initialized successfully")
            except Exception as e:
                import traceback
                error_msg = f"Error initializing Motion_tokenizer: {str(e)}\n{traceback.format_exc()}"
                debug_print(error_msg)
                raise RuntimeError(error_msg) from e
        
        return self.motion_tokenizer
    
    def _get_audio_tokenizer(self):
        """Lazy load Audio_tokenizer."""
        if self.audio_tokenizer is None:
            try:
                # Fix import path for WavTokenizer
                wavtokenizer_base = self.parent_dir / 'utils' / 'salsa_utils' / 'libs' / 'WavTokenizer'
                
                # Add WavTokenizer directory to path so 'decoder' package can be found
                if str(wavtokenizer_base) not in sys.path:
                    sys.path.insert(0, str(wavtokenizer_base))
                
                from options.option_llm import get_args_parser
                from utils.salsa_utils.salsa_dataloader import Audio_tokenizer
                
                # Create args
                args = get_args_parser()
                args.device = 'cpu'  # Use CPU for visualization
                args.is_MDM = self.is_MDM
                args.parent_dir = str(self.parent_dir)
                
                debug_print("Initializing Audio_tokenizer...")
                self.audio_tokenizer = Audio_tokenizer(args)
                debug_print("Audio_tokenizer initialized successfully")
            except Exception as e:
                import traceback
                error_msg = f"Error initializing Audio_tokenizer: {str(e)}\n{traceback.format_exc()}"
                debug_print(error_msg)
                raise RuntimeError(error_msg) from e
        
        return self.audio_tokenizer
    
    def visualize_audio_wavtokenizer(self, idx: int) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize audio tokenization and reconstruction using WavTokenizer.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (original_audio_path, reconstructed_audio_path, debug_msg)
        """
        debug_msg = ""
        try:
            if not self.loader:
                return None, None, "Error: No LMDB loaded"
            
            # Get sample
            sample = self.loader.get_sample(idx)
            if not sample:
                return None, None, f"Error: Could not load sample {idx}"
            
            debug_msg += f"Processing audio WavTokenizer for sample {idx}...\n"
            
            # Get raw audio
            audio_raw = sample.get('audio_raw')
            if audio_raw is None:
                error_msg = "Raw audio not found in sample"
                debug_msg += error_msg + "\n"
                return None, None, debug_msg
            
            # Convert to numpy if needed
            if isinstance(audio_raw, torch.Tensor):
                audio_raw = audio_raw.detach().cpu().numpy()
            
            # Handle shape: might be (channels, samples) or (samples,)
            if len(audio_raw.shape) > 1:
                # If stereo/multi-channel, convert to mono
                audio_raw = audio_raw.mean(axis=0) if audio_raw.shape[0] > 1 else audio_raw.squeeze()
            else:
                audio_raw = audio_raw.squeeze()
            
            # Save original audio
            original_audio_path = os.path.join(self.temp_dir, f"original_audio_{idx}.wav")
            from visualization.visualization_utils import save_audio_waveform
            save_audio_waveform(audio_raw, original_audio_path, sample_rate=24000)
            debug_msg += f"Original audio saved to {original_audio_path}\n"
            
            # Initialize tokenizer
            try:
                tokenizer = self._get_audio_tokenizer()
            except Exception as e:
                error_msg = f"Error loading Audio_tokenizer: {str(e)}"
                debug_msg += error_msg + "\n"
                return original_audio_path, None, debug_msg
            
            # Tokenize and detokenize
            debug_msg += "Tokenizing audio...\n"
            audio_tensor = torch.from_numpy(audio_raw).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
            # Move audio tensor to the same device as the model
            model_device = next(tokenizer.wavtokenizer.parameters()).device
            audio_tensor = audio_tensor.to(model_device)
            debug_msg += f"Audio tensor device: {audio_tensor.device}, Model device: {model_device}\n"
            
            # Tokenize using encode_infer (returns features and discrete codes)
            bandwidth_id = torch.tensor([0]).to(model_device)
            features, discrete_code = tokenizer.wavtokenizer.encode_infer(
                audio_tensor,
                bandwidth_id=bandwidth_id
            )
            debug_msg += f"Audio tokens (discrete_code) shape: {discrete_code.shape}\n"
            
            # Detokenize: convert codes to features, then decode features to audio
            debug_msg += "Detokenizing audio...\n"
            # Convert discrete codes back to features
            features_from_codes = tokenizer.wavtokenizer.codes_to_features(discrete_code)
            # Decode features to audio
            reconstructed_audio = tokenizer.wavtokenizer.decode(features_from_codes, bandwidth_id=bandwidth_id)
            
            # Move back to CPU for saving
            if isinstance(reconstructed_audio, torch.Tensor):
                reconstructed_audio = reconstructed_audio.detach().cpu()
            
            # Convert to numpy
            if isinstance(reconstructed_audio, torch.Tensor):
                reconstructed_audio = reconstructed_audio.detach().cpu().numpy()
            
            # Handle shape
            if len(reconstructed_audio.shape) > 1:
                reconstructed_audio = reconstructed_audio.squeeze()
            
            # Save reconstructed audio
            reconstructed_audio_path = os.path.join(self.temp_dir, f"reconstructed_audio_{idx}.wav")
            save_audio_waveform(reconstructed_audio, reconstructed_audio_path, sample_rate=24000)
            debug_msg += f"Reconstructed audio saved to {reconstructed_audio_path}\n"
            debug_msg += "Audio WavTokenizer processing complete!\n"
            
            return original_audio_path, reconstructed_audio_path, debug_msg
            
        except Exception as e:
            import traceback
            error_msg = f"Error in visualize_audio_wavtokenizer: {str(e)}\n{traceback.format_exc()}"
            debug_msg += error_msg
            debug_print(error_msg)
            return None, None, debug_msg
    
    def visualize_vqvae_reconstruction(self, idx: int) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str]:
        """
        Visualize VQVAE reconstruction for leader and follower.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (leader_original, leader_reconstructed, leader_overlapped,
                     follower_original, follower_reconstructed, follower_overlapped, debug_msg)
        """
        debug_msg = ""
        try:
            if not self.loader:
                return None, None, None, None, None, None, "Error: No LMDB loaded"
            
            # Get sample
            sample = self.loader.get_sample(idx)
            if not sample:
                return None, None, None, None, None, None, f"Error: Could not load sample {idx}"
            
            debug_msg += f"Visualizing VQVAE reconstruction for sample {idx}...\n"
            
            # Initialize tokenizer
            try:
                tokenizer = self._get_motion_tokenizer()
            except Exception as e:
                error_msg = f"Error loading Motion_tokenizer: {str(e)}"
                debug_msg += error_msg + "\n"
                return None, None, None, None, None, None, debug_msg
            
            # Get VQ tokens
            vq_tokens_L = sample.get('vq_tokens_L')
            vq_tokens_F = sample.get('vq_tokens_F')
            
            if vq_tokens_L is None or vq_tokens_F is None:
                error_msg = "VQ tokens not found in sample"
                debug_msg += error_msg + "\n"
                return None, None, None, None, None, None, debug_msg
            
            # Get original keypoints
            original_keypoints_L = sample.get('poses_keypoints3d_L')
            original_keypoints_F = sample.get('poses_keypoints3d_F')
            
            if original_keypoints_L is None or original_keypoints_F is None:
                error_msg = "Original keypoints not found in sample"
                debug_msg += error_msg + "\n"
                return None, None, None, None, None, None, debug_msg
            
            # Decode reconstructed motions
            debug_msg += "Decoding leader motion from VQ tokens...\n"
            reconstructed_keypoints_L = decode_motion_from_vq_tokens(
                vq_tokens_L, tokenizer, device='cpu'
            )
            
            debug_msg += "Decoding follower motion from VQ tokens...\n"
            reconstructed_keypoints_F = decode_motion_from_vq_tokens(
                vq_tokens_F, tokenizer, device='cpu'
            )
            
            # Convert to numpy if needed
            if isinstance(original_keypoints_L, torch.Tensor):
                original_keypoints_L = original_keypoints_L.detach().cpu().numpy()
            if isinstance(original_keypoints_F, torch.Tensor):
                original_keypoints_F = original_keypoints_F.detach().cpu().numpy()
            
            # Get video ID for titles
            vid_id = sample.get('aux_info', {}).get('vid', f'sample_{idx}')
            
            # Render Leader videos
            debug_msg += "Rendering leader videos...\n"
            leader_original_path = os.path.join(self.temp_dir, f"leader_original_vqvae_{idx}.mp4")
            leader_reconstructed_path = os.path.join(self.temp_dir, f"leader_reconstructed_vqvae_{idx}.mp4")
            leader_overlapped_path = os.path.join(self.temp_dir, f"leader_overlapped_vqvae_{idx}.mp4")
            
            render_skeleton_from_keypoints(
                original_keypoints_L,
                leader_original_path,
                title=f"Leader Original: {vid_id}",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            render_skeleton_from_keypoints(
                reconstructed_keypoints_L,
                leader_reconstructed_path,
                title=f"Leader Reconstructed: {vid_id}",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            render_vqvae_reconstruction_comparison(
                original_keypoints_L,
                reconstructed_keypoints_L,
                leader_overlapped_path,
                title=f"Leader: Original vs Reconstructed",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            # Render Follower videos
            debug_msg += "Rendering follower videos...\n"
            follower_original_path = os.path.join(self.temp_dir, f"follower_original_vqvae_{idx}.mp4")
            follower_reconstructed_path = os.path.join(self.temp_dir, f"follower_reconstructed_vqvae_{idx}.mp4")
            follower_overlapped_path = os.path.join(self.temp_dir, f"follower_overlapped_vqvae_{idx}.mp4")
            
            render_skeleton_from_keypoints(
                original_keypoints_F,
                follower_original_path,
                title=f"Follower Original: {vid_id}",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            render_skeleton_from_keypoints(
                reconstructed_keypoints_F,
                follower_reconstructed_path,
                title=f"Follower Reconstructed: {vid_id}",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            render_vqvae_reconstruction_comparison(
                original_keypoints_F,
                reconstructed_keypoints_F,
                follower_overlapped_path,
                title=f"Follower: Original vs Reconstructed",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            debug_msg += "VQVAE reconstruction visualization complete!\n"
            return (
                leader_original_path if os.path.exists(leader_original_path) else None,
                leader_reconstructed_path if os.path.exists(leader_reconstructed_path) else None,
                leader_overlapped_path if os.path.exists(leader_overlapped_path) else None,
                follower_original_path if os.path.exists(follower_original_path) else None,
                follower_reconstructed_path if os.path.exists(follower_reconstructed_path) else None,
                follower_overlapped_path if os.path.exists(follower_overlapped_path) else None,
                debug_msg
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Error in visualize_vqvae_reconstruction: {str(e)}\n{traceback.format_exc()}"
            debug_msg += error_msg
            debug_print(error_msg)
            return None, None, None, None, None, None, debug_msg
    
    def visualize_relative_motion(self, idx: int) -> Tuple[Optional[str], Optional[str], str]:
        """
        Visualize relative motion: Leader-Follower and Follower-Leader.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (leader_minus_follower_path, follower_minus_leader_path, debug_msg)
        """
        debug_msg = ""
        try:
            if not self.loader:
                return None, None, "Error: No LMDB loaded"
            
            # Get sample
            sample = self.loader.get_sample(idx)
            if not sample:
                return None, None, f"Error: Could not load sample {idx}"
            
            debug_msg += f"Visualizing relative motion for sample {idx}...\n"
            
            # Get keypoints
            leader_keypoints = sample.get('poses_keypoints3d_L')
            follower_keypoints = sample.get('poses_keypoints3d_F')
            
            if leader_keypoints is None or follower_keypoints is None:
                error_msg = "Keypoints not found in sample"
                debug_msg += error_msg + "\n"
                return None, None, debug_msg
            
            # Convert to numpy if needed
            if isinstance(leader_keypoints, torch.Tensor):
                leader_keypoints = leader_keypoints.detach().cpu().numpy()
            if isinstance(follower_keypoints, torch.Tensor):
                follower_keypoints = follower_keypoints.detach().cpu().numpy()
            
            # Ensure same shape for subtraction
            min_frames = min(leader_keypoints.shape[0], follower_keypoints.shape[0])
            leader_keypoints = leader_keypoints[:min_frames]
            follower_keypoints = follower_keypoints[:min_frames]
            
            # Compute relative motions
            leader_minus_follower = leader_keypoints - follower_keypoints
            follower_minus_leader = follower_keypoints - leader_keypoints
            
            # Get video ID for titles
            vid_id = sample.get('aux_info', {}).get('vid', f'sample_{idx}')
            
            # Render relative motions
            debug_msg += "Rendering relative motions...\n"
            leader_minus_follower_path = os.path.join(self.temp_dir, f"leader_minus_follower_{idx}.mp4")
            follower_minus_leader_path = os.path.join(self.temp_dir, f"follower_minus_leader_{idx}.mp4")
            
            render_skeleton_from_keypoints(
                leader_minus_follower,
                leader_minus_follower_path,
                title=f"Leader - Follower: {vid_id}",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            render_skeleton_from_keypoints(
                follower_minus_leader,
                follower_minus_leader_path,
                title=f"Follower - Leader: {vid_id}",
                fps=20,
                figsize=(6, 6),
                dpi=100
            )
            
            debug_msg += "Relative motion visualization complete!\n"
            return (
                leader_minus_follower_path if os.path.exists(leader_minus_follower_path) else None,
                follower_minus_leader_path if os.path.exists(follower_minus_leader_path) else None,
                debug_msg
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Error in visualize_relative_motion: {str(e)}\n{traceback.format_exc()}"
            debug_msg += error_msg
            debug_print(error_msg)
            return None, None, debug_msg


def create_interface():
    """Create the Gradio interface."""
    app = VisualizationApp()
    
    # Custom CSS for better UI styling
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
    h3 {
        font-size: 24px !important;
        font-weight: 500 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    .markdown-text {
        font-size: 14px !important;
        color: #666;
    }
    label {
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    .sample-info textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
        line-height: 1.4;
    }
    .debug-output textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 11px !important;
        line-height: 1.3;
    }
    button {
        font-size: 13px !important;
        padding: 8px 16px !important;
    }
    .video-player {
        min-width: 300px;
    }
    """
    
    with gr.Blocks(title="Salsa Dataset Visualization") as demo:
        # Add custom CSS using gr.HTML or by injecting it
        gr.HTML(f"<style>{custom_css}</style>", visible=False)
        gr.Markdown("# 🕺 Salsa Dataset Visualization Tool")
        gr.Markdown("Load an LMDB file and explore skeleton data, motion tokens, and audio.", elem_classes="markdown-text")
        
        with gr.Row():
            with gr.Column(scale=2):
                lmdb_path = gr.Textbox(
                    label="LMDB Directory Path",
                    value="dataset_processed_New/lmdb_Salsa_pair/lmdb_train",
                    placeholder="Enter path to LMDB directory (e.g., dataset_processed_New/lmdb_Salsa_pair/lmdb_train)",
                )
                is_MDM = gr.Checkbox(label="Is MDM Format", value=False)
                load_btn = gr.Button("Load LMDB", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                total_samples = gr.Number(label="Total Samples", value=0, interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Navigation", elem_classes="h3")
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
                
                with gr.Row():
                    show_leader = gr.Checkbox(label="Show Leader", value=True)
                    show_follower = gr.Checkbox(label="Show Follower", value=True)
                    show_combined = gr.Checkbox(label="Show Combined", value=False)
                    visualize_btn = gr.Button("Visualize", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Skeleton Visualizations", elem_classes="h3")
                with gr.Row():
                    leader_video = gr.Video(label="Leader Skeleton", scale=1, elem_classes="video-player")
                    follower_video = gr.Video(label="Follower Skeleton", scale=1, elem_classes="video-player")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Combined Visualization", elem_classes="h3")
                with gr.Row():
                    original_video = gr.Video(label="Original Video", scale=1, elem_classes="video-player")
                    combined_video = gr.Video(label="Combined: Dancing Together", scale=1, elem_classes="video-player")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Experiment: Relative", elem_classes="h3")
                relative_btn = gr.Button("Generate Relative Motion", variant="primary")
                with gr.Row():
                    leader_minus_follower_video = gr.Video(label="Leader - Follower", scale=1, elem_classes="video-player")
                    follower_minus_leader_video = gr.Video(label="Follower - Leader", scale=1, elem_classes="video-player")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Motion VQVAE Reconstruction", elem_classes="h3")
                vqvae_btn = gr.Button("Generate VQVAE Reconstruction", variant="primary")
                with gr.Row():
                    gr.Markdown("**Leader:**")
                with gr.Row():
                    leader_original_video = gr.Video(label="Original", scale=1, elem_classes="video-player")
                    leader_reconstructed_video = gr.Video(label="Reconstructed", scale=1, elem_classes="video-player")
                    leader_overlapped_video = gr.Video(label="Overlapped", scale=1, elem_classes="video-player")
                with gr.Row():
                    gr.Markdown("**Follower:**")
                with gr.Row():
                    follower_original_video = gr.Video(label="Original", scale=1, elem_classes="video-player")
                    follower_reconstructed_video = gr.Video(label="Reconstructed", scale=1, elem_classes="video-player")
                    follower_overlapped_video = gr.Video(label="Overlapped", scale=1, elem_classes="video-player")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Audio WavTokenizer", elem_classes="h3")
                audio_wavtokenizer_btn = gr.Button("Generate Audio Reconstruction", variant="primary")
                with gr.Row():
                    original_audio = gr.Audio(label="Original Audio", type="filepath")
                    reconstructed_audio = gr.Audio(label="Reconstructed Audio (Tokenized & Detokenized)", type="filepath")
        
        with gr.Row():
            with gr.Column(scale=2):
                sample_info = gr.Textbox(
                    label="Sample Information",
                    lines=8,
                    interactive=False,
                    elem_classes=["sample-info"]
                )
            with gr.Column(scale=1):
                debug_output = gr.Textbox(
                    label="Debug Output",
                    lines=8,
                    interactive=False,
                    value="Debug messages will appear here...",
                    elem_classes=["debug-output"]
                )
        
        # Event handlers
        def on_load(lmdb_path_val, is_MDM_val):
            status, idx, total = app.load_lmdb(lmdb_path_val, is_MDM_val)
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
        
        def on_visualize(idx_val, show_leader_val, show_follower_val, show_combined_val):
            debug_msg = ""
            try:
                idx = int(idx_val) if idx_val is not None else 0
                debug_print(f"on_visualize called: idx={idx}, show_leader={show_leader_val}, show_follower={show_follower_val}, show_combined={show_combined_val}")
                debug_msg += f"Visualizing sample {idx}...\n"
                
                leader_vid, follower_vid, combined_vid, original_vid, info = app.visualize_sample(
                    idx,
                    show_leader_val if show_leader_val is not None else True,
                    show_follower_val if show_follower_val is not None else True,
                    show_combined_val if show_combined_val is not None else False
                )
                
                debug_print(f"Visualization complete. Leader: {leader_vid}, Follower: {follower_vid}, Combined: {combined_vid}, Original: {original_vid}")
                debug_msg += f"Visualization complete.\n"
                debug_msg += f"Leader video: {leader_vid}\n"
                debug_msg += f"Follower video: {follower_vid}\n"
                debug_msg += f"Combined video: {combined_vid}\n"
                debug_msg += f"Original video: {original_vid}\n"
                if leader_vid and os.path.exists(leader_vid):
                    file_size = os.path.getsize(leader_vid)
                    debug_msg += f"Leader file exists: {file_size} bytes\n"
                    debug_print(f"Leader file exists: {file_size} bytes")
                if follower_vid and os.path.exists(follower_vid):
                    file_size = os.path.getsize(follower_vid)
                    debug_msg += f"Follower file exists: {file_size} bytes\n"
                    debug_print(f"Follower file exists: {file_size} bytes")
                if combined_vid and os.path.exists(combined_vid):
                    file_size = os.path.getsize(combined_vid)
                    debug_msg += f"Combined file exists: {file_size} bytes\n"
                    debug_print(f"Combined file exists: {file_size} bytes")
                
                debug_print(f"Visualization complete. Leader: {leader_vid}, Follower: {follower_vid}, Combined: {combined_vid}, Original: {original_vid}")
                return leader_vid, follower_vid, combined_vid, original_vid, info, debug_msg
            except Exception as e:
                import traceback
                error_msg = f"Error in on_visualize: {str(e)}\n{traceback.format_exc()}"
                debug_msg += error_msg
                debug_print(error_msg)
                return None, None, None, None, error_msg, debug_msg
        
        load_btn.click(
            fn=on_load,
            inputs=[lmdb_path, is_MDM],
            outputs=[load_status, total_samples, sample_idx]
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
        
        def on_go(idx_val, show_leader_val, show_follower_val, show_combined_val):
            return on_visualize(idx_val, show_leader_val, show_follower_val, show_combined_val)
        
        go_btn.click(
            fn=on_go,
            inputs=[sample_idx, show_leader, show_follower, show_combined],
            outputs=[leader_video, follower_video, combined_video, original_video, sample_info, debug_output]
        )
        
        visualize_btn.click(
            fn=on_visualize,
            inputs=[sample_idx, show_leader, show_follower, show_combined],
            outputs=[leader_video, follower_video, combined_video, original_video, sample_info, debug_output]
        )
        
        # Auto-visualize when index changes
        sample_idx.change(
            fn=on_visualize,
            inputs=[sample_idx, show_leader, show_follower, show_combined],
            outputs=[leader_video, follower_video, combined_video, original_video, sample_info, debug_output]
        )
        
        def on_vqvae_visualize(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app.visualize_vqvae_reconstruction(idx)
            except Exception as e:
                import traceback
                error_msg = f"Error in on_vqvae_visualize: {str(e)}\n{traceback.format_exc()}"
                return None, None, None, None, None, None, error_msg
        
        vqvae_btn.click(
            fn=on_vqvae_visualize,
            inputs=[sample_idx],
            outputs=[
                leader_original_video,
                leader_reconstructed_video,
                leader_overlapped_video,
                follower_original_video,
                follower_reconstructed_video,
                follower_overlapped_video,
                debug_output
            ]
        )
        
        def on_audio_wavtokenizer_visualize(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app.visualize_audio_wavtokenizer(idx)
            except Exception as e:
                import traceback
                error_msg = f"Error in on_audio_wavtokenizer_visualize: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        audio_wavtokenizer_btn.click(
            fn=on_audio_wavtokenizer_visualize,
            inputs=[sample_idx],
            outputs=[original_audio, reconstructed_audio, debug_output]
        )
        
        def on_relative_visualize(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app.visualize_relative_motion(idx)
            except Exception as e:
                import traceback
                error_msg = f"Error in on_relative_visualize: {str(e)}\n{traceback.format_exc()}"
                return None, None, error_msg
        
        relative_btn.click(
            fn=on_relative_visualize,
            inputs=[sample_idx],
            outputs=[leader_minus_follower_video, follower_minus_leader_video, debug_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7861)

