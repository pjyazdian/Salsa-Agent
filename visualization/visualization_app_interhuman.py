"""
Comprehensive visualization app for Salsa dataset with InterHuman representation support.
Shows all stored data including tokens, InterHuman motions, relationship features, and reconstructions.
"""
# DEBUG flag for detailed reconstruction debugging
DEBUG = True

import os
import sys
from pathlib import Path

# CRITICAL: Salsa-Agent root must stay at sys.path[0] so "import models.vqvae" finds Salsa-Agent/models/vqvae.py.
# motion_representation also has a "models" package (no vqvae.py); if it were first, the import would fail.
parent_dir = Path(__file__).parent.parent
parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.insert(0, parent_dir_str)

# Add WavTokenizer and motion_representation after the root so they don't shadow Salsa-Agent/models
wavtokenizer_base = parent_dir / 'utils' / 'salsa_utils' / 'libs' / 'WavTokenizer'
if str(wavtokenizer_base) not in sys.path:
    sys.path.insert(1, str(wavtokenizer_base))

import gradio as gr
import numpy as np
import torch
import tempfile
import time
import pyarrow
import pickle
from typing import Optional, Tuple, Dict
import json
import pandas as pd
import warnings

from visualization.visualization_utils import (
    render_skeleton_from_keypoints,
    render_combined_skeletons,
    create_video_with_audio,
    save_audio_waveform,
    DEBUG,
    debug_print
)
from visualization.joints_to_mesh_utils import keypoints_to_mesh_video

# Import args parser (Salsa_Dataset will be imported lazily to avoid WavTokenizer import issues)
from options.option_llm import get_args_parser

# motion_representation must NOT be inserted at 0, or "import models.vqvae" finds motion_representation/models/ (no vqvae.py)
motion_rep_path = Path(__file__).parent.parent / 'motion_representation'
if str(motion_rep_path) not in sys.path:
    sys.path.insert(1, str(motion_rep_path))  # insert at 1 so Salsa-Agent root stays at 0

try:
    from motion_representation.visualization.vae_visualization_app import VAEVisualizationApp
    from motion_representation.models.motion_model import MotionModel
    from motion_representation.utils.relationship_features import (
        salsa_to_interhuman, extract_interhuman_relationship_features
    )
    from in2in.utils.utils import rigid_transform
    from in2in.utils.plot import plot_3d_motion as plot_3d_motion_interhuman
    from in2in.utils.paramUtil import HML_KINEMATIC_CHAIN
    from in2in.utils.quaternion import qbetween_np
    INTERHUMAN_AVAILABLE = True
except ImportError as e:
    INTERHUMAN_AVAILABLE = False
    qbetween_np = None
    INTERHUMAN_AVAILABLE = False
    print(f"Warning: InterHuman visualization dependencies not available: {e}")


# Debug flag for concatenation function
DEBUG_CONCATENATION = True

# Dataset README Glossary: 31 move classes + Siete + 2 styling + 4 error classes = 37 total
README_MOVE_CLASSES = [
    "Arm lock", "Basic step", "Body shake", "Body roll", "Change of Directions",
    "Check", "Comb", "Copa", "Dile que no", "Hand throw", "Right turn",
    "Drawing circle", "Enchufla", "Walks around", "Suzy Q", "Hip movement",
    "Kicks", "Lasso", "Natural top", "Left turn", "Mambo", "Open break",
    "Point", "Sliding", "Standing", "Steps", "Swing", "Walk",
    "XBL (Cross Body Lead)", "Indescribable", "Markers Swap issue", "Siete",
]
README_STYLING_CLASSES = ["Lady styling", "Man styling"]
README_ERROR_CLASSES = ["Misinterpreted signal", "Misstep", "Mixed signals", "Off beat"]
ALL_LABEL_NAMES = README_MOVE_CLASSES + README_STYLING_CLASSES + README_ERROR_CLASSES  # 37 total

# LLM-Inference / Latency task labels -> training_utils task keys
LLM_INFERENCE_TASK_MAP = {
    "Leader + Rel to Follower": "leader_rel_to_follower",
    "Follower + Rel to Leader": "follower_rel_to_leader",
    "Caption + Leader + Rel to Follower": "caption_leader_rel_to_follower",
    "Caption + Follower + Rel to Leader": "caption_follower_rel_to_leader",
    "Pair to Relationship": "pair_to_relationship",
    "Caption to Leader": "caption_to_leader",
    "Caption to Follower": "caption_to_follower",
    "Leader to Follower": "leader_to_follower",
    "Follower to Leader": "follower_to_leader",
    "Motion completion (Leader)": "motion_completion_leader",
    "Motion completion (Follower)": "motion_completion_follower",
    "Leader motion to Leader MotionScript": "leader_motion_to_motionscript",
    "Follower motion to Follower MotionScript": "follower_motion_to_motionscript",
    "Leader MotionScript to Leader motion": "motionscript_to_leader_motion",
    "Follower MotionScript to Follower motion": "motionscript_to_follower_motion",
    "Caption to Leader MotionScript": "caption_to_leader_motionscript",
    "Caption to Follower MotionScript": "caption_to_follower_motionscript",
    "Caption to Both MotionScripts": "caption_to_both_motionscripts",
    "Leader MotionScript + Rel to Follower MotionScript": "leader_motionscript_rel_to_follower_motionscript",
    "Follower MotionScript + Rel to Leader MotionScript": "follower_motionscript_rel_to_leader_motionscript",
    "MotionScript completion (Leader)": "motionscript_completion_leader",
    "MotionScript completion (Follower)": "motionscript_completion_follower",
    "Caption + Leader MotionScript to Follower MotionScript": "caption_leader_motionscript_to_follower_motionscript",
    "Caption + Follower MotionScript to Leader MotionScript": "caption_follower_motionscript_to_leader_motionscript",
}
LLM_INFERENCE_TASK_CHOICES = list(LLM_INFERENCE_TASK_MAP.keys())

# Map dataloader move_class / error_class keys to README display names
LABEL_TO_README = {
    "right turn": "Right turn", "Left turn": "Left turn", "Body Shake": "Body shake",
    "Suzy": "Suzy Q", "XBL": "XBL (Cross Body Lead)", "walks around": "Walks around",
    "Drawing circle": "Drawing circle",
}

def concatenate_windows_with_continuity(
    leader_windows, follower_windows, relationship_windows,
    root_quat_inits_L=None, root_pos_inits_L=None, root_quat_inits_F=None, root_pos_inits_F=None
):
    """
    Concatenate multiple windows of InterHuman motion with continuity.
    
    Each window starts from origin in canonical frame. This function transforms
    subsequent windows to continue from where the previous window ended.
    
    Args:
        leader_windows: List of (T, 262) numpy arrays - Leader motion for each window
        follower_windows: List of (T, 262) numpy arrays - Follower motion for each window
        relationship_windows: List of (T, 4) numpy arrays - Relationship features for each window
        root_quat_inits_L: Optional list of (4,) numpy arrays - Leader root quaternion from frame 0 of each window
        root_pos_inits_L: Optional list of (3,) numpy arrays - Leader root position from frame 0 of each window
        root_quat_inits_F: Optional list of (4,) numpy arrays - Follower root quaternion from frame 0 of each window
        root_pos_inits_F: Optional list of (3,) numpy arrays - Follower root position from frame 0 of each window
    
    Returns:
        leader_continuous: (total_frames, 262) numpy array - Concatenated leader motion with continuity
        follower_continuous: (total_frames, 262) numpy array - Concatenated follower motion with continuity
        relationship_continuous: (total_frames, 4) numpy array - Concatenated relationship features
    """
    # Check if dependencies are available
    if qbetween_np is None or rigid_transform is None:
        raise ImportError("InterHuman dependencies (qbetween_np, rigid_transform) are not available. Make sure in2IN is properly installed.")
    
    # Setup debug logging
    debug_log_path = None
    if DEBUG_CONCATENATION:
        import tempfile
        debug_log_path = os.path.join(tempfile.gettempdir(), "concatenation_debug.txt")
        with open(debug_log_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CONCATENATION DEBUG LOG\n")
            f.write("="*80 + "\n\n")
    
    def debug_log(msg):
        if DEBUG_CONCATENATION and debug_log_path:
            with open(debug_log_path, 'a') as f:
                f.write(msg + "\n")
    debug_log(f"Starting concatenation with {len(leader_windows)} windows")
    
    if len(leader_windows) == 0:
        debug_log("No windows to concatenate, returning empty arrays")
        return np.array([]), np.array([]), np.array([])
    
    if len(leader_windows) == 1:
        # Single window: no transformation needed
        debug_log("Single window, no transformation needed")
        return leader_windows[0], follower_windows[0], relationship_windows[0]
    
    n_joints = 22
    r_hip, l_hip = 2, 1  # Joint indices for right and left hip
    
    # Helper function to extract root position and yaw from a motion frame
    def extract_root_pos_yaw(motion_frame):
        """Extract root position and yaw from a single InterHuman motion frame (262-dim)."""
        # Root position is first 3 dims
        root_pos = motion_frame[:3]  # (3,)
        
        # Extract hip positions to compute forward direction
        r_hip_pos = motion_frame[r_hip*3:(r_hip+1)*3]  # (3,)
        l_hip_pos = motion_frame[l_hip*3:(l_hip+1)*3]  # (3,)
        
        # Compute across vector (hip-based, same as relationship_features.py)
        across = r_hip_pos - l_hip_pos  # (3,)
        across_norm = np.linalg.norm(across)
        if across_norm > 1e-6:
            across = across / across_norm  # Normalize
        else:
            # Fallback: use default forward direction if hips are too close
            across = np.array([1, 0, 0])
        
        # Compute forward direction: cross(y_axis, across)
        y_axis = np.array([0, 1, 0])
        forward = np.cross(y_axis, across)  # (3,)
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward = forward / forward_norm  # Normalize
        else:
            # Fallback: default forward (Z+)
            forward = np.array([0, 0, 1])
        
        # Compute root quaternion from forward direction (facing Z+)
        target = np.array([0, 0, 1])  # Target direction (Z+)
        root_quat = qbetween_np(forward.reshape(1, -1), target.reshape(1, -1))[0]  # (4,)
        
        # Extract yaw angle (half-angle from quaternion)
        yaw_half = np.arctan2(root_quat[2], root_quat[0])  # scalar
        
        return root_pos, yaw_half, root_quat
    
    # Process windows sequentially
    transformed_leader_windows = []
    transformed_follower_windows = []
    transformed_relationship_windows = []
    
    # First window: leader stays in canonical; follower is placed relative to leader
    # using the CURRENT window relationship (frame 0), then kept at origin.
    transformed_leader_windows.append(leader_windows[0].copy())
    first_relationship_frame = relationship_windows[0][0]
    first_rel_w = first_relationship_frame[0]
    first_rel_z_quat = first_relationship_frame[1]
    first_relationship_angle_half = np.arctan2(first_rel_z_quat, first_rel_w)
    first_relationship_x = first_relationship_frame[2]
    first_relationship_z = first_relationship_frame[3]
    first_relationship_transform = np.array(
        [first_relationship_angle_half, first_relationship_x, first_relationship_z],
        dtype=np.float32
    )
    first_follower_transformed = rigid_transform(first_relationship_transform, follower_windows[0].copy())
    transformed_follower_windows.append(first_follower_transformed)
    transformed_relationship_windows.append(relationship_windows[0].copy())
    debug_log("\n--- Window 0 ---")
    debug_log("First window: leader canonical, follower aligned by current relationship")
    debug_log(
        f"Relationship transform (current window frame 0): "
        f"angle_half={np.degrees(first_relationship_angle_half):.2f}°, "
        f"x={first_relationship_x:.3f}, z={first_relationship_z:.3f}"
    )
    
    # Track last frame of previous window to determine new world origin.
    # Extrapolate ONE step beyond the last frame to account for the missing transition
    # frame at each window boundary (20 raw frames → 19 IH frames loses one frame).
    _w0 = transformed_leader_windows[-1]
    _pos_last, _yaw_last, _ = extract_root_pos_yaw(_w0[-1])
    _pos_prev, _yaw_prev, _ = extract_root_pos_yaw(_w0[-2])
    new_world_origin_pos = _pos_last + (_pos_last - _pos_prev)
    new_world_origin_yaw_half = _yaw_last + (_yaw_last - _yaw_prev)
    
    for w in range(1, len(leader_windows)):
        # Get first frame of current window (in canonical frame, starts from origin)
        curr_window_first_frame = leader_windows[w][0]  # (262,)
        curr_first_frame_pos, curr_first_frame_yaw_half, _ = extract_root_pos_yaw(curr_window_first_frame)
        
        # LEADER: Simple transform from canonical to new world origin (last frame of previous window)
        # Compute window transform: move leader from canonical origin to new world origin
        window_pos_offset_xz = new_world_origin_pos[[0, 2]] - curr_first_frame_pos[[0, 2]]  # (2,) - XZ only
        window_yaw_offset_half = new_world_origin_yaw_half - curr_first_frame_yaw_half  # scalar
        
        # Create window transform: [angle_half, x, z] for rigid_transform
        window_transform = np.array([window_yaw_offset_half, window_pos_offset_xz[0], window_pos_offset_xz[1]], dtype=np.float32)
        
        # Transform leader to new world origin
        leader_transformed = rigid_transform(window_transform, leader_windows[w].copy())  # (T, 262)
        transformed_leader_windows.append(leader_transformed)
        
        # FOLLOWER: Apply current window relationship in canonical space,
        # then move both leader and follower to the previous window's last frame.
        # Relationship comes from CURRENT window's first frame (frame 0).
        curr_relationship_first_frame = relationship_windows[w][0]  # (4,) - [w, z, x, z]
        
        # Convert relationship from [w, z, x, z] format to [angle_half, x, z] format
        # Relationship quaternion [w, z] = [cos(θ/2), sin(θ/2)], so angle_half = arctan2(z, w)
        rel_w = curr_relationship_first_frame[0]
        rel_z_quat = curr_relationship_first_frame[1]
        relationship_angle_half = np.arctan2(rel_z_quat, rel_w)  # Extract half-angle from quaternion
        relationship_x = curr_relationship_first_frame[2]
        relationship_z = curr_relationship_first_frame[3]
        
        # Apply transforms: relationship first (in canonical space), then window transform
        # This matches in2IN's approach: relationship positions follower relative to leader in canonical,
        # then window transform moves both to world space together
        
        # Step 1: Apply relationship transform to canonical follower (positions follower relative to leader in canonical space)
        relationship_transform = np.array([relationship_angle_half, relationship_x, relationship_z], dtype=np.float32)
        follower_after_relationship = rigid_transform(relationship_transform, follower_windows[w].copy())  # (T, 262)
        
        # Step 2: Apply window transform to move follower to world space (same transform as leader)
        # This preserves the relationship because both leader and follower are transformed by the same window transform
        follower_after_relationship_for_window = follower_after_relationship.copy()
        follower_transformed = rigid_transform(window_transform, follower_after_relationship_for_window)  # (T, 262)
        transformed_follower_windows.append(follower_transformed)
        
        # Relationship features remain the same (they're relative, not absolute)
        transformed_relationship_windows.append(relationship_windows[w].copy())
        
        # Comprehensive debug logging
        debug_log(f"\n--- Window {w} ---")
        debug_log(f"New world origin (from prev window last frame): pos={new_world_origin_pos}, yaw_half={np.degrees(new_world_origin_yaw_half):.2f}°")
        debug_log(f"Window transform (leader): angle_half={np.degrees(window_yaw_offset_half):.2f}°, x={window_pos_offset_xz[0]:.3f}, z={window_pos_offset_xz[1]:.3f}")
        debug_log(
            f"Relationship transform (from CURRENT window frame 0): "
            f"angle_half={np.degrees(relationship_angle_half):.2f}°, "
            f"x={relationship_x:.3f}, z={relationship_z:.3f}"
        )
        
        # Debug: Check what happens at each step
        follower_canonical_first = follower_windows[w][0]
        follower_canonical_root = follower_canonical_first[:3].copy()
        debug_log(f"Follower canonical (first frame): root={follower_canonical_root}")
        
        follower_after_rel_first = follower_after_relationship[0]
        follower_after_rel_root = follower_after_rel_first[:3].copy()
        _, follower_after_rel_yaw_half, _ = extract_root_pos_yaw(follower_after_rel_first)
        debug_log(f"Follower after relationship transform: root={follower_after_rel_root}, yaw_half={np.degrees(follower_after_rel_yaw_half):.2f}°")
        debug_log(f"  Relationship transform moved follower by: {follower_after_rel_root - follower_canonical_root}")
        debug_log(f"  Relationship transform: angle_half={np.degrees(relationship_angle_half):.2f}°, x={relationship_x:.3f}, z={relationship_z:.3f}")
        
        # Debug: Check what should happen when we apply window transform
        # rigid_transform does: rotate by -window_angle, then translate by [window_x, window_z]
        # So follower_after_rel_root should be rotated by -window_angle, then translated
        cos_window = np.cos(window_yaw_offset_half)
        sin_window = np.sin(window_yaw_offset_half)
        follower_after_rel_pos_xz = follower_after_rel_root[[0, 2]]
        # Rotate by -window_angle: x' = x*cos(angle) + z*sin(angle), z' = -x*sin(angle) + z*cos(angle)
        follower_rotated_x = follower_after_rel_pos_xz[0] * cos_window + follower_after_rel_pos_xz[1] * sin_window
        follower_rotated_z = -follower_after_rel_pos_xz[0] * sin_window + follower_after_rel_pos_xz[1] * cos_window
        follower_expected_after_window_xz = np.array([follower_rotated_x, follower_rotated_z]) + window_pos_offset_xz
        debug_log(f"  Expected after window transform (manual calculation):")
        debug_log(f"    Rotated position: x={follower_rotated_x:.3f}, z={follower_rotated_z:.3f}")
        debug_log(f"    After translation: x={follower_expected_after_window_xz[0]:.3f}, z={follower_expected_after_window_xz[1]:.3f}")
        
        follower_final_first = follower_transformed[0]
        follower_final_root = follower_final_first[:3].copy()
        follower_final_pos_xz = follower_final_root[[0, 2]]
        debug_log(f"Follower after window transform (actual): root={follower_final_root}")
        debug_log(f"  Actual position: x={follower_final_pos_xz[0]:.3f}, z={follower_final_pos_xz[1]:.3f}")
        window_movement = follower_final_root - follower_after_rel_root
        debug_log(f"  Window transform moved follower by: {window_movement}")
        debug_log(f"  Window transform should move by: x={window_pos_offset_xz[0]:.3f}, z={window_pos_offset_xz[1]:.3f}")
        debug_log(f"  Position error (actual vs expected): {np.linalg.norm(follower_final_pos_xz - follower_expected_after_window_xz):.6f}")
        
        # Check if follower is positioned correctly relative to leader
        leader_final_first = leader_transformed[0]
        leader_final_root = leader_final_first[:3]
        actual_rel_xz = (follower_final_root - leader_final_root)[[0, 2]]
        debug_log(f"  Leader final position: {leader_final_root}")
        debug_log(f"  Actual relationship in world space: x={actual_rel_xz[0]:.3f}, z={actual_rel_xz[1]:.3f}")
        # Expected relationship in world space: rotate relationship offset by window angle
        relationship_x_world_expected = relationship_x * np.cos(window_yaw_offset_half) - relationship_z * np.sin(window_yaw_offset_half)
        relationship_z_world_expected = relationship_x * np.sin(window_yaw_offset_half) + relationship_z * np.cos(window_yaw_offset_half)
        debug_log(f"  Expected relationship (rotated by window angle): x={relationship_x_world_expected:.3f}, z={relationship_z_world_expected:.3f}")
        debug_log(f"  Relationship error: {np.linalg.norm(actual_rel_xz - np.array([relationship_x_world_expected, relationship_z_world_expected])):.6f}")
        
        # Check leader position for reference
        leader_final_first = leader_transformed[0]
        leader_final_root = leader_final_first[:3]
        debug_log(f"Leader final (first frame): root={leader_final_root}")
        debug_log(f"Follower should be at: leader_root + relationship_offset (in world space)")
        debug_log(f"  Expected relationship offset in world space: x={relationship_x:.3f}, z={relationship_z:.3f} (needs rotation)")
        
        # Update new world origin for next iteration – extrapolate one step ahead
        # to account for the missing transition frame between consecutive windows.
        _pos_last, _yaw_last, _ = extract_root_pos_yaw(leader_transformed[-1])
        _pos_prev, _yaw_prev, _ = extract_root_pos_yaw(leader_transformed[-2])
        new_world_origin_pos = _pos_last + (_pos_last - _pos_prev)
        new_world_origin_yaw_half = _yaw_last + (_yaw_last - _yaw_prev)
        debug_log(f"Updated world origin for next window: pos={new_world_origin_pos}, yaw_half={np.degrees(new_world_origin_yaw_half):.2f}°")
        
        # Check continuity at boundary
        if w > 0:
            prev_follower_last = transformed_follower_windows[-2][-1]  # Last frame of previous window
            curr_follower_first = follower_transformed[0]  # First frame of current window
            prev_leader_last = transformed_leader_windows[-2][-1]
            curr_leader_first = leader_transformed[0]
            
            prev_follower_root = prev_follower_last[:3]
            curr_follower_root = curr_follower_first[:3]
            prev_leader_root = prev_leader_last[:3]
            curr_leader_root = curr_leader_first[:3]
            
            follower_jump = np.linalg.norm(curr_follower_root - prev_follower_root)
            leader_jump = np.linalg.norm(curr_leader_root - prev_leader_root)
            
            # Compute actual relationship at boundary from positions
            actual_rel_at_boundary_xz = (prev_follower_root - prev_leader_root)[[0, 2]]
            _, prev_leader_yaw, _ = extract_root_pos_yaw(prev_leader_last)
            _, prev_follower_yaw, _ = extract_root_pos_yaw(prev_follower_last)
            actual_rel_yaw = prev_follower_yaw - prev_leader_yaw
            while actual_rel_yaw > np.pi:
                actual_rel_yaw -= 2 * np.pi
            while actual_rel_yaw < -np.pi:
                actual_rel_yaw += 2 * np.pi
            
            debug_log(f"\nContinuity check at boundary:")
            debug_log(f"  Leader jump: {leader_jump:.6f} (should be ~0)")
            debug_log(f"  Follower jump: {follower_jump:.6f} (should be ~0)")
            debug_log(f"  Prev follower root: {prev_follower_root}")
            debug_log(f"  Curr follower root: {curr_follower_root}")
            debug_log(f"  Prev leader root: {prev_leader_root}")
            debug_log(f"  Curr leader root: {curr_leader_root}")
            debug_log(f"  Actual relationship at boundary (from positions):")
            debug_log(f"    angle_half={np.degrees(actual_rel_yaw/2):.2f}°, x={actual_rel_at_boundary_xz[0]:.3f}, z={actual_rel_at_boundary_xz[1]:.3f}")
            debug_log(f"  Relationship used (from current window first frame):")
            debug_log(f"    angle_half={np.degrees(relationship_angle_half):.2f}°, x={relationship_x:.3f}, z={relationship_z:.3f}")
            debug_log(f"  Relationship mismatch:")
            debug_log(f"    angle_diff={np.degrees((actual_rel_yaw/2) - relationship_angle_half):.2f}°, x_diff={actual_rel_at_boundary_xz[0] - relationship_x:.3f}, z_diff={actual_rel_at_boundary_xz[1] - relationship_z:.3f}")
    
    # Concatenate all windows, inserting one linearly-interpolated frame at each boundary
    # to compensate for the raw frame lost in the 20-frame → 19-frame IH conversion.
    all_leader = [transformed_leader_windows[0]]
    all_follower = [transformed_follower_windows[0]]
    all_rel = [transformed_relationship_windows[0]]
    for w in range(1, len(transformed_leader_windows)):
        interp_L = (transformed_leader_windows[w - 1][-1:] + transformed_leader_windows[w][:1]) * 0.5
        interp_F = (transformed_follower_windows[w - 1][-1:] + transformed_follower_windows[w][:1]) * 0.5
        interp_R = (transformed_relationship_windows[w - 1][-1:] + transformed_relationship_windows[w][:1]) * 0.5
        all_leader.extend([interp_L, transformed_leader_windows[w]])
        all_follower.extend([interp_F, transformed_follower_windows[w]])
        all_rel.extend([interp_R, transformed_relationship_windows[w]])
    leader_continuous = np.concatenate(all_leader, axis=0)
    follower_continuous = np.concatenate(all_follower, axis=0)
    relationship_continuous = np.concatenate(all_rel, axis=0)
    
    debug_log(f"\n--- Final Summary ---")
    debug_log(f"Total frames: {len(leader_continuous)}")
    debug_log(f"Leader shape: {leader_continuous.shape}")
    debug_log(f"Follower shape: {follower_continuous.shape}")
    debug_log(f"Relationship shape: {relationship_continuous.shape}")
    debug_log(f"Debug log saved to: {debug_log_path}")
    
    return leader_continuous, follower_continuous, relationship_continuous


class InterHumanVisualizationApp:
    """Main visualization application with InterHuman support."""
    
    def __init__(self):
        self.dataset = None  # Will be Salsa_Dataset instance
        self.current_idx = 0
        self.temp_dir = tempfile.mkdtemp()
        self.is_MDM = False
        self.parent_dir = Path(__file__).parent.parent
        
        # InterHuman tokenizers (lazy loaded)
        self.interhuman_motion_tokenizer = None
        self.relationship_tokenizer = None
        self.interhuman_normalization_stats = None
        self.relationship_normalization_stats = None
        
        # Audio tokenizer (lazy loaded)
        self.wavtokenizer = None
        
        # Cached LLM for inference / latency (reload only when checkpoint changes)
        self._llm_model = None
        self._llm_ckpt_path = None
        
        # Store last computed statistics for dropdown access
        self.last_computed_stats = None
    
    def load_lmdb(self, lmdb_path: str, is_MDM: bool) -> Tuple[str, int, int]:
        """Load an LMDB file using Salsa_Dataset (automatically creates cache if needed)."""
        try:
            if not os.path.exists(lmdb_path):
                return f"Error: LMDB path does not exist: {lmdb_path}", 0, 0
            
            # Lazy import Salsa_Dataset to avoid import issues
            # The parent directory should already be in sys.path from the top of the file
            # Ensure Salsa-Agent root is first and motion_representation is NOT in path during this import.
            # motion_representation has its own models/ package (no vqvae.py); if it's in path first, "import models.vqvae" fails.
            parent_dir_str = str(self.parent_dir)
            motion_rep_str = str(self.parent_dir / 'motion_representation')
            while motion_rep_str in sys.path:
                sys.path.remove(motion_rep_str)
            if parent_dir_str in sys.path:
                sys.path.remove(parent_dir_str)
            sys.path.insert(0, parent_dir_str)

            # Test if we can import models.vqvae directly (this is what salsa_dataloader.py needs)
            try:
                import models.vqvae as test_vqvae
                del test_vqvae  # Clean up
            except ImportError as test_e:
                # Re-add motion_representation so other code still works
                if motion_rep_str not in sys.path:
                    sys.path.insert(1, motion_rep_str)
                return (f"Error: Cannot import models.vqvae. This is required by Salsa_Dataset.\n"
                       f"Error: {str(test_e)}\n"
                       f"Parent dir in sys.path: {parent_dir_str in sys.path}\n"
                       f"Parent dir: {parent_dir_str}\n"
                       f"Current sys.path entries: {sys.path[:5]}\n"
                       f"\nPlease ensure you're running from the Salsa-Agent directory or that the path is set correctly."), 0, 0

            # Re-add motion_representation so Salsa_Dataset and rest of app can import it
            if motion_rep_str not in sys.path:
                sys.path.insert(1, motion_rep_str)
            
            # Fix import path for WavTokenizer (already done at top, but ensure it's there)
            wavtokenizer_base = self.parent_dir / 'utils' / 'salsa_utils' / 'libs' / 'WavTokenizer'
            if str(wavtokenizer_base) not in sys.path:
                sys.path.insert(0, str(wavtokenizer_base))
            
            try:
                from utils.salsa_utils.salsa_dataloader import Salsa_Dataset
            except ImportError as e:
                import traceback
                error_details = traceback.format_exc()
                return (f"Error importing Salsa_Dataset: {str(e)}\n\n"
                       f"Traceback:\n{error_details}\n\n"
                       f"Note: models.vqvae import test passed, so the issue is elsewhere.\n"
                       f"Please check the full traceback above."), 0, 0
            
            # Create args for Salsa_Dataset
            args = get_args_parser()
            args.is_MDM = is_MDM
            args.parent_dir = str(self.parent_dir)
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use Salsa_Dataset with custom cache suffix for visualization
            # This will automatically create cache if it doesn't exist
            cache_suffix = "_salsa_dataset_cache"  # Different cache name for visualization
            
            print(f"Loading dataset from: {lmdb_path}")
            print(f"Cache will be created at: {lmdb_path}{cache_suffix}")
            if is_MDM:
                print(f"  (MDM format: {lmdb_path}{cache_suffix}_MDM)")
            
            self.dataset = Salsa_Dataset(
                args,
                lmdb_dir=lmdb_path,
                n_poses=100,  # Same as training
                subdivision_stride=50,  # Same as training
                pose_resampling_fps=20,  # Same as training
                cache_suffix=cache_suffix
            )
            
            self.is_MDM = is_MDM
            self.current_idx = 0
            
            total_samples = len(self.dataset)
            cache_path = lmdb_path + cache_suffix
            if is_MDM:
                cache_path += '_MDM'
            
            status_msg = f"Loaded dataset with {total_samples} samples\n"
            status_msg += f"Cache location: {cache_path}\n"
            if os.path.exists(cache_path):
                status_msg += f"Using existing cache"
            else:
                status_msg += f"Cache created automatically from raw LMDB"
            
            return status_msg, 0, total_samples
        except Exception as e:
            import traceback
            error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
            debug_print(error_msg)
            return error_msg, 0, 0
    
    def _get_sample_from_dataset(self, idx: int) -> dict:
        """Get a sample from the dataset and convert to dict format."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        # Access raw LMDB to get full sample (including InterHuman data)
        with self.dataset.lmdb_env.begin(write=False) as txn:
            key = "{:010}".format(idx).encode("ascii")
            sample_bytes = txn.get(key)
            if sample_bytes is None:
                raise KeyError(f"Sample {idx} not found")
            # Suppress deprecation warning for pyarrow.deserialize (data is serialized with pyarrow)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*pyarrow.deserialize.*")
            sample = pyarrow.deserialize(sample_bytes)
        
        # Convert tuple to dict
        return self._tuple_to_dict(sample)
    
    def _tuple_to_dict(self, sample) -> dict:
        """Convert sample tuple to dict format for compatibility."""
        interhuman_data = None
        sample_len = len(sample) if hasattr(sample, '__len__') else 0
        
        ms_motioncodes_L = None
        ms_motioncodes_F = None
        if not self.is_MDM:
            if sample_len == 14:
                # Format with InterHuman + MotionScript timeline motioncodes
                poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                 poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                 audio_tokens, audio_raw, aux_info, interhuman_data, \
                 ms_motioncodes_L, ms_motioncodes_F = sample
            elif sample_len == 12:
                # New format with InterHuman (no motioncodes)
                poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                 poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                 audio_tokens, audio_raw, aux_info, interhuman_data = sample
            elif sample_len == 11:
                # Old format without InterHuman
                poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                 poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                 audio_tokens, audio_raw, aux_info = sample
            else:
                raise ValueError(f"Unexpected sample length: {sample_len} (expected 11, 12 or 14 for non-MDM)")
            HML3D_L = None
            HML3D_F = None
        else:
            if sample_len == 14:
                # New format with InterHuman
                poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                    poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                    audio_tokens, audio_raw, aux_info, interhuman_data = sample
            elif sample_len == 13:
                # Old format without InterHuman
                poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                    poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                    audio_tokens, audio_raw, aux_info = sample
            else:
                raise ValueError(f"Unexpected sample length: {sample_len} (expected 13 or 14 for MDM)")
        
        return {
            'poses_keypoints3d_L': poses_keypoints3d_L,
            'poses_rotmat_L': poses_rotmat_L,
            'poses_keypoints3d_F': poses_keypoints3d_F,
            'poses_rotmat_F': poses_rotmat_F,
            'ms_desc_L': ms_desc_L,
            'ms_des_F': ms_des_F,
            'vq_tokens_L': vq_tokens_L,
            'vq_tokens_F': vq_tokens_F,
            'audio_tokens': audio_tokens,
            'audio_raw': audio_raw,
            'aux_info': aux_info,
            'HML3D_L': HML3D_L,
            'HML3D_F': HML3D_F,
            'interhuman_data': interhuman_data,
            'ms_motioncodes_L': ms_motioncodes_L,
            'ms_motioncodes_F': ms_motioncodes_F,
        }

    def _generate_motionscript_timeline_gifs(self, idx: int, show_timeline: bool):
        """Generate MotionScript timeline GIFs for leader and follower using MotionScript's create_gif_with_blinking.
        Returns (leader_gif_path, follower_gif_path, status_message).
        """
        def _err(msg):
            return None, None, msg
        if not show_timeline:
            return None, None, "Checkbox is off; enable it to generate timeline."
        try:
            sample = self._get_sample_from_dataset(idx)
        except Exception as e:
            import traceback
            return None, None, f"Failed to load sample: {e}\n{traceback.format_exc()}"
        ms_L = sample.get('ms_motioncodes_L')
        ms_F = sample.get('ms_motioncodes_F')
        poses_L = sample.get('poses_keypoints3d_L')
        total_frames = int(poses_L.shape[0]) if poses_L is not None and hasattr(poses_L, 'shape') else 0
        if total_frames <= 0:
            return _err("No pose frames in sample (total_frames <= 0).")
        if (ms_L is None or len(ms_L) == 0) and (ms_F is None or len(ms_F) == 0):
            return _err(
                "No MotionScript motion codes for this sample. "
                "Use a cache built with is_MDM=False and rebuild the cache to include motion codes."
            )
        try:
            from utils.salsa_utils.libs.MotionScript.MS_Algorithms import create_gif_with_blinking, merge_gifs_vertical
        except ImportError as e:
            return _err(f"Could not import MotionScript timeline: {e}")
        import copy
        path_L = None
        path_F = None
        errors = []
        poses_F = sample.get('poses_keypoints3d_F')

        def _render_and_merge(keypoints, timeline_gif_path, merged_path, role_label):
            """Render skeleton to mp4, then merge vertically (animation on top, timeline below)."""
            if keypoints is None or not hasattr(keypoints, 'shape') or keypoints.size == 0:
                return None
            anim_mp4 = os.path.join(self.temp_dir, f"motionscript_anim_{role_label}_{idx}.mp4")
            try:
                render_skeleton_from_keypoints(
                    np.asarray(keypoints, dtype=np.float32),
                    anim_mp4,
                    title=f"{role_label} (MotionScript)",
                    fps=20,
                    radius=4,
                    figsize=(6, 6),
                    dpi=100,
                )
            except Exception as e:
                import traceback
                errors.append(f"{role_label} animation: {e}\n{traceback.format_exc()}")
                return timeline_gif_path
            if not os.path.exists(anim_mp4):
                return timeline_gif_path
            try:
                merge_gifs_vertical(anim_mp4, timeline_gif_path, merged_path)
                return merged_path if os.path.exists(merged_path) else timeline_gif_path
            except Exception as e:
                import traceback
                errors.append(f"{role_label} merge: {e}\n{traceback.format_exc()}")
                return timeline_gif_path

        if ms_L is not None and len(ms_L) > 0:
            try:
                motioncodes_L = copy.deepcopy(ms_L)
                out_L = os.path.join(self.temp_dir, f"motionscript_timeline_leader_{idx}.gif")
                create_gif_with_blinking(motioncodes_L, total_frames=total_frames, outname=out_L)
                if os.path.exists(out_L):
                    merged_L = os.path.join(self.temp_dir, f"motionscript_merged_leader_{idx}.gif")
                    path_L = _render_and_merge(poses_L, out_L, merged_L, "Leader")
            except Exception as e:
                import traceback
                errors.append(f"Leader: {e}\n{traceback.format_exc()}")
        if ms_F is not None and len(ms_F) > 0:
            try:
                motioncodes_F = copy.deepcopy(ms_F)
                out_F = os.path.join(self.temp_dir, f"motionscript_timeline_follower_{idx}.gif")
                create_gif_with_blinking(motioncodes_F, total_frames=total_frames, outname=out_F)
                if os.path.exists(out_F):
                    merged_F = os.path.join(self.temp_dir, f"motionscript_merged_follower_{idx}.gif")
                    path_F = _render_and_merge(poses_F, out_F, merged_F, "Follower")
            except Exception as e:
                import traceback
                errors.append(f"Follower: {e}\n{traceback.format_exc()}")
        status = "Generated leader (animation + timeline)." if path_L else "Leader: no output."
        status += " Generated follower (animation + timeline)." if path_F else " Follower: no output."
        if errors:
            status += "\n\nErrors:\n" + "\n---\n".join(errors)
        return path_L, path_F, status

    def _run_motionscript_stat_analysis(self, max_samples: int = 50) -> str:
        """Run MotionScript motioncode statistical analysis on the loaded dataset (leader motions).
        Returns text summary and saves PDFs to temp dir; reads back statistics.txt for display.
        """
        if self.dataset is None:
            return "Error: No dataset loaded. Load an LMDB first."
        if self.is_MDM:
            return "Error: MotionScript motioncode stats require a non-MDM cache (Is MDM Format unchecked)."
        try:
            import utils.salsa_utils.libs.MotionScript.captioning_motion_Salsa as MS_Salsa
            from utils.salsa_utils.libs.MotionScript import captioning as captioning_py
        except ImportError as e:
            return f"Error: MotionScript not available: {e}"
        from scipy.spatial.transform import Rotation as R
        from collections import defaultdict

        n_total = len(self.dataset)
        n_run = min(int(max_samples), n_total) if max_samples else n_total
        all_motion_stats = defaultdict(list)
        errors = []
        skipped = 0
        for idx in range(n_run):
            try:
                sample = self._get_sample_from_dataset(idx)
            except Exception as e:
                errors.append(f"Sample {idx}: {e}")
                skipped += 1
                continue
            poses_kp = sample.get("poses_keypoints3d_L")
            poses_rotmat = sample.get("poses_rotmat_L")
            if poses_kp is None or poses_rotmat is None or not hasattr(poses_kp, "shape") or poses_kp.size == 0:
                skipped += 1
                if idx == 0:
                    errors.append(f"Sample 0: missing or empty poses_keypoints3d_L/poses_rotmat_L (keys present: {list(sample.keys())})")
                continue
            poses_kp = np.asarray(poses_kp, dtype=np.float64).copy()
            poses_rotmat = np.asarray(poses_rotmat, dtype=np.float64).copy()
            T = poses_rotmat.shape[0]
            # Cache stores rotmat as (T, 498): [trans(3) | 55*9 flattened rotmats]; or (T, J, 3, 3)
            if poses_rotmat.ndim == 2 and poses_rotmat.shape[1] == 498:
                trans = poses_rotmat[:, :3].copy()
                flat_rot = poses_rotmat[:, 3:].copy()  # (T, 495) -> (T, 55, 3, 3)
                rotmat_4d = flat_rot.reshape(T, 55, 3, 3)
                J_use = 22
                rotmat_4d = rotmat_4d[:, :J_use]
            elif poses_rotmat.ndim == 4 and poses_rotmat.shape[2:4] == (3, 3):
                J_use = poses_rotmat.shape[1]
                rotmat_4d = poses_rotmat
                trans = poses_kp[:, 0, :].copy()
            else:
                if idx == 0:
                    errors.append(f"Sample 0: rotmat shape {poses_rotmat.shape} not (T,498) or (T,J,3,3)")
                skipped += 1
                continue
            rotvec = np.zeros((T, J_use * 3), dtype=np.float64)
            for t in range(T):
                for j in range(J_use):
                    # .copy() so scipy gets a writable buffer (cache arrays can be read-only)
                    rotvec[t, j * 3 : (j + 1) * 3] = R.from_matrix(np.asarray(rotmat_4d[t, j], dtype=np.float64).copy()).as_rotvec()
            input_loaded = {
                "poses": rotvec,
                "3d_keypoints": poses_kp,
                "trans": trans,
                "body_betas": None,
                "body_vertices": None,
                "body_faces": None,
            }
            try:
                m_interpretations = MS_Salsa.MotionScript_Forward_Salsa(
                    input_loaded, motion_id=f"stat_{idx}", motion_stats=True
                )
            except Exception as e:
                errors.append(f"Sample {idx} MotionScript: {e}")
                skipped += 1
                continue
            if not isinstance(m_interpretations, dict):
                skipped += 1
                if idx == 0:
                    ty = type(m_interpretations).__name__
                    err_details = repr(m_interpretations)[:200] if m_interpretations is not None else "None"
                    errors.append(f"Sample 0: MotionScript returned {ty} (expected dict): {err_details}")
                continue
            # Aggregate by joint-set index: step2 expects m_interpretations[k][j] = list of interpretations for joint set j
            for k in m_interpretations:
                for j, joint_set_list in enumerate(m_interpretations[k]):
                    while len(all_motion_stats[k]) <= j:
                        all_motion_stats[k].append([])
                    all_motion_stats[k][j].extend(joint_set_list)

        if not all_motion_stats:
            err_msg = (
                f"No motioncode stats collected (skipped {skipped}/{n_run} samples). "
                "Possible causes: (1) samples missing or empty poses_keypoints3d_L/poses_rotmat_L, "
                "(2) rotmat shape not (T,498) or (T,J,3,3), (3) MotionScript returned no stats (e.g. very short sequences)."
            )
            if errors:
                err_msg += "\n\nErrors / first-sample info:\n" + "\n".join(errors[:10])
            return err_msg

        stat_dir = os.path.join(self.temp_dir, "motionscript_statistics")
        os.makedirs(stat_dir, exist_ok=True)
        try:
            captioning_py.motioncode_stat_analysis_step2_visualization(dict(all_motion_stats), stat_dir)
        except Exception as e:
            import traceback
            return f"Error in step2 visualization: {e}\n{traceback.format_exc()}"
        stats_file = os.path.join(stat_dir, "statistics.txt")
        if os.path.isfile(stats_file):
            with open(stats_file, "r") as f:
                text = f.read()
        else:
            text = "statistics.txt not written."
        pdfs = [f for f in os.listdir(stat_dir) if f.endswith(".pdf")]
        if pdfs:
            text += f"\n\nPlots saved in: {stat_dir}\nFiles: " + ", ".join(sorted(pdfs))
        text = f"Analyzed {n_run} samples (of {n_total}).\n\n{text}"
        if errors:
            text += "\n\nSample errors (first 5):\n" + "\n".join(errors[:5])
        return text

    def _load_interhuman_tokenizers(self):
        """Lazy load InterHuman and Relationship VQVAE tokenizers."""
        if self.interhuman_motion_tokenizer is not None:
            return  # Already loaded
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        motion_rep_path = self.parent_dir / 'motion_representation'
        
        # Load InterHuman Motion VQVAE
        interhuman_ckpt_dir = motion_rep_path / 'checkpoints_VQVAE_GRU_InterHuman'
        import glob
        checkpoints = glob.glob(str(interhuman_ckpt_dir / 'checkpoint_epoch_*.pth'))
        best_path = interhuman_ckpt_dir / 'best_checkpoint.pth'
        
        if best_path.exists():
            interhuman_ckpt_path = str(best_path)
        elif checkpoints:
            interhuman_ckpt_path = max(checkpoints, key=os.path.getctime)
        else:
            raise FileNotFoundError(f"InterHuman VQVAE checkpoint not found in {interhuman_ckpt_dir}")
        
        print(f"Loading InterHuman Motion VQVAE from {interhuman_ckpt_path}")
        interhuman_ckpt = torch.load(interhuman_ckpt_path, map_location='cpu')
        interhuman_config = interhuman_ckpt['config']
        
        self.interhuman_motion_tokenizer = MotionModel(
            input_dim=interhuman_config['input_dim'],
            hidden_dim=interhuman_config['hidden_dim'],
            num_layers=interhuman_config['num_layers'],
            latent_dim=interhuman_config['latent_dim'],
            seq_len=interhuman_config['seq_len'],
            dropout=interhuman_config['dropout'],
            encoder_type=interhuman_config.get('encoder_type', 'gru'),
            decoder_type=interhuman_config.get('decoder_type', 'gru'),
            use_vqvae=True,
            nb_code=interhuman_config.get('nb_code', 512),
            quantizer=interhuman_config.get('quantizer', 'ema_reset'),
            vq_mu=interhuman_config.get('vq_mu', 0.95),
        ).to(device)
        self.interhuman_motion_tokenizer.load_state_dict(interhuman_ckpt['model_state_dict'])
        self.interhuman_motion_tokenizer.eval()
        
        if DEBUG:
            print(f"Loaded InterHuman Motion Tokenizer:")
            print(f"  input_dim: {interhuman_config['input_dim']}")
            print(f"  seq_len: {interhuman_config['seq_len']}")
            print(f"  use_vqvae: {self.interhuman_motion_tokenizer.use_vqvae}")
            print(f"  nb_code: {interhuman_config.get('nb_code', 'N/A')}")
            print(f"  latent_dim: {interhuman_config['latent_dim']}")
        
        # Load Relationship VQVAE
        relationship_ckpt_dir = motion_rep_path / 'checkpoints_VQVAE_GRU_Relationship'
        checkpoints = glob.glob(str(relationship_ckpt_dir / 'checkpoint_epoch_*.pth'))
        best_path = relationship_ckpt_dir / 'best_checkpoint.pth'
        
        if best_path.exists():
            relationship_ckpt_path = str(best_path)
        elif checkpoints:
            relationship_ckpt_path = max(checkpoints, key=os.path.getctime)
        else:
            raise FileNotFoundError(f"Relationship VQVAE checkpoint not found in {relationship_ckpt_dir}")
        
        print(f"Loading Relationship VQVAE from {relationship_ckpt_path}")
        relationship_ckpt = torch.load(relationship_ckpt_path, map_location='cpu')
        relationship_config = relationship_ckpt['config']
        
        self.relationship_tokenizer = MotionModel(
            input_dim=relationship_config['input_dim'],
            hidden_dim=relationship_config['hidden_dim'],
            num_layers=relationship_config['num_layers'],
            latent_dim=relationship_config['latent_dim'],
            seq_len=relationship_config['seq_len'],
            dropout=relationship_config['dropout'],
            encoder_type=relationship_config.get('encoder_type', 'gru'),
            decoder_type=relationship_config.get('decoder_type', 'gru'),
            use_vqvae=True,
            nb_code=relationship_config.get('nb_code', 512),
            quantizer=relationship_config.get('quantizer', 'ema_reset'),
            vq_mu=relationship_config.get('vq_mu', 0.95),
        ).to(device)
        self.relationship_tokenizer.load_state_dict(relationship_ckpt['model_state_dict'])
        self.relationship_tokenizer.eval()
        
        # Load normalization statistics
        # Try multiple possible cache locations (matching vae_visualization_app.py approach)
        import pickle
        
        # First, try to get cache_dir from dataset if available
        cache_dir = None
        if self.dataset is not None:
            # Check if dataset has cache_dir attribute
            if hasattr(self.dataset, 'cache_dir'):
                cache_dir = Path(self.dataset.cache_dir)
            elif hasattr(self.dataset, 'lmdb_env'):
                # Try to infer from lmdb_env path
                if hasattr(self.dataset.lmdb_env, 'path'):
                    cache_dir = Path(self.dataset.lmdb_env.path()).parent
        
        # If not found, try common locations
        possible_cache_dirs = []
        if cache_dir and cache_dir.exists():
            possible_cache_dirs.append(cache_dir)
        
        # Add other possible locations (try both with and without /dd/)
        base_cache_dd = self.parent_dir / 'dataset_processed_New' / 'lmdb_Salsa_pair' / 'dd'
        base_cache_no_dd = self.parent_dir / 'dataset_processed_New' / 'lmdb_Salsa_pair'
        possible_cache_dirs.extend([
            base_cache_no_dd / 'lmdb_train_interhuman_20frames_cache',  # Actual location (no /dd/)
            base_cache_dd / 'lmdb_train_interhuman_20frames_cache',  # Alternative location
            base_cache_dd / 'lmdb_train_salsa_dataset_cache',
            base_cache_no_dd / 'lmdb_train_salsa_dataset_cache',
        ])
        
        # Try to find stats files in any of these locations
        interhuman_stats_path = None
        relationship_stats_path = None
        
        for cache_dir_candidate in possible_cache_dirs:
            if DEBUG:
                print(f"Checking cache directory: {cache_dir_candidate}")
            if cache_dir_candidate.exists():
                ih_path = cache_dir_candidate / 'normalization_stats_interhuman.pkl'
                rel_path = cache_dir_candidate / 'normalization_stats_relationship.pkl'
                if ih_path.exists() and interhuman_stats_path is None:
                    interhuman_stats_path = ih_path
                    if DEBUG:
                        print(f"  Found InterHuman stats at: {ih_path}")
                if rel_path.exists() and relationship_stats_path is None:
                    relationship_stats_path = rel_path
                    if DEBUG:
                        print(f"  Found Relationship stats at: {rel_path}")
        
        EXPECTED_INTERHUMAN_DIM = 262  # Must match InterHuman representation (salsa_to_interhuman output)
        if interhuman_stats_path and interhuman_stats_path.exists():
            with open(interhuman_stats_path, 'rb') as f:
                self.interhuman_normalization_stats = pickle.load(f)
            if isinstance(self.interhuman_normalization_stats, dict) and 'mean' in self.interhuman_normalization_stats and 'std' in self.interhuman_normalization_stats:
                mean_arr = np.asarray(self.interhuman_normalization_stats['mean'])
                std_arr = np.asarray(self.interhuman_normalization_stats['std'])
                if mean_arr.ndim != 1 or std_arr.ndim != 1 or mean_arr.shape[0] != EXPECTED_INTERHUMAN_DIM or std_arr.shape[0] != EXPECTED_INTERHUMAN_DIM:
                    print(f"WARNING: InterHuman normalization stats have wrong shape (expected 1D length {EXPECTED_INTERHUMAN_DIM}). "
                          f"Got mean.shape={mean_arr.shape}, std.shape={std_arr.shape}. Denormalization will be SKIPPED (poses may look wrong).")
                    self.interhuman_normalization_stats = None
                else:
                    # Avoid zero std (would make denorm wrong); match dataset behavior for constant dims
                    epsilon = 1e-8
                    std_safe = np.where(std_arr < epsilon, 1.0, std_arr).astype(std_arr.dtype)
                    self.interhuman_normalization_stats = {**self.interhuman_normalization_stats, 'std': std_safe}
            if DEBUG and self.interhuman_normalization_stats is not None:
                print(f"Loaded InterHuman normalization stats from: {interhuman_stats_path}")
                if isinstance(self.interhuman_normalization_stats, dict):
                    print(f"  Stats keys: {list(self.interhuman_normalization_stats.keys())}")
                    if 'mean' in self.interhuman_normalization_stats:
                        mean_arr = self.interhuman_normalization_stats['mean']
                        print(f"  mean shape: {mean_arr.shape}, dtype: {mean_arr.dtype}")
                        print(f"  mean sample (first 10): {mean_arr[:10] if len(mean_arr) >= 10 else mean_arr}")
                    if 'std' in self.interhuman_normalization_stats:
                        std_arr = self.interhuman_normalization_stats['std']
                        print(f"  std shape: {std_arr.shape}, dtype: {std_arr.dtype}")
                        print(f"  std sample (first 10): {std_arr[:10] if len(std_arr) >= 10 else std_arr}")
        else:
            if DEBUG:
                print(f"WARNING: InterHuman normalization stats not found at: {interhuman_stats_path}")
        
        EXPECTED_RELATIONSHIP_DIM = 4  # [w, z, x, z] - relationship features
        if relationship_stats_path and relationship_stats_path.exists():
            with open(relationship_stats_path, 'rb') as f:
                self.relationship_normalization_stats = pickle.load(f)
            if isinstance(self.relationship_normalization_stats, dict) and 'mean' in self.relationship_normalization_stats and 'std' in self.relationship_normalization_stats:
                mean_arr = np.asarray(self.relationship_normalization_stats['mean'])
                std_arr = np.asarray(self.relationship_normalization_stats['std'])
                if mean_arr.ndim != 1 or std_arr.ndim != 1 or mean_arr.shape[0] != EXPECTED_RELATIONSHIP_DIM or std_arr.shape[0] != EXPECTED_RELATIONSHIP_DIM:
                    print(f"WARNING: Relationship normalization stats have wrong shape (expected 1D length {EXPECTED_RELATIONSHIP_DIM}). "
                          f"Got mean.shape={mean_arr.shape}, std.shape={std_arr.shape}. Denormalization will be SKIPPED.")
                    self.relationship_normalization_stats = None
                else:
                    epsilon = 1e-8
                    std_safe = np.where(std_arr < epsilon, 1.0, std_arr).astype(std_arr.dtype)
                    self.relationship_normalization_stats = {**self.relationship_normalization_stats, 'std': std_safe}
            if DEBUG and self.relationship_normalization_stats is not None:
                print(f"Loaded Relationship normalization stats from: {relationship_stats_path}")
                if isinstance(self.relationship_normalization_stats, dict):
                    print(f"  Stats keys: {list(self.relationship_normalization_stats.keys())}")
                    if 'mean' in self.relationship_normalization_stats:
                        mean_arr = self.relationship_normalization_stats['mean']
                        print(f"  mean shape: {mean_arr.shape}, dtype: {mean_arr.dtype}")
                        print(f"  mean values: {mean_arr}")
                    if 'std' in self.relationship_normalization_stats:
                        std_arr = self.relationship_normalization_stats['std']
                        print(f"  std shape: {std_arr.shape}, dtype: {std_arr.dtype}")
                        print(f"  std values: {std_arr}")
        else:
            if DEBUG:
                print(f"WARNING: Relationship normalization stats not found!")
                print(f"  Searched in: {[str(d) for d in possible_cache_dirs]}")
                if relationship_stats_path:
                    print(f"  Last attempted path: {relationship_stats_path}")
        
        print("InterHuman tokenizers loaded successfully!")
    
    def get_sample_data_summary(self, idx: int) -> str:
        """Get comprehensive summary of all data stored in sample."""
        if self.dataset is None:
            return "No dataset loaded"
        
        try:
            sample = self._get_sample_from_dataset(idx)
            lines = []
            
            # Basic info
            aux_info = sample.get('aux_info', {})
            lines.append("=" * 80)
            lines.append("SAMPLE DATA SUMMARY")
            lines.append("=" * 80)
            lines.append(f"\nSample Index: {idx}")
            lines.append(f"Video ID: {aux_info.get('vid', 'N/A')}")
            lines.append(f"Time Range: {aux_info.get('start_time', 'N/A'):.2f}s - {aux_info.get('end_time', 'N/A'):.2f}s")
            lines.append(f"Frame Range: {aux_info.get('start_frame_no', 'N/A')} - {aux_info.get('end_frame_no', 'N/A')}")
            
            # Keypoints
            if sample.get('poses_keypoints3d_L') is not None:
                kp_L = sample['poses_keypoints3d_L']
                lines.append(f"\nLeader Keypoints: {kp_L.shape if hasattr(kp_L, 'shape') else 'N/A'}")
            if sample.get('poses_keypoints3d_F') is not None:
                kp_F = sample['poses_keypoints3d_F']
                lines.append(f"Follower Keypoints: {kp_F.shape if hasattr(kp_F, 'shape') else 'N/A'}")
            
            # Motion Scripts
            ms_L = sample.get('ms_desc_L')
            ms_F = sample.get('ms_des_F')
            if ms_L:
                lines.append(f"\nLeader Motion Script: {ms_L if isinstance(ms_L, str) else 'Available'}")
            if ms_F:
                lines.append(f"Follower Motion Script: {ms_F if isinstance(ms_F, str) else 'Available'}")
            
            # VQ Tokens (Legacy HumanML3D)
            vq_L = sample.get('vq_tokens_L')
            vq_F = sample.get('vq_tokens_F')
            if vq_L is not None:
                vq_L_arr = np.array(vq_L) if not isinstance(vq_L, np.ndarray) else vq_L
                lines.append(f"\nLegacy VQ Tokens (HumanML3D):")
                lines.append(f"  Leader: shape={vq_L_arr.shape}, range=[{vq_L_arr.min()}, {vq_L_arr.max()}]")
            if vq_F is not None:
                vq_F_arr = np.array(vq_F) if not isinstance(vq_F, np.ndarray) else vq_F
                lines.append(f"  Follower: shape={vq_F_arr.shape}, range=[{vq_F_arr.min()}, {vq_F_arr.max()}]")
            
            # InterHuman Data
            interhuman_data = sample.get('interhuman_data')
            if interhuman_data is not None:
                lines.append(f"\n{'='*80}")
                lines.append("INTERHUMAN DATA")
                lines.append("=" * 80)
                
                # InterHuman Motions
                if 'leader_motion_ih' in interhuman_data:
                    motion_L = np.array(interhuman_data['leader_motion_ih'])
                    lines.append(f"\nLeader InterHuman Motion: shape={motion_L.shape}, dtype={motion_L.dtype}")
                    lines.append(f"  Range: [{motion_L.min():.4f}, {motion_L.max():.4f}]")
                
                if 'follower_motion_ih' in interhuman_data:
                    motion_F = np.array(interhuman_data['follower_motion_ih'])
                    lines.append(f"Follower InterHuman Motion: shape={motion_F.shape}, dtype={motion_F.dtype}")
                    lines.append(f"  Range: [{motion_F.min():.4f}, {motion_F.max():.4f}]")
                
                # Relationship Features
                if 'relationship_features' in interhuman_data:
                    rel = np.array(interhuman_data['relationship_features'])
                    lines.append(f"\nRelationship Features: shape={rel.shape}, dtype={rel.dtype}")
                    lines.append(f"  Format: [w, z, x, z] per frame")
                    lines.append(f"  Range: [{rel.min():.4f}, {rel.max():.4f}]")
                
                # Tokens
                if 'leader_tokens' in interhuman_data:
                    tokens_L = np.array(interhuman_data['leader_tokens'])
                    lines.append(f"\nInterHuman Motion Tokens:")
                    lines.append(f"  Leader: {tokens_L} (shape={tokens_L.shape})")
                
                if 'follower_tokens' in interhuman_data:
                    tokens_F = np.array(interhuman_data['follower_tokens'])
                    lines.append(f"  Follower: {tokens_F} (shape={tokens_F.shape})")
                
                if 'relationship_tokens' in interhuman_data:
                    tokens_R = np.array(interhuman_data['relationship_tokens'])
                    lines.append(f"  Relationship: {tokens_R} (shape={tokens_R.shape})")
                
                # Root transforms
                if 'root_quat_init_L' in interhuman_data:
                    lines.append(f"\nRoot Transforms (Frame 0):")
                    lines.append(f"  Leader quaternion: {interhuman_data['root_quat_init_L']}")
                    lines.append(f"  Leader position: {interhuman_data['root_pos_init_L']}")
                    lines.append(f"  Follower quaternion: {interhuman_data['root_quat_init_F']}")
                    lines.append(f"  Follower position: {interhuman_data['root_pos_init_F']}")
            else:
                lines.append(f"\n{'='*80}")
                lines.append("INTERHUMAN DATA: Not available (old cache format)")
                lines.append("=" * 80)
            
            # Audio
            audio_tokens = sample.get('audio_tokens')
            audio_raw = sample.get('audio_raw')
            if audio_tokens is not None:
                audio_tokens_arr = np.array(audio_tokens) if not isinstance(audio_tokens, np.ndarray) else audio_tokens
                lines.append(f"\nAudio Tokens: shape={audio_tokens_arr.shape}")
            if audio_raw is not None:
                audio_raw_arr = np.array(audio_raw) if not isinstance(audio_raw, np.ndarray) else audio_raw
                lines.append(f"Audio Raw: shape={audio_raw_arr.shape}")
            
            return "\n".join(lines)
        except Exception as e:
            import traceback
            return f"Error loading sample: {str(e)}\n{traceback.format_exc()}"
    
    def visualize_interhuman_pair(
        self,
        idx: int,
        use_continuous_concatenation: bool = False,
        use_mesh: bool = False,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
        """
        Visualize InterHuman representation as two-person dancing.
        Shows: original InterHuman motions and combined. Optionally mesh (SMPL) when use_mesh=True.
        """
        if self.dataset is None:
            return None, None, None, None, "Error: No dataset loaded"
        
        if not INTERHUMAN_AVAILABLE:
            return None, None, None, None, "Error: InterHuman visualization dependencies not available"
        
        try:
            sample = self._get_sample_from_dataset(idx)
            interhuman_data = sample.get('interhuman_data')
            
            if interhuman_data is None:
                return None, None, None, None, "Error: InterHuman data not available in this sample (old cache format)"
            
            # Extract InterHuman motions - these are already stored in cache
            # Following vae_visualization_app.py approach: just visualize stored data
            leader_motion_ih = np.array(interhuman_data['leader_motion_ih'], dtype=np.float32)
            follower_motion_ih = np.array(interhuman_data['follower_motion_ih'], dtype=np.float32)
            relationship_features = np.array(interhuman_data['relationship_features'], dtype=np.float32)
            
            # Get token info to check cache format
            leader_tokens_array = np.array(interhuman_data['leader_tokens'])
            follower_tokens_array = np.array(interhuman_data['follower_tokens'])
            relationship_tokens_array = np.array(interhuman_data['relationship_tokens'])
            num_tokens = len(leader_tokens_array)
            seq_len = len(leader_motion_ih)
            expected_tokens = (seq_len + 19) // 19  # Approximate
            
            # Check if this is old cache format (1 token for long sequence)
            cache_warning = ""
            if num_tokens == 1 and seq_len > 20:
                cache_warning = f"\n⚠️  WARNING: Old cache format detected!\n"
                cache_warning += f"  Sequence has {seq_len} frames but only {num_tokens} token.\n"
                cache_warning += f"  Expected ~{expected_tokens} tokens (one per 20-frame window).\n"
                cache_warning += f"  Please regenerate cache to get proper window-by-window tokenization.\n"
            
            # Extract root transforms (from first window)
            root_quat_L = np.array(interhuman_data['root_quat_init_L'], dtype=np.float32)
            root_pos_L = np.array(interhuman_data['root_pos_init_L'], dtype=np.float32)
            root_quat_F = np.array(interhuman_data['root_quat_init_F'], dtype=np.float32)
            root_pos_F = np.array(interhuman_data['root_pos_init_F'], dtype=np.float32)
            
            # Apply continuous concatenation if requested
            if use_continuous_concatenation and num_tokens > 1:
                # Split full sequence back into windows (19 frames per window after salsa_to_interhuman)
                window_size_after_processing = 19
                leader_windows = []
                follower_windows = []
                relationship_windows = []
                root_quat_inits_L = []
                root_pos_inits_L = []
                root_quat_inits_F = []
                root_pos_inits_F = []
                
                for w in range(num_tokens):
                    window_start = w * window_size_after_processing
                    window_end = min(window_start + window_size_after_processing, seq_len)
                    
                    leader_windows.append(leader_motion_ih[window_start:window_end])
                    follower_windows.append(follower_motion_ih[window_start:window_end])
                    relationship_windows.append(relationship_features[window_start:window_end])
                    
                    # For root transforms, we only have frame 0 of first window
                    if w == 0:
                        root_quat_inits_L.append(root_quat_L)
                        root_pos_inits_L.append(root_pos_L)
                        root_quat_inits_F.append(root_quat_F)
                        root_pos_inits_F.append(root_pos_F)
                    else:
                        # For subsequent windows, use identity (not used by concatenation function)
                        root_quat_inits_L.append(np.array([1, 0, 0, 0], dtype=np.float32))
                        root_pos_inits_L.append(np.array([0, 0, 0], dtype=np.float32))
                        root_quat_inits_F.append(np.array([1, 0, 0, 0], dtype=np.float32))
                        root_pos_inits_F.append(np.array([0, 0, 0], dtype=np.float32))
                
                # Apply continuous concatenation
                leader_motion_ih, follower_motion_ih, relationship_features = concatenate_windows_with_continuity(
                    leader_windows, follower_windows, relationship_windows,
                    root_quat_inits_L, root_pos_inits_L, root_quat_inits_F, root_pos_inits_F
                )
                seq_len = len(leader_motion_ih)  # Update seq_len after concatenation
            
            # No reconstruction here - just visualize stored InterHuman motions
            # (Reconstruction is handled separately in visualize_reconstruction_from_tokens)
            
            # Extract keypoints from InterHuman motions (following vae_visualization_app.py approach)
            # InterHuman format: first 66 dims (22*3) are joint positions
            n_joints = 22
            n_joints = 22
            leader_motion_ih_array = np.asarray(leader_motion_ih, dtype=np.float32)
            follower_motion_ih_array = np.asarray(follower_motion_ih, dtype=np.float32)
            
            # Verify shapes
            if leader_motion_ih_array.ndim != 2 or leader_motion_ih_array.shape[1] < n_joints*3:
                return None, None, None, f"Error: Leader motion has wrong shape: {leader_motion_ih_array.shape}, expected (seq_len, 262)"
            if follower_motion_ih_array.ndim != 2 or follower_motion_ih_array.shape[1] < n_joints*3:
                return None, None, None, f"Error: Follower motion has wrong shape: {follower_motion_ih_array.shape}, expected (seq_len, 262)"
            
            # Extract first 66 dims and reshape to (seq_len, 22, 3)
            leader_keypoints_flat = leader_motion_ih_array[:, :n_joints*3]  # (seq_len, 66)
            leader_keypoints = leader_keypoints_flat.reshape(-1, n_joints, 3)  # (seq_len, 22, 3)
            
            follower_keypoints_flat = follower_motion_ih_array[:, :n_joints*3]  # (seq_len, 66)
            follower_keypoints = follower_keypoints_flat.reshape(-1, n_joints, 3)  # (seq_len, 22, 3)
            
            # Align follower for visualization.
            # If continuous concatenation was applied, follower is already aligned per-window.
            # Otherwise, align using relationship from frame 0.
            relationship_features_array = np.asarray(relationship_features, dtype=np.float32)
            rel_w = float(relationship_features_array[0, 0])
            rel_z = float(relationship_features_array[0, 1])
            rel_x = float(relationship_features_array[0, 2])
            rel_z_pos = float(relationship_features_array[0, 3])
            
            # Convert to rigid_transform format: [angle_half, x, z]
            angle_half = np.arctan2(rel_z, rel_w)
            relative_transform = np.array([angle_half, rel_x, rel_z_pos], dtype=np.float32)
            
            if use_continuous_concatenation and num_tokens > 1:
                follower_aligned_motion = follower_motion_ih_array
            else:
                # Apply rigid_transform to follower motion (full 262-dim)
                follower_aligned_motion = rigid_transform(relative_transform, follower_motion_ih_array.copy())
            
            follower_aligned_keypoints_flat = follower_aligned_motion[:, :n_joints*3]  # (seq_len, 66)
            follower_aligned_keypoints = follower_aligned_keypoints_flat.reshape(-1, n_joints, 3)  # (seq_len, 22, 3)
            
            # Ensure keypoints are numpy arrays with correct dtype
            leader_keypoints = np.asarray(leader_keypoints, dtype=np.float32)
            follower_aligned_keypoints = np.asarray(follower_aligned_keypoints, dtype=np.float32)
            
            # Final verification
            if leader_keypoints.ndim != 3:
                return None, None, None, None, f"Error: Leader keypoints wrong dimensions: {leader_keypoints.ndim}, expected 3"
            if follower_aligned_keypoints.ndim != 3:
                return None, None, None, None, f"Error: Follower keypoints wrong dimensions: {follower_aligned_keypoints.ndim}, expected 3"
            
            vid_id = sample.get('aux_info', {}).get('vid', f'sample_{idx}')
            
            # Create video paths
            leader_video_path = os.path.join(self.temp_dir, f"interhuman_leader_{idx}.mp4")
            follower_video_path = os.path.join(self.temp_dir, f"interhuman_follower_{idx}.mp4")
            combined_video_path = os.path.join(self.temp_dir, f"interhuman_combined_{idx}.mp4")
            
            # Visualize original leader (canonicalized)
            # plot_3d_motion_interhuman expects mp_joints: list of (seq_len, 22, 3) keypoint arrays
            plot_3d_motion_interhuman(
                save_path=leader_video_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[leader_keypoints],  # List of (19, 22, 3) arrays
                title=f"Leader (Canonicalized InterHuman)",
                fps=20,
                radius=4
            )
            
            # Visualize original follower (aligned to leader's space)
            plot_3d_motion_interhuman(
                save_path=follower_video_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[follower_aligned_keypoints],  # List of (19, 22, 3) arrays
                title=f"Follower (Aligned to Leader Space)",
                fps=20,
                radius=4
            )
            
            # Combined visualization: both dancers together (following vae_visualization_app.py)
            # Use plot_3d_motion with both keypoints as a list
            plot_3d_motion_interhuman(
                save_path=combined_video_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[leader_keypoints, follower_aligned_keypoints],  # List of (seq_len, 22, 3) arrays
                title=f"Together: {vid_id} (InterHuman)",
                fps=20,
                radius=6,
                figsize=(12, 12)
            )
            
            
            info = f"InterHuman Visualization\n"
            info += f"{'='*60}\n"
            info += f"Sample: {idx}\n"
            info += f"Video: {vid_id}\n"
            info += cache_warning
            info += f"\nSequence Info:\n"
            info += f"  Total frames: {seq_len}\n"
            info += f"  Number of tokens: {num_tokens}\n"
            info += f"  Expected tokens: ~{expected_tokens} (one per 20-frame window)\n"
            info += f"  Frames per window (after processing): 19\n"
            info += f"\nMotion Shapes:\n"
            info += f"  Leader Motion: {leader_motion_ih.shape} (canonicalized)\n"
            info += f"  Follower Motion: {follower_motion_ih.shape} (canonicalized)\n"
            info += f"  Follower Aligned: {follower_aligned_motion.shape} (in leader's space)\n"
            info += f"\nTokens ({num_tokens} tokens total):\n"
            info += f"  Leader: {leader_tokens_array.tolist()}\n"
            info += f"  Follower: {follower_tokens_array.tolist()}\n"
            info += f"  Relationship: {relationship_tokens_array.tolist()}\n"
            info += f"\nRelationship Transform (Frame 0):\n"
            info += f"  Angle (half): {np.degrees(angle_half):.2f}°\n"
            
            mesh_video_path = None
            if use_mesh:
                try:
                    mesh_video_path = os.path.join(self.temp_dir, f"interhuman_mesh_{idx}.mp4")
                    out = keypoints_to_mesh_video(
                        leader_keypoints,
                        follower_aligned_keypoints,
                        mesh_video_path,
                        fps=20,
                    )
                    if out is None:
                        mesh_video_path = None
                        info += "\nMesh: skipped or failed (check priorMDM body_models and pyrender)."
                    else:
                        info += "\nMesh: produced (2-person SMPL)."
                except Exception as mesh_err:
                    import traceback
                    mesh_video_path = None
                    info += f"\nMesh: failed — {mesh_err}\n{traceback.format_exc()}"
            
            return leader_video_path, follower_video_path, combined_video_path, mesh_video_path, info
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing InterHuman: {str(e)}\n{traceback.format_exc()}"
            return None, None, None, None, error_msg
    
    def visualize_reconstruction_from_tokens(
        self,
        idx: int,
        use_continuous_concatenation: bool = False,
        use_actual_relation: bool = False,
        use_canonical_seed: bool = False,
        smooth_boundaries: bool = False,
        smooth_boundary_half_kernel: int = 4,
        leader_tokens_override=None,
        follower_tokens_override=None,
        relationship_tokens_override=None,
        output_suffix: str = "",
        use_mesh: bool = False,
        human_study_call: bool = False,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[object], Optional[str], str]:
        """Visualize motion reconstructed from InterHuman and Relationship tokens.
        Optional overrides: pass list/array of tokens to use instead of sample's (e.g. LLM-predicted tokens).
        output_suffix: appended to temp filenames (e.g. '_pred', '_gt') so multiple calls don't overwrite.
        use_mesh: when True, also render a 2-person SMPL mesh video from reconstructed joints.
        human_study_call: when True, returns (leader_kp, follower_kp, info, motion1_262, motion2_262, None)
          for Human Study refinement; leader_kp/follower_kp (T, 22, 3), motion1/2 (T, 262).
        """
        if self.dataset is None:
            return None, None, None, None, None, "Error: No dataset loaded"
        
        if not INTERHUMAN_AVAILABLE:
            return None, None, None, None, None, "Error: InterHuman visualization dependencies not available"
        
        try:
            sample = self._get_sample_from_dataset(idx)
            interhuman_data = sample.get('interhuman_data')
            
            if interhuman_data is None:
                return None, None, None, None, None, "Error: InterHuman data not available in this sample (old cache format)"
            
            if self.interhuman_motion_tokenizer is None:
                self._load_interhuman_tokenizers()
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # Get original motions - these are now full sequences (e.g., 99 frames = 5 windows * 19 frames)
            # Ensure they're numpy arrays
            leader_motion_ih = np.array(interhuman_data['leader_motion_ih'])  # (num_windows*19, 262)
            follower_motion_ih = np.array(interhuman_data['follower_motion_ih'])  # (num_windows*19, 262)
            relationship_features = np.array(interhuman_data['relationship_features'])  # (num_windows*19, 4)
            seq_len = len(leader_motion_ih)
            
            # Extract root transforms (from first window), same as ground-truth visualization
            root_quat_L = np.array(interhuman_data['root_quat_init_L'], dtype=np.float32)
            root_pos_L = np.array(interhuman_data['root_pos_init_L'], dtype=np.float32)
            root_quat_F = np.array(interhuman_data['root_quat_init_F'], dtype=np.float32)
            root_pos_F = np.array(interhuman_data['root_pos_init_F'], dtype=np.float32)
            
            # Get tokens - use overrides if provided (e.g. LLM-predicted), else sample's tokens
            leader_tokens = np.array(leader_tokens_override if leader_tokens_override is not None else interhuman_data['leader_tokens']).ravel()
            follower_tokens = np.array(follower_tokens_override if follower_tokens_override is not None else interhuman_data['follower_tokens']).ravel()
            relationship_tokens = np.array(relationship_tokens_override if relationship_tokens_override is not None else interhuman_data['relationship_tokens']).ravel()
            num_tokens = len(leader_tokens)
            # Cap num_windows so we don't index past motion arrays (important when using LLM-predicted token overrides)
            window_size_after_processing = 19
            max_windows_from_motion = max(1, len(leader_motion_ih) // window_size_after_processing)
            num_windows = min(len(leader_tokens), len(follower_tokens), len(relationship_tokens), max_windows_from_motion)
            leader_tokens = leader_tokens[:num_windows]
            follower_tokens = follower_tokens[:num_windows]
            relationship_tokens = relationship_tokens[:num_windows]
            # Continuous concatenation is applied after reconstruction only.
            # window_size_after_processing already set above (19)
            if DEBUG:
                print("\n" + "="*80)
                print("DEBUG: RECONSTRUCTION FROM TOKENS")
                print("="*80)
                print(f"Sample index: {idx}")
                print(f"Number of windows: {num_windows}")
                print(f"Window size after processing: {window_size_after_processing}")
                print(f"\nOriginal Motion Shapes:")
                print(f"  leader_motion_ih: {leader_motion_ih.shape}")
                print(f"  follower_motion_ih: {follower_motion_ih.shape}")
                print(f"  relationship_features: {relationship_features.shape}")
                print(f"\nTokens:")
                print(f"  leader_tokens: {leader_tokens.tolist()} (shape: {leader_tokens.shape})")
                print(f"  follower_tokens: {follower_tokens.tolist()} (shape: {follower_tokens.shape})")
                print(f"  relationship_tokens: {relationship_tokens.tolist()} (shape: {relationship_tokens.shape})")
                print(f"\nOriginal Motion Sample Values (first 3 frames, first 10 dims):")
                print(f"  leader_motion_ih[0:3, :10]:\n{leader_motion_ih[0:3, :10]}")
                print(f"  follower_motion_ih[0:3, :10]:\n{follower_motion_ih[0:3, :10]}")
                print(f"  relationship_features[0:3]:\n{relationship_features[0:3]}")
            
            # Check for empty tokens
            if num_windows == 0:
                return None, None, None, None, None, f"Error: No tokens found in sample {idx}. Cache may be corrupted or empty."
            
            # Normalization stats (must match decoder output dims: InterHuman 262, relationship 4)
            epsilon = 1e-8
            expected_ih_dim = getattr(self.interhuman_motion_tokenizer, 'input_dim', 262)
            expected_rel_dim = getattr(self.relationship_tokenizer, 'input_dim', 4)
            if self.interhuman_normalization_stats is not None:
                mean_ih = torch.from_numpy(self.interhuman_normalization_stats['mean']).float()
                std_ih = torch.from_numpy(self.interhuman_normalization_stats['std']).float()
                if mean_ih.shape[0] != expected_ih_dim or std_ih.shape[0] != expected_ih_dim:
                    print(f"WARNING: InterHuman stats dim ({mean_ih.shape[0]}) != tokenizer input_dim ({expected_ih_dim}). Skipping denorm (poses would be wrong).")
                    mean_ih = None
                    std_ih = None
                if DEBUG and mean_ih is not None:
                    print(f"\nInterHuman Normalization Stats:")
                    print(f"  mean_ih shape: {mean_ih.shape}, sample (first 10): {mean_ih[:10].numpy()}")
                    print(f"  std_ih shape: {std_ih.shape}, sample (first 10): {std_ih[:10].numpy()}")
                    print(f"  epsilon: {epsilon}")
            else:
                mean_ih = None
                std_ih = None
                if DEBUG:
                    print("\nWARNING: No InterHuman normalization stats loaded!")
            
            if self.relationship_normalization_stats is not None:
                mean_rel = torch.from_numpy(self.relationship_normalization_stats['mean']).float()
                std_rel = torch.from_numpy(self.relationship_normalization_stats['std']).float()
                if mean_rel.shape[0] != expected_rel_dim or std_rel.shape[0] != expected_rel_dim:
                    print(f"WARNING: Relationship stats dim ({mean_rel.shape[0]}) != tokenizer input_dim ({expected_rel_dim}). Skipping denorm.")
                    mean_rel = None
                    std_rel = None
                if DEBUG and mean_rel is not None:
                    print(f"\nRelationship Normalization Stats:")
                    print(f"  mean_rel shape: {mean_rel.shape}, values: {mean_rel.numpy()}")
                    print(f"  std_rel shape: {std_rel.shape}, values: {std_rel.numpy()}")
            else:
                mean_rel = None
                std_rel = None
                if DEBUG:
                    print("\nWARNING: No Relationship normalization stats loaded!")
            
            # Helper: extract root pos and yaw (half-angle) from a frame
            def extract_root_pos_yaw(motion_frame):
                root_pos = motion_frame[:3]
                r_hip, l_hip = 2, 1
                positions = motion_frame[:22 * 3].reshape(22, 3)
                across = positions[r_hip] - positions[l_hip]
                across = across / (np.linalg.norm(across) + 1e-8)
                forward = np.cross(np.array([0, 1, 0]), across)
                forward = forward / (np.linalg.norm(forward) + 1e-8)
                target = np.array([0, 0, 1])
                root_quat = qbetween_np(forward.reshape(1, -1), target.reshape(1, -1))[0]
                yaw_half = np.arctan2(root_quat[2], root_quat[0])
                return root_pos, yaw_half, root_quat

            def canonicalize_frame(frame_1xD):
                """Move root to XZ=0, yaw=0 (canonical). Keeps root Y and body pose."""
                pos, yaw_half, _ = extract_root_pos_yaw(frame_1xD[0])
                inv_transform = np.array([-yaw_half, -pos[0], -pos[2]], dtype=np.float32)
                return rigid_transform(inv_transform, frame_1xD.copy())

            # Helper: compute relationship features from world-space leader/follower frames
            def compute_actual_relationship_from_world(leader_frame, follower_frame):
                # Root positions
                leader_root = leader_frame[:3]
                follower_root = follower_frame[:3]
                
                # Estimate yaw from hip-based forward direction (same as concatenate helper)
                r_hip, l_hip = 2, 1
                leader_pos = leader_frame[:22 * 3].reshape(22, 3)
                follower_pos = follower_frame[:22 * 3].reshape(22, 3)
                
                def frame_yaw_half(joint_pos):
                    across = joint_pos[r_hip] - joint_pos[l_hip]
                    across = across / (np.linalg.norm(across) + 1e-8)
                    forward = np.cross(np.array([0, 1, 0]), across)
                    forward = forward / (np.linalg.norm(forward) + 1e-8)
                    target = np.array([0, 0, 1])
                    root_quat = qbetween_np(forward.reshape(1, -1), target.reshape(1, -1))[0]
                    return np.arctan2(root_quat[2], root_quat[0])
                
                leader_yaw_half = frame_yaw_half(leader_pos)
                follower_yaw_half = frame_yaw_half(follower_pos)
                
                # Relative yaw (half-angle)
                rel_yaw_half = follower_yaw_half - leader_yaw_half
                rel_w = np.cos(rel_yaw_half)
                rel_z = np.sin(rel_yaw_half)
                
                # Relative position in leader's local frame
                delta = (follower_root - leader_root)[[0, 2]]
                leader_yaw = 2.0 * leader_yaw_half
                cos_y = np.cos(leader_yaw)
                sin_y = np.sin(leader_yaw)
                rel_x = delta[0] * cos_y + delta[1] * sin_y
                rel_z_pos = -delta[0] * sin_y + delta[1] * cos_y
                
                return np.array([rel_w, rel_z, rel_x, rel_z_pos], dtype=np.float32)
            
            # Reconstruct from tokens window-by-window, keep a list of decoded windows.
            with torch.no_grad():
                leader_recon_windows = []
                follower_recon_windows = []
                relationship_recon_windows = []

                leader_prev_last = None
                follower_prev_last = None
                rel_prev_last = None
                
                # Track world-space continuity for "actual relation" option
                world_origin_pos = None
                world_origin_yaw_half = None

                for w in range(num_windows):
                    window_start = w * window_size_after_processing

                    # Conditioning: first window uses GT first frame; subsequent windows use previous recon last frame.
                    if w == 0:
                        if window_start < len(leader_motion_ih):
                            leader_first = leader_motion_ih[window_start:window_start + 1]
                            follower_first = follower_motion_ih[window_start:window_start + 1]
                            rel_first = relationship_features[window_start:window_start + 1]
                        else:
                            leader_first = np.zeros((1, leader_motion_ih.shape[1]), dtype=np.float32)
                            follower_first = np.zeros((1, follower_motion_ih.shape[1]), dtype=np.float32)
                            rel_first = np.zeros((1, relationship_features.shape[1]), dtype=np.float32)
                    else:
                        leader_first = leader_prev_last
                        follower_first = follower_prev_last
                        rel_first = rel_prev_last
                    
                    if DEBUG and w > 0:
                        print(f"\n[DEBUG][REL] Window {w}: rel_first (denorm) = {rel_first[0]}")

                    if mean_ih is not None:
                        leader_first_norm = (torch.from_numpy(leader_first).float() - mean_ih) / (std_ih + epsilon)
                        follower_first_norm = (torch.from_numpy(follower_first).float() - mean_ih) / (std_ih + epsilon)
                    else:
                        leader_first_norm = torch.from_numpy(leader_first).float()
                        follower_first_norm = torch.from_numpy(follower_first).float()

                    if mean_rel is not None:
                        rel_first_norm = (torch.from_numpy(rel_first).float() - mean_rel) / (std_rel + epsilon)
                    else:
                        rel_first_norm = torch.from_numpy(rel_first).float()
                    
                    if DEBUG and w > 0:
                        print(f"[DEBUG][REL] Window {w}: rel_first_norm = {rel_first_norm[0].cpu().numpy()}")

                    leader_token_tensor = torch.tensor([leader_tokens[w]], dtype=torch.long).to(device)
                    follower_token_tensor = torch.tensor([follower_tokens[w]], dtype=torch.long).to(device)
                    rel_token_tensor = torch.tensor([relationship_tokens[w]], dtype=torch.long).to(device)

                    leader_latent = self.interhuman_motion_tokenizer.vq_layer.dequantize(leader_token_tensor)
                    follower_latent = self.interhuman_motion_tokenizer.vq_layer.dequantize(follower_token_tensor)
                    rel_latent = self.relationship_tokenizer.vq_layer.dequantize(rel_token_tensor)

                    leader_recon = self.interhuman_motion_tokenizer.decoder(leader_latent, leader_first_norm.to(device))[0]
                    follower_recon = self.interhuman_motion_tokenizer.decoder(follower_latent, follower_first_norm.to(device))[0]
                    rel_recon = self.relationship_tokenizer.decoder(rel_latent, rel_first_norm.to(device))[0]

                    actual_window_len = min(window_size_after_processing, len(leader_motion_ih) - window_start)
                    if actual_window_len <= 0:
                        actual_window_len = window_size_after_processing

                    leader_recon_np = leader_recon[:actual_window_len].cpu().numpy()
                    follower_recon_np = follower_recon[:actual_window_len].cpu().numpy()
                    rel_recon_np = rel_recon[:actual_window_len].cpu().numpy()

                    # Denormalize only if decoder output dim matches stats (avoids wrong poses from shape mismatch)
                    use_ih_denorm = mean_ih is not None and leader_recon_np.shape[1] == mean_ih.shape[0]
                    use_rel_denorm = mean_rel is not None and rel_recon_np.shape[1] == mean_rel.shape[0]
                    if mean_ih is not None and not use_ih_denorm and w == 0:
                        print(f"WARNING: Decoder output dim ({leader_recon_np.shape[1]}) != InterHuman stats dim ({mean_ih.shape[0]}). Skipping motion denorm.")
                    if mean_rel is not None and not use_rel_denorm and w == 0:
                        print(f"WARNING: Decoder output dim ({rel_recon_np.shape[1]}) != relationship stats dim ({mean_rel.shape[0]}). Skipping relationship denorm.")

                    if use_ih_denorm:
                        leader_recon_denorm_window = leader_recon_np * std_ih.numpy() + mean_ih.numpy()
                        follower_recon_denorm_window = follower_recon_np * std_ih.numpy() + mean_ih.numpy()
                    else:
                        leader_recon_denorm_window = leader_recon_np.copy()
                        follower_recon_denorm_window = follower_recon_np.copy()

                    if use_rel_denorm:
                        rel_recon_denorm_window = rel_recon_np * std_rel.numpy() + mean_rel.numpy()
                    else:
                        rel_recon_denorm_window = rel_recon_np.copy()
                    
                    if DEBUG and w > 0:
                        print(f"[DEBUG][REL] Window {w}: rel_recon_denorm_window[0] (pre-override) = {rel_recon_denorm_window[0]}")

                    # Enforce continuity for relationship when using actual relation input
                    if use_actual_relation and w > 0 and rel_first is not None:
                        rel_recon_denorm_window[0] = rel_first[0]
                        if DEBUG:
                            print(f"[DEBUG][REL] Window {w}: enforced frame0 = rel_first")

                    if DEBUG and w > 0:
                        diff_frame0 = rel_recon_denorm_window[0] - rel_first[0]
                        print(f"[DEBUG][REL] Window {w}: frame0 diff after enforce = {diff_frame0}")

                    leader_recon_windows.append(leader_recon_denorm_window)
                    follower_recon_windows.append(follower_recon_denorm_window)
                    relationship_recon_windows.append(rel_recon_denorm_window)

                    # Update conditioning for next window with last reconstructed frame (denormalized).
                    # If use_canonical_seed: canonicalize the frame (root → origin, yaw → 0) so the
                    # decoder receives a frame matching its training distribution, reducing backward-motion
                    # artifacts caused by non-canonical conditioning at window boundaries.
                    if use_canonical_seed:
                        leader_prev_last = canonicalize_frame(leader_recon_denorm_window[-1:])
                        follower_prev_last = canonicalize_frame(follower_recon_denorm_window[-1:])
                    else:
                        leader_prev_last = leader_recon_denorm_window[-1:]
                        follower_prev_last = follower_recon_denorm_window[-1:]
                    
                    # Optionally compute actual relation from world-space aligned motions
                    if use_actual_relation:
                        # Align this window into world space (same logic as concatenate)
                        curr_leader_first = leader_recon_denorm_window[0]
                        curr_leader_first_pos, curr_leader_first_yaw_half, _ = extract_root_pos_yaw(curr_leader_first)
                        
                        if w == 0:
                            # First window: leader is canonical, follower aligned by frame-0 relationship
                            rel_frame0 = rel_recon_denorm_window[0]
                            rel_angle_half = np.arctan2(rel_frame0[1], rel_frame0[0])
                            rel_transform = np.array([rel_angle_half, rel_frame0[2], rel_frame0[3]], dtype=np.float32)
                            
                            leader_world = leader_recon_denorm_window.copy()
                            follower_world = rigid_transform(rel_transform, follower_recon_denorm_window.copy())
                            
                            # Extrapolate world origin (same logic as concatenate_windows_with_continuity)
                            _pos_last, _yaw_last, _ = extract_root_pos_yaw(leader_world[-1])
                            _pos_prev, _yaw_prev, _ = extract_root_pos_yaw(leader_world[-2])
                            world_origin_pos = _pos_last + (_pos_last - _pos_prev)
                            world_origin_yaw_half = _yaw_last + (_yaw_last - _yaw_prev)
                        else:
                            window_pos_offset_xz = world_origin_pos[[0, 2]] - curr_leader_first_pos[[0, 2]]
                            window_yaw_offset_half = world_origin_yaw_half - curr_leader_first_yaw_half
                            window_transform = np.array(
                                [window_yaw_offset_half, window_pos_offset_xz[0], window_pos_offset_xz[1]],
                                dtype=np.float32
                            )
                            
                            leader_world = rigid_transform(window_transform, leader_recon_denorm_window.copy())
                            rel_frame0 = rel_recon_denorm_window[0]
                            rel_angle_half = np.arctan2(rel_frame0[1], rel_frame0[0])
                            rel_transform = np.array([rel_angle_half, rel_frame0[2], rel_frame0[3]], dtype=np.float32)
                            follower_aligned = rigid_transform(rel_transform, follower_recon_denorm_window.copy())
                            follower_world = rigid_transform(window_transform, follower_aligned.copy())
                            
                            # Extrapolate world origin (same logic as concatenate_windows_with_continuity)
                            _pos_last, _yaw_last, _ = extract_root_pos_yaw(leader_world[-1])
                            _pos_prev, _yaw_prev, _ = extract_root_pos_yaw(leader_world[-2])
                            world_origin_pos = _pos_last + (_pos_last - _pos_prev)
                            world_origin_yaw_half = _yaw_last + (_yaw_last - _yaw_prev)
                        
                        # Compute relationship from extrapolated positions so it is consistent
                        # with the extrapolated world origin used by concatenate_windows_with_continuity.
                        _leader_extrap = leader_world[-1] + (leader_world[-1] - leader_world[-2])
                        _follower_extrap = follower_world[-1] + (follower_world[-1] - follower_world[-2])
                        actual_rel_last = compute_actual_relationship_from_world(
                            _leader_extrap, _follower_extrap
                        )
                        rel_prev_last = actual_rel_last[None, :]
                        if DEBUG:
                            print(f"[DEBUG][REL] Window {w}: actual_rel_last (denorm) = {actual_rel_last}")
                    else:
                        rel_prev_last = rel_recon_denorm_window[-1:]
                        if DEBUG and w > 0:
                            print(f"[DEBUG][REL] Window {w}: rel_prev_last (denorm, from recon) = {rel_prev_last[0]}")

            # Concatenate denormalized windows for downstream processing/visualization
            leader_recon_denorm = np.concatenate(leader_recon_windows, axis=0)
            follower_recon_denorm = np.concatenate(follower_recon_windows, axis=0)
            rel_recon_denorm = np.concatenate(relationship_recon_windows, axis=0)
            
            # Apply continuous concatenation to reconstructed outputs if requested
            if use_continuous_concatenation and num_tokens > 1:
                leader_recon_denorm, follower_recon_denorm, rel_recon_denorm = concatenate_windows_with_continuity(
                    leader_recon_windows, follower_recon_windows, relationship_recon_windows
                )
            
            # Gaussian smoothing around each window boundary to reduce visible jumps.
            # Boundaries are at every window_size_after_processing frames; with continuous
            # concatenation an extra interpolated frame shifts each boundary by one.
            if smooth_boundaries and num_windows > 1:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    hk = max(1, int(smooth_boundary_half_kernel))
                    sigma = hk / 2.0
                    stride = window_size_after_processing
                    if use_continuous_concatenation:
                        # Each window contributes stride frames + 1 interpolated = stride+1 frames total
                        # (except window 0 which contributes just stride frames)
                        # Seam (interpolated) frame positions: stride, stride+(stride+1), ...
                        boundary_frames = [stride + (stride + 1) * k for k in range(num_windows - 1)]
                    else:
                        boundary_frames = [stride * k for k in range(1, num_windows)]
                    T = len(leader_recon_denorm)
                    for bidx in boundary_frames:
                        s = max(0, bidx - hk)
                        e = min(T, bidx + hk + 1)
                        if e > s:
                            leader_recon_denorm[s:e] = gaussian_filter1d(leader_recon_denorm[s:e], sigma=sigma, axis=0)
                            follower_recon_denorm[s:e] = gaussian_filter1d(follower_recon_denorm[s:e], sigma=sigma, axis=0)
                            rel_recon_denorm[s:e] = gaussian_filter1d(rel_recon_denorm[s:e], sigma=sigma, axis=0)
                except ImportError:
                    print("scipy not available; boundary smoothing skipped.")
            
            if DEBUG:
                for boundary_idx in range(1, num_windows):
                    boundary_frame_idx = boundary_idx * window_size_after_processing
                    if boundary_frame_idx < len(rel_recon_denorm):
                        prev_last = rel_recon_denorm[boundary_frame_idx - 1]
                        curr_first = rel_recon_denorm[boundary_frame_idx]
                        diff = curr_first - prev_last
                        print(f"[DEBUG][REL] Boundary {boundary_idx}: prev_last={prev_last}, curr_first={curr_first}, diff={diff}")

            # If requested, compute actual relationship sequence from reconstructed motions (world space)
            rel_recon_for_plot = rel_recon_denorm
            if use_actual_relation:
                rel_actual = np.zeros_like(rel_recon_denorm)
                for t in range(len(rel_recon_denorm)):
                    rel_actual[t] = compute_actual_relationship_from_world(
                        leader_recon_denorm[t], follower_recon_denorm[t]
                    )
                rel_recon_for_plot = rel_actual
                if DEBUG:
                    for boundary_idx in range(1, num_windows):
                        boundary_frame_idx = boundary_idx * window_size_after_processing
                        if boundary_frame_idx < len(rel_actual):
                            prev_last = rel_actual[boundary_frame_idx - 1]
                            curr_first = rel_actual[boundary_frame_idx]
                            diff = curr_first - prev_last
                            print(f"[DEBUG][REL] Actual Boundary {boundary_idx}: prev_last={prev_last}, curr_first={curr_first}, diff={diff}")
            
            if DEBUG:
                print(f"\n{'='*80}")
                print("DEBUG: FINAL RECONSTRUCTION SUMMARY")
                print(f"{'='*80}")
                print(f"Final shapes:")
                print(f"  leader_recon_denorm: {leader_recon_denorm.shape}")
                print(f"  follower_recon_denorm: {follower_recon_denorm.shape}")
                print(f"  rel_recon_denorm: {rel_recon_denorm.shape}")
                print(f"\nOriginal shapes:")
                print(f"  leader_motion_ih: {leader_motion_ih.shape}")
                print(f"  follower_motion_ih: {follower_motion_ih.shape}")
                print(f"  relationship_features: {relationship_features.shape}")
                
                # Compute MSE for comparison
                min_len = min(len(leader_motion_ih), len(leader_recon_denorm))
                leader_mse = np.mean((leader_motion_ih[:min_len] - leader_recon_denorm[:min_len]) ** 2)
                follower_mse = np.mean((follower_motion_ih[:min_len] - follower_recon_denorm[:min_len]) ** 2)
                rel_mse = np.mean((relationship_features[:min_len] - rel_recon_denorm[:min_len]) ** 2)
                
                # Per-dimension MSE for first few dimensions
                leader_per_dim_mse = np.mean((leader_motion_ih[:min_len] - leader_recon_denorm[:min_len]) ** 2, axis=0)
                print(f"\nReconstruction Errors (MSE):")
                print(f"  Leader MSE: {leader_mse:.6f}")
                print(f"  Follower MSE: {follower_mse:.6f}")
                print(f"  Relationship MSE: {rel_mse:.6f}")
                print(f"\nLeader Per-Dimension MSE (first 10 dims):")
                print(f"  {leader_per_dim_mse[:10]}")
                print(f"  Max per-dim MSE: {leader_per_dim_mse.max():.6f} (dim {np.argmax(leader_per_dim_mse)})")
                print(f"  Min per-dim MSE: {leader_per_dim_mse.min():.6f} (dim {np.argmin(leader_per_dim_mse)})")
                print(f"{'='*80}\n")
            
            # Extract keypoints from reconstructed motions (full sequence)
            n_joints = 22
            leader_recon_keypoints = leader_recon_denorm[:, :n_joints*3].reshape(-1, n_joints, 3)  # (num_windows*19, 22, 3)
            follower_recon_keypoints = follower_recon_denorm[:, :n_joints*3].reshape(-1, n_joints, 3)  # (num_windows*19, 22, 3)
            
            if DEBUG:
                print(f"Extracted keypoints shapes:")
                print(f"  leader_recon_keypoints: {leader_recon_keypoints.shape}")
                print(f"  follower_recon_keypoints: {follower_recon_keypoints.shape}")
                print(f"  leader_recon_keypoints[0, 0] (first joint, first frame): {leader_recon_keypoints[0, 0]}")
                print(f"  Original leader keypoints[0, 0] (for comparison): {leader_motion_ih[0, :3]}")
            
            # Align reconstructed follower for visualization.
            # If continuous concatenation was applied, follower is already aligned per-window.
            rel_recon_w = rel_recon_denorm[0, 0]
            rel_recon_z = rel_recon_denorm[0, 1]
            rel_recon_x = rel_recon_denorm[0, 2]
            rel_recon_z_pos = rel_recon_denorm[0, 3]
            angle_half_recon = np.arctan2(rel_recon_z, rel_recon_w)
            relative_transform_recon = np.array([angle_half_recon, rel_recon_x, rel_recon_z_pos])

            if use_continuous_concatenation and num_tokens > 1:
                follower_recon_aligned_motion = follower_recon_denorm
            else:
                follower_recon_aligned_motion = rigid_transform(relative_transform_recon, follower_recon_denorm.copy())

            follower_recon_aligned_keypoints = follower_recon_aligned_motion[:, :n_joints*3].reshape(-1, n_joints, 3)
            
            if human_study_call:
                info_short = f"Reconstructed {leader_recon_keypoints.shape[0]} frames (idx={idx}, {num_windows} windows)."
                return (np.asarray(leader_recon_keypoints, dtype=np.float32),
                        np.asarray(follower_recon_aligned_keypoints, dtype=np.float32),
                        info_short,
                        np.asarray(leader_recon_denorm, dtype=np.float32),
                        np.asarray(follower_recon_aligned_motion, dtype=np.float32),
                        None)
            
            vid_id = sample.get('aux_info', {}).get('vid', f'sample_{idx}')
            
            # Create videos (output_suffix avoids overwriting when e.g. LLM-Inference calls pred then GT)
            recon_leader_path = os.path.join(self.temp_dir, f"recon_leader_{idx}{output_suffix}.mp4")
            recon_follower_path = os.path.join(self.temp_dir, f"recon_follower_{idx}{output_suffix}.mp4")
            recon_combined_path = os.path.join(self.temp_dir, f"recon_combined_{idx}{output_suffix}.mp4")
            
            # Visualize reconstructed leader
            # plot_3d_motion_interhuman expects mp_joints: list of (seq_len, 22, 3) keypoint arrays
            leader_recon_keypoints_array = np.asarray(leader_recon_keypoints, dtype=np.float32)
            if leader_recon_keypoints_array.ndim != 3 or leader_recon_keypoints_array.shape[1] != 22 or leader_recon_keypoints_array.shape[2] != 3:
                return None, None, None, None, None, f"Error: Reconstructed leader keypoints have wrong shape: {leader_recon_keypoints_array.shape}, expected (seq_len, 22, 3)"
            
            plot_3d_motion_interhuman(
                save_path=recon_leader_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[leader_recon_keypoints_array],  # List of (num_windows*19, 22, 3) arrays
                title=f"Leader Reconstructed ({num_windows} tokens: {leader_tokens.tolist()})",
                fps=20,
                radius=4
            )
            
            # Visualize reconstructed follower (aligned)
            follower_recon_aligned_keypoints_array = np.asarray(follower_recon_aligned_keypoints, dtype=np.float32)
            if follower_recon_aligned_keypoints_array.ndim != 3 or follower_recon_aligned_keypoints_array.shape[1] != 22 or follower_recon_aligned_keypoints_array.shape[2] != 3:
                return None, None, None, None, None, f"Error: Reconstructed follower keypoints have wrong shape: {follower_recon_aligned_keypoints_array.shape}, expected (seq_len, 22, 3)"
            
            plot_3d_motion_interhuman(
                save_path=recon_follower_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[follower_recon_aligned_keypoints_array],  # List of (num_windows*19, 22, 3) arrays
                title=f"Follower Reconstructed ({num_windows} tokens: {follower_tokens.tolist()})",
                fps=20,
                radius=4
            )
            
            # Combined reconstructed visualization (following vae_visualization_app.py)
            # Use plot_3d_motion with both keypoints as a list
            plot_3d_motion_interhuman(
                save_path=recon_combined_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[leader_recon_keypoints, follower_recon_aligned_keypoints],  # List of (seq_len, 22, 3) arrays
                title=f"Reconstructed Together: {vid_id}",
                fps=20,
                radius=6,
                figsize=(12, 12)
            )
            
            # Ensure shapes match for error computation
            min_len = min(len(leader_motion_ih), len(leader_recon_denorm))
            leader_motion_ih_trimmed = leader_motion_ih[:min_len]
            leader_recon_denorm_trimmed = leader_recon_denorm[:min_len]
            follower_motion_ih_trimmed = follower_motion_ih[:min_len]
            follower_recon_denorm_trimmed = follower_recon_denorm[:min_len]
            relationship_features_trimmed = relationship_features[:min_len]
            rel_recon_denorm_trimmed = rel_recon_denorm[:min_len]
            
            # Compute reconstruction error
            leader_error = np.mean((leader_motion_ih_trimmed - leader_recon_denorm_trimmed) ** 2)
            follower_error = np.mean((follower_motion_ih_trimmed - follower_recon_denorm_trimmed) ** 2)
            rel_error = np.mean((relationship_features_trimmed - rel_recon_denorm_trimmed) ** 2)
            
            # Use first window's relationship for alignment
            rel_recon_w = rel_recon_denorm[0, 0]
            rel_recon_z = rel_recon_denorm[0, 1]
            rel_recon_x = rel_recon_denorm[0, 2]
            rel_recon_z_pos = rel_recon_denorm[0, 3]
            angle_half_recon = np.arctan2(rel_recon_z, rel_recon_w)
            
            # Build relationship comparison chart (GT vs reconstructed)
            relationship_fig = None
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # Use full sequences (ground truth and reconstructed)
                gt_rel = relationship_features[:len(rel_recon_denorm)]
                recon_rel = rel_recon_for_plot

                # Convert [w, z, x, z] -> [yaw_rad, x, z]
                gt_yaw = np.arctan2(gt_rel[:, 1], gt_rel[:, 0])
                recon_yaw = np.arctan2(recon_rel[:, 1], recon_rel[:, 0])
                gt_viz = np.stack([gt_yaw, gt_rel[:, 2], gt_rel[:, 3]], axis=1)
                recon_viz = np.stack([recon_yaw, recon_rel[:, 2], recon_rel[:, 3]], axis=1)

                dim_names = [
                    "Relative Yaw (radians)",
                    "Relative X Position",
                    "Relative Z Position"
                ]

                seq_len_rel = gt_viz.shape[0]
                time_axis = np.arange(seq_len_rel)

                relationship_fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=dim_names,
                    horizontal_spacing=0.12
                )

                for dim in range(3):
                    relationship_fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=gt_viz[:, dim],
                            mode='lines',
                            name='Ground Truth',
                            line=dict(color='blue', width=2),
                            showlegend=(dim == 0)
                        ),
                        row=1, col=dim + 1
                    )
                    relationship_fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=recon_viz[:, dim],
                            mode='lines',
                            name='Reconstructed',
                            line=dict(color='red', width=2, dash='dash'),
                            showlegend=(dim == 0)
                        ),
                        row=1, col=dim + 1
                    )
                    relationship_fig.update_xaxes(title_text="Frame", row=1, col=dim + 1)
                    relationship_fig.update_yaxes(title_text="Value", row=1, col=dim + 1)

                relationship_fig.update_layout(
                    title="Relationship Features: Ground Truth vs Reconstructed",
                    height=350,
                    template="plotly_white",
                    margin=dict(l=30, r=30, t=50, b=30)
                )
            except Exception as plot_err:
                if DEBUG:
                    print(f"Warning: Failed to create relationship plot: {plot_err}")

            info = f"Reconstruction from Tokens\n"
            info += f"{'='*60}\n"
            info += f"Sample: {idx}\n"
            info += f"Video: {vid_id}\n"
            info += f"\nSequence Info:\n"
            info += f"  Original frames: {len(leader_motion_ih)}\n"
            info += f"  Number of windows: {num_windows}\n"
            info += f"  Frames per window (after processing): {window_size_after_processing}\n"
            info += f"\nLong Sequence Generation (Short):\n"
            info += f"  Decode tokens window-by-window with autoregressive conditioning\n"
            info += f"  (first frame = GT for window 0, then last recon frame for next)\n"
            info += f"  Concatenate windows, optionally apply continuous alignment.\n"
            info += f"\nTokens Used ({num_windows} tokens total):\n"
            info += f"  Leader: {leader_tokens.tolist()}\n"
            info += f"  Follower: {follower_tokens.tolist()}\n"
            info += f"  Relationship: {relationship_tokens.tolist()}\n"
            info += f"\nReconstruction Errors (MSE):\n"
            info += f"  Leader Motion: {leader_error:.6f}\n"
            info += f"  Follower Motion: {follower_error:.6f}\n"
            info += f"  Relationship: {rel_error:.6f}\n"
            info += f"\nReconstructed Relationship Transform (Frame 0):\n"
            info += f"  Angle (half): {np.degrees(angle_half_recon):.2f}°\n"
            
            recon_mesh_path = None
            if use_mesh:
                try:
                    recon_mesh_path = os.path.join(self.temp_dir, f"recon_mesh_{idx}{output_suffix}.mp4")
                    out = keypoints_to_mesh_video(
                        leader_recon_keypoints_array,
                        follower_recon_aligned_keypoints_array,
                        recon_mesh_path,
                        fps=20,
                    )
                    if out is None:
                        recon_mesh_path = None
                        info += "\nMesh: skipped or failed (check priorMDM body_models and pyrender)."
                    else:
                        info += "\nMesh: produced (2-person SMPL)."
                except Exception as mesh_err:
                    import traceback
                    recon_mesh_path = None
                    info += f"\nMesh: failed — {mesh_err}\n{traceback.format_exc()}"
            
            return recon_leader_path, recon_follower_path, recon_combined_path, relationship_fig, recon_mesh_path, info
            
        except Exception as e:
            import traceback
            return None, None, None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def get_reconstructed_keypoints(
        self,
        idx: int,
        use_continuous_concatenation: bool = True,
        use_actual_relation: bool = True,
        leader_tokens_override=None,
        follower_tokens_override=None,
        relationship_tokens_override=None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Get (leader_kp, follower_kp) as (T, 22, 3) numpy arrays from token reconstruction.
        Optional overrides for LLM-predicted tokens. Returns (leader_kp, follower_kp, info_str);
        on error returns (None, None, error_msg).
        """
        result = self.visualize_reconstruction_from_tokens(
            idx,
            use_continuous_concatenation=use_continuous_concatenation,
            use_actual_relation=use_actual_relation,
            leader_tokens_override=leader_tokens_override,
            follower_tokens_override=follower_tokens_override,
            relationship_tokens_override=relationship_tokens_override,
            output_suffix="_kp",
            use_mesh=False,
            human_study_call=True,
        )
        # result is (leader_kp, follower_kp, info, leader_262, follower_262, None) or (..., error_str) on error
        if len(result) >= 6 and result[0] is not None and hasattr(result[0], "shape"):
            return result[0], result[1], result[2] or ""
        if len(result) >= 6 and isinstance(result[5], str) and result[5]:
            return None, None, result[5]
        return None, None, "Unknown error from reconstruction"

    def get_actual_gt_keypoints(
        self,
        idx: int,
        use_continuous_concatenation: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Get actual GT keypoints from InterHuman data cache (no VQ-VAE reconstruction).
        Returns (leader_kp, follower_kp, info) with (T, 22, 3) arrays; on error (None, None, error_msg)."""
        if not INTERHUMAN_AVAILABLE:
            return None, None, "InterHuman visualization dependencies not available"
        if self.dataset is None:
            return None, None, "No dataset loaded"
        try:
            sample = self._get_sample_from_dataset(idx)
            interhuman_data = sample.get('interhuman_data')
            if interhuman_data is None:
                return None, None, "No InterHuman data in sample"
            leader_motion_ih = np.array(interhuman_data['leader_motion_ih'], dtype=np.float32)
            follower_motion_ih = np.array(interhuman_data['follower_motion_ih'], dtype=np.float32)
            relationship_features = np.array(interhuman_data['relationship_features'], dtype=np.float32)
            leader_tokens_array = np.array(interhuman_data['leader_tokens'])
            num_tokens = len(leader_tokens_array)
            seq_len = len(leader_motion_ih)
            root_quat_L = np.array(interhuman_data['root_quat_init_L'], dtype=np.float32)
            root_pos_L = np.array(interhuman_data['root_pos_init_L'], dtype=np.float32)
            root_quat_F = np.array(interhuman_data['root_quat_init_F'], dtype=np.float32)
            root_pos_F = np.array(interhuman_data['root_pos_init_F'], dtype=np.float32)
            if use_continuous_concatenation and num_tokens > 1:
                window_size_after_processing = 19
                leader_windows, follower_windows, relationship_windows = [], [], []
                root_quat_inits_L, root_pos_inits_L = [], []
                root_quat_inits_F, root_pos_inits_F = [], []
                for w in range(num_tokens):
                    window_start = w * window_size_after_processing
                    window_end = min(window_start + window_size_after_processing, seq_len)
                    leader_windows.append(leader_motion_ih[window_start:window_end])
                    follower_windows.append(follower_motion_ih[window_start:window_end])
                    relationship_windows.append(relationship_features[window_start:window_end])
                    if w == 0:
                        root_quat_inits_L.append(root_quat_L)
                        root_pos_inits_L.append(root_pos_L)
                        root_quat_inits_F.append(root_quat_F)
                        root_pos_inits_F.append(root_pos_F)
                    else:
                        root_quat_inits_L.append(np.array([1, 0, 0, 0], dtype=np.float32))
                        root_pos_inits_L.append(np.zeros(3, dtype=np.float32))
                        root_quat_inits_F.append(np.array([1, 0, 0, 0], dtype=np.float32))
                        root_pos_inits_F.append(np.zeros(3, dtype=np.float32))
                leader_motion_ih, follower_motion_ih, relationship_features = concatenate_windows_with_continuity(
                    leader_windows, follower_windows, relationship_windows,
                    root_quat_inits_L, root_pos_inits_L, root_quat_inits_F, root_pos_inits_F,
                )
            n_joints = 22
            leader_motion_ih_array = np.asarray(leader_motion_ih, dtype=np.float32)
            follower_motion_ih_array = np.asarray(follower_motion_ih, dtype=np.float32)
            if leader_motion_ih_array.ndim != 2 or leader_motion_ih_array.shape[1] < n_joints * 3:
                return None, None, f"Leader motion wrong shape: {leader_motion_ih_array.shape}"
            if follower_motion_ih_array.ndim != 2 or follower_motion_ih_array.shape[1] < n_joints * 3:
                return None, None, f"Follower motion wrong shape: {follower_motion_ih_array.shape}"
            leader_keypoints = leader_motion_ih_array[:, : n_joints * 3].reshape(-1, n_joints, 3).astype(np.float32)
            relationship_features_array = np.asarray(relationship_features, dtype=np.float32)
            rel_w = float(relationship_features_array[0, 0])
            rel_z = float(relationship_features_array[0, 1])
            rel_x = float(relationship_features_array[0, 2])
            angle_half = np.arctan2(rel_z, rel_w)
            relative_transform = np.array([angle_half, rel_x, relationship_features_array[0, 3]], dtype=np.float32)
            if use_continuous_concatenation and num_tokens > 1:
                follower_aligned_motion = follower_motion_ih_array
            else:
                follower_aligned_motion = rigid_transform(relative_transform, follower_motion_ih_array.copy())
            follower_aligned_keypoints = follower_aligned_motion[:, : n_joints * 3].reshape(-1, n_joints, 3).astype(np.float32)
            vid_id = sample.get('aux_info', {}).get('vid', f'sample_{idx}')
            return leader_keypoints, follower_aligned_keypoints, vid_id
        except Exception as e:
            import traceback
            return None, None, f"{e}\n{traceback.format_exc()}"

    def get_reconstructed_keypoints_and_motion(
        self,
        idx: int,
        use_continuous_concatenation: bool = True,
        use_actual_relation: bool = True,
        leader_tokens_override=None,
        follower_tokens_override=None,
        relationship_tokens_override=None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
        """Get (leader_kp, follower_kp, leader_motion_262, follower_motion_262, info).
        Same as get_reconstructed_keypoints but also returns InterHuman 262-d motion for refinement."""
        result = self.visualize_reconstruction_from_tokens(
            idx,
            use_continuous_concatenation=use_continuous_concatenation,
            use_actual_relation=use_actual_relation,
            leader_tokens_override=leader_tokens_override,
            follower_tokens_override=follower_tokens_override,
            relationship_tokens_override=relationship_tokens_override,
            output_suffix="_kp",
            use_mesh=False,
            human_study_call=True,
        )
        if len(result) >= 6 and result[0] is not None and result[3] is not None and result[4] is not None:
            return result[0], result[1], result[3], result[4], result[2] or ""
        if len(result) >= 6 and isinstance(result[5], str) and result[5]:
            return None, None, None, None, result[5]
        return None, None, None, None, "Unknown error from reconstruction"
    
    def visualize_legacy_format(self, idx: int, show_leader: bool, show_follower: bool, show_combined: bool) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        """Visualize legacy HumanML3D format (for comparison)."""
        if self.dataset is None:
            return None, None, None, "Error: No dataset loaded"
        
        try:
            sample = self._get_sample_from_dataset(idx)
            aux_info = sample.get('aux_info', {})
            vid_id = aux_info.get('vid', f'sample_{idx}')
            
            leader_video = None
            follower_video = None
            combined_video = None
            info_lines = []
            
            if show_leader and sample.get('poses_keypoints3d_L') is not None:
                leader_keypoints = sample['poses_keypoints3d_L']
                if isinstance(leader_keypoints, torch.Tensor):
                    leader_keypoints = leader_keypoints.cpu().numpy()
                leader_video_path = os.path.join(self.temp_dir, f"legacy_leader_{idx}.mp4")
                render_skeleton_from_keypoints(
                    leader_keypoints,
                    leader_video_path,
                    title=f"Leader (Legacy HumanML3D): {vid_id}",
                    fps=20,
                    radius=4,
                    figsize=(6, 6),
                    dpi=100
                )
                if os.path.exists(leader_video_path):
                    leader_video = leader_video_path
                    info_lines.append(f"Leader: {leader_keypoints.shape[0]} frames, {leader_keypoints.shape[1]} joints")
            
            # Extract keypoints from reconstructed motions (full sequence)
            n_joints = 22
            leader_recon_keypoints = leader_recon_denorm[:, :n_joints*3].reshape(-1, n_joints, 3)  # (num_windows*19, 22, 3)
            follower_recon_keypoints = follower_recon_denorm[:, :n_joints*3].reshape(-1, n_joints, 3)  # (num_windows*19, 22, 3)
            
            if DEBUG:
                print(f"Extracted keypoints shapes:")
                print(f"  leader_recon_keypoints: {leader_recon_keypoints.shape}")
                print(f"  follower_recon_keypoints: {follower_recon_keypoints.shape}")
                print(f"  leader_recon_keypoints[0, 0] (first joint, first frame): {leader_recon_keypoints[0, 0]}")
                print(f"  Original leader keypoints[0, 0] (for comparison): {leader_motion_ih[0, :3]}")
            
            # Apply rigid_transform to reconstructed follower using reconstructed relationship (frame 0)
            rel_recon_w = rel_recon_denorm[0, 0]
            rel_recon_z = rel_recon_denorm[0, 1]
            rel_recon_x = rel_recon_denorm[0, 2]
            rel_recon_z_pos = rel_recon_denorm[0, 3]
            angle_half_recon = np.arctan2(rel_recon_z, rel_recon_w)
            relative_transform_recon = np.array([angle_half_recon, rel_recon_x, rel_recon_z_pos])
            follower_recon_aligned_motion = rigid_transform(relative_transform_recon, follower_recon_denorm.copy())
            follower_recon_aligned_keypoints = follower_recon_aligned_motion[:, :n_joints*3].reshape(-1, n_joints, 3)
            
            if human_study_call:
                info_short = f"Reconstructed {leader_recon_keypoints.shape[0]} frames (idx={idx}, {num_windows} windows)."
                return (np.asarray(leader_recon_keypoints, dtype=np.float32),
                        np.asarray(follower_recon_aligned_keypoints, dtype=np.float32),
                        info_short,
                        np.asarray(leader_recon_denorm, dtype=np.float32),
                        np.asarray(follower_recon_aligned_motion, dtype=np.float32),
                        None)
            
            vid_id = sample.get('aux_info', {}).get('vid', f'sample_{idx}')
            
            # Create videos (output_suffix avoids overwriting when e.g. LLM-Inference calls pred then GT)
            recon_leader_path = os.path.join(self.temp_dir, f"recon_leader_{idx}{output_suffix}.mp4")
            recon_follower_path = os.path.join(self.temp_dir, f"recon_follower_{idx}{output_suffix}.mp4")
            recon_combined_path = os.path.join(self.temp_dir, f"recon_combined_{idx}{output_suffix}.mp4")
            
            # Visualize reconstructed leader
            # plot_3d_motion_interhuman expects mp_joints: list of (seq_len, 22, 3) keypoint arrays
            leader_recon_keypoints_array = np.asarray(leader_recon_keypoints, dtype=np.float32)
            if leader_recon_keypoints_array.ndim != 3 or leader_recon_keypoints_array.shape[1] != 22 or leader_recon_keypoints_array.shape[2] != 3:
                return None, None, None, f"Error: Reconstructed leader keypoints have wrong shape: {leader_recon_keypoints_array.shape}, expected (seq_len, 22, 3)"
            
            plot_3d_motion_interhuman(
                save_path=recon_leader_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[leader_recon_keypoints_array],  # List of (num_windows*19, 22, 3) arrays
                title=f"Leader Reconstructed ({num_windows} tokens: {leader_tokens.tolist()})",
                fps=20,
                radius=4
            )
            
            # Visualize reconstructed follower (aligned)
            follower_recon_aligned_keypoints_array = np.asarray(follower_recon_aligned_keypoints, dtype=np.float32)
            if follower_recon_aligned_keypoints_array.ndim != 3 or follower_recon_aligned_keypoints_array.shape[1] != 22 or follower_recon_aligned_keypoints_array.shape[2] != 3:
                return None, None, None, f"Error: Reconstructed follower keypoints have wrong shape: {follower_recon_aligned_keypoints_array.shape}, expected (seq_len, 22, 3)"
            
            plot_3d_motion_interhuman(
                save_path=recon_follower_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[follower_recon_aligned_keypoints_array],  # List of (num_windows*19, 22, 3) arrays
                title=f"Follower Reconstructed ({num_windows} tokens: {follower_tokens.tolist()})",
                fps=20,
                radius=4
            )
            
            # Combined reconstructed visualization (following vae_visualization_app.py)
            # Use plot_3d_motion with both keypoints as a list
            plot_3d_motion_interhuman(
                save_path=recon_combined_path,
                kinematic_tree=HML_KINEMATIC_CHAIN,
                mp_joints=[leader_recon_keypoints, follower_recon_aligned_keypoints],  # List of (seq_len, 22, 3) arrays
                title=f"Reconstructed Together: {vid_id}",
                fps=20,
                radius=6,
                figsize=(12, 12)
            )
            
            # Ensure shapes match for error computation
            min_len = min(len(leader_motion_ih), len(leader_recon_denorm))
            leader_motion_ih_trimmed = leader_motion_ih[:min_len]
            leader_recon_denorm_trimmed = leader_recon_denorm[:min_len]
            follower_motion_ih_trimmed = follower_motion_ih[:min_len]
            follower_recon_denorm_trimmed = follower_recon_denorm[:min_len]
            relationship_features_trimmed = relationship_features[:min_len]
            rel_recon_denorm_trimmed = rel_recon_denorm[:min_len]
            
            # Compute reconstruction error
            leader_error = np.mean((leader_motion_ih_trimmed - leader_recon_denorm_trimmed) ** 2)
            follower_error = np.mean((follower_motion_ih_trimmed - follower_recon_denorm_trimmed) ** 2)
            rel_error = np.mean((relationship_features_trimmed - rel_recon_denorm_trimmed) ** 2)
            
            # Use first window's relationship for alignment
            rel_recon_w = rel_recon_denorm[0, 0]
            rel_recon_z = rel_recon_denorm[0, 1]
            rel_recon_x = rel_recon_denorm[0, 2]
            rel_recon_z_pos = rel_recon_denorm[0, 3]
            angle_half_recon = np.arctan2(rel_recon_z, rel_recon_w)
            
            info = f"Reconstruction from Tokens\n"
            info += f"{'='*60}\n"
            info += f"Sample: {idx}\n"
            info += f"Video: {vid_id}\n"
            info += f"\nSequence Info:\n"
            info += f"  Original frames: {len(leader_motion_ih)}\n"
            info += f"  Number of windows: {num_windows}\n"
            info += f"  Frames per window (after processing): {window_size_after_processing}\n"
            info += f"\nTokens Used ({num_windows} tokens total):\n"
            info += f"  Leader: {leader_tokens.tolist()}\n"
            info += f"  Follower: {follower_tokens.tolist()}\n"
            info += f"  Relationship: {relationship_tokens.tolist()}\n"
            info += f"\nReconstruction Errors (MSE):\n"
            info += f"  Leader Motion: {leader_error:.6f}\n"
            info += f"  Follower Motion: {follower_error:.6f}\n"
            info += f"  Relationship: {rel_error:.6f}\n"
            info += f"\nReconstructed Relationship Transform (Frame 0):\n"
            info += f"  Angle (half): {np.degrees(angle_half_recon):.2f}°\n"
            info += f"  Position: x={rel_recon_x:.3f}, z={rel_recon_z_pos:.3f}\n"
            
            return recon_leader_path, recon_follower_path, recon_combined_path, info
            
        except Exception as e:
            import traceback
            return None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def visualize_legacy_format(self, idx: int, show_leader: bool, show_follower: bool, show_combined: bool) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        """Visualize legacy HumanML3D format (for comparison)."""
        if self.dataset is None:
            return None, None, None, "Error: No dataset loaded"
        
        try:
            sample = self._get_sample_from_dataset(idx)
            aux_info = sample.get('aux_info', {})
            vid_id = aux_info.get('vid', f'sample_{idx}')
            
            leader_video = None
            follower_video = None
            combined_video = None
            info_lines = []
            
            if show_leader and sample.get('poses_keypoints3d_L') is not None:
                leader_keypoints = sample['poses_keypoints3d_L']
                if isinstance(leader_keypoints, torch.Tensor):
                    leader_keypoints = leader_keypoints.cpu().numpy()
                leader_video_path = os.path.join(self.temp_dir, f"legacy_leader_{idx}.mp4")
                render_skeleton_from_keypoints(
                    leader_keypoints,
                    leader_video_path,
                    title=f"Leader (Legacy HumanML3D): {vid_id}",
                    fps=20,
                    radius=4,
                    figsize=(6, 6),
                    dpi=100
                )
                if os.path.exists(leader_video_path):
                    leader_video = leader_video_path
                    info_lines.append(f"Leader: {leader_keypoints.shape[0]} frames, {leader_keypoints.shape[1]} joints")
            
            if show_follower and sample.get('poses_keypoints3d_F') is not None:
                follower_keypoints = sample['poses_keypoints3d_F']
                if isinstance(follower_keypoints, torch.Tensor):
                    follower_keypoints = follower_keypoints.cpu().numpy()
                follower_video_path = os.path.join(self.temp_dir, f"legacy_follower_{idx}.mp4")
                render_skeleton_from_keypoints(
                    follower_keypoints,
                    follower_video_path,
                    title=f"Follower (Legacy HumanML3D): {vid_id}",
                    fps=20,
                    radius=4,
                    figsize=(6, 6),
                    dpi=100
                )
                if os.path.exists(follower_video_path):
                    follower_video = follower_video_path
                    info_lines.append(f"Follower: {follower_keypoints.shape[0]} frames, {follower_keypoints.shape[1]} joints")
            
            if show_combined and sample.get('poses_keypoints3d_L') is not None and sample.get('poses_keypoints3d_F') is not None:
                combined_video_path = os.path.join(self.temp_dir, f"legacy_combined_{idx}.mp4")
                leader_kp = sample['poses_keypoints3d_L']
                follower_kp = sample['poses_keypoints3d_F']
                if isinstance(leader_kp, torch.Tensor):
                    leader_kp = leader_kp.cpu().numpy()
                if isinstance(follower_kp, torch.Tensor):
                    follower_kp = follower_kp.cpu().numpy()
                render_combined_skeletons(
                    leader_kp,
                    follower_kp,
                    combined_video_path,
                    title=f"Together (Legacy HumanML3D): {vid_id}",
                    fps=20,
                    figsize=(8, 6),
                    dpi=100
                )
                if os.path.exists(combined_video_path):
                    combined_video = combined_video_path
            
            info = f"Legacy HumanML3D Visualization\n"
            info += f"{'='*60}\n"
            info += f"Sample: {idx}\n"
            info += f"Video: {vid_id}\n"
            if info_lines:
                info += "\n" + "\n".join(info_lines)
            
            return leader_video, follower_video, combined_video, info
            
        except Exception as e:
            import traceback
            return None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def _get_wavtokenizer(self):
        """Lazy load WavTokenizer for audio decoding."""
        if self.wavtokenizer is not None:
            return self.wavtokenizer
        
        try:
            from utils.salsa_utils.libs.WavTokenizer.decoder.pretrained import WavTokenizer
            
            # Get paths
            WavTokenizer_relativeroot = self.parent_dir / 'utils' / 'salsa_utils' / 'libs' / 'WavTokenizer'
            config_path = WavTokenizer_relativeroot / 'configs' / 'wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
            model_path = WavTokenizer_relativeroot / 'results' / 'train' / 'wavtokenizer_large_unify_600_24k.ckpt'
            
            if not config_path.exists():
                raise FileNotFoundError(f"WavTokenizer config not found: {config_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"WavTokenizer model not found: {model_path}")
            
            # Initialize WavTokenizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.wavtokenizer = WavTokenizer.from_pretrained0802(str(config_path), str(model_path))
            self.wavtokenizer = self.wavtokenizer.to(device)
            self.wavtokenizer.eval()
            
            print("WavTokenizer loaded successfully!")
            return self.wavtokenizer
        except Exception as e:
            import traceback
            error_msg = f"Error loading WavTokenizer: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None
    
    def visualize_audio_comparison(self, idx: int) -> Tuple[Optional[str], Optional[str], Optional[object], str]:
        """
        Visualize audio comparison: Ground Truth vs Wavetokenizer decoded.
        
        Returns:
            gt_audio_path: Path to ground truth audio file
            decoded_audio_path: Path to decoded audio file (from tokens)
            waveform_plot: Plotly figure comparing waveforms
            info: Information string
        """
        if self.dataset is None:
            return None, None, None, "Error: No dataset loaded"
        
        try:
            sample = self._get_sample_from_dataset(idx)
            aux_info = sample.get('aux_info', {})
            vid_id = aux_info.get('vid', f'sample_{idx}')
            
            # Get audio data
            audio_raw = sample.get('audio_raw')
            audio_tokens = sample.get('audio_tokens')
            
            if audio_raw is None:
                return None, None, None, "Error: No ground truth audio (audio_raw) available"
            
            if audio_tokens is None:
                return None, None, None, "Error: No audio tokens available"
            
            # Convert audio_raw to numpy if needed
            if isinstance(audio_raw, torch.Tensor):
                audio_raw = audio_raw.detach().cpu().numpy()
            audio_raw = np.array(audio_raw)
            
            # Handle multi-channel audio
            if len(audio_raw.shape) > 1:
                audio_raw = audio_raw.mean(axis=0) if audio_raw.shape[0] > 1 else audio_raw.squeeze()
            audio_raw = audio_raw.squeeze()
            
            # Save ground truth audio
            gt_audio_path = os.path.join(self.temp_dir, f"audio_gt_{idx}.wav")
            save_audio_waveform(audio_raw, gt_audio_path, sample_rate=24000)
            
            # Decode audio from tokens
            decoded_audio = None
            decoded_audio_path = None
            decode_error_msg = None
            
            wavtokenizer = self._get_wavtokenizer()
            if wavtokenizer is not None:
                try:
                    from visualization.visualization_utils import decode_audio_tokens
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    decoded_audio = decode_audio_tokens(audio_tokens, wavtokenizer, device=device)
                    
                    if decoded_audio is not None:
                        decoded_audio_path = os.path.join(self.temp_dir, f"audio_decoded_{idx}.wav")
                        save_audio_waveform(decoded_audio, decoded_audio_path, sample_rate=24000)
                except Exception as decode_err:
                    import traceback
                    decode_error_msg = f"{decode_err}\n{traceback.format_exc()}"
                    print(f"Error decoding audio tokens: {decode_error_msg}")
            else:
                return None, None, None, "Error: Could not load WavTokenizer. Please check model paths."
            
            # Create waveform comparison plot
            waveform_plot = None
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Prepare data for plotting
                sample_rate = 24000
                duration_gt = len(audio_raw) / sample_rate
                time_gt = np.linspace(0, duration_gt, len(audio_raw))
                
                if decoded_audio is not None:
                    duration_decoded = len(decoded_audio) / sample_rate
                    time_decoded = np.linspace(0, duration_decoded, len(decoded_audio))
                    
                    # Create subplots: GT on top, Decoded on bottom
                    waveform_plot = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Ground Truth Audio", "Wavetokenizer Decoded Audio"),
                        vertical_spacing=0.15,
                        shared_xaxes=True
                    )
                    
                    # Ground truth waveform
                    waveform_plot.add_trace(
                        go.Scatter(
                            x=time_gt,
                            y=audio_raw,
                            mode='lines',
                            name='Ground Truth',
                            line=dict(color='blue', width=1),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Decoded waveform
                    waveform_plot.add_trace(
                        go.Scatter(
                            x=time_decoded,
                            y=decoded_audio,
                            mode='lines',
                            name='Decoded',
                            line=dict(color='red', width=1),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # Update axes
                    waveform_plot.update_xaxes(title_text="Time (seconds)", row=2, col=1)
                    waveform_plot.update_yaxes(title_text="Amplitude", row=1, col=1)
                    waveform_plot.update_yaxes(title_text="Amplitude", row=2, col=1)
                    
                    waveform_plot.update_layout(
                        title="Audio Comparison: Ground Truth vs Wavetokenizer Decoded",
                        height=600,
                        template="plotly_white",
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                else:
                    # Only ground truth available
                    waveform_plot = go.Figure()
                    waveform_plot.add_trace(
                        go.Scatter(
                            x=time_gt,
                            y=audio_raw,
                            mode='lines',
                            name='Ground Truth',
                            line=dict(color='blue', width=1)
                        )
                    )
                    waveform_plot.update_layout(
                        title="Ground Truth Audio (Decoding Failed)",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude",
                        height=400,
                        template="plotly_white"
                    )
            except Exception as plot_err:
                if DEBUG:
                    print(f"Warning: Failed to create waveform plot: {plot_err}")
            
            # Build info string
            info = f"Audio Comparison\n"
            info += f"{'='*60}\n"
            info += f"Sample: {idx}\n"
            info += f"Video: {vid_id}\n"
            info += f"\nGround Truth Audio:\n"
            info += f"  Shape: {audio_raw.shape}\n"
            info += f"  Duration: {len(audio_raw) / 24000:.3f} seconds\n"
            info += f"  Sample Rate: 24000 Hz\n"
            info += f"  Range: [{audio_raw.min():.4f}, {audio_raw.max():.4f}]\n"
            
            if audio_tokens is not None:
                audio_tokens_arr = np.array(audio_tokens) if not isinstance(audio_tokens, np.ndarray) else audio_tokens
                info += f"\nAudio Tokens:\n"
                info += f"  Shape: {audio_tokens_arr.shape}\n"
                info += f"  Range: [{audio_tokens_arr.min()}, {audio_tokens_arr.max()}]\n"
            
            if decoded_audio is not None:
                info += f"\nDecoded Audio (from tokens):\n"
                info += f"  Shape: {decoded_audio.shape}\n"
                info += f"  Duration: {len(decoded_audio) / 24000:.3f} seconds\n"
                info += f"  Range: [{decoded_audio.min():.4f}, {decoded_audio.max():.4f}]\n"
                
                # Compute similarity metrics
                min_len = min(len(audio_raw), len(decoded_audio))
                if min_len > 0:
                    gt_trimmed = audio_raw[:min_len]
                    decoded_trimmed = decoded_audio[:min_len]
                    
                    # MSE
                    mse = np.mean((gt_trimmed - decoded_trimmed) ** 2)
                    info += f"\nComparison Metrics:\n"
                    info += f"  MSE: {mse:.6f}\n"
                    
                    # Correlation
                    if np.std(gt_trimmed) > 1e-6 and np.std(decoded_trimmed) > 1e-6:
                        correlation = np.corrcoef(gt_trimmed, decoded_trimmed)[0, 1]
                        info += f"  Correlation: {correlation:.4f}\n"
            else:
                info += f"\nDecoded Audio: Failed to decode\n"
                if decode_error_msg:
                    info += f"\nDecode error:\n{decode_error_msg}\n"
            
            return gt_audio_path, decoded_audio_path, waveform_plot, info
            
        except Exception as e:
            import traceback
            return None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def _parse_vid_metadata(self, vid: str) -> dict:
        """Parse vid string to extract metadata: Pair, Song, Take, Roles."""
        metadata = {
            'pair': None,
            'song': None,
            'take': None,
            'leader_id': None,
            'follower_id': None,
            'level': None,
            'split': None
        }
        
        try:
            # Format: "Pair3_song2_take2_leader,Pair3_song2_take2_follower"
            # Or: "Pair1_8_7_take1_1_leader_subject,Pair1_8_7_take1_1_follower_subject"
            if ',' in vid:
                parts = vid.split(',')
                leader_part = parts[0].strip()
                follower_part = parts[1].strip() if len(parts) > 1 else None
            else:
                leader_part = vid
                follower_part = None
            
            # Extract Pair
            if 'pair' in leader_part.lower():
                pair_match = [p for p in leader_part.split('_') if 'pair' in p.lower()]
                if pair_match:
                    pair_str = pair_match[0].lower()
                    try:
                        pair_num = int(''.join(filter(str.isdigit, pair_str)))
                        metadata['pair'] = f"Pair{pair_num}"
                    except:
                        pass
            
            # Extract Song
            if 'song' in leader_part.lower():
                song_match = [p for p in leader_part.split('_') if 'song' in p.lower()]
                if song_match:
                    song_str = song_match[0].lower()
                    try:
                        song_num = int(''.join(filter(str.isdigit, song_str)))
                        metadata['song'] = f"Song{song_num}"
                    except:
                        pass
            
            # Extract Take
            if 'take' in leader_part.lower():
                take_match = [p for p in leader_part.split('_') if 'take' in p.lower()]
                if take_match:
                    take_str = take_match[0].lower()
                    try:
                        take_num = int(''.join(filter(str.isdigit, take_str)))
                        metadata['take'] = f"Take{take_num}"
                    except:
                        pass
            
            # Take ID for annotation lookup: PairX_songY_takeZ (annotations in Dataset/compas3d only)
            take_id = leader_part
            for suf in ('_leader', '_leader_subject', '_follower', '_follower_subject'):
                if take_id.lower().endswith(suf):
                    take_id = take_id[:-len(suf)].rstrip('_')
                    break
            if len(take_id.split('_')) >= 3 and 'pair' in take_id.lower().split('_')[0]:
                metadata['leader_id'] = take_id
                if follower_part:
                    tf = follower_part
                    for suf in ('_leader', '_leader_subject', '_follower', '_follower_subject'):
                        if tf.lower().endswith(suf):
                            tf = tf[:-len(suf)].rstrip('_')
                            break
                    metadata['follower_id'] = tf if len(tf.split('_')) >= 3 else None
            
            # Get proficiency level from PAIR2LEVEL
            if metadata['pair']:
                pair_key = metadata['pair'].lower()
                try:
                    from utils.salsa_utils.salsa_dataloader import PAIR2LEVEL
                    metadata['level'] = PAIR2LEVEL.get(pair_key, 'unknown')
                except ImportError:
                    # Fallback mapping
                    pair_levels = {
                        'pair1': 'beginner', 'pair2': 'intermediate', 'pair3': 'beginner',
                        'pair4': 'intermediate', 'pair5': 'professional', 'pair6': 'intermediate',
                        'pair7': 'professional', 'pair8': 'beginner', 'pair9': 'professional'
                    }
                    metadata['level'] = pair_levels.get(pair_key, 'unknown')
            
            # Determine split (train/val/test)
            # Default to 'train' - could be enhanced to check actual splits_map if needed
            if metadata['pair'] and metadata['song'] and metadata['take']:
                metadata['split'] = 'train'  # Default, could be enhanced
            
        except Exception as e:
            if DEBUG:
                print(f"Error parsing vid metadata: {e}")
        
        return metadata
    
    def get_metadata_info(self, idx: int) -> dict:
        """Get comprehensive metadata for a sample."""
        if self.dataset is None:
            return {'error': 'No dataset loaded'}
        
        try:
            sample = self._get_sample_from_dataset(idx)
            aux_info = sample.get('aux_info', {})
            vid = aux_info.get('vid', '')
            
            # Parse vid metadata
            vid_metadata = self._parse_vid_metadata(vid)
            
            # Annotations only from cache (aux_info); no on-the-fly loading
            matched_annotations = {'moves': [], 'errors': [], 'styling_leader': [], 'styling_follower': []}
            if 'dance_moves' in aux_info:
                matched_annotations['moves'] = aux_info.get('dance_moves', [])
                matched_annotations['errors'] = aux_info.get('errors', [])
                matched_annotations['styling_leader'] = aux_info.get('styling_leader', [])
                matched_annotations['styling_follower'] = aux_info.get('styling_follower', [])
            
            out = {
                'vid': vid,
                'pair': vid_metadata.get('pair', 'Unknown'),
                'song': vid_metadata.get('song', 'Unknown'),
                'take': vid_metadata.get('take', 'Unknown'),
                'level': vid_metadata.get('level', 'Unknown'),
                'split': vid_metadata.get('split', 'Unknown'),
                'start_time': aux_info.get('start_time', 0),
                'end_time': aux_info.get('end_time', 0),
                'start_frame': aux_info.get('start_frame_no', 0),
                'end_frame': aux_info.get('end_frame_no', 0),
                'duration': aux_info.get('end_time', 0) - aux_info.get('start_time', 0),
                'moves': matched_annotations['moves'],
                'errors': matched_annotations['errors'],
                'styling_leader': matched_annotations['styling_leader'],
                'styling_follower': matched_annotations['styling_follower'],
                'annotations_loaded': 'dance_moves' in aux_info
            }
            if sample.get('ms_desc_L') is not None or sample.get('ms_des_F') is not None:
                out['motionscript_leader'] = sample.get('ms_desc_L') or ''
                out['motionscript_follower'] = sample.get('ms_des_F') or ''
            return out
        except Exception as e:
            import traceback
            return {'error': f"Error loading metadata: {str(e)}\n{traceback.format_exc()}"}
    
    def create_video_with_metadata_overlay(self, video_path: str, metadata: dict, output_path: Optional[str] = None) -> Optional[str]:
        """Create video with metadata overlay showing role, move names, and level badge."""
        if not os.path.exists(video_path):
            return None
        
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"video_with_overlay_{os.path.basename(video_path)}")
        
        try:
            try:
                from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            except ImportError:
                if DEBUG:
                    print("MoviePy not available, skipping video overlay")
                return video_path
            
            # Load video
            video = VideoFileClip(video_path)
            fps = video.fps
            duration = video.duration
            
            # Get moves, errors, styling sorted by time
            moves = sorted(metadata.get('moves', []), key=lambda x: x.get('overlap_start', 0))
            errors = sorted(metadata.get('errors', []), key=lambda x: x.get('overlap_start', 0))
            styling_leader = sorted(metadata.get('styling_leader', []), key=lambda x: x.get('overlap_start', 0))
            styling_follower = sorted(metadata.get('styling_follower', []), key=lambda x: x.get('overlap_start', 0))
            
            overlay_clips = []
            window_start_time = metadata.get('start_time', 0)
            level = metadata.get('level', 'unknown')
            
            # Level badge (top right, always visible)
            level_colors = {
                'beginner': '#4CAF50',
                'intermediate': '#FF9800',
                'professional': '#F44336'
            }
            level_color = level_colors.get(level, '#9E9E9E')
            level_badge = TextClip(
                level.upper(),
                fontsize=26,
                color='white',
                font='Arial-Bold',
                bg_color=level_color,
                size=(160, 45),
                method='caption',
                align='center'
            ).set_position(('right', 'top')).set_duration(duration).set_start(0)
            overlay_clips.append(level_badge)
            
            # Pair/Song/Take info (bottom left, always visible)
            pair = metadata.get('pair', 'Unknown')
            song = metadata.get('song', 'Unknown')
            take = metadata.get('take', 'Unknown')
            info_text = f"{pair} | {song} | {take}"
            info_clip = TextClip(
                info_text,
                fontsize=18,
                color='white',
                font='Arial',
                bg_color='rgba(0,0,0,0.7)',
                size=(300, 35),
                method='caption',
                align='left'
            ).set_position(('left', 'bottom-40')).set_duration(duration).set_start(0)
            overlay_clips.append(info_clip)
            
            # Create overlay for each move segment
            for move in moves:
                move_start_abs = move.get('overlap_start', 0)
                move_end_abs = move.get('overlap_end', 0)
                start_t = max(0, move_start_abs - window_start_time)
                end_t = min(duration, move_end_abs - window_start_time)
                
                if end_t > start_t and start_t < duration:
                    move_text = move.get('description', '')
                    role_text = "Together"
                    
                    # Determine role from description
                    desc_lower = move_text.lower()
                    if 'leader' in desc_lower and 'follower' not in desc_lower:
                        role_text = "Leader"
                    elif 'follower' in desc_lower and 'leader' not in desc_lower:
                        role_text = "Follower"
                    elif 'leader' in desc_lower and 'follower' in desc_lower:
                        role_text = "Together"
                    
                    # Role badge (top left)
                    role_clip = TextClip(
                        role_text,
                        fontsize=28,
                        color='white',
                        font='Arial-Bold',
                        bg_color='rgba(0,0,0,0.75)',
                        size=(160, 45),
                        method='caption',
                        align='center'
                    ).set_position(('left', 'top')).set_duration(end_t - start_t).set_start(start_t)
                    
                    # Move description (top center)
                    display_text = move_text[:55] + "..." if len(move_text) > 55 else move_text
                    move_clip = TextClip(
                        display_text,
                        fontsize=18,
                        color='yellow',
                        font='Arial',
                        bg_color='rgba(0,0,0,0.75)',
                        size=(min(550, video.w - 20), 70),
                        method='caption',
                        align='center'
                    ).set_position(('center', 'top+50')).set_duration(end_t - start_t).set_start(start_t)
                    
                    overlay_clips.extend([role_clip, move_clip])
            
            # Add error indicators (red markers at bottom)
            for error in errors:
                error_start_abs = error.get('overlap_start', 0)
                error_end_abs = error.get('overlap_end', 0)
                start_t = max(0, error_start_abs - window_start_time)
                end_t = min(duration, error_end_abs - window_start_time)
                
                if end_t > start_t and start_t < duration:
                    error_indicator = TextClip(
                        "⚠",
                        fontsize=30,
                        color='red',
                        font='Arial-Bold',
                        bg_color='rgba(255,0,0,0.3)',
                        size=(40, 40),
                        method='caption',
                        align='center'
                    ).set_position(('right', 'bottom-50')).set_duration(end_t - start_t).set_start(start_t)
                    overlay_clips.append(error_indicator)
            
            # Composite video
            if overlay_clips:
                final_video = CompositeVideoClip([video] + overlay_clips)
            else:
                final_video = video
            
            final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            video.close()
            final_video.close()
            
            return output_path
        except Exception as e:
            import traceback
            if DEBUG:
                print(f"Error creating video overlay: {e}\n{traceback.format_exc()}")
            return video_path
    
    def create_timeline_visualization(self, metadata: dict) -> Optional[object]:
        """Create a timeline plot showing moves, errors, and styling."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            duration = metadata.get('duration', 5.0)
            moves = metadata.get('moves', [])
            errors = metadata.get('errors', [])
            styling_leader = metadata.get('styling_leader', [])
            styling_follower = metadata.get('styling_follower', [])
            window_start = metadata.get('start_time', 0)
            
            # Create subplots: one for moves, one for errors/styling
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Dance Moves", "Errors", "Styling"),
                vertical_spacing=0.15,
                shared_xaxes=True,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Moves timeline - show as horizontal bars
            if moves:
                move_y_positions = np.linspace(0.5, -0.5, len(moves))
                for i, move in enumerate(moves):
                    start_rel = max(0, move.get('overlap_start', 0) - window_start)
                    end_rel = min(duration, move.get('overlap_end', 0) - window_start)
                    move_text = move.get('description', '')[:35]
                    
                    if end_rel > start_rel:
                        fig.add_trace(
                            go.Scatter(
                                x=[start_rel, end_rel, end_rel, start_rel, start_rel],
                                y=[move_y_positions[i]-0.2, move_y_positions[i]-0.2, move_y_positions[i]+0.2, move_y_positions[i]+0.2, move_y_positions[i]-0.2],
                                fill='toself',
                                fillcolor='rgba(100, 149, 237, 0.7)',
                                line=dict(color='blue', width=2),
                                mode='lines',
                                name=move_text,
                                showlegend=False,
                                hovertemplate=f"<b>{move_text}</b><br>Start: {start_rel:.2f}s<br>End: {end_rel:.2f}s<extra></extra>"
                            ),
                            row=1, col=1
                        )
                        # Add text annotation in the middle
                        if (end_rel - start_rel) > 0.5:  # Only show text if bar is wide enough
                            fig.add_annotation(
                                x=(start_rel + end_rel) / 2,
                                y=move_y_positions[i],
                                text=move_text,
                                showarrow=False,
                                font=dict(size=9, color='white'),
                                bgcolor='rgba(0,0,0,0.5)',
                                bordercolor='white',
                                borderwidth=1,
                                row=1, col=1
                            )
            
            # Errors timeline
            for error in errors:
                start_rel = max(0, error.get('overlap_start', 0) - window_start)
                end_rel = min(duration, error.get('overlap_end', 0) - window_start)
                error_text = error.get('description', '')[:30]
                
                if end_rel > start_rel:
                    fig.add_trace(
                        go.Scatter(
                            x=[start_rel, end_rel],
                            y=[0, 0],
                            mode='markers+lines',
                            marker=dict(size=12, color='red', symbol='x'),
                            line=dict(color='red', width=2),
                            name=error_text,
                            showlegend=False,
                            hovertemplate=f"<b>Error: {error_text}</b><br>Time: {start_rel:.2f}s - {end_rel:.2f}s<extra></extra>"
                        ),
                        row=2, col=1
                    )
            
            # Styling timeline
            for style in styling_leader:
                start_rel = max(0, style.get('overlap_start', 0) - window_start)
                end_rel = min(duration, style.get('overlap_end', 0) - window_start)
                if end_rel > start_rel:
                    fig.add_trace(
                        go.Scatter(
                            x=[start_rel, end_rel],
                            y=[0.15, 0.15],
                            mode='markers+lines',
                            marker=dict(size=10, color='orange', symbol='star'),
                            line=dict(color='orange', width=2),
                            name='Leader Styling',
                            showlegend=False,
                            hovertemplate=f"<b>Leader Styling</b><br>Time: {start_rel:.2f}s - {end_rel:.2f}s<extra></extra>"
                        ),
                        row=3, col=1
                    )
            
            for style in styling_follower:
                start_rel = max(0, style.get('overlap_start', 0) - window_start)
                end_rel = min(duration, style.get('overlap_end', 0) - window_start)
                if end_rel > start_rel:
                    fig.add_trace(
                        go.Scatter(
                            x=[start_rel, end_rel],
                            y=[-0.15, -0.15],
                            mode='markers+lines',
                            marker=dict(size=10, color='pink', symbol='star'),
                            line=dict(color='pink', width=2),
                            name='Follower Styling',
                            showlegend=False,
                            hovertemplate=f"<b>Follower Styling</b><br>Time: {start_rel:.2f}s - {end_rel:.2f}s<extra></extra>"
                        ),
                        row=3, col=1
                    )
            
            # Update axes
            fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
            fig.update_yaxes(title_text="", row=1, col=1, showticklabels=False)
            fig.update_yaxes(title_text="", row=2, col=1, showticklabels=False, range=[-0.5, 0.5])
            fig.update_yaxes(title_text="", row=3, col=1, showticklabels=False, range=[-0.5, 0.5])
            
            fig.update_layout(
                title="Timeline: Moves, Errors, and Styling",
                height=400,
                template="plotly_white",
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis_range=[0, duration]
            )
            
            return fig
        except Exception as e:
            if DEBUG:
                print(f"Error creating timeline: {e}")
            return None
    
    def get_dataset_statistics(self) -> dict:
        """Get statistics about the entire dataset, including annotation-specific stats."""
        if self.dataset is None:
            return {'error': 'No dataset loaded'}
        
        try:
            total_samples = len(self.dataset)
            # All README move + error classes, initialized to 0
            move_class_dist = {c: 0 for c in ALL_LABEL_NAMES}
            stats = {
                'total_samples': total_samples,
                'pairs': {},
                'levels': {'beginner': 0, 'intermediate': 0, 'professional': 0, 'unknown': 0},
                'songs': {},
                'moves_count': 0,
                'errors_count': 0,
                'styling_count': 0,
                'samples_with_annotations': 0,
                'samples_without_annotations': 0,
                'samples_with_errors': 0,
                'samples_with_styling': 0,
                'move_class_dist': move_class_dist,
                'unclassified_move_count': 0,
                'unclassified_error_count': 0,
                'metadata_errors': 0,
                'sample_indices_with_errors': [],
            }
            
            sample_size = min(200, total_samples)
            sample_indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)
            
            for idx in sample_indices:
                try:
                    metadata = self.get_metadata_info(idx)
                    if 'error' in metadata:
                        stats['metadata_errors'] += 1
                        if len(stats['sample_indices_with_errors']) < 20:
                            stats['sample_indices_with_errors'].append(int(idx))
                        continue
                    
                    pair = metadata.get('pair', 'Unknown')
                    stats['pairs'][pair] = stats['pairs'].get(pair, 0) + 1
                    level = metadata.get('level', 'unknown')
                    stats['levels'][level] = stats['levels'].get(level, 0) + 1
                    song = metadata.get('song', 'Unknown')
                    stats['songs'][song] = stats['songs'].get(song, 0) + 1
                    
                    moves = metadata.get('moves', [])
                    errors = metadata.get('errors', [])
                    styling_l = metadata.get('styling_leader', [])
                    styling_f = metadata.get('styling_follower', [])
                    
                    stats['moves_count'] += len(moves)
                    stats['errors_count'] += len(errors)
                    stats['styling_count'] += len(styling_l) + len(styling_f)
                    
                    # Coverage = has at least one move, error, or styling in window (exclude empty)
                    n_ann = len(moves) + len(errors) + len(styling_l) + len(styling_f)
                    if n_ann > 0:
                        stats['samples_with_annotations'] += 1
                    else:
                        stats['samples_without_annotations'] += 1
                    if len(errors) > 0:
                        stats['samples_with_errors'] += 1
                    if len(styling_l) + len(styling_f) > 0:
                        stats['samples_with_styling'] += 1
                    
                    for m in moves:
                        mc = m.get('move_class')
                        if mc:
                            k = LABEL_TO_README.get(mc, mc)
                            if k in stats['move_class_dist']:
                                stats['move_class_dist'][k] += 1
                            else:
                                stats['unclassified_move_count'] += 1
                        else:
                            stats['unclassified_move_count'] += 1
                    for e in errors:
                        ec = e.get('error_class')
                        if ec:
                            k = LABEL_TO_README.get(ec, ec)
                            if k in stats['move_class_dist']:
                                stats['move_class_dist'][k] += 1
                            else:
                                stats['unclassified_error_count'] += 1
                        else:
                            stats['unclassified_error_count'] += 1
                    # Count styling classes
                    if styling_l:
                        if 'Lady styling' in stats['move_class_dist']:
                            stats['move_class_dist']['Lady styling'] += len(styling_l)
                    if styling_f:
                        if 'Man styling' in stats['move_class_dist']:
                            stats['move_class_dist']['Man styling'] += len(styling_f)
                except Exception as e:
                    stats['metadata_errors'] += 1
                    if len(stats['sample_indices_with_errors']) < 20:
                        stats['sample_indices_with_errors'].append(int(idx))
                    continue
            
            scale_factor = total_samples / max(1, sample_size)
            stats['moves_count'] = int(stats['moves_count'] * scale_factor)
            stats['errors_count'] = int(stats['errors_count'] * scale_factor)
            stats['styling_count'] = int(stats['styling_count'] * scale_factor)
            stats['samples_with_annotations'] = int(stats['samples_with_annotations'] * scale_factor)
            stats['samples_without_annotations'] = max(0, total_samples - stats['samples_with_annotations'])
            stats['samples_with_errors'] = int(stats['samples_with_errors'] * scale_factor)
            stats['samples_with_styling'] = int(stats['samples_with_styling'] * scale_factor)
            stats['unclassified_move_count'] = int(stats['unclassified_move_count'] * scale_factor)
            stats['unclassified_error_count'] = int(stats.get('unclassified_error_count', 0) * scale_factor)
            stats['metadata_errors'] = int(stats['metadata_errors'] * scale_factor)
            
            # Token similarity analysis per class
            stats['token_analysis'] = self._analyze_tokens_by_class(sample_indices)
            
            # Store for dropdown access
            self.last_computed_stats = stats
            
            return stats
        except Exception as e:
            import traceback
            return {'error': f"Error computing statistics: {str(e)}\n{traceback.format_exc()}"}
    
    def _analyze_tokens_by_class(self, sample_indices: np.ndarray) -> dict:
        """Analyze motion tokens (leader, follower, relationship) per move/error class."""
        if self.dataset is None:
            return {}
        
        # Collect tokens per class
        tokens_by_class = {c: {'leader': [], 'follower': [], 'relationship': []} for c in ALL_LABEL_NAMES}
        
        for idx in sample_indices:
            try:
                sample = self._get_sample_from_dataset(idx)
                interhuman_data = sample.get('interhuman_data', {})
                if not interhuman_data:
                    continue
                
                leader_tokens = interhuman_data.get('leader_tokens')
                follower_tokens = interhuman_data.get('follower_tokens')
                relationship_tokens = interhuman_data.get('relationship_tokens')
                
                if leader_tokens is None or follower_tokens is None or relationship_tokens is None:
                    continue
                
                metadata = self.get_metadata_info(idx)
                if 'error' in metadata:
                    continue
                
                # Get classes for this sample (moves, errors, styling)
                classes_in_sample = set()
                for move in metadata.get('moves', []):
                    mc = move.get('move_class')
                    if mc:
                        k = LABEL_TO_README.get(mc, mc)
                        if k in tokens_by_class:
                            classes_in_sample.add(k)
                for error in metadata.get('errors', []):
                    ec = error.get('error_class')
                    if ec:
                        k = LABEL_TO_README.get(ec, ec)
                        if k in tokens_by_class:
                            classes_in_sample.add(k)
                # Add styling classes
                styling_leader = metadata.get('styling_leader', [])
                styling_follower = metadata.get('styling_follower', [])
                if styling_leader:
                    if 'Lady styling' in tokens_by_class:
                        classes_in_sample.add('Lady styling')
                if styling_follower:
                    if 'Man styling' in tokens_by_class:
                        classes_in_sample.add('Man styling')
                
                # Store token sequences per class (for similarity metrics)
                leader_tokens_arr = np.array(leader_tokens).flatten().tolist()
                follower_tokens_arr = np.array(follower_tokens).flatten().tolist()
                relationship_tokens_arr = np.array(relationship_tokens).flatten().tolist()
                
                for cls in classes_in_sample:
                    # Store full sequences for similarity computation
                    if 'sequences' not in tokens_by_class[cls]:
                        tokens_by_class[cls]['sequences'] = []
                    tokens_by_class[cls]['sequences'].append({
                        'idx': int(idx),
                        'leader': leader_tokens_arr,
                        'follower': follower_tokens_arr,
                        'relationship': relationship_tokens_arr
                    })
                    # Also store flattened for frequency analysis
                    tokens_by_class[cls]['leader'].extend(leader_tokens_arr)
                    tokens_by_class[cls]['follower'].extend(follower_tokens_arr)
                    tokens_by_class[cls]['relationship'].extend(relationship_tokens_arr)
            except:
                continue
        
        # Compute statistics per class
        analysis = {}
        for cls, tokens in tokens_by_class.items():
            if not tokens['leader']:
                continue
            
            # Token frequency distributions
            leader_freq = {}
            follower_freq = {}
            relationship_freq = {}
            
            for t in tokens['leader']:
                leader_freq[t] = leader_freq.get(t, 0) + 1
            for t in tokens['follower']:
                follower_freq[t] = follower_freq.get(t, 0) + 1
            for t in tokens['relationship']:
                relationship_freq[t] = relationship_freq.get(t, 0) + 1
            
            # Most common tokens
            leader_top = sorted(leader_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            follower_top = sorted(follower_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            relationship_top = sorted(relationship_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Unique token counts
            unique_leader = len(set(tokens['leader']))
            unique_follower = len(set(tokens['follower']))
            unique_relationship = len(set(tokens['relationship']))
            
            # Token diversity (entropy-like measure)
            total_leader = len(tokens['leader'])
            total_follower = len(tokens['follower'])
            total_relationship = len(tokens['relationship'])
            
            leader_entropy = -sum((freq/total_leader) * np.log2(freq/total_leader) 
                                 for freq in leader_freq.values() if total_leader > 0) if total_leader > 0 else 0
            follower_entropy = -sum((freq/total_follower) * np.log2(freq/total_follower) 
                                   for freq in follower_freq.values() if total_follower > 0) if total_follower > 0 else 0
            relationship_entropy = -sum((freq/total_relationship) * np.log2(freq/total_relationship) 
                                       for freq in relationship_freq.values() if total_relationship > 0) if total_relationship > 0 else 0
            
            analysis[cls] = {
                'leader': {
                    'top_tokens': leader_top,
                    'unique_count': unique_leader,
                    'total_count': total_leader,
                    'entropy': leader_entropy,
                    'freq_dist': leader_freq
                },
                'follower': {
                    'top_tokens': follower_top,
                    'unique_count': unique_follower,
                    'total_count': total_follower,
                    'entropy': follower_entropy,
                    'freq_dist': follower_freq
                },
                'relationship': {
                    'top_tokens': relationship_top,
                    'unique_count': unique_relationship,
                    'total_count': total_relationship,
                    'entropy': relationship_entropy,
                    'freq_dist': relationship_freq
                }
            }
        
        # Compute similarity metrics for each class
        for cls, cls_data in analysis.items():
            sequences = tokens_by_class[cls].get('sequences', [])
            if len(sequences) < 2:
                # Need at least 2 samples to compute similarity
                cls_data['similarity_metrics'] = None
                continue
            
            # Compute pairwise similarities for each token type
            similarity_results = self._compute_token_similarity_metrics(sequences)
            cls_data['similarity_metrics'] = similarity_results
            cls_data['num_samples'] = len(sequences)
        
        return analysis
    
    def _compute_token_similarity_metrics(self, sequences: list) -> dict:
        """Compute similarity metrics between token sequences.
        
        Metrics: Edit Distance, Hamming Distance, Jaccard, BLEU, ROUGE
        """
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.metrics.distance import edit_distance
        except ImportError:
            return {'error': 'nltk not available'}
        
        # Download required nltk data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # ROUGE-1 implementation (unigram recall)
        def rouge_1(reference, candidate):
            ref_ngrams = set(reference)
            cand_ngrams = set(candidate)
            if len(ref_ngrams) == 0:
                return 0.0
            overlap = len(ref_ngrams & cand_ngrams)
            return overlap / len(ref_ngrams)
        
        results = {
            'leader': {'edit_dist': [], 'hamming': [], 'jaccard': [], 'bleu': [], 'rouge': []},
            'follower': {'edit_dist': [], 'hamming': [], 'jaccard': [], 'bleu': [], 'rouge': []},
            'relationship': {'edit_dist': [], 'hamming': [], 'jaccard': [], 'bleu': [], 'rouge': []}
        }
        
        smoothing = SmoothingFunction().method1
        
        # Convert tokens to strings for nltk
        def tokens_to_str(tokens):
            return [str(t) for t in tokens]
        
        # Compute pairwise similarities
        n = len(sequences)
        for i in range(n):
            for j in range(i + 1, n):
                seq1 = sequences[i]
                seq2 = sequences[j]
                
                for token_type in ['leader', 'follower', 'relationship']:
                    tokens1 = tokens_to_str(seq1[token_type])
                    tokens2 = tokens_to_str(seq2[token_type])
                    
                    # Edit Distance (Levenshtein)
                    edit_dist = edit_distance(tokens1, tokens2)
                    results[token_type]['edit_dist'].append(edit_dist)
                    
                    # Hamming Distance (for equal length sequences)
                    if len(tokens1) == len(tokens2):
                        hamming = sum(t1 != t2 for t1, t2 in zip(tokens1, tokens2))
                    else:
                        hamming = None  # Skip if lengths differ
                    if hamming is not None:
                        results[token_type]['hamming'].append(hamming)
                    
                    # Jaccard Similarity
                    set1 = set(tokens1)
                    set2 = set(tokens2)
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard = intersection / union if union > 0 else 0.0
                    results[token_type]['jaccard'].append(jaccard)
                    
                    # BLEU (using sentence_bleu with smoothing)
                    try:
                        bleu = sentence_bleu([tokens1], tokens2, smoothing_function=smoothing)
                        results[token_type]['bleu'].append(bleu)
                    except:
                        pass
                    
                    # ROUGE-1 (unigram recall)
                    try:
                        rouge1 = rouge_1(tokens1, tokens2)
                        results[token_type]['rouge'].append(rouge1)
                    except:
                        pass
        
        # Compute statistics
        stats = {}
        for token_type in ['leader', 'follower', 'relationship']:
            stats[token_type] = {}
            for metric_name, values in results[token_type].items():
                if values:
                    stats[token_type][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
                else:
                    stats[token_type][metric_name] = None
        
        return stats
    
    def _compute_inter_class_similarity(self, token_analysis: dict) -> dict:
        """Compute similarity metrics between samples from different classes (inter-class).
        
        Returns average similarity between classes for comparison with intra-class similarity.
        """
        if not token_analysis:
            return {}
        
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.metrics.distance import edit_distance
        except ImportError:
            return {'error': 'nltk not available'}
        
        # Download required nltk data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        smoothing = SmoothingFunction().method1
        
        def rouge_1(reference, candidate):
            ref_ngrams = set(reference)
            cand_ngrams = set(candidate)
            if len(ref_ngrams) == 0:
                return 0.0
            overlap = len(ref_ngrams & cand_ngrams)
            return overlap / len(ref_ngrams)
        
        def tokens_to_str(tokens):
            return [str(t) for t in tokens]
        
        # Collect all sequences with their class labels
        all_sequences = []
        for cls, cls_data in token_analysis.items():
            sequences = cls_data.get('sequences', [])
            for seq in sequences:
                all_sequences.append({'class': cls, 'seq': seq})
        
        # Compute inter-class similarities (samples from different classes)
        inter_class_results = {
            'leader': {'edit_dist': [], 'hamming': [], 'jaccard': [], 'bleu': [], 'rouge': []},
            'follower': {'edit_dist': [], 'hamming': [], 'jaccard': [], 'bleu': [], 'rouge': []},
            'relationship': {'edit_dist': [], 'hamming': [], 'jaccard': [], 'bleu': [], 'rouge': []}
        }
        
        # Sample pairs from different classes (limit to avoid too many computations)
        n_total = len(all_sequences)
        max_pairs = 500  # Limit inter-class pairs for performance
        pair_count = 0
        
        for i in range(min(n_total, 100)):  # Sample up to 100 sequences
            for j in range(i + 1, min(n_total, 200)):
                if all_sequences[i]['class'] != all_sequences[j]['class']:
                    if pair_count >= max_pairs:
                        break
                    pair_count += 1
                    
                    seq1 = all_sequences[i]['seq']
                    seq2 = all_sequences[j]['seq']
                    
                    for token_type in ['leader', 'follower', 'relationship']:
                        tokens1 = tokens_to_str(seq1[token_type])
                        tokens2 = tokens_to_str(seq2[token_type])
                        
                        # Edit Distance
                        edit_dist = edit_distance(tokens1, tokens2)
                        inter_class_results[token_type]['edit_dist'].append(edit_dist)
                        
                        # Hamming
                        if len(tokens1) == len(tokens2):
                            hamming = sum(t1 != t2 for t1, t2 in zip(tokens1, tokens2))
                            inter_class_results[token_type]['hamming'].append(hamming)
                        
                        # Jaccard
                        set1 = set(tokens1)
                        set2 = set(tokens2)
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        jaccard = intersection / union if union > 0 else 0.0
                        inter_class_results[token_type]['jaccard'].append(jaccard)
                        
                        # BLEU
                        try:
                            bleu = sentence_bleu([tokens1], tokens2, smoothing_function=smoothing)
                            inter_class_results[token_type]['bleu'].append(bleu)
                        except:
                            pass
                        
                        # ROUGE
                        try:
                            rouge1 = rouge_1(tokens1, tokens2)
                            inter_class_results[token_type]['rouge'].append(rouge1)
                        except:
                            pass
            if pair_count >= max_pairs:
                break
        
        # Compute average inter-class statistics
        inter_class_stats = {}
        for token_type in ['leader', 'follower', 'relationship']:
            inter_class_stats[token_type] = {}
            for metric_name, values in inter_class_results[token_type].items():
                if values:
                    inter_class_stats[token_type][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
                else:
                    inter_class_stats[token_type][metric_name] = None
        
        return inter_class_stats
    
    def _create_metric_comparison_chart(self, token_analysis: dict, metric_name: str) -> tuple:
        """Create a bar chart comparing a specific metric across all classes.
        
        Returns: (figure, description_text)
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Metric mapping
        metric_mapping = {
            "Jaccard Similarity": ("jaccard", "Jaccard Similarity: Ratio of shared tokens to total unique tokens. Range [0,1], higher = more similar."),
            "BLEU": ("bleu", "BLEU: Precision-based n-gram overlap score. Range [0,1], higher = more similar (translation quality metric)."),
            "ROUGE-1": ("rouge", "ROUGE-1: Recall-based unigram overlap. Range [0,1], higher = more similar (summarization metric)."),
            "Edit Distance": ("edit_dist", "Edit Distance (Levenshtein): Minimum operations (insert/delete/substitute) to transform one sequence into another. Lower = more similar."),
            "Hamming Distance": ("hamming", "Hamming Distance: Count of positions where tokens differ (only for equal-length sequences). Lower = more similar.")
        }
        
        if metric_name not in metric_mapping:
            return None, "Unknown metric"
        
        metric_key, description = metric_mapping[metric_name]
        
        # Get classes with similarity data
        classes_with_data = []
        leader_values = []
        follower_values = []
        relationship_values = []
        
        for cls in ALL_LABEL_NAMES:
            if cls in token_analysis:
                sim_metrics = token_analysis[cls].get('similarity_metrics')
                if sim_metrics and 'error' not in sim_metrics:
                    leader_data = sim_metrics['leader'].get(metric_key)
                    follower_data = sim_metrics['follower'].get(metric_key)
                    relationship_data = sim_metrics['relationship'].get(metric_key)
                    
                    if leader_data and leader_data.get('mean') is not None:
                        classes_with_data.append(cls[:20])  # Truncate for display
                        # For distance metrics (edit_dist, hamming), show raw values (lower = more similar)
                        # For similarity metrics (jaccard, bleu, rouge), show raw values (higher = more similar)
                        leader_val = leader_data['mean']
                        follower_val = follower_data['mean'] if follower_data and follower_data.get('mean') is not None else 0.0
                        relationship_val = relationship_data['mean'] if relationship_data and relationship_data.get('mean') is not None else 0.0
                        
                        leader_values.append(leader_val)
                        follower_values.append(follower_val)
                        relationship_values.append(relationship_val)
        
        if not classes_with_data:
            return None, description
        
        # Create grouped bar chart
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Leader Tokens', 'Follower Tokens', 'Relationship Tokens'],
            horizontal_spacing=0.12
        )
        
        for col_idx, (token_type, values) in enumerate([
            ('leader', leader_values),
            ('follower', follower_values),
            ('relationship', relationship_values)
        ], 1):
            fig.add_trace(
                go.Bar(
                    x=classes_with_data,
                    y=values,
                    name=token_type.capitalize(),
                    marker_color={'leader': 'blue', 'follower': 'green', 'relationship': 'red'}[token_type],
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
            fig.update_xaxes(title_text="Class", row=1, col=col_idx, tickangle=-45)
            fig.update_yaxes(title_text=metric_name, row=1, col=col_idx)
        
        fig.update_layout(
            title=f"{metric_name} Comparison Across Classes",
            height=500,
            template="plotly_white"
        )
        
        return fig, description
    
    def _create_radar_chart_all_metrics(self, token_analysis: dict):
        """Create a radar chart showing all metrics for all classes.
        
        Uses polar coordinates to show multiple metrics simultaneously.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if not token_analysis:
            return None
        
        # Get classes with similarity data
        classes_with_data = []
        for cls in ALL_LABEL_NAMES:
            if cls in token_analysis:
                sim_metrics = token_analysis[cls].get('similarity_metrics')
                if sim_metrics and 'error' not in sim_metrics:
                    classes_with_data.append(cls)
        
        if not classes_with_data:
            return None
        
        # Limit to top 12 classes by sample count to avoid clutter
        classes_with_data = sorted(classes_with_data, 
                                   key=lambda x: token_analysis[x].get('num_samples', 0), 
                                   reverse=True)[:12]
        
        # Metrics to include (similarity metrics only, normalized to [0,1])
        metrics = ['jaccard', 'bleu', 'rouge']
        metric_labels = ['Jaccard', 'BLEU', 'ROUGE-1']
        token_types = ['leader', 'follower', 'relationship']
        
        # Create radar charts - one per token type
        fig = go.Figure()
        
        # We'll create separate subplots for each token type using domain
        # Or create one comprehensive radar chart with all classes
        # Let's create one radar chart with all classes, using different colors
        
        # Prepare data for all classes
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                 'olive', 'cyan', 'magenta', 'yellow']
        
        for cls_idx, cls in enumerate(classes_with_data):
            sim_metrics = token_analysis[cls]['similarity_metrics']
            
            # Average across all token types for each metric
            avg_values = []
            for metric_key in metrics:
                metric_vals = []
                for token_type in token_types:
                    metric_data = sim_metrics[token_type].get(metric_key)
                    if metric_data and metric_data.get('mean') is not None:
                        metric_vals.append(metric_data['mean'])
                if metric_vals:
                    avg_values.append(np.mean(metric_vals))
                else:
                    avg_values.append(0.0)
            
            # Close the polygon
            avg_values.append(avg_values[0])
            theta_labels = metric_labels + [metric_labels[0]]
            
            # Add trace for this class
            fig.add_trace(go.Scatterpolar(
                r=avg_values,
                theta=theta_labels,
                fill='toself' if cls_idx < 6 else 'none',
                name=cls[:20],
                line=dict(
                    width=2 if cls_idx < 6 else 1,
                    color=colors[cls_idx % len(colors)]
                ),
                opacity=0.8 if cls_idx < 6 else 0.5
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2,
                    showticklabels=True,
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11),
                    rotation=90,
                    direction='counterclockwise'
                )
            ),
            title="Radar Chart: All Metrics (Jaccard, BLEU, ROUGE-1) Across All Classes<br><sub>Average across Leader, Follower, and Relationship tokens</sub>",
            height=700,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.15
            )
        )
        
        return fig
    
    def get_move_class_counts(self) -> dict:
        """Get counts of samples per class (moves + styling + errors) for dropdown."""
        if self.dataset is None:
            return {}
        counts = {c: 0 for c in ALL_LABEL_NAMES}
        total_samples = len(self.dataset)
        sample_size = min(500, total_samples)
        sample_indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)
        for idx in sample_indices:
            try:
                metadata = self.get_metadata_info(idx)
                if 'error' in metadata:
                    continue
                for move in metadata.get('moves', []):
                    mc = move.get('move_class')
                    if mc:
                        k = LABEL_TO_README.get(mc, mc)
                        if k in counts:
                            counts[k] += 1
                for error in metadata.get('errors', []):
                    ec = error.get('error_class')
                    if ec:
                        k = LABEL_TO_README.get(ec, ec)
                        if k in counts:
                            counts[k] += 1
                if metadata.get('styling_leader'):
                    if 'Lady styling' in counts:
                        counts['Lady styling'] += 1
                if metadata.get('styling_follower'):
                    if 'Man styling' in counts:
                        counts['Man styling'] += 1
            except:
                continue
        scale_factor = total_samples / max(1, sample_size)
        return {k: int(v * scale_factor) for k, v in counts.items()}
    
    def search_samples(self, filters: dict) -> list:
        """Search for samples matching filters."""
        if self.dataset is None:
            return []
        
        try:
            total_samples = len(self.dataset)
            matching_indices = []
            
            # Limit search to avoid loading all samples
            max_search = min(500, total_samples)
            
            for idx in range(max_search):
                try:
                    metadata = self.get_metadata_info(idx)
                    if 'error' in metadata:
                        continue
                    
                    # Apply filters
                    match = True
                    
                    if filters.get('level') and metadata.get('level') != filters['level']:
                        match = False
                    
                    if filters.get('pair') and metadata.get('pair') != filters['pair']:
                        match = False
                    
                    if filters.get('song') and metadata.get('song') != filters['song']:
                        match = False
                    
                    if filters.get('has_errors') and len(metadata.get('errors', [])) == 0:
                        match = False
                    
                    if filters.get('has_styling') and len(metadata.get('styling_leader', [])) + len(metadata.get('styling_follower', [])) == 0:
                        match = False
                    
                    if filters.get('move_keyword'):
                        keyword = filters['move_keyword'].lower()
                        moves = metadata.get('moves', [])
                        if not any(keyword in move.get('description', '').lower() for move in moves):
                            match = False
                    
                    if match:
                        matching_indices.append({
                            'index': idx,
                            'pair': metadata.get('pair', 'Unknown'),
                            'song': metadata.get('song', 'Unknown'),
                            'take': metadata.get('take', 'Unknown'),
                            'level': metadata.get('level', 'Unknown'),
                            'moves_count': len(metadata.get('moves', [])),
                            'errors_count': len(metadata.get('errors', []))
                        })
                except:
                    continue
            
            return matching_indices
        except Exception as e:
            if DEBUG:
                print(f"Error searching samples: {e}")
            return []

    def get_llm_model(self, ckpt_path: str):
        """Load MotionLLM once per checkpoint. Returns (model, error_msg)."""
        ckpt_path = os.path.abspath(ckpt_path) if ckpt_path else ""
        if self._llm_model is not None and self._llm_ckpt_path == ckpt_path:
            return self._llm_model, None
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return None, f"Checkpoint not found: {ckpt_path}"
        try:
            from models.mllm import MotionLLM
            from options.option_llm import get_args_parser
            args = get_args_parser()
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.motion_repr_type = "interhuman"
            ckpt_config = MotionLLM.load_config_from_checkpoint(ckpt_path)
            args.include_audio = ckpt_config.get("include_audio", False)
            args.include_motionscript = ckpt_config.get("include_motionscript", False)
            model = MotionLLM(args)
            model.load_model(ckpt_path)
            model.llm.eval()
            self._llm_model = model
            self._llm_ckpt_path = ckpt_path
            return model, None
        except Exception as e:
            import traceback
            return None, f"Model load error: {e}\n{traceback.format_exc()}"

    def _build_llm_prompt_for_sample(
        self,
        idx: int,
        task_choice: str,
        include_audio_val: bool,
        include_motionscript_val: bool,
        output_motionscript_first_val: bool,
        model,
    ):
        """Build InterHuman prompt/target for one sample. Returns (prompt, gt_target, task_key, error)."""
        from models.training_utils import build_prompt_interhuman_salsa
        sample = self._get_sample_from_dataset(idx)
        interhuman_data = sample.get("interhuman_data")
        if interhuman_data is None:
            return None, None, None, "No InterHuman data for this sample."
        leader_tokens = interhuman_data.get("leader_tokens")
        follower_tokens = interhuman_data.get("follower_tokens")
        relationship_tokens = interhuman_data.get("relationship_tokens")
        if leader_tokens is None or follower_tokens is None or relationship_tokens is None:
            return None, None, None, "Sample missing leader_tokens, follower_tokens, or relationship_tokens."
        leader_tokens = np.asarray(leader_tokens).ravel().tolist()
        follower_tokens = np.asarray(follower_tokens).ravel().tolist()
        relationship_tokens = np.asarray(relationship_tokens).ravel().tolist()
        audio_tokens = sample.get("audio_tokens")
        if audio_tokens is not None:
            audio_tokens = np.asarray(audio_tokens).ravel().tolist()
        task_key = LLM_INFERENCE_TASK_MAP.get(task_choice, "leader_rel_to_follower")
        metadata = self.get_metadata_info(idx)
        move_annotations = metadata.get("moves", []) if metadata and "error" not in metadata else []
        level = metadata.get("level") if metadata and "error" not in metadata else None
        caption = metadata.get("caption") if metadata and "error" not in metadata else None
        ms_L = sample.get("ms_desc_L")
        ms_F = sample.get("ms_des_F")
        if ms_L is not None and not isinstance(ms_L, str):
            ms_L = " --> ".join(str(x) for x in ms_L) if (isinstance(ms_L, (list, tuple)) and ms_L) else ""
        if ms_F is not None and not isinstance(ms_F, str):
            ms_F = " --> ".join(str(x) for x in ms_F) if (isinstance(ms_F, (list, tuple)) and ms_F) else ""
        ms_L = (ms_L or "").strip() if isinstance(ms_L, str) else ""
        ms_F = (ms_F or "").strip() if isinstance(ms_F, str) else ""
        include_ms = bool(include_motionscript_val) and bool(ms_L or ms_F) and bool(getattr(model, "include_motionscript", False))
        effective_include_audio = bool(include_audio_val) and bool(getattr(model, "include_audio", False))
        prompt_text, gt_target_text = build_prompt_interhuman_salsa(
            leader_tokens=leader_tokens,
            follower_tokens=follower_tokens,
            relationship_tokens=relationship_tokens,
            task=task_key,
            move_annotations=move_annotations,
            level=level,
            caption=caption,
            audio_tokens=audio_tokens,
            include_audio=effective_include_audio,
            include_motionscript=include_ms,
            motionscript_leader=(ms_L or None) if include_ms else None,
            motionscript_follower=(ms_F or None) if include_ms else None,
            output_motionscript_first=bool(output_motionscript_first_val),
        )
        return prompt_text, gt_target_text, task_key, None

    def _timed_llm_generate(self, model, prompt_text, task_key, max_new_tokens):
        """CUDA-synced wall time of generate_Payam_interhuman only. Returns (elapsed_s, pred_dict)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_dict = model.generate_Payam_interhuman(prompt_text, task_key, max_new_tokens=int(max_new_tokens))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed, pred_dict

    def run_llm_latency_benchmark(
        self,
        start_idx,
        task_choice,
        include_audio_val,
        include_motionscript_val,
        output_motionscript_first_val,
        ckpt_path,
        n_samples,
        max_new_tokens,
        visualize_one,
        warmup,
        progress=None,
    ):
        """
        Measure average LLM inference time over N samples.
        Times only generate_Payam_interhuman (CUDA-synced). Excludes checkpoint load,
        dataset I/O, prompt construction, token reconstruction, and video rendering.
        """
        if progress is None:
            progress = lambda *a, **k: None
        if self.dataset is None:
            return "", None, "Load LMDB first."
        n_total = len(self.dataset)
        if n_total <= 0:
            return "", None, "Dataset is empty."
        try:
            start_idx = int(start_idx) if start_idx is not None else 0
            start_idx = start_idx % n_total
        except (TypeError, ValueError):
            start_idx = 0
        try:
            n_samples = int(n_samples) if n_samples not in (None, "") else 10
            n_samples = max(1, min(100, n_samples))
        except (TypeError, ValueError):
            n_samples = 10
        try:
            max_new_tokens = int(max_new_tokens) if max_new_tokens not in (None, "") else 150
            max_new_tokens = max(8, min(1024, max_new_tokens))
        except (TypeError, ValueError):
            max_new_tokens = 150
        want_viz = bool(visualize_one)
        warmup = bool(warmup)

        progress(0, desc="Loading LLM checkpoint (not timed)…")
        model, err = self.get_llm_model(ckpt_path)
        if err:
            return "", None, err

        n_extra = (1 if warmup else 0) + (1 if (want_viz and not warmup) else 0)
        n_needed = n_samples + n_extra

        progress(0.05, desc="Building prompts (not timed)…")
        prompts = []
        used_indices = []
        skipped = 0
        scan_i = 0
        max_scan = n_total * 2
        while len(prompts) < n_needed and scan_i < max_scan:
            idx = (start_idx + scan_i) % n_total
            scan_i += 1
            prompt_text, _gt, task_key, perr = self._build_llm_prompt_for_sample(
                idx,
                task_choice,
                include_audio_val,
                include_motionscript_val,
                output_motionscript_first_val,
                model,
            )
            if perr or not prompt_text:
                skipped += 1
                continue
            prompts.append((idx, prompt_text, task_key))
            used_indices.append(idx)
        if len(prompts) < n_needed:
            return "", None, (
                f"Could not build enough InterHuman prompts ({len(prompts)}/{n_needed}). "
                f"Skipped {skipped} samples without InterHuman data."
            )

        from models.training_utils import INTERHUMAN_TASK_OUTPUT_TYPE

        video_path = None
        cursor = 0
        try:
            if warmup:
                progress(0.08, desc="Warmup (untimed)…")
                idx_w, prompt_w, task_w = prompts[cursor]
                _, pred_w = self._timed_llm_generate(model, prompt_w, task_w, max_new_tokens)
                if want_viz and pred_w is not None:
                    progress(0.12, desc="Rendering sanity-check video (not timed)…")
                    out_type = INTERHUMAN_TASK_OUTPUT_TYPE.get(task_w, "follower")
                    leader_override = (pred_w.get("leader_tokens") or []) if out_type == "leader" else None
                    follower_override = (pred_w.get("follower_tokens") or []) if out_type == "follower" else None
                    rel_override = (pred_w.get("relationship_tokens") or []) if out_type == "relationship" else None
                    _, _, pred_combined, _, _, _ = self.visualize_reconstruction_from_tokens(
                        idx_w,
                        use_continuous_concatenation=True,
                        use_actual_relation=True,
                        leader_tokens_override=leader_override,
                        follower_tokens_override=follower_override,
                        relationship_tokens_override=rel_override,
                        output_suffix="_latency",
                        use_mesh=False,
                    )
                    video_path = pred_combined
                cursor += 1
            elif want_viz:
                progress(0.12, desc="Sanity-check sample (excluded from average)…")
                idx_v, prompt_v, task_v = prompts[cursor]
                _, pred_v = self._timed_llm_generate(model, prompt_v, task_v, max_new_tokens)
                out_type = INTERHUMAN_TASK_OUTPUT_TYPE.get(task_v, "follower")
                leader_override = (pred_v.get("leader_tokens") or []) if out_type == "leader" else None
                follower_override = (pred_v.get("follower_tokens") or []) if out_type == "follower" else None
                rel_override = (pred_v.get("relationship_tokens") or []) if out_type == "relationship" else None
                _, _, pred_combined, _, _, _ = self.visualize_reconstruction_from_tokens(
                    idx_v,
                    use_continuous_concatenation=True,
                    use_actual_relation=True,
                    leader_tokens_override=leader_override,
                    follower_tokens_override=follower_override,
                    relationship_tokens_override=rel_override,
                    output_suffix="_latency",
                    use_mesh=False,
                )
                video_path = pred_combined
                cursor += 1

            times = []
            n_pred_tokens = []
            timed_indices = used_indices[cursor : cursor + n_samples]
            for i in range(n_samples):
                progress((i + 1) / (n_samples + 1), desc=f"Timed inference {i + 1}/{n_samples}")
                _idx, prompt_text, task_key = prompts[cursor + i]
                elapsed, pred_dict = self._timed_llm_generate(model, prompt_text, task_key, max_new_tokens)
                times.append(elapsed)
                out_type = INTERHUMAN_TASK_OUTPUT_TYPE.get(task_key, "follower")
                if out_type == "relationship":
                    toks = pred_dict.get("relationship_tokens") or []
                else:
                    toks = pred_dict.get(f"{out_type}_tokens") or []
                n_pred_tokens.append(len(toks))

            progress(1.0, desc="Done")
            arr = np.asarray(times, dtype=np.float64)
            n = int(arr.size)
            mean_s = float(arr.mean())
            std_s = float(arr.std(ddof=1)) if n > 1 else 0.0
            device = "cpu"
            if torch.cuda.is_available():
                try:
                    device = f"cuda ({torch.cuda.get_device_name(0)})"
                except Exception:
                    device = "cuda"
            ckpt_label = ckpt_path
            try:
                ckpt_label = os.path.relpath(os.path.abspath(ckpt_path), str(self.parent_dir))
            except ValueError:
                pass
            mean_tok = float(np.mean(n_pred_tokens)) if n_pred_tokens else 0.0
            lines = [
                "Latency — LLM inference only (generate_Payam_interhuman)",
                "Excluded: checkpoint load, dataset I/O, prompt construction, VQ decode, visualization/rendering.",
                "",
                f"Checkpoint:     {ckpt_label}",
                f"Device:         {device}",
                f"Task:           {task_choice} ({LLM_INFERENCE_TASK_MAP.get(task_choice, 'leader_rel_to_follower')})",
                f"max_new_tokens: {max_new_tokens}",
                f"Include audio:  {bool(include_audio_val)}",
                f"Include MS:     {bool(include_motionscript_val)}",
                f"MS first:       {bool(output_motionscript_first_val)}",
                f"Timed samples:  {n} (requested {n_samples})",
                f"Warmup:         {'yes (1 untimed run)' if warmup else 'no'}",
                f"Sanity video:   {'yes (excluded from average)' if video_path else 'no'}",
                f"Sample indices: {', '.join(str(i) for i in timed_indices)}",
                "",
                f"Mean:           {mean_s:.4f} s",
                f"Std:            {std_s:.4f} s",
                f"Min:            {float(arr.min()):.4f} s",
                f"Max:            {float(arr.max()):.4f} s",
                f"Total (timed):  {float(arr.sum()):.4f} s",
                f"Throughput:     {(n / float(arr.sum())) if float(arr.sum()) > 0 else 0.0:.4f} samples/s",
                f"Mean pred toks: {mean_tok:.1f}",
                "",
                f"Per-sample (s): {', '.join(f'{t:.4f}' for t in times)}",
                f"Per-sample toks:{', '.join(str(int(t)) for t in n_pred_tokens)}",
            ]
            return "\n".join(lines), video_path, ""
        except Exception as e:
            import traceback
            return "", video_path, f"{e}\n{traceback.format_exc()}"


def create_interface():
    """Create the Gradio interface with tabs."""
    app = InterHumanVisualizationApp()
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1600px;
    }
    .tab-nav {
        font-size: 14px;
    }
    """
    
    with gr.Blocks(title="Salsa Dataset Visualization - InterHuman") as demo:
        # Inject custom CSS
        gr.HTML(f"<style>{custom_css}</style>", visible=False)
        gr.Markdown("# 🕺 Salsa Dataset Visualization Tool - InterHuman Support")
        gr.Markdown("Load dataset and explore all stored data including InterHuman representation, tokens, and reconstructions.")
        
        # Dataset Loading Section
        with gr.Row():
            with gr.Column(scale=2):
                lmdb_path = gr.Textbox(
                    label="LMDB Directory Path",
                    value="dataset_processed_New/lmdb_Salsa_pair/lmdb_train",
                    placeholder="Enter path to LMDB directory (will auto-detect cache if available)",
                    info="Enter the base LMDB directory. The app will automatically look for processed cache directories (ending with '_cache' or '_cache_MDM'). If cache doesn't exist, you may need to run DataPreprocessor first."
                )
                is_MDM = gr.Checkbox(label="Is MDM Format", value=False)
                load_btn = gr.Button("Load LMDB", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False, lines=5)
            
            with gr.Column(scale=1):
                total_samples = gr.Number(label="Total Samples", value=0, interactive=False)
                sample_idx = gr.Number(
                    label="Sample Index",
                    value=0,
                    minimum=0,
                    maximum=0,
                    step=1,
                    precision=0
                )
        
        # Navigation
        with gr.Row():
            prev_btn = gr.Button("◀ Previous", size="sm")
            next_btn = gr.Button("Next ▶", size="sm")
            go_btn = gr.Button("Go to Index", size="sm")
        
        # Tabs for different sections
        with gr.Tabs() as tabs:
            # Tab 1: Data Summary
            with gr.Tab("📊 Data Summary"):
                data_summary = gr.Textbox(
                    label="Complete Data Summary",
                    lines=30,
                    interactive=False,
                    value="Load a dataset and select a sample to see data summary..."
                )
            
            # Tab 2: InterHuman Visualization
            with gr.Tab("🎭 InterHuman Visualization"):
                gr.Markdown("### Original InterHuman Motions (Canonicalized)")
                use_continuous_interhuman = gr.Checkbox(
                    label="Use Continuous Concatenation",
                    value=True,
                    info="Transform windows to flow continuously (for multi-window sequences). Unchecked: each window starts from origin (good for debugging)."
                )
                use_mesh_interhuman = gr.Checkbox(
                    label="Also produce mesh visualization (2-person SMPL)",
                    value=False,
                    info="Render a 2-person SMPL mesh video from joints. Requires priorMDM body_models and pyrender."
                )
                with gr.Row():
                    interhuman_leader_video = gr.Video(label="Leader (InterHuman)", scale=1)
                    interhuman_follower_video = gr.Video(label="Follower (Aligned)", scale=1)
                interhuman_combined_video = gr.Video(label="Together (Combined)", scale=1)
                interhuman_mesh_video = gr.Video(label="Mesh (2-person SMPL)", scale=1, visible=True)
                interhuman_info = gr.Textbox(label="Info", lines=10, interactive=False)
                interhuman_btn = gr.Button("Visualize InterHuman", variant="primary")
                gr.Markdown("### MotionScript: Motion Codes on Timeline")
                show_motionscript_timeline = gr.Checkbox(
                    label="Show MotionScript timeline (motion codes on timeline)",
                    value=False,
                    info="Generate timeline GIFs with motion codes for leader and follower (requires non-MDM cache with MotionScript)."
                )
                with gr.Row():
                    motionscript_timeline_leader = gr.Video(label="Leader MotionScript Timeline", scale=1)
                    motionscript_timeline_follower = gr.Video(label="Follower MotionScript Timeline", scale=1)
                motionscript_timeline_info = gr.Textbox(label="MotionScript Timeline status / errors", lines=4, interactive=False)
                motionscript_timeline_btn = gr.Button("Generate MotionScript Timeline", variant="secondary")
            
            # Tab 3: Token Reconstruction
            with gr.Tab("🔄 Token Reconstruction"):
                gr.Markdown("### Reconstructed Motion from Tokens")
                use_continuous_recon = gr.Checkbox(
                    label="Use Continuous Concatenation",
                    value=True,
                    info="Transform windows to flow continuously (for multi-window sequences). Unchecked: each window starts from origin (good for debugging)."
                )
                use_actual_relation = gr.Checkbox(
                    label="Use Actual Relation (from reconstructed motions)",
                    value=False,
                    info="Compute relationship input for the next window from the last reconstructed leader/follower frame (world space)."
                )
                use_canonical_seed = gr.Checkbox(
                    label="Canonicalize boundary seed frame",
                    value=False,
                    info="Before passing the last frame of window W as conditioning for window W+1's decoder, move it to canonical (root → origin, yaw → 0). Reduces backward-motion artifacts at window boundaries."
                )
                with gr.Row():
                    smooth_boundaries = gr.Checkbox(
                        label="Smooth window boundaries",
                        value=False,
                        info="Apply a Gaussian filter over a small neighbourhood of frames around each window boundary to reduce visible jumps."
                    )
                    smooth_boundary_half_kernel = gr.Slider(
                        minimum=1, maximum=10, step=1, value=4,
                        label="Smoothing half-kernel (frames each side)",
                        info="Number of frames on each side of each boundary to include in the smoothing window."
                    )
                use_mesh_recon = gr.Checkbox(
                    label="Also produce mesh visualization (2-person SMPL)",
                    value=False,
                    info="Render a 2-person SMPL mesh video from reconstructed joints. Requires priorMDM body_models and pyrender."
                )
                with gr.Row():
                    recon_leader_video = gr.Video(label="Leader Reconstructed", scale=1)
                    recon_follower_video = gr.Video(label="Follower Reconstructed", scale=1)
                    recon_combined_video = gr.Video(label="Combined Reconstructed", scale=1)
                recon_mesh_video = gr.Video(label="Mesh (2-person SMPL)", scale=1, visible=True)
                relationship_plot = gr.Plot(label="Relationship Features (GT vs Reconstructed)")
                recon_info = gr.Textbox(label="Reconstruction Info", lines=10, interactive=False)
                recon_btn = gr.Button("Reconstruct from Tokens", variant="primary")
            
            # Tab 4: Audio Comparison
            with gr.Tab("🎵 Audio Comparison"):
                gr.Markdown("### Ground Truth vs Wavetokenizer Decoded Audio")
                with gr.Row():
                    audio_gt = gr.Audio(label="Ground Truth Audio", type="filepath", scale=1)
                    audio_decoded = gr.Audio(label="Wavetokenizer Decoded Audio", type="filepath", scale=1)
                audio_waveform_plot = gr.Plot(label="Waveform Comparison")
                audio_info = gr.Textbox(label="Audio Info", lines=15, interactive=False)
                audio_btn = gr.Button("Compare Audio", variant="primary")
            
            # Tab 5: Metadata
            with gr.Tab("📋 Metadata"):
                with gr.Tabs() as metadata_tabs:
                    # Sub-tab 1: Current Sample Metadata
                    with gr.Tab("📊 Current Sample"):
                        gr.Markdown("### Sample Metadata and Annotations")
                        with gr.Row():
                            with gr.Column(scale=1):
                                metadata_basic_info = gr.Textbox(
                                    label="Basic Information",
                                    lines=10,
                                    interactive=False,
                                    value="Load a sample to see metadata..."
                                )
                                metadata_level_badge = gr.HTML(label="Proficiency Level")
                            with gr.Column(scale=1):
                                metadata_moves = gr.Dataframe(
                                    label="Dance Moves in Window",
                                    headers=["Move", "Start", "End", "Duration"],
                                    interactive=False
                                )
                                metadata_errors = gr.Dataframe(
                                    label="Errors in Window",
                                    headers=["Error", "Start", "End"],
                                    interactive=False
                                )
                        with gr.Row():
                            metadata_styling = gr.Dataframe(
                                label="Styling Segments",
                                headers=["Role", "Description", "Start", "End"],
                                interactive=False
                            )
                        with gr.Row():
                            metadata_video_overlay = gr.Video(
                                label="Video with Metadata Overlay (Level, Role, Moves)",
                                scale=1
                            )
                        with gr.Row():
                            metadata_timeline_plot = gr.Plot(
                                label="Timeline: Moves, Errors, and Styling"
                            )
                        with gr.Row():
                            metadata_info_text = gr.Textbox(
                                label="Detailed Metadata",
                                lines=10,
                                interactive=False
                            )
                        metadata_btn = gr.Button("Load Metadata", variant="primary")
                    
                    # Sub-tab 2: Statistics
                    with gr.Tab("📈 Statistics"):
                        gr.Markdown("### Dataset Statistics")
                        with gr.Row():
                            stats_levels_plot = gr.Plot(label="Distribution by Proficiency Level")
                            stats_pairs_plot = gr.Plot(label="Distribution by Pair")
                        with gr.Row():
                            stats_songs_plot = gr.Plot(label="Distribution by Song")
                            stats_summary = gr.Textbox(
                                label="Statistics Summary",
                                lines=12,
                                interactive=False
                            )
                        gr.Markdown("### Annotation Statistics")
                        with gr.Row():
                            stats_annotation_coverage_plot = gr.Plot(
                                label="Annotation Coverage (Samples With vs Without Annotations)"
                            )
                            stats_move_class_plot = gr.Plot(
                                label="Move Class Distribution"
                            )
                        with gr.Row():
                            stats_annotation_summary = gr.Textbox(
                                label="Annotation Stats (Load/Assign, Unclassified, Exceptions)",
                                lines=18,
                                interactive=False
                            )
                        gr.Markdown("### Token Similarity Analysis by Class")
                        with gr.Row():
                            stats_token_analysis_plot = gr.Plot(
                                label="Token Frequency by Class (Top Tokens)"
                            )
                            stats_token_entropy_plot = gr.Plot(
                                label="Token Diversity (Entropy) by Class"
                            )
                        with gr.Row():
                            stats_token_summary = gr.Textbox(
                                label="Token Analysis Summary",
                                lines=15,
                                interactive=False
                            )
                        gr.Markdown("### Inter-Class vs Intra-Class Similarity Comparison")
                        with gr.Row():
                            stats_inter_intra_comparison_plot = gr.Plot(
                                label="Intra-Class vs Inter-Class Similarity (All Classes)"
                            )
                            stats_class_similarity_radar = gr.Plot(
                                label="Similarity Metrics Comparison Across Classes"
                            )
                        gr.Markdown("### Metric Comparison Across Classes")
                        with gr.Row():
                            with gr.Column(scale=1):
                                stats_metric_nav_prev = gr.Button("◀ Previous Metric", variant="secondary")
                                stats_metric_dropdown = gr.Dropdown(
                                    label="Select Metric",
                                    choices=["Jaccard Similarity", "BLEU", "ROUGE-1", "Edit Distance", "Hamming Distance"],
                                    value="Jaccard Similarity",
                                    interactive=True
                                )
                                stats_metric_nav_next = gr.Button("Next Metric ▶", variant="secondary")
                                stats_metric_description = gr.Textbox(
                                    label="Metric Description",
                                    lines=3,
                                    interactive=False
                                )
                            with gr.Column(scale=3):
                                stats_metric_comparison_plot = gr.Plot(
                                    label="Metric Comparison Across Classes"
                                )
                        gr.Markdown("### Radar Chart: All Metrics Across All Classes")
                        with gr.Row():
                            stats_radar_chart = gr.Plot(
                                label="Radar Chart: All Metrics for All Classes"
                            )
                        gr.Markdown("### Class-Specific Token Analysis")
                        with gr.Row():
                            stats_class_dropdown = gr.Dropdown(
                                label="Select Class",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                        with gr.Row():
                            stats_class_token_dist_plot = gr.Plot(
                                label="Token Distribution for Selected Class"
                            )
                            stats_class_similarity_plot = gr.Plot(
                                label="Intra-Class Similarity Metrics for Selected Class"
                            )
                        with gr.Row():
                            stats_class_metrics_text = gr.Textbox(
                                label="Similarity Metrics Explanation",
                                lines=20,
                                interactive=False
                            )
                        stats_btn = gr.Button("Compute Statistics", variant="primary")
                        gr.Markdown("### MotionScript motioncode statistics")
                        gr.Markdown(
                            "Run MotionScript statistical analysis on the loaded dataset (leader motions). "
                            "Shows distribution of motioncode types and threshold/classification stats (helps check if thresholds are appropriate). "
                            "Uses up to the specified number of samples; requires non-MDM cache."
                        )
                        with gr.Row():
                            motionscript_stat_max_samples = gr.Number(
                                label="Max samples to analyze",
                                value=50,
                                minimum=1,
                                maximum=500,
                                step=1,
                                interactive=True
                            )
                            motionscript_stat_btn = gr.Button("Run MotionScript motioncode analysis", variant="secondary")
                        motionscript_stat_output = gr.Textbox(
                            label="MotionScript statistics result",
                            lines=25,
                            interactive=False,
                            value="Click the button to run analysis..."
                        )
                    
                    # Sub-tab 3: Search & Filter
                    with gr.Tab("🔍 Search & Filter"):
                        gr.Markdown("### Search Samples by Criteria")
                        with gr.Row():
                            with gr.Column(scale=1):
                                search_level = gr.Dropdown(
                                    label="Proficiency Level",
                                    choices=["All", "beginner", "intermediate", "professional"],
                                    value="All",
                                    interactive=True
                                )
                                search_pair = gr.Dropdown(
                                    label="Pair",
                                    choices=["All", "Pair1", "Pair2", "Pair3", "Pair4", "Pair5", "Pair6", "Pair7", "Pair8", "Pair9"],
                                    value="All",
                                    interactive=True
                                )
                                search_song = gr.Dropdown(
                                    label="Song",
                                    choices=["All", "Song1", "Song2", "Song3", "Song4"],
                                    value="All",
                                    interactive=True
                                )
                            with gr.Column(scale=1):
                                search_move_class = gr.Dropdown(
                                    label="Move Class",
                                    choices=["All"],
                                    value="All",
                                    interactive=True
                                )
                                search_has_errors = gr.Checkbox(
                                    label="Has Errors",
                                    value=False
                                )
                                search_has_styling = gr.Checkbox(
                                    label="Has Styling",
                                    value=False
                                )
                        search_btn = gr.Button("Search", variant="primary")
                        search_results = gr.Dataframe(
                            label="Matching Samples",
                            headers=["Index", "Pair", "Song", "Take", "Level", "Moves", "Errors"],
                            interactive=True,
                            type="pandas"
                        )
                        with gr.Row():
                            search_selected_idx = gr.Number(
                                label="Selected Sample Index",
                                value=0,
                                precision=0,
                                interactive=True
                            )
                            search_visualize_btn = gr.Button("Visualize Selected Sample", variant="primary")
                        search_video = gr.Video(label="Visualization")
                        search_info = gr.Textbox(label="Sample Info", lines=5, interactive=False)
            
            # Tab 6: Example Prompts (InterHuman pipeline)
            with gr.Tab("📝 Example-Prompts"):
                gr.Markdown("### InterHuman prompt preview – use the **Sample Index** at the top, pick a task, then click Generate.")
                with gr.Row():
                    prompts_task = gr.Dropdown(
                        label="Task",
                        choices=[
                            "Leader + Rel → Follower",
                            "Follower + Rel → Leader",
                            "Caption + Leader + Rel → Follower",
                            "Caption + Follower + Rel → Leader",
                            "Pair (Leader+Follower) → Relationship",
                            "Caption → Leader",
                            "Caption → Follower",
                            "Leader → Follower",
                            "Follower → Leader",
                            "Motion completion (Leader)",
                            "Motion completion (Follower)",
                            "Leader motion → Leader MotionScript",
                            "Follower motion → Follower MotionScript",
                            "Leader MotionScript → Leader motion",
                            "Follower MotionScript → Follower motion",
                            "Caption → Leader MotionScript",
                            "Caption → Follower MotionScript",
                            "Caption → Both MotionScripts",
                            "Leader MotionScript + Rel → Follower MotionScript",
                            "Follower MotionScript + Rel → Leader MotionScript",
                            "MotionScript completion (Leader)",
                            "MotionScript completion (Follower)",
                            "Caption + Leader MotionScript → Follower MotionScript",
                            "Caption + Follower MotionScript → Leader MotionScript",
                        ],
                        value="Leader + Rel → Follower"
                    )
                    prompts_include_audio = gr.Checkbox(label="Include audio", value=False)
                    prompts_include_motionscript = gr.Checkbox(label="Include MotionScript", value=False)
                    prompts_output_motionscript_first = gr.Checkbox(
                        label="Output MotionScript first",
                        value=False,
                        info="For leader/follower tasks: predict MotionScript then motion tokens (only when MotionScript is available)."
                    )
                    prompts_btn = gr.Button("Generate Prompts", variant="primary")
                with gr.Row():
                    prompts_prompt_text = gr.Textbox(
                        label="Prompt (input)",
                        lines=16,
                        interactive=False
                    )
                    prompts_target_text = gr.Textbox(
                        label="Target (model output)",
                        lines=8,
                        interactive=False
                    )
                prompts_raw_info = gr.Textbox(
                    label="Raw info",
                    lines=3,
                    interactive=False,
                    value=""
                )
            
            # Tab 8: Legacy Visualization
            # Tab 7: LLM-Inference
            with gr.Tab("LLM-Inference"):
                gr.Markdown("### Run LLM inference – compare predicted vs ground truth.")
                with gr.Row():
                    llm_task = gr.Dropdown(
                        label="Task",
                        choices=LLM_INFERENCE_TASK_CHOICES,
                        value="Leader + Rel to Follower"
                    )
                    llm_include_audio = gr.Checkbox(label="Include audio", value=False)
                    llm_include_motionscript = gr.Checkbox(label="Include MotionScript", value=False)
                    llm_output_motionscript_first = gr.Checkbox(
                        label="Output MotionScript first",
                        value=False,
                        info="For leader/follower tasks: predict MotionScript then motion (only when MotionScript is included)."
                    )
                use_mesh_llm = gr.Checkbox(
                    label="Also produce mesh visualization (2-person SMPL)",
                    value=False,
                    info="Render 2-person SMPL mesh for predicted and ground truth. Requires priorMDM body_models and pyrender."
                )
                llm_ckpt = gr.Textbox(label="LLM checkpoint path", value="output_trained/pretrain_all/Xmotionllm_epoch10.pth")
                llm_run_btn = gr.Button("Run LLM Inference", variant="primary")
                gr.Markdown("Prompts")
                with gr.Row():
                    llm_prompt_text = gr.Textbox(label="Input prompt", lines=12, interactive=False)
                    llm_gt_target = gr.Textbox(label="Ground truth target", lines=8, interactive=False)
                    llm_pred_target = gr.Textbox(label="Predicted target", lines=8, interactive=False)
                gr.Markdown("Side-by-side: Predicted vs Ground truth")
                with gr.Row():
                    llm_pred_video = gr.Video(label="Predicted pair", scale=1)
                    llm_gt_video = gr.Video(label="Ground truth pair", scale=1)
                with gr.Row():
                    llm_pred_mesh_video = gr.Video(label="Predicted mesh (2-person SMPL)", scale=1)
                    llm_gt_mesh_video = gr.Video(label="Ground truth mesh (2-person SMPL)", scale=1)
                llm_info = gr.Textbox(label="Info", lines=8, interactive=False)

            with gr.Tab("Latency"):
                gr.Markdown(
                    "Measure **LLM inference latency** (`generate_Payam_interhuman` only, CUDA-synced). "
                    "**Not included:** checkpoint load, dataset I/O, prompt construction, VQ-VAE decode, or video rendering. "
                    "Load LMDB first. Timed runs **cycle through samples** from the current index. "
                    "Warmup and the optional sanity-check video are extra runs and are **excluded** from the average. "
                    "`max_new_tokens` defaults to **150** (same as the LLM-Inference tab)."
                )
                lat_task = gr.Dropdown(
                    label="Task",
                    choices=LLM_INFERENCE_TASK_CHOICES,
                    value="Leader + Rel to Follower",
                )
                with gr.Row():
                    lat_include_audio = gr.Checkbox(label="Include audio", value=False)
                    lat_include_motionscript = gr.Checkbox(label="Include MotionScript", value=False)
                    lat_output_motionscript_first = gr.Checkbox(
                        label="Output MotionScript first",
                        value=False,
                    )
                lat_ckpt = gr.Textbox(
                    label="LLM checkpoint path",
                    value="output_trained/pretrain_all/Xmotionllm_epoch10.pth",
                )
                with gr.Row():
                    lat_n = gr.Number(value=10, label="Number of timed samples", precision=0)
                    lat_max_tokens = gr.Number(
                        value=150,
                        label="max_new_tokens",
                        precision=0,
                        info="LLM generation budget (default 150, same as LLM-Inference).",
                    )
                with gr.Row():
                    lat_warmup = gr.Checkbox(
                        value=True,
                        label="Warmup (1 extra untimed run; recommended so CUDA kernels are compiled before timing)",
                    )
                    lat_viz = gr.Checkbox(
                        value=False,
                        label="Visualize one sample for sanity check (excluded from time calculation)",
                    )
                lat_btn = gr.Button("Measure latency", variant="primary")
                lat_report = gr.Textbox(label="Latency report", interactive=False, lines=24)
                lat_video = gr.Video(label="Sanity-check predicted pair (not timed)")
                lat_status = gr.Textbox(label="Message", interactive=False)

            with gr.Tab("📹 Legacy (HumanML3D)"):
                gr.Markdown("### Original HumanML3D Visualization")
                with gr.Row():
                    show_leader = gr.Checkbox(label="Show Leader", value=True)
                    show_follower = gr.Checkbox(label="Show Follower", value=True)
                    show_combined = gr.Checkbox(label="Show Combined", value=False)
                with gr.Row():
                    legacy_leader_video = gr.Video(label="Leader", scale=1)
                    legacy_follower_video = gr.Video(label="Follower", scale=1)
                    legacy_combined_video = gr.Video(label="Combined", scale=1)
                legacy_info = gr.Textbox(label="Info", lines=5, interactive=False)
                legacy_btn = gr.Button("Visualize Legacy Format", variant="primary")
        
        # Event handlers
        def on_load(lmdb_path_val, is_MDM_val):
            try:
                status, idx, total = app.load_lmdb(lmdb_path_val, is_MDM_val)
                max_val = max(0, int(total) - 1) if total > 0 else 0
                return status, total, gr.update(maximum=max_val, value=0)
            except Exception as e:
                import traceback
                error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
                return error_msg, 0, gr.update(maximum=0, value=0)
        
        def on_navigate(direction, current_idx_val, total_samples_val):
            try:
                current = int(current_idx_val) if current_idx_val is not None else 0
                total = int(total_samples_val) if total_samples_val is not None else 0
                if direction == 'next':
                    new_idx = min(current + 1, total - 1)
                else:
                    new_idx = max(current - 1, 0)
                app.current_idx = new_idx
                return new_idx
            except:
                return 0
        
        def on_data_summary(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app.get_sample_data_summary(idx)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def on_interhuman_visualize(idx_val, use_continuous_val, use_mesh_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                use_continuous = bool(use_continuous_val) if use_continuous_val is not None else True
                use_mesh = bool(use_mesh_val) if use_mesh_val is not None else False
                return app.visualize_interhuman_pair(idx, use_continuous_concatenation=use_continuous, use_mesh=use_mesh)
            except Exception as e:
                import traceback
                return None, None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        def on_reconstruct(idx_val, use_continuous_val, use_actual_relation_val, use_canonical_seed_val, smooth_val, smooth_kernel_val, use_mesh_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                use_continuous = bool(use_continuous_val) if use_continuous_val is not None else False
                use_actual = bool(use_actual_relation_val) if use_actual_relation_val is not None else False
                use_can_seed = bool(use_canonical_seed_val) if use_canonical_seed_val is not None else False
                do_smooth = bool(smooth_val) if smooth_val is not None else False
                half_kernel = int(smooth_kernel_val) if smooth_kernel_val is not None else 4
                use_mesh = bool(use_mesh_val) if use_mesh_val is not None else False
                return app.visualize_reconstruction_from_tokens(
                    idx,
                    use_continuous_concatenation=use_continuous,
                    use_actual_relation=use_actual,
                    use_canonical_seed=use_can_seed,
                    smooth_boundaries=do_smooth,
                    smooth_boundary_half_kernel=half_kernel,
                    use_mesh=use_mesh,
                )
            except Exception as e:
                import traceback
                return None, None, None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        # Wire up events
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
        
        go_btn.click(fn=on_data_summary, inputs=[sample_idx], outputs=[data_summary])
        
        # Auto-update data summary when index changes
        sample_idx.change(fn=on_data_summary, inputs=[sample_idx], outputs=[data_summary])
        
        # InterHuman visualization
        interhuman_btn.click(
            fn=on_interhuman_visualize,
            inputs=[sample_idx, use_continuous_interhuman, use_mesh_interhuman],
            outputs=[interhuman_leader_video, interhuman_follower_video, interhuman_combined_video, interhuman_mesh_video, interhuman_info]
        )

        # MotionScript timeline (motion codes on timeline)
        def on_motionscript_timeline(idx_val, show_timeline_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app._generate_motionscript_timeline_gifs(idx, bool(show_timeline_val))
            except Exception as e:
                import traceback
                err = f"MotionScript timeline error: {e}\n{traceback.format_exc()}"
                debug_print(err)
                return None, None, err

        motionscript_timeline_btn.click(
            fn=on_motionscript_timeline,
            inputs=[sample_idx, show_motionscript_timeline],
            outputs=[motionscript_timeline_leader, motionscript_timeline_follower, motionscript_timeline_info]
        )
        
        # Reconstruction
        recon_btn.click(
            fn=on_reconstruct,
            inputs=[sample_idx, use_continuous_recon, use_actual_relation, use_canonical_seed, smooth_boundaries, smooth_boundary_half_kernel, use_mesh_recon],
            outputs=[recon_leader_video, recon_follower_video, recon_combined_video, relationship_plot, recon_mesh_video, recon_info]
        )
        
        # Audio comparison
        def on_audio_compare(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app.visualize_audio_comparison(idx)
            except Exception as e:
                import traceback
                return None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        audio_btn.click(
            fn=on_audio_compare,
            inputs=[sample_idx],
            outputs=[audio_gt, audio_decoded, audio_waveform_plot, audio_info]
        )
        
        # Example-Prompts: build InterHuman prompt/target text for current sample and task
        def on_prompts_generate(idx_val, task_choice, include_audio_val, include_motionscript_val, output_motionscript_first_val):
            try:
                from models.training_utils import build_prompt_interhuman_salsa
                idx = int(idx_val) if idx_val is not None else 0
                sample = app._get_sample_from_dataset(idx)
                interhuman_data = sample.get("interhuman_data")
                if interhuman_data is None:
                    return "", "", "No InterHuman data for this sample (cache may lack it)."
                leader_tokens = interhuman_data.get("leader_tokens")
                follower_tokens = interhuman_data.get("follower_tokens")
                relationship_tokens = interhuman_data.get("relationship_tokens")
                if leader_tokens is None or follower_tokens is None or relationship_tokens is None:
                    return "", "", "Sample missing leader_tokens, follower_tokens, or relationship_tokens."
                leader_tokens = np.asarray(leader_tokens).ravel().tolist()
                follower_tokens = np.asarray(follower_tokens).ravel().tolist()
                relationship_tokens = np.asarray(relationship_tokens).ravel().tolist()
                audio_tokens = sample.get("audio_tokens")
                if audio_tokens is not None:
                    audio_tokens = np.asarray(audio_tokens).ravel().tolist()
                ms_L = sample.get("ms_desc_L")
                ms_F = sample.get("ms_des_F")
                if ms_L is not None and not isinstance(ms_L, str):
                    ms_L = " --> ".join(str(x) for x in ms_L) if (isinstance(ms_L, (list, tuple)) and ms_L) else ""
                if ms_F is not None and not isinstance(ms_F, str):
                    ms_F = " --> ".join(str(x) for x in ms_F) if (isinstance(ms_F, (list, tuple)) and ms_F) else ""
                ms_L = (ms_L or "").strip() if isinstance(ms_L, str) else ""
                ms_F = (ms_F or "").strip() if isinstance(ms_F, str) else ""
                task_map = {
                    "Leader + Rel → Follower": "leader_rel_to_follower",
                    "Follower + Rel → Leader": "follower_rel_to_leader",
                    "Caption + Leader + Rel → Follower": "caption_leader_rel_to_follower",
                    "Caption + Follower + Rel → Leader": "caption_follower_rel_to_leader",
                    "Pair (Leader+Follower) → Relationship": "pair_to_relationship",
                    "Caption → Leader": "caption_to_leader",
                    "Caption → Follower": "caption_to_follower",
                    "Leader → Follower": "leader_to_follower",
                    "Follower → Leader": "follower_to_leader",
                    "Motion completion (Leader)": "motion_completion_leader",
                    "Motion completion (Follower)": "motion_completion_follower",
                    "Leader motion → Leader MotionScript": "leader_motion_to_motionscript",
                    "Follower motion → Follower MotionScript": "follower_motion_to_motionscript",
                    "Leader MotionScript → Leader motion": "motionscript_to_leader_motion",
                    "Follower MotionScript → Follower motion": "motionscript_to_follower_motion",
                    "Caption → Leader MotionScript": "caption_to_leader_motionscript",
                    "Caption → Follower MotionScript": "caption_to_follower_motionscript",
                    "Caption → Both MotionScripts": "caption_to_both_motionscripts",
                    "Leader MotionScript + Rel → Follower MotionScript": "leader_motionscript_rel_to_follower_motionscript",
                    "Follower MotionScript + Rel → Leader MotionScript": "follower_motionscript_rel_to_leader_motionscript",
                    "MotionScript completion (Leader)": "motionscript_completion_leader",
                    "MotionScript completion (Follower)": "motionscript_completion_follower",
                    "Caption + Leader MotionScript → Follower MotionScript": "caption_leader_motionscript_to_follower_motionscript",
                    "Caption + Follower MotionScript → Leader MotionScript": "caption_follower_motionscript_to_leader_motionscript",
                }
                task_key = task_map.get(task_choice, "leader_rel_to_follower")
                metadata = app.get_metadata_info(idx)
                move_annotations = None
                level = None
                if metadata and "error" not in metadata:
                    move_annotations = metadata.get("moves", [])
                    level = metadata.get("level")
                prompt_text, target_text = build_prompt_interhuman_salsa(
                    leader_tokens=leader_tokens,
                    follower_tokens=follower_tokens,
                    relationship_tokens=relationship_tokens,
                    task=task_key,
                    move_annotations=move_annotations,
                    level=level,
                    audio_tokens=audio_tokens,
                    include_audio=bool(include_audio_val),
                    include_motionscript=bool(include_motionscript_val),
                    motionscript_leader=ms_L or None,
                    motionscript_follower=ms_F or None,
                    output_motionscript_first=bool(output_motionscript_first_val),
                )
                raw_info = f"Sample {idx} | Leader tokens: {len(leader_tokens)}, Rel: {len(relationship_tokens)}, Follower: {len(follower_tokens)}"
                if include_audio_val and audio_tokens:
                    raw_info += f" | Audio tokens: {len(audio_tokens)}"
                if include_motionscript_val and (ms_L or ms_F):
                    raw_info += " | MotionScript included"
                if output_motionscript_first_val and (ms_L or ms_F):
                    raw_info += " | Output: MotionScript first, then motion"
                return prompt_text, target_text, raw_info
            except Exception as e:
                import traceback
                err = f"Error: {str(e)}\n{traceback.format_exc()}"
                return "", "", err

        prompts_btn.click(
            fn=on_prompts_generate,
            inputs=[sample_idx, prompts_task, prompts_include_audio, prompts_include_motionscript, prompts_output_motionscript_first],
            outputs=[prompts_prompt_text, prompts_target_text, prompts_raw_info]
        )

        # LLM-Inference tab handler (tab UI must be added above Legacy tab)
        def on_llm_inference(idx_val, task_choice, include_audio_val, include_motionscript_val, output_motionscript_first_val, use_mesh_val, ckpt_path):
            try:
                from models.training_utils import (
                    build_prompt_interhuman_salsa,
                    INTERHUMAN_PROMPT_DELIMITERS,
                    INTERHUMAN_TASK_OUTPUT_TYPE,
                )
                from models.mllm import MotionLLM
                from options.option_llm import get_args_parser
                idx = int(idx_val) if idx_val is not None else 0
                use_mesh = bool(use_mesh_val) if use_mesh_val is not None else False
                sample = app._get_sample_from_dataset(idx)
                interhuman_data = sample.get("interhuman_data")
                if interhuman_data is None:
                    return "", "", "", None, None, None, None, "No InterHuman data for this sample."
                leader_tokens = np.asarray(interhuman_data.get("leader_tokens")).ravel().tolist()
                follower_tokens = np.asarray(interhuman_data.get("follower_tokens")).ravel().tolist()
                relationship_tokens = np.asarray(interhuman_data.get("relationship_tokens")).ravel().tolist()
                audio_tokens = sample.get("audio_tokens")
                if audio_tokens is not None:
                    audio_tokens = np.asarray(audio_tokens).ravel().tolist()
                task_key = LLM_INFERENCE_TASK_MAP.get(task_choice, "leader_rel_to_follower")
                metadata = app.get_metadata_info(idx)
                move_annotations = metadata.get("moves", []) if metadata and "error" not in metadata else []
                level = metadata.get("level") if metadata and "error" not in metadata else None
                caption = metadata.get("caption") if metadata and "error" not in metadata else None
                ms_L = sample.get("ms_desc_L")
                ms_F = sample.get("ms_des_F")
                if ms_L is not None and not isinstance(ms_L, str):
                    ms_L = " --> ".join(str(x) for x in ms_L) if (isinstance(ms_L, (list, tuple)) and ms_L) else ""
                if ms_F is not None and not isinstance(ms_F, str):
                    ms_F = " --> ".join(str(x) for x in ms_F) if (isinstance(ms_F, (list, tuple)) and ms_F) else ""
                ms_L = (ms_L or "").strip() if isinstance(ms_L, str) else ""
                ms_F = (ms_F or "").strip() if isinstance(ms_F, str) else ""
                args = get_args_parser()
                args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                args.motion_repr_type = "interhuman"
                if not ckpt_path or not os.path.isfile(ckpt_path):
                    return "", "", "", None, None, None, None, f"Checkpoint not found: {ckpt_path}"
                ckpt_config = MotionLLM.load_config_from_checkpoint(ckpt_path)
                args.include_audio = ckpt_config.get("include_audio", False)
                args.include_motionscript = ckpt_config.get("include_motionscript", False)
                model = MotionLLM(args)
                model.load_model(ckpt_path)
                model.llm.eval()
                # Build prompt only with modalities the model was trained with (from checkpoint)
                include_ms = bool(include_motionscript_val) and bool(ms_L or ms_F) and model.include_motionscript
                effective_include_audio = bool(include_audio_val) and model.include_audio
                prompt_text, gt_target_text = build_prompt_interhuman_salsa(
                    leader_tokens=leader_tokens,
                    follower_tokens=follower_tokens,
                    relationship_tokens=relationship_tokens,
                    task=task_key,
                    move_annotations=move_annotations,
                    level=level,
                    caption=caption,
                    audio_tokens=audio_tokens,
                    include_audio=effective_include_audio,
                    include_motionscript=include_ms,
                    motionscript_leader=(ms_L or None) if include_ms else None,
                    motionscript_follower=(ms_F or None) if include_ms else None,
                    output_motionscript_first=bool(output_motionscript_first_val),
                )
                pred_dict = model.generate_Payam_interhuman(prompt_text, task_key, max_new_tokens=150)
                D = INTERHUMAN_PROMPT_DELIMITERS
                out_type = INTERHUMAN_TASK_OUTPUT_TYPE.get(task_key, "follower")
                if out_type == "relationship":
                    pred_tokens = pred_dict.get("relationship_tokens") or []
                    pred_target_str = " ".join(D["rel_token"].format(t) for t in pred_tokens) + " " + D["relationship_close"]
                else:
                    pred_tokens = pred_dict.get(f"{out_type}_tokens") or []
                    label = "leader_motion_close" if out_type == "leader" else "follower_motion_close"
                    pred_target_str = " ".join(D["ih_token"].format(t) for t in pred_tokens) + " " + D[label]
                leader_override = (pred_dict.get("leader_tokens") or []) if out_type == "leader" else None
                follower_override = (pred_dict.get("follower_tokens") or []) if out_type == "follower" else None
                rel_override = (pred_dict.get("relationship_tokens") or []) if out_type == "relationship" else None
                # Match Token Reconstruction tab: continuous concatenation + actual relation for correct pair alignment
                _, _, pred_combined, _, pred_mesh_path, pred_info = app.visualize_reconstruction_from_tokens(
                    idx,
                    use_continuous_concatenation=True,
                    use_actual_relation=True,
                    leader_tokens_override=leader_override,
                    follower_tokens_override=follower_override,
                    relationship_tokens_override=rel_override,
                    output_suffix="_pred",
                    use_mesh=use_mesh,
                )
                _, _, gt_combined, _, gt_mesh_path, gt_info = app.visualize_reconstruction_from_tokens(
                    idx,
                    use_continuous_concatenation=True,
                    use_actual_relation=True,
                    output_suffix="_gt",
                    use_mesh=use_mesh,
                )
                info = f"Task: {task_key}\nPredicted {len(pred_tokens)} tokens.\n{pred_info}\n---\n{gt_info}"
                return prompt_text, gt_target_text, pred_target_str, pred_combined, gt_combined, pred_mesh_path, gt_mesh_path, info
            except Exception as e:
                import traceback
                return "", "", "", None, None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"

        llm_run_btn.click(
            fn=on_llm_inference,
            inputs=[sample_idx, llm_task, llm_include_audio, llm_include_motionscript, llm_output_motionscript_first, use_mesh_llm, llm_ckpt],
            outputs=[llm_prompt_text, llm_gt_target, llm_pred_target, llm_pred_video, llm_gt_video, llm_pred_mesh_video, llm_gt_mesh_video, llm_info]
        )

        def do_latency(
            idx_val, task_choice, include_audio_val, include_motionscript_val,
            output_motionscript_first_val, ckpt_path, n, max_tok, viz, warm,
            progress=gr.Progress(),
        ):
            report, video, err = app.run_llm_latency_benchmark(
                idx_val,
                task_choice,
                include_audio_val,
                include_motionscript_val,
                output_motionscript_first_val,
                ckpt_path,
                n,
                max_tok,
                viz,
                warm,
                progress=progress,
            )
            return report, video, err or ""

        lat_btn.click(
            fn=do_latency,
            inputs=[
                sample_idx,
                lat_task,
                lat_include_audio,
                lat_include_motionscript,
                lat_output_motionscript_first,
                lat_ckpt,
                lat_n,
                lat_max_tokens,
                lat_viz,
                lat_warmup,
            ],
            outputs=[lat_report, lat_video, lat_status],
        )
        
        # Metadata visualization
        def on_metadata_load(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                metadata = app.get_metadata_info(idx)
                
                if 'error' in metadata:
                    error_html = f"<div style='color: red;'>{metadata['error']}</div>"
                    return (
                        metadata['error'],
                        error_html,
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        None,
                        None,
                        metadata['error']
                    )
                
                # Basic info
                basic_info = f"Pair: {metadata.get('pair', 'Unknown')}\n"
                basic_info += f"Song: {metadata.get('song', 'Unknown')}\n"
                basic_info += f"Take: {metadata.get('take', 'Unknown')}\n"
                basic_info += f"Level: {metadata.get('level', 'Unknown')}\n"
                basic_info += f"Split: {metadata.get('split', 'Unknown')}\n"
                basic_info += f"Duration: {metadata.get('duration', 0):.2f}s\n"
                basic_info += f"Frame Range: {metadata.get('start_frame', 0)} - {metadata.get('end_frame', 0)}\n"
                basic_info += f"Time Range: {metadata.get('start_time', 0):.2f}s - {metadata.get('end_time', 0):.2f}s\n"
                basic_info += f"Annotations: {'Loaded' if metadata.get('annotations_loaded') else 'Not found'}"
                
                # Level badge
                level = metadata.get('level', 'unknown')
                level_colors = {
                    'beginner': '#4CAF50',
                    'intermediate': '#FF9800',
                    'professional': '#F44336'
                }
                color = level_colors.get(level, '#9E9E9E')
                level_badge = f"<div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 18px;'>{level.upper()}</div>"
                
                # Moves dataframe: Class, Description (full), Start, End, Duration
                moves_data = []
                for move in metadata.get('moves', []):
                    mc = move.get('move_class') or ''
                    mc_display = LABEL_TO_README.get(mc, mc)
                    desc = move.get('description', '') or ''
                    moves_data.append([
                        mc_display,
                        desc,
                        f"{move.get('overlap_start', 0):.2f}s",
                        f"{move.get('overlap_end', 0):.2f}s",
                        f"{move.get('overlap_end', 0) - move.get('overlap_start', 0):.2f}s"
                    ])
                moves_df = pd.DataFrame(moves_data, columns=["Class", "Description", "Start", "End", "Duration"]) if moves_data else pd.DataFrame(columns=["Class", "Description", "Start", "End", "Duration"])
                
                # Errors dataframe: Class, Description (full), Start, End
                errors_data = []
                for error in metadata.get('errors', []):
                    ec = error.get('error_class') or ''
                    ec_display = LABEL_TO_README.get(ec, ec)
                    desc = error.get('description', '') or ''
                    errors_data.append([
                        ec_display,
                        desc,
                        f"{error.get('overlap_start', 0):.2f}s",
                        f"{error.get('overlap_end', 0):.2f}s"
                    ])
                errors_df = pd.DataFrame(errors_data, columns=["Class", "Description", "Start", "End"]) if errors_data else pd.DataFrame(columns=["Class", "Description", "Start", "End"])
                
                # Styling dataframe
                styling_data = []
                for style in metadata.get('styling_leader', []):
                    styling_data.append([
                        "Leader",
                        style.get('description', '')[:50],
                        f"{style.get('overlap_start', 0):.2f}s",
                        f"{style.get('overlap_end', 0):.2f}s"
                    ])
                for style in metadata.get('styling_follower', []):
                    styling_data.append([
                        "Follower",
                        style.get('description', '')[:50],
                        f"{style.get('overlap_start', 0):.2f}s",
                        f"{style.get('overlap_end', 0):.2f}s"
                    ])
                styling_df = pd.DataFrame(styling_data, columns=["Role", "Description", "Start", "End"]) if styling_data else pd.DataFrame(columns=["Role", "Description", "Start", "End"])
                
                # Detailed info
                detailed_info = f"Metadata for Sample {idx}\n"
                detailed_info += f"{'='*60}\n"
                detailed_info += f"Video ID: {metadata.get('vid', 'N/A')}\n"
                detailed_info += f"\nMoves: {len(metadata.get('moves', []))}\n"
                detailed_info += f"Errors: {len(metadata.get('errors', []))}\n"
                detailed_info += f"Styling (Leader): {len(metadata.get('styling_leader', []))}\n"
                detailed_info += f"Styling (Follower): {len(metadata.get('styling_follower', []))}\n"
                if 'motionscript_leader' in metadata or 'motionscript_follower' in metadata:
                    def _fmt_ms(x):
                        if x is None: return '(none)'
                        if isinstance(x, list): return ' --> '.join(str(e) for e in x) if x else '(empty)'
                        return str(x)
                    detailed_info += f"\nMotionScript (Leader):\n{_fmt_ms(metadata.get('motionscript_leader'))}\n"
                    detailed_info += f"\nMotionScript (Follower):\n{_fmt_ms(metadata.get('motionscript_follower'))}\n"
                
                # Create timeline visualization
                timeline_plot = app.create_timeline_visualization(metadata)
                
                # Try to create video overlay using the actual generated InterHuman combined video
                overlay_video = None
                try:
                    # Use the actual combined video from InterHuman visualization (ensures alignment)
                    combined_video_path = os.path.join(app.temp_dir, f"interhuman_combined_{idx}.mp4")
                    if not os.path.exists(combined_video_path):
                        # Generate it if it doesn't exist
                        leader_vid, follower_vid, combined_vid, _, _ = app.visualize_interhuman_pair(idx, use_continuous_concatenation=False)
                        if combined_vid and os.path.exists(combined_vid):
                            combined_video_path = combined_vid
                    
                    if os.path.exists(combined_video_path):
                        overlay_video_path = app.create_video_with_metadata_overlay(combined_video_path, metadata)
                        if overlay_video_path and os.path.exists(overlay_video_path):
                            overlay_video = overlay_video_path
                except Exception as e:
                    if DEBUG:
                        print(f"Could not create video overlay: {e}")
                
                return (
                    basic_info,
                    level_badge,
                    moves_df,
                    errors_df,
                    styling_df,
                    overlay_video,
                    timeline_plot,
                    detailed_info
                )
            except Exception as e:
                import traceback
                error_msg = f"Error loading metadata: {str(e)}\n{traceback.format_exc()}"
                error_html = f"<div style='color: red;'>{error_msg}</div>"
                return (
                    error_msg,
                    error_html,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    None,
                    None,
                    error_msg
                )
        
        metadata_btn.click(
            fn=on_metadata_load,
            inputs=[sample_idx],
            outputs=[metadata_basic_info, metadata_level_badge, metadata_moves, metadata_errors, metadata_styling, metadata_video_overlay, metadata_timeline_plot, metadata_info_text]
        )
        
        # Statistics computation
        def on_compute_statistics():
            try:
                stats = app.get_dataset_statistics()
                
                if 'error' in stats:
                    err = stats['error']
                    return (None, None, None, err, None, None, err, None, None, err, 
                           None, None, gr.update(choices=[], value=None), None, None, err,
                           None, "", None)
                
                import plotly.graph_objects as go
                
                levels_data = stats.get('levels', {})
                levels_fig = go.Figure(data=[
                    go.Bar(
                        x=list(levels_data.keys()),
                        y=list(levels_data.values()),
                        marker_color=['#4CAF50', '#FF9800', '#F44336', '#9E9E9E'][:len(levels_data)]
                    )
                ])
                levels_fig.update_layout(
                    title="Distribution by Proficiency Level",
                    xaxis_title="Level",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                
                pairs_data = stats.get('pairs', {})
                pairs_fig = go.Figure(data=[
                    go.Bar(
                        x=list(pairs_data.keys()),
                        y=list(pairs_data.values()),
                        marker_color='steelblue'
                    )
                ])
                pairs_fig.update_layout(
                    title="Distribution by Pair",
                    xaxis_title="Pair",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                
                songs_data = stats.get('songs', {})
                songs_fig = go.Figure(data=[
                    go.Bar(
                        x=list(songs_data.keys()),
                        y=list(songs_data.values()),
                        marker_color='coral'
                    )
                ])
                songs_fig.update_layout(
                    title="Distribution by Song",
                    xaxis_title="Song",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                
                summary = f"Dataset Statistics\n{'='*60}\n"
                summary += f"Total Samples: {stats.get('total_samples', 0)}\n\n"
                summary += f"Proficiency Levels:\n"
                for level, count in levels_data.items():
                    summary += f"  {level}: {count}\n"
                summary += f"\nPairs:\n"
                for pair, count in pairs_data.items():
                    summary += f"  {pair}: {count}\n"
                summary += f"\nSongs:\n"
                for song, count in songs_data.items():
                    summary += f"  {song}: {count}\n"
                summary += f"\nMoves (est.): {stats.get('moves_count', 0)}  |  Errors (est.): {stats.get('errors_count', 0)}  |  Styling (est.): {stats.get('styling_count', 0)}\n"
                
                # Annotation coverage plot (with vs without annotations)
                with_ann = stats.get('samples_with_annotations', 0)
                without_ann = stats.get('samples_without_annotations', 0)
                coverage_fig = go.Figure(data=[
                    go.Pie(
                        labels=['With annotations', 'Without annotations'],
                        values=[with_ann, without_ann],
                        hole=0.4,
                        marker_colors=['#2ecc71', '#e74c3c'],
                        textinfo='label+percent+value'
                    )
                ])
                coverage_fig.update_layout(
                    title="Annotation Coverage",
                    template="plotly_white",
                    showlegend=True,
                    height=320
                )
                
                # Label distribution (moves + errors): all README classes, zeros for missing
                move_class_data = stats.get('move_class_dist', {})
                mc_keys = list(ALL_LABEL_NAMES)
                mc_vals = [int(move_class_data.get(k, 0)) for k in mc_keys]
                move_class_fig = go.Figure(data=[
                    go.Bar(
                        x=mc_keys,
                        y=mc_vals,
                        marker_color='teal',
                        text=mc_vals,
                        textposition='auto'
                    )
                ])
                move_class_fig.update_layout(
                    title="Label distribution (moves + errors)",
                    xaxis_title="Class",
                    yaxis_title="Count",
                    template="plotly_white",
                    height=320,
                    xaxis_tickangle=-45
                )
                
                # Annotation stats summary
                ann_summary = "Annotation Statistics\n"
                ann_summary += "=" * 60 + "\n"
                ann_summary += "Samples with annotations = windows that have at least one move, error, or styling label.\n"
                ann_summary += f"Samples with annotations:    {with_ann}\n"
                ann_summary += f"Samples without annotations: {without_ann}\n"
                ann_summary += f"Samples with errors:         {stats.get('samples_with_errors', 0)}\n"
                ann_summary += f"Samples with styling:        {stats.get('samples_with_styling', 0)}\n"
                ann_summary += f"Unclassified moves (est.):   {stats.get('unclassified_move_count', 0)}\n"
                ann_summary += f"Unclassified errors (est.):  {stats.get('unclassified_error_count', 0)}\n"
                ann_summary += f"Metadata/load errors (est.): {stats.get('metadata_errors', 0)}\n"
                err_indices = stats.get('sample_indices_with_errors', [])
                if err_indices:
                    ann_summary += f"\nSample indices with errors (first few): {err_indices[:15]}\n"
                ann_summary += "\nLabel distribution (moves + errors, see chart above)."
                
                # Token similarity analysis visualization
                token_analysis = stats.get('token_analysis', {})
                token_freq_fig = None
                token_entropy_fig = None
                token_summary = "Token Similarity Analysis\n" + "=" * 60 + "\n"
                
                if token_analysis:
                    from plotly.subplots import make_subplots
                    
                    # Token frequency heatmap (top tokens per class)
                    classes_with_data = [c for c in ALL_LABEL_NAMES if c in token_analysis]
                    if classes_with_data:
                        # Get top 10 tokens across all classes
                        all_tokens = set()
                        for cls_data in token_analysis.values():
                            all_tokens.update([t[0] for t in cls_data['leader']['top_tokens']])
                            all_tokens.update([t[0] for t in cls_data['follower']['top_tokens']])
                            all_tokens.update([t[0] for t in cls_data['relationship']['top_tokens']])
                        top_tokens_global = sorted(list(all_tokens))[:15]
                        
                        # Create heatmap data
                        leader_freq_matrix = []
                        follower_freq_matrix = []
                        relationship_freq_matrix = []
                        
                        for cls in classes_with_data[:10]:  # Top 10 classes
                            cls_data = token_analysis[cls]
                            leader_row = [cls_data['leader']['freq_dist'].get(t, 0) / max(cls_data['leader']['total_count'], 1) 
                                         for t in top_tokens_global]
                            follower_row = [cls_data['follower']['freq_dist'].get(t, 0) / max(cls_data['follower']['total_count'], 1) 
                                           for t in top_tokens_global]
                            relationship_row = [cls_data['relationship']['freq_dist'].get(t, 0) / max(cls_data['relationship']['total_count'], 1) 
                                               for t in top_tokens_global]
                            leader_freq_matrix.append(leader_row)
                            follower_freq_matrix.append(follower_row)
                            relationship_freq_matrix.append(relationship_row)
                        
                        # Create subplots for leader, follower, relationship
                        token_freq_fig = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=['Leader Tokens', 'Follower Tokens', 'Relationship Tokens'],
                            horizontal_spacing=0.15
                        )
                        
                        for idx, (matrix, name) in enumerate([(leader_freq_matrix, 'Leader'), 
                                                               (follower_freq_matrix, 'Follower'),
                                                               (relationship_freq_matrix, 'Relationship')], 1):
                            token_freq_fig.add_trace(
                                go.Heatmap(
                                    z=matrix,
                                    x=[f"T{t}" for t in top_tokens_global],
                                    y=[c[:20] for c in classes_with_data[:10]],
                                    colorscale='Viridis',
                                    showscale=(idx == 1),
                                    colorbar=dict(title="Frequency", x=0.33*idx-0.16 if idx > 1 else 1.02)
                                ),
                                row=1, col=idx
                            )
                        
                        token_freq_fig.update_layout(
                            title="Token Frequency by Class (Normalized)",
                            height=400,
                            template="plotly_white"
                        )
                        
                        # Entropy (diversity) plot
                        leader_entropies = [token_analysis[c]['leader']['entropy'] for c in classes_with_data]
                        follower_entropies = [token_analysis[c]['follower']['entropy'] for c in classes_with_data]
                        relationship_entropies = [token_analysis[c]['relationship']['entropy'] for c in classes_with_data]
                        
                        token_entropy_fig = go.Figure()
                        token_entropy_fig.add_trace(go.Bar(
                            x=[c[:20] for c in classes_with_data],
                            y=leader_entropies,
                            name='Leader',
                            marker_color='blue'
                        ))
                        token_entropy_fig.add_trace(go.Bar(
                            x=[c[:20] for c in classes_with_data],
                            y=follower_entropies,
                            name='Follower',
                            marker_color='green'
                        ))
                        token_entropy_fig.add_trace(go.Bar(
                            x=[c[:20] for c in classes_with_data],
                            y=relationship_entropies,
                            name='Relationship',
                            marker_color='red'
                        ))
                        token_entropy_fig.update_layout(
                            title="Token Diversity (Entropy) by Class",
                            xaxis_title="Class",
                            yaxis_title="Entropy (bits)",
                            template="plotly_white",
                            height=400,
                            barmode='group',
                            xaxis_tickangle=-45
                        )
                        
                        # Summary text
                        token_summary += f"Analyzed {len(classes_with_data)} classes with token data.\n\n"
                        for cls in sorted(classes_with_data, key=lambda x: token_analysis[x]['leader']['total_count'], reverse=True)[:5]:
                            cls_data = token_analysis[cls]
                            token_summary += f"{cls}:\n"
                            token_summary += f"  Leader: {cls_data['leader']['unique_count']} unique tokens, "
                            token_summary += f"entropy={cls_data['leader']['entropy']:.2f}, "
                            token_summary += f"top={cls_data['leader']['top_tokens'][0][0] if cls_data['leader']['top_tokens'] else 'N/A'}\n"
                            token_summary += f"  Follower: {cls_data['follower']['unique_count']} unique tokens, "
                            token_summary += f"entropy={cls_data['follower']['entropy']:.2f}, "
                            token_summary += f"top={cls_data['follower']['top_tokens'][0][0] if cls_data['follower']['top_tokens'] else 'N/A'}\n"
                            token_summary += f"  Relationship: {cls_data['relationship']['unique_count']} unique tokens, "
                            token_summary += f"entropy={cls_data['relationship']['entropy']:.2f}, "
                            token_summary += f"top={cls_data['relationship']['top_tokens'][0][0] if cls_data['relationship']['top_tokens'] else 'N/A'}\n\n"
                else:
                    token_summary += "No token data available. Ensure samples have interhuman_data with tokens."
                
                # Inter-class vs Intra-class similarity comparison
                inter_class_sim = stats.get('inter_class_similarity', {})
                inter_intra_comparison_fig = None
                class_similarity_radar_fig = None
                
                if token_analysis and inter_class_sim and 'error' not in inter_class_sim:
                    classes_with_data = [c for c in ALL_LABEL_NAMES if c in token_analysis]
                    classes_with_similarity = [c for c in classes_with_data 
                                               if token_analysis[c].get('similarity_metrics') and 
                                               'error' not in token_analysis[c].get('similarity_metrics', {})]
                    
                    if classes_with_similarity:
                        # Compare intra-class (within class) vs inter-class (between classes) similarity
                        metrics_to_compare = ['jaccard', 'bleu', 'rouge']  # Similarity metrics (higher = more similar)
                        metric_labels = ['Jaccard', 'BLEU', 'ROUGE-1']
                        token_types = ['leader', 'follower', 'relationship']
                        
                        # Collect intra-class averages per class
                        intra_class_means = {m: {t: [] for t in token_types} for m in metrics_to_compare}
                        class_names_list = []
                        
                        for cls in classes_with_similarity[:15]:  # Top 15 classes
                            class_names_list.append(cls[:20])  # Truncate for display
                            sim_metrics = token_analysis[cls]['similarity_metrics']
                            for metric_name in metrics_to_compare:
                                for token_type in token_types:
                                    metric_data = sim_metrics[token_type].get(metric_name)
                                    if metric_data and metric_data.get('mean') is not None:
                                        intra_class_means[metric_name][token_type].append(metric_data['mean'])
                                    else:
                                        intra_class_means[metric_name][token_type].append(0.0)
                        
                        # Get inter-class averages
                        inter_class_means = {}
                        for metric_name in metrics_to_compare:
                            inter_class_means[metric_name] = {}
                            for token_type in token_types:
                                inter_data = inter_class_sim[token_type].get(metric_name)
                                if inter_data and inter_data.get('mean') is not None:
                                    inter_class_means[metric_name][token_type] = inter_data['mean']
                                else:
                                    inter_class_means[metric_name][token_type] = 0.0
                        
                        # Create comparison chart: Intra-class vs Inter-class
                        inter_intra_comparison_fig = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=['Leader Tokens', 'Follower Tokens', 'Relationship Tokens'],
                            horizontal_spacing=0.12
                        )
                        
                        for col_idx, token_type in enumerate(token_types, 1):
                            x_metrics = []
                            intra_vals = []
                            inter_vals = []
                            
                            for metric_name, metric_label in zip(metrics_to_compare, metric_labels):
                                # Average intra-class similarity across all classes for this metric
                                intra_avg = np.mean([v for v in intra_class_means[metric_name][token_type] if v > 0]) if intra_class_means[metric_name][token_type] else 0.0
                                inter_avg = inter_class_means[metric_name].get(token_type, 0.0)
                                
                                x_metrics.append(metric_label)
                                intra_vals.append(intra_avg)
                                inter_vals.append(inter_avg)
                            
                            inter_intra_comparison_fig.add_trace(
                                go.Bar(name='Intra-Class (within class)', x=x_metrics, y=intra_vals, 
                                      marker_color='steelblue', showlegend=(col_idx == 1)),
                                row=1, col=col_idx
                            )
                            inter_intra_comparison_fig.add_trace(
                                go.Bar(name='Inter-Class (between classes)', x=x_metrics, y=inter_vals,
                                      marker_color='coral', showlegend=(col_idx == 1)),
                                row=1, col=col_idx
                            )
                            inter_intra_comparison_fig.update_xaxes(title_text="Metric", row=1, col=col_idx)
                            inter_intra_comparison_fig.update_yaxes(title_text="Similarity Score", row=1, col=col_idx)
                        
                        inter_intra_comparison_fig.update_layout(
                            title="Intra-Class vs Inter-Class Similarity Comparison",
                            height=400,
                            template="plotly_white",
                            barmode='group'
                        )
                        
                        # Radar chart: Compare similarity metrics across top classes
                        if len(classes_with_similarity) > 0:
                            top_classes = sorted(classes_with_similarity, 
                                               key=lambda x: token_analysis[x].get('num_samples', 0), 
                                               reverse=True)[:10]
                            
                            # Create radar chart data
                            categories = []
                            leader_values = []
                            follower_values = []
                            relationship_values = []
                            
                            for cls in top_classes:
                                categories.append(cls[:15])  # Truncate for display
                                sim_metrics = token_analysis[cls]['similarity_metrics']
                                
                                # Average Jaccard, BLEU, ROUGE for each token type
                                leader_avg = np.mean([
                                    sim_metrics['leader'].get('jaccard', {}).get('mean', 0) or 0,
                                    sim_metrics['leader'].get('bleu', {}).get('mean', 0) or 0,
                                    sim_metrics['leader'].get('rouge', {}).get('mean', 0) or 0
                                ])
                                follower_avg = np.mean([
                                    sim_metrics['follower'].get('jaccard', {}).get('mean', 0) or 0,
                                    sim_metrics['follower'].get('bleu', {}).get('mean', 0) or 0,
                                    sim_metrics['follower'].get('rouge', {}).get('mean', 0) or 0
                                ])
                                relationship_avg = np.mean([
                                    sim_metrics['relationship'].get('jaccard', {}).get('mean', 0) or 0,
                                    sim_metrics['relationship'].get('bleu', {}).get('mean', 0) or 0,
                                    sim_metrics['relationship'].get('rouge', {}).get('mean', 0) or 0
                                ])
                                
                                leader_values.append(leader_avg)
                                follower_values.append(follower_avg)
                                relationship_values.append(relationship_avg)
                            
                            # Create grouped bar chart (radar charts are complex in plotly, use grouped bars)
                            class_similarity_radar_fig = go.Figure()
                            class_similarity_radar_fig.add_trace(go.Bar(
                                x=categories,
                                y=leader_values,
                                name='Leader',
                                marker_color='blue'
                            ))
                            class_similarity_radar_fig.add_trace(go.Bar(
                                x=categories,
                                y=follower_values,
                                name='Follower',
                                marker_color='green'
                            ))
                            class_similarity_radar_fig.add_trace(go.Bar(
                                x=categories,
                                y=relationship_values,
                                name='Relationship',
                                marker_color='red'
                            ))
                            class_similarity_radar_fig.update_layout(
                                title="Average Similarity (Jaccard+BLEU+ROUGE) Across Classes",
                                xaxis_title="Class",
                                yaxis_title="Average Similarity Score",
                                template="plotly_white",
                                height=400,
                                barmode='group',
                                xaxis_tickangle=-45
                            )
                
                # Prepare dropdown choices and initial class-specific plots
                class_dropdown_choices = []
                class_token_dist_fig = None
                class_similarity_fig = None
                class_metrics_text = "Select a class from the dropdown above to see detailed token analysis.\n\n"
                class_metrics_text += "Similarity Metrics Explained:\n"
                class_metrics_text += "=" * 60 + "\n"
                class_metrics_text += "• Edit Distance (Levenshtein): Minimum operations (insert/delete/substitute) to transform one sequence into another. Lower = more similar.\n"
                class_metrics_text += "• Hamming Distance: Count of positions where tokens differ (only for equal-length sequences). Lower = more similar.\n"
                class_metrics_text += "• Jaccard Similarity: Ratio of shared tokens to total unique tokens. Range [0,1], higher = more similar.\n"
                class_metrics_text += "• BLEU: Precision-based n-gram overlap score. Range [0,1], higher = more similar (translation quality metric).\n"
                class_metrics_text += "• ROUGE-1: Recall-based unigram overlap. Range [0,1], higher = more similar (summarization metric).\n"
                
                if token_analysis:
                    classes_with_data = [c for c in ALL_LABEL_NAMES if c in token_analysis]
                    for cls in classes_with_data:
                        num_samples = token_analysis[cls].get('num_samples', 0)
                        if num_samples > 0:
                            class_dropdown_choices.append(f"{cls} ({num_samples} samples)")
                
                # Metric comparison across classes (initial: Jaccard Similarity)
                metric_comparison_fig = None
                metric_description_text = ""
                if token_analysis:
                    metric_comparison_fig, metric_description_text = app._create_metric_comparison_chart(
                        token_analysis, "Jaccard Similarity"
                    )
                
                # Radar chart: all metrics across all classes
                radar_chart_fig = app._create_radar_chart_all_metrics(token_analysis)
                
                return (levels_fig, pairs_fig, songs_fig, summary, coverage_fig, move_class_fig, ann_summary,
                       token_freq_fig, token_entropy_fig, token_summary,
                       inter_intra_comparison_fig, class_similarity_radar_fig,
                       gr.update(choices=class_dropdown_choices, value=class_dropdown_choices[0] if class_dropdown_choices else None),
                       class_token_dist_fig, class_similarity_fig, class_metrics_text,
                       metric_comparison_fig, metric_description_text, radar_chart_fig)
            except Exception as e:
                import traceback
                err = f"Error computing statistics: {str(e)}\n{traceback.format_exc()}"
                return (None, None, None, err, None, None, err, None, None, err,
                       None, None,
                       gr.update(choices=[], value=None), None, None, err,
                       None, "", None)
        
        stats_btn.click(
            fn=on_compute_statistics,
            inputs=[],
            outputs=[
                stats_levels_plot,
                stats_pairs_plot,
                stats_songs_plot,
                stats_summary,
                stats_annotation_coverage_plot,
                stats_move_class_plot,
                stats_annotation_summary,
                stats_token_analysis_plot,
                stats_token_entropy_plot,
                stats_token_summary,
                stats_inter_intra_comparison_plot,
                stats_class_similarity_radar,
                stats_class_dropdown,
                stats_class_token_dist_plot,
                stats_class_similarity_plot,
                stats_class_metrics_text,
                stats_metric_comparison_plot,
                stats_metric_description,
                stats_radar_chart,
            ]
        )

        def on_motionscript_stat_analysis(max_samples_val):
            try:
                n = int(max_samples_val) if max_samples_val is not None else 50
                return app._run_motionscript_stat_analysis(max_samples=n)
            except Exception as e:
                import traceback
                return f"Error: {e}\n{traceback.format_exc()}"

        motionscript_stat_btn.click(
            fn=on_motionscript_stat_analysis,
            inputs=[motionscript_stat_max_samples],
            outputs=[motionscript_stat_output]
        )
        
        # Handler for class selection dropdown
        def on_class_selected(class_choice):
            if not class_choice or class_choice == "None":
                return None, None, "Select a class to see analysis."
            
            # Extract class name from "Class (N samples)" format - handle nested parentheses like "XBL (Cross Body Lead) (45 samples)"
            # Find the last occurrence of " (" followed by a number and " samples)"
            import re
            match = re.search(r'^(.+?)\s+\(\d+\s+samples\)$', class_choice)
            if match:
                class_name = match.group(1).strip()
            elif " (" in class_choice:
                # Fallback: split on last " (" 
                parts = class_choice.rsplit(" (", 1)
                class_name = parts[0]
            else:
                class_name = class_choice
            
            # Get statistics from stored cache (avoid recomputation)
            try:
                if app.last_computed_stats is None:
                    stats = app.get_dataset_statistics()
                else:
                    stats = app.last_computed_stats
                token_analysis = stats.get('token_analysis', {})
                
                # Try exact match first
                if class_name not in token_analysis:
                    # Try fuzzy matching: check if any stored class name matches
                    found = False
                    # Try exact match after trimming
                    class_name_trimmed = class_name.strip()
                    if class_name_trimmed in token_analysis:
                        class_name = class_name_trimmed
                        found = True
                    else:
                        # Try partial matching
                        for stored_name in token_analysis.keys():
                            if stored_name == class_name or stored_name.startswith(class_name) or class_name.startswith(stored_name):
                                class_name = stored_name
                                found = True
                                break
                            # Also try reverse mapping
                            for key, val in LABEL_TO_README.items():
                                if val == class_name and stored_name == val:
                                    class_name = stored_name
                                    found = True
                                    break
                    if not found:
                        available = list(token_analysis.keys())[:10]
                        return None, None, f"No token data for class: '{class_name}'\n\nAvailable classes (first 10):\n" + "\n".join(available)
                
                cls_data = token_analysis[class_name]
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Get inter-class similarity for comparison
                inter_class_sim = stats.get('inter_class_similarity', {})
                
                # Token distribution chart for this class
                leader_freq = cls_data['leader']['freq_dist']
                follower_freq = cls_data['follower']['freq_dist']
                relationship_freq = cls_data['relationship']['freq_dist']
                
                # Get top 20 tokens for each type
                leader_top = sorted(leader_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                follower_top = sorted(follower_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                relationship_top = sorted(relationship_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                
                token_dist_fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=['Leader Tokens', 'Follower Tokens', 'Relationship Tokens'],
                    horizontal_spacing=0.12
                )
                
                for idx, (tokens_data, name) in enumerate([(leader_top, 'Leader'), (follower_top, 'Follower'), (relationship_top, 'Relationship')], 1):
                    tokens = [f"T{t[0]}" for t in tokens_data]
                    counts = [t[1] for t in tokens_data]
                    token_dist_fig.add_trace(
                        go.Bar(x=tokens, y=counts, name=name, showlegend=(idx == 1)),
                        row=1, col=idx
                    )
                    token_dist_fig.update_xaxes(title_text="Token", row=1, col=idx, tickangle=-45)
                    token_dist_fig.update_yaxes(title_text="Frequency", row=1, col=idx)
                
                token_dist_fig.update_layout(
                    title=f"Token Distribution: {class_name}",
                    height=400,
                    template="plotly_white"
                )
                
                # Similarity metrics visualization
                similarity_metrics = cls_data.get('similarity_metrics')
                if similarity_metrics and 'error' not in similarity_metrics:
                    # Create grouped bar chart for each metric
                    metrics_names = ['edit_dist', 'hamming', 'jaccard', 'bleu', 'rouge']
                    metric_labels = ['Edit Distance', 'Hamming Distance', 'Jaccard Similarity', 'BLEU', 'ROUGE-1']
                    token_types = ['leader', 'follower', 'relationship']
                    
                    similarity_fig = go.Figure()
                    
                    x_pos = []
                    y_vals = []
                    colors = ['blue', 'green', 'red']
                    names = []
                    
                    for metric_idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metric_labels)):
                        for type_idx, token_type in enumerate(token_types):
                            metric_data = similarity_metrics[token_type].get(metric_name)
                            if metric_data and metric_data.get('mean') is not None:
                                x_pos.append(f"{metric_label}\n({token_type})")
                                # Normalize edit/hamming distance (invert for similarity)
                                if metric_name in ['edit_dist', 'hamming']:
                                    # Use inverse normalized (lower is better, so we show 1/(1+mean))
                                    val = 1.0 / (1.0 + metric_data['mean'])
                                else:
                                    val = metric_data['mean']
                                y_vals.append(val)
                                names.append(token_type)
                    
                    if y_vals:
                        similarity_fig.add_trace(go.Bar(
                            x=x_pos,
                            y=y_vals,
                            marker_color=[colors[token_types.index(n)] if n in token_types else 'gray' for n in names],
                            name='Similarity Score'
                        ))
                        similarity_fig.update_layout(
                            title=f"Similarity Metrics: {class_name}",
                            xaxis_title="Metric",
                            yaxis_title="Similarity Score (normalized)",
                            template="plotly_white",
                            height=400,
                            xaxis_tickangle=-45
                        )
                    else:
                        similarity_fig = None
                else:
                    similarity_fig = None
                
                # Metrics explanation text
                metrics_text = f"Token Analysis for: {class_name}\n"
                metrics_text += "=" * 60 + "\n"
                metrics_text += f"Number of samples: {cls_data.get('num_samples', 0)}\n\n"
                
                if similarity_metrics and 'error' not in similarity_metrics:
                    metrics_text += "Intra-Class Similarity Metrics (pairwise between samples in this class):\n"
                    metrics_text += "-" * 60 + "\n"
                    metrics_names = ['edit_dist', 'hamming', 'jaccard', 'bleu', 'rouge']
                    metric_labels = ['Edit Distance', 'Hamming Distance', 'Jaccard Similarity', 'BLEU', 'ROUGE-1']
                    
                    for token_type in ['leader', 'follower', 'relationship']:
                        metrics_text += f"\n{token_type.capitalize()} Tokens:\n"
                        type_metrics = similarity_metrics[token_type]
                        for metric_name, metric_label in zip(metrics_names, metric_labels):
                            metric_data = type_metrics.get(metric_name)
                            if metric_data:
                                metrics_text += f"  {metric_label}:\n"
                                metrics_text += f"    Mean: {metric_data['mean']:.4f}, Std: {metric_data['std']:.4f}\n"
                                metrics_text += f"    Range: [{metric_data['min']:.4f}, {metric_data['max']:.4f}]\n"
                                metrics_text += f"    Median: {metric_data['median']:.4f}\n"
                    
                    # Add inter-class comparison
                    if inter_class_sim and 'error' not in inter_class_sim:
                        metrics_text += "\n" + "=" * 60 + "\n"
                        metrics_text += "Inter-Class Comparison (average similarity between different classes):\n"
                        metrics_text += "-" * 60 + "\n"
                        for token_type in ['leader', 'follower', 'relationship']:
                            metrics_text += f"\n{token_type.capitalize()} Tokens (Inter-Class):\n"
                            inter_type = inter_class_sim.get(token_type, {})
                            for metric_name, metric_label in zip(['jaccard', 'bleu', 'rouge'], ['Jaccard', 'BLEU', 'ROUGE-1']):
                                inter_data = inter_type.get(metric_name)
                                if inter_data:
                                    metrics_text += f"  {metric_label}: {inter_data['mean']:.4f} (avg)\n"
                else:
                    metrics_text += "Similarity metrics not available (need at least 2 samples).\n"
                
                metrics_text += "\n" + "=" * 60 + "\n"
                metrics_text += "Metric Explanations:\n"
                metrics_text += "• Edit Distance: Lower = more similar (operations to transform sequences)\n"
                metrics_text += "• Hamming Distance: Lower = more similar (position differences, equal length only)\n"
                metrics_text += "• Jaccard: Higher = more similar [0,1] (shared tokens / total unique)\n"
                metrics_text += "• BLEU: Higher = more similar [0,1] (n-gram precision)\n"
                metrics_text += "• ROUGE-1: Higher = more similar [0,1] (unigram recall)\n"
                
                return token_dist_fig, similarity_fig, metrics_text
            except Exception as e:
                import traceback
                return None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        stats_class_dropdown.change(
            fn=on_class_selected,
            inputs=[stats_class_dropdown],
            outputs=[stats_class_token_dist_plot, stats_class_similarity_plot, stats_class_metrics_text]
        )
        
        # Metric navigation handlers
        metric_list = ["Jaccard Similarity", "BLEU", "ROUGE-1", "Edit Distance", "Hamming Distance"]
        
        def on_metric_selected(metric_name):
            """Handler for metric dropdown selection."""
            if app.last_computed_stats is None:
                stats = app.get_dataset_statistics()
            else:
                stats = app.last_computed_stats
            token_analysis = stats.get('token_analysis', {})
            fig, desc = app._create_metric_comparison_chart(token_analysis, metric_name)
            return fig, desc
        
        def on_metric_prev(current_metric):
            """Navigate to previous metric."""
            if not current_metric or current_metric not in metric_list:
                current_metric = metric_list[0]
            current_idx = metric_list.index(current_metric)
            prev_idx = (current_idx - 1) % len(metric_list)
            new_metric = metric_list[prev_idx]
            
            if app.last_computed_stats is None:
                stats = app.get_dataset_statistics()
            else:
                stats = app.last_computed_stats
            token_analysis = stats.get('token_analysis', {})
            fig, desc = app._create_metric_comparison_chart(token_analysis, new_metric)
            return gr.update(value=new_metric), fig, desc
        
        def on_metric_next(current_metric):
            """Navigate to next metric."""
            if not current_metric or current_metric not in metric_list:
                current_metric = metric_list[0]
            current_idx = metric_list.index(current_metric)
            next_idx = (current_idx + 1) % len(metric_list)
            new_metric = metric_list[next_idx]
            
            if app.last_computed_stats is None:
                stats = app.get_dataset_statistics()
            else:
                stats = app.last_computed_stats
            token_analysis = stats.get('token_analysis', {})
            fig, desc = app._create_metric_comparison_chart(token_analysis, new_metric)
            return gr.update(value=new_metric), fig, desc
        
        # Connect handlers
        stats_metric_dropdown.change(
            fn=on_metric_selected,
            inputs=[stats_metric_dropdown],
            outputs=[stats_metric_comparison_plot, stats_metric_description]
        )
        
        stats_metric_nav_prev.click(
            fn=on_metric_prev,
            inputs=[stats_metric_dropdown],
            outputs=[stats_metric_dropdown, stats_metric_comparison_plot, stats_metric_description]
        )
        
        stats_metric_nav_next.click(
            fn=on_metric_next,
            inputs=[stats_metric_dropdown],
            outputs=[stats_metric_dropdown, stats_metric_comparison_plot, stats_metric_description]
        )
        
        # Search functionality
        def on_search(level_val, pair_val, song_val, move_class_val, has_errors_val, has_styling_val):
            try:
                filters = {}
                if level_val and level_val != "All":
                    filters['level'] = level_val
                if pair_val and pair_val != "All":
                    filters['pair'] = pair_val
                if song_val and song_val != "All":
                    filters['song'] = song_val
                if move_class_val and move_class_val != "All":
                    # Extract class name from "Class (count)" format
                    if " (" in move_class_val:
                        class_name = move_class_val.split(" (")[0]
                    else:
                        class_name = move_class_val
                    filters['move_class'] = class_name
                if has_errors_val:
                    filters['has_errors'] = True
                if has_styling_val:
                    filters['has_styling'] = True
                
                results = app.search_samples(filters)
                
                if results:
                    # Convert to DataFrame with proper column order
                    df = pd.DataFrame(results)
                    # Ensure columns are in the right order
                    if not df.empty:
                        df = df[["index", "pair", "song", "take", "level", "moves_count", "errors_count"]]
                        df.columns = ["Index", "Pair", "Song", "Take", "Level", "Moves", "Errors"]
                    return df
                else:
                    return pd.DataFrame(columns=["Index", "Pair", "Song", "Take", "Level", "Moves", "Errors"])
            except Exception as e:
                import traceback
                if DEBUG:
                    print(f"Error in search: {e}\n{traceback.format_exc()}")
                return pd.DataFrame(columns=["Index", "Pair", "Song", "Take", "Level", "Moves", "Errors"])
        
        def on_search_visualize(idx_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                # Use same visualization as ground truth (InterHuman tab) with continuous concatenation
                leader_vid, follower_vid, combined_vid, _, info = app.visualize_interhuman_pair(idx, use_continuous_concatenation=True)
                return combined_vid, info
            except Exception as e:
                import traceback
                error_msg = f"Error visualizing: {str(e)}\n{traceback.format_exc()}"
                return None, error_msg
        
        search_btn.click(
            fn=on_search,
            inputs=[search_level, search_pair, search_song, search_move_class, search_has_errors, search_has_styling],
            outputs=[search_results]
        )
        
        search_visualize_btn.click(
            fn=on_search_visualize,
            inputs=[search_selected_idx],
            outputs=[search_video, search_info]
        )
        
        # When user selects a row in search results, update the selected index
        def on_search_result_select(evt: gr.SelectData):
            try:
                if evt.value is not None:
                    # Get the index from the selected row (first column)
                    selected_idx = int(evt.value.iloc[0, 0]) if hasattr(evt.value, 'iloc') else int(evt.value[0])
                    return selected_idx
            except Exception as e:
                if DEBUG:
                    print(f"Error selecting from search results: {e}")
            return 0
        
        # Connect search results selection to update selected index
        try:
            search_results.select(
                fn=on_search_result_select,
                outputs=[search_selected_idx]
            )
        except:
            # Gradio version might not support select event, user can manually enter index
            pass
        
        # Legacy visualization
        def on_legacy_visualize(idx_val, show_leader_val, show_follower_val, show_combined_val):
            try:
                idx = int(idx_val) if idx_val is not None else 0
                return app.visualize_legacy_format(idx, show_leader_val, show_follower_val, show_combined_val)
            except Exception as e:
                import traceback
                return None, None, None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        legacy_btn.click(
            fn=on_legacy_visualize,
            inputs=[sample_idx, show_leader, show_follower, show_combined],
            outputs=[legacy_leader_video, legacy_follower_video, legacy_combined_video, legacy_info]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7862)