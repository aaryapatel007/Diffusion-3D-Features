"""
Apply motion transfer using correspondence between two point clouds.

This script takes motion displacements from a source skeleton and applies them
to a target skeleton using pre-computed correspondences.

Usage:
    python apply_motion_transfer.py \
        --source_delta data/predicted_3d_relative_delta.npy \
        --target_frame0 data/joint_displacements_abs.npy \
        --correspondence skeleton_correspondence_swapped.npy \
        --output data/animated_target_from_source.npy
"""

import numpy as np
import argparse


def load_npy(filepath):
    """Load numpy array, handling object arrays."""
    data = np.load(filepath, allow_pickle=True)
    if data.dtype == object:
        data = np.array(data.tolist(), dtype=np.float32)
    return data


def apply_motion_transfer(source_delta, target_frame0, correspondence):
    """
    Apply motion from source to target using correspondence.
    
    Args:
        source_delta: (num_frames, num_source_points, 3) - displacements per frame
        target_frame0: (num_target_points, 3) - base pose of target
        correspondence: (num_target_points,) - maps each target point to source point index
    
    Returns:
        animated_target: (num_frames, num_target_points, 3) - animated target positions
    """
    num_frames = source_delta.shape[0]
    num_target_points = len(correspondence)
    
    animated_target = np.zeros((num_frames, num_target_points, 3), dtype=np.float32)
    
    for frame in range(num_frames):
        for target_idx in range(num_target_points):
            source_idx = correspondence[target_idx]
            displacement = source_delta[frame, source_idx, :]
            animated_target[frame, target_idx, :] = target_frame0[target_idx, :] + displacement
    
    return animated_target


def main():
    parser = argparse.ArgumentParser(description="Apply motion transfer using correspondence")
    parser.add_argument("--source_delta", type=str, required=True, 
                        help="Path to source motion deltas (num_frames, num_source_points, 3)")
    parser.add_argument("--target_frame0", type=str, required=True,
                        help="Path to target positions (uses frame 0 as base pose)")
    parser.add_argument("--correspondence", type=str, required=True,
                        help="Path to correspondence file (maps target -> source)")
    parser.add_argument("--output", type=str, default="animated_target.npy",
                        help="Output path for animated target")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading source deltas: {args.source_delta}")
    source_delta = load_npy(args.source_delta)
    print(f"  Shape: {source_delta.shape}")
    
    print(f"Loading target positions: {args.target_frame0}")
    target_positions = load_npy(args.target_frame0)
    # Use frame 0 as base pose
    if target_positions.ndim == 3:
        target_frame0 = target_positions[0]
    else:
        target_frame0 = target_positions
    print(f"  Frame 0 shape: {target_frame0.shape}")
    
    print(f"Loading correspondence: {args.correspondence}")
    correspondence = np.load(args.correspondence)
    print(f"  Shape: {correspondence.shape}")
    
    # Validate
    num_target_points = len(correspondence)
    num_source_points = source_delta.shape[1]
    print(f"\nTransferring motion from {num_source_points} source points to {num_target_points} target points")
    
    if correspondence.max() >= num_source_points:
        raise ValueError(f"Correspondence has invalid index {correspondence.max()} >= {num_source_points}")
    
    # Apply motion transfer
    print("Applying motion transfer...")
    animated_target = apply_motion_transfer(source_delta, target_frame0, correspondence)
    print(f"  Output shape: {animated_target.shape}")
    
    # Save
    np.save(args.output, animated_target)
    print(f"\nSaved to: {args.output}")
    
    # Show sample
    print("\nSample - Target point 0:")
    print(f"  Base position: {target_frame0[0]}")
    print(f"  Corresponding source point: {correspondence[0]}")
    print(f"  Displacement at frame 1: {source_delta[1, correspondence[0]]}")
    print(f"  Animated position at frame 1: {animated_target[1, 0]}")


if __name__ == "__main__":
    main()
