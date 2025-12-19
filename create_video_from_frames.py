"""
Create MP4 video from rendered PNG frames.

Usage:
    python create_video_from_frames.py --input_dir output_renders --output transferred_motion.mp4 --fps 24
"""

import cv2
import os
import re
import argparse


def create_video_from_frames(input_dir, output_path, fps=24, frame_prefix="front_frame"):
    """
    Create MP4 video from PNG frames.
    
    Args:
        input_dir: Directory containing PNG frames
        output_path: Output video path
        fps: Frames per second
        frame_prefix: Prefix of frame filenames (e.g., "front_frame")
    """
    # Get all matching PNG files
    files = [f for f in os.listdir(input_dir) 
             if f.startswith(frame_prefix) and f.endswith('.png')]
    
    if not files:
        raise ValueError(f"No frames found with prefix '{frame_prefix}' in {input_dir}")
    
    # Sort by frame number
    def get_frame_num(f):
        match = re.search(rf'{frame_prefix}(\d+)\.png', f)
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_frame_num)
    print(f"Found {len(files)} frames")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(input_dir, files[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {files[0]}")
    
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame
    for f in files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is not None:
            out.write(img)
    
    out.release()
    
    # Check file size
    size = os.path.getsize(output_path)
    print(f"Video saved to: {output_path}")
    print(f"File size: {size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Create MP4 video from PNG frames")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing PNG frames")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second")
    parser.add_argument("--prefix", type=str, default="front_frame",
                        help="Prefix of frame filenames")
    args = parser.parse_args()
    
    create_video_from_frames(args.input_dir, args.output, args.fps, args.prefix)


if __name__ == "__main__":
    main()
