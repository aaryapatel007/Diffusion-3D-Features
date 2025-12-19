"""
Visualize 3D point cloud animation as a video.

Usage:
    python visualize_animation.py \
        --input data/animated_target_from_source.npy \
        --output animation.mp4 \
        --fps 24
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def load_npy(filepath):
    """Load numpy array, handling object arrays."""
    data = np.load(filepath, allow_pickle=True)
    if data.dtype == object:
        data = np.array(data.tolist(), dtype=np.float32)
    return data


def create_animation(points, output_path, fps=24, elev=20, azim=45, point_size=20, zoom=1.0):
    """
    Create video animation of 3D point cloud.
    
    Args:
        points: (num_frames, num_points, 3) - animated point positions
        output_path: Path to save video
        fps: Frames per second
        elev: Camera elevation angle
        azim: Camera azimuth angle
        point_size: Size of points in scatter plot
        zoom: Zoom factor (>1 zooms in, <1 zooms out)
    """
    num_frames, num_points, _ = points.shape
    
    # Compute global bounds for consistent axis limits
    x_min, x_max = points[:, :, 0].min(), points[:, :, 0].max()
    y_min, y_max = points[:, :, 1].min(), points[:, :, 1].max()
    z_min, z_max = points[:, :, 2].min(), points[:, :, 2].max()
    
    # Apply zoom by reducing the range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    x_range = (x_max - x_min) / zoom
    y_range = (y_max - y_min) / zoom
    z_range = (z_max - z_min) / zoom
    
    # Add padding
    padding = 0.1
    
    x_min = x_center - x_range / 2 * (1 + padding)
    x_max = x_center + x_range / 2 * (1 + padding)
    y_min = y_center - y_range / 2 * (1 + padding)
    y_max = y_center + y_range / 2 * (1 + padding)
    z_min = z_center - z_range / 2 * (1 + padding)
    z_max = z_center + z_range / 2 * (1 + padding)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], [], s=point_size, c='blue', alpha=0.8)
    
    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        return scatter,
    
    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame + 1}/{num_frames}')
        ax.view_init(elev=elev, azim=azim)
        
        # Color points by height (z-coordinate) for better visualization
        colors = plt.cm.viridis((points[frame, :, 2] - z_min) / (z_max - z_min))
        ax.scatter(points[frame, :, 0], 
                   points[frame, :, 1], 
                   points[frame, :, 2], 
                   s=point_size, c=colors, alpha=0.8)
        return scatter,
    
    print(f"Creating animation with {num_frames} frames at {fps} fps...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=num_frames, interval=1000/fps, blit=False)
    
    # Save video
    print(f"Saving to {output_path}...")
    if output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    
    anim.save(output_path, writer=writer)
    print(f"Saved animation to {output_path}")
    plt.close()


def create_side_by_side_animation(points1, points2, output_path, fps=24, 
                                   title1="Source", title2="Target",
                                   elev=20, azim=45, point_size=20, zoom=1.0):
    """
    Create side-by-side video animation of two 3D point clouds.
    
    Args:
        points1: (num_frames, num_points1, 3) - first animation
        points2: (num_frames, num_points2, 3) - second animation
        output_path: Path to save video
        fps: Frames per second
        zoom: Zoom factor (>1 zooms in, <1 zooms out)
    """
    num_frames = min(points1.shape[0], points2.shape[0])
    
    # Compute global bounds
    all_points = np.concatenate([points1.reshape(-1, 3), points2.reshape(-1, 3)], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # Apply zoom by reducing the range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    x_range = (x_max - x_min) / zoom
    y_range = (y_max - y_min) / zoom
    z_range = (z_max - z_min) / zoom
    
    # Add padding
    padding = 0.1
    
    x_min = x_center - x_range / 2 * (1 + padding)
    x_max = x_center + x_range / 2 * (1 + padding)
    y_min = y_center - y_range / 2 * (1 + padding)
    y_max = y_center + y_range / 2 * (1 + padding)
    z_min = z_center - z_range / 2 * (1 + padding)
    z_max = z_center + z_range / 2 * (1 + padding)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    def update(frame):
        for ax, pts, title in [(ax1, points1, title1), (ax2, points2, title2)]:
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} - Frame {frame + 1}/{num_frames}')
            ax.view_init(elev=elev, azim=azim)
            
            colors = plt.cm.viridis((pts[frame, :, 2] - z_min) / (z_max - z_min))
            ax.scatter(pts[frame, :, 0], 
                       pts[frame, :, 1], 
                       pts[frame, :, 2], 
                       s=point_size, c=colors, alpha=0.8)
    
    print(f"Creating side-by-side animation with {num_frames} frames at {fps} fps...")
    anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                   interval=1000/fps, blit=False)
    
    print(f"Saving to {output_path}...")
    if output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    
    anim.save(output_path, writer=writer)
    print(f"Saved animation to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D point cloud animation")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to animated points (num_frames, num_points, 3)")
    parser.add_argument("--input2", type=str, default=None,
                        help="Optional second animation for side-by-side comparison")
    parser.add_argument("--output", type=str, default="animation.mp4",
                        help="Output video path (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--elev", type=float, default=20, help="Camera elevation angle")
    parser.add_argument("--azim", type=float, default=45, help="Camera azimuth angle")
    parser.add_argument("--point_size", type=int, default=20, help="Point size")
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (>1 zooms in)")
    parser.add_argument("--title1", type=str, default="Animation 1", help="Title for first animation")
    parser.add_argument("--title2", type=str, default="Animation 2", help="Title for second animation")
    args = parser.parse_args()
    
    print(f"Loading animation: {args.input}")
    points = load_npy(args.input)
    print(f"  Shape: {points.shape}")
    
    if args.input2:
        print(f"Loading second animation: {args.input2}")
        points2 = load_npy(args.input2)
        print(f"  Shape: {points2.shape}")
        create_side_by_side_animation(points, points2, args.output, 
                                       fps=args.fps, title1=args.title1, title2=args.title2,
                                       elev=args.elev, azim=args.azim, point_size=args.point_size,
                                       zoom=args.zoom)
    else:
        create_animation(points, args.output, fps=args.fps, 
                         elev=args.elev, azim=args.azim, point_size=args.point_size,
                         zoom=args.zoom)


if __name__ == "__main__":
    main()
