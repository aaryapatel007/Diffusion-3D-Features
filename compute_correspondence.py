"""
Compute correspondence between two 3D point clouds using Diffusion-3D-Features (Diff3F).

This script uses cosine similarity of Diff3F features to find point correspondences
between two point clouds that may have different numbers of points.

Usage:
    python compute_correspondence.py --source source.ply --target target.ply --prompt "object description"
"""

import torch
import numpy as np
import argparse
from pytorch3d.structures import Pointclouds

from diff3f_copy_copy import get_features_per_vertex
from utils import cosine_similarity, get_colors
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino

try:
    import meshplot as mp
    # Check if running in Jupyter notebook
    def _in_notebook():
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                return False
            if 'IPKernelApp' not in get_ipython().config:
                return False
            return True
        except:
            return False
    MESHPLOT_AVAILABLE = _in_notebook()  # Only use meshplot in notebooks
except ImportError:
    MESHPLOT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def convert_pointcloud_to_torch_pointcloud(pc, device):
    """Convert MeshContainer point cloud to PyTorch3D Pointclouds."""
    points = torch.tensor(pc.vert, dtype=torch.float32)
    colors = torch.ones_like(points) * 0.8
    pointcloud = Pointclouds(points=[points.to(device)], features=[colors.to(device)])
    return pointcloud


def convert_numpy_to_torch_pointcloud(points_np, device):
    """Convert numpy array of points to PyTorch3D Pointclouds."""
    points = torch.tensor(points_np, dtype=torch.float32)
    colors = torch.ones_like(points) * 0.8
    pointcloud = Pointclouds(points=[points.to(device)], features=[colors.to(device)])
    return pointcloud


class NumpyPointCloud:
    """Simple wrapper to hold numpy points like MeshContainer."""
    def __init__(self, points):
        self.vert = points


def compute_features(device, pipe, dino_model, mesh_container, prompt, 
                     num_views=100, H=512, W=512, tolerance=0.004, 
                     num_images_per_prompt=1, use_normal_map=True):
    """
    Compute Diff3F features for a point cloud.
    
    Args:
        device: CUDA device
        pipe: Stable Diffusion ControlNet pipeline
        dino_model: DINOv2 model
        mesh_container: MeshContainer or NumpyPointCloud with point cloud data
        prompt: Text prompt describing the object
        num_views: Number of rendered views for feature extraction
        H, W: Render resolution
        tolerance: Tolerance for vertex visibility
        num_images_per_prompt: Number of diffusion images per prompt
        use_normal_map: Whether to use normal maps
    
    Returns:
        features: Tensor of shape (N_vertices, 2048)
    """
    if isinstance(mesh_container, NumpyPointCloud):
        pointcloud = convert_numpy_to_torch_pointcloud(mesh_container.vert, device=device)
    else:
        pointcloud = convert_pointcloud_to_torch_pointcloud(mesh_container, device=device)
    mesh_vertices = pointcloud.points_list()[0]
    
    features = get_features_per_vertex(
        device=device,
        pipe=pipe,
        dino_model=dino_model,
        mesh=pointcloud,
        prompt=prompt,
        mesh_vertices=mesh_vertices,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
        ply=True
    )
    return features.cpu()


def compute_correspondence(f_source, f_target, device):
    """
    Compute correspondence between two point clouds using cosine similarity.
    
    Args:
        f_source: Source features, shape (N_source, 2048)
        f_target: Target features, shape (N_target, 2048)
        device: CUDA device
    
    Returns:
        correspondence: Array of shape (N_target,) where correspondence[i] is the
                        index in source that corresponds to target point i
    """
    # Compute cosine similarity matrix: (N_source, N_target)
    similarity = cosine_similarity(f_source.to(device), f_target.to(device))
    
    # For each target point, find the best matching source point
    correspondence = torch.argmax(similarity, dim=0).cpu().numpy()
    
    return correspondence, similarity.cpu().numpy()


def load_point_cloud(filepath):
    """Load point cloud from PLY, OBJ, or NPY file."""
    if filepath.endswith('.npy'):
        points = np.load(filepath, allow_pickle=True)
        # Handle object arrays (convert to float)
        if points.dtype == object:
            points = np.array(points.tolist(), dtype=np.float32)
        # If 3D array (T, N, 3), use first frame
        points = points[0] if points.ndim == 3 else points
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"NPY file must have shape (N, 3), got {points.shape}")
        return NumpyPointCloud(points.astype(np.float32))
    else:
        return MeshContainer().load_from_file(filepath)


def visualize_correspondence_meshplot(source_pc, target_pc, correspondence, output_path=None):
    """
    Visualize correspondence using meshplot (interactive 3D visualization).
    Corresponding points share the same color.
    
    Args:
        source_pc: Source point cloud (MeshContainer or NumpyPointCloud)
        target_pc: Target point cloud (MeshContainer or NumpyPointCloud)
        correspondence: Array mapping target points to source points
        output_path: Optional path to save visualization
    """
    if not MESHPLOT_AVAILABLE:
        print("meshplot not available. Install with: pip install meshplot")
        return
    
    # Generate colors for source based on vertex positions
    cmap_source = get_colors(source_pc.vert)
    # Target gets colors from corresponding source points
    cmap_target = cmap_source[correspondence]
    
    # Check if source has faces (mesh) or just points
    if hasattr(source_pc, 'face') and source_pc.face is not None and len(source_pc.face) > 0:
        # Source is a mesh
        d = mp.subplot(source_pc.vert, source_pc.face, c=cmap_source, s=[2, 2, 0])
    else:
        # Source is a point cloud
        d = mp.subplot(source_pc.vert, c=cmap_source, s=[2, 2, 0])
    
    # Target is always treated as point cloud
    mp.subplot(target_pc.vert, c=cmap_target, s=[2, 2, 1], data=d)
    
    print("Visualization displayed. Matching colors indicate corresponding points.")


def visualize_correspondence_matplotlib(source_pc, target_pc, correspondence, output_path=None):
    """
    Visualize correspondence using matplotlib (static 3D visualization).
    Corresponding points share the same color.
    
    Args:
        source_pc: Source point cloud (MeshContainer or NumpyPointCloud)
        target_pc: Target point cloud (MeshContainer or NumpyPointCloud)
        correspondence: Array mapping target points to source points
        output_path: Optional path to save visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Generate colors for source based on vertex positions
    cmap_source = get_colors(source_pc.vert)
    # Target gets colors from corresponding source points
    cmap_target = cmap_source[correspondence]
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot source
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(source_pc.vert[:, 0], source_pc.vert[:, 1], source_pc.vert[:, 2], 
                c=cmap_source, s=20)
    ax1.set_title(f'Source ({len(source_pc.vert)} points)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot target
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(target_pc.vert[:, 0], target_pc.vert[:, 1], target_pc.vert[:, 2], 
                c=cmap_target, s=20)
    ax2.set_title(f'Target ({len(target_pc.vert)} points)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.suptitle('Correspondence Visualization\n(Matching colors = corresponding points)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_correspondence(source_pc, target_pc, correspondence, output_path=None, use_matplotlib=False):
    """
    Visualize correspondence between two point clouds.
    
    Args:
        source_pc: Source point cloud
        target_pc: Target point cloud
        correspondence: Array mapping target points to source points
        output_path: Optional path to save visualization
        use_matplotlib: If True, use matplotlib instead of meshplot
    """
    if use_matplotlib or not MESHPLOT_AVAILABLE:
        visualize_correspondence_matplotlib(source_pc, target_pc, correspondence, output_path)
    else:
        visualize_correspondence_meshplot(source_pc, target_pc, correspondence, output_path)


def main():
    parser = argparse.ArgumentParser(description="Compute correspondence between two 3D point clouds")
    parser.add_argument("--source", type=str, required=True, help="Path to source point cloud (PLY/OBJ/NPY)")
    parser.add_argument("--target", type=str, required=True, help="Path to target point cloud (PLY/OBJ/NPY)")
    parser.add_argument("--prompt_source", type=str, required=True, help="Text prompt describing the source object")
    parser.add_argument("--prompt_target", type=str, required=True, help="Text prompt describing the target object")
    parser.add_argument("--output", type=str, default="correspondence.npy", help="Output file for correspondence")
    parser.add_argument("--num_views", type=int, default=100, help="Number of rendered views")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--visualize", action="store_true", help="Visualize correspondence after computing")
    parser.add_argument("--use_matplotlib", action="store_true", help="Use matplotlib instead of meshplot for visualization")
    parser.add_argument("--vis_output", type=str, default=None, help="Path to save visualization image (matplotlib only)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    
    print("Initializing models...")
    pipe = init_pipe(device)
    dino_model = init_dino(device)
    
    print(f"Loading source point cloud: {args.source}")
    source_pc = load_point_cloud(args.source)
    print(f"  Source has {len(source_pc.vert)} points")
    
    print(f"Loading target point cloud: {args.target}")
    target_pc = load_point_cloud(args.target)
    print(f"  Target has {len(target_pc.vert)} points")
    
    print("Computing source features...")
    f_source = compute_features(device, pipe, dino_model, source_pc, args.prompt_source, 
                                 num_views=args.num_views)
    print(f"  Source features shape: {f_source.shape}")
    
    print("Computing target features...")
    f_target = compute_features(device, pipe, dino_model, target_pc, args.prompt_target,
                                 num_views=args.num_views)
    print(f"  Target features shape: {f_target.shape}")
    
    print("Computing correspondence...")
    correspondence, similarity = compute_correspondence(f_source, f_target, device)
    
    print(f"Correspondence shape: {correspondence.shape}")
    print(f"  Each target point (0 to {len(target_pc.vert)-1}) maps to a source point index")
    print(f"  Example: target[0] -> source[{correspondence[0]}]")
    
    # Save results
    np.save(args.output, correspondence)
    print(f"Saved correspondence to {args.output}")
    
    # Also save similarity matrix for analysis
    sim_output = args.output.replace(".npy", "_similarity.npy")
    np.save(sim_output, similarity)
    print(f"Saved similarity matrix to {sim_output}")
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing correspondence...")
        visualize_correspondence(source_pc, target_pc, correspondence, 
                                  output_path=args.vis_output, 
                                  use_matplotlib=args.use_matplotlib)
    
    return correspondence, similarity


if __name__ == "__main__":
    main()
