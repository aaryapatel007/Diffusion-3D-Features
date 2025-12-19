"""Blender script to visualize correspondence between two skeletons with matching colors."""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple
import bpy
import numpy as np
from mathutils import Vector

def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()
    bpy.ops.object.camera_add()
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"
    bpy.context.scene.camera = new_camera


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def _create_light(
    name: str,
    light_type: str,
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = False
    light_data.energy = energy
    return light_object


def randomize_lighting():
    """Sets up lighting for the scene."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    _create_light("Key_Light", "SUN", (0, 0, 0), (0.785398, 0, -0.785398), 4)
    _create_light("Fill_Light", "SUN", (0, 0, 0), (0.785398, 0, 2.35619), 3)
    _create_light("Rim_Light", "SUN", (0, 0, 0), (-0.785398, 0, -3.92699), 4)
    _create_light("Bottom_Light", "SUN", (0, 0, 0), (3.14159, 0, 0), 2)


def load_points(filepath: str) -> np.ndarray:
    """Load points from npy file, taking frame 0 if 3D array."""
    data = np.load(filepath, allow_pickle=True)
    if data.dtype == object:
        data = np.array(data.tolist(), dtype=np.float32)
    if data.ndim == 3:
        data = data[0]  # Take first frame
    return data.astype(np.float32)


def get_distinct_colors(n: int) -> List[Tuple[float, float, float, float]]:
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (saturation=0.8, value=0.9 for nice colors)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((r, g, b, 1.0))
    return colors


def apply_color_to_object(obj: bpy.types.Object, color: Tuple[float, float, float, float]) -> None:
    """Applies the given color to the object."""
    mat = bpy.data.materials.new(name=f"Material_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def set_camera_front(camera_dist: float = 2.0):
    """Set camera to front view."""
    direction = Vector((0, 1, 0)).normalized()
    camera_pos = -camera_dist * direction
    bpy.context.scene.camera.location = camera_pos
    rot_quat = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()


def create_correspondence_visualization(
    source_file: str,
    target_file: str,
    correspondence_file: str,
    output_path: str,
    marker_radius: float = 0.015,
    separation: float = 0.5,
    rotate_source: bool = False,
    display_scale: float = 0.5,
    camera_dist: float = 2.5,
):
    """
    Create side-by-side visualization of source and target with matching colors.
    
    Args:
        source_file: Path to source points (70 points)
        target_file: Path to target points (78 points)
        correspondence_file: Path to correspondence array (78 -> 70 mapping)
        output_path: Path to save the rendered image
        marker_radius: Radius of joint markers
        separation: Distance between source and target in X axis
        rotate_source: If True, rotate source 90 degrees around X to match target orientation
        display_scale: Size of normalized skeletons (larger = bigger skeletons)
        camera_dist: Camera distance (smaller = more zoomed in)
    """
    reset_cameras()
    reset_scene()
    randomize_lighting()
    
    # Load data
    source_points = load_points(source_file)  # (70, 3)
    target_points = load_points(target_file)  # (78, 3)
    correspondence = np.load(correspondence_file)  # (78,) maps target -> source
    
    print(f"Source points: {source_points.shape}")
    print(f"Target points: {target_points.shape}")
    print(f"Correspondence: {correspondence.shape}")
    
    num_source = source_points.shape[0]
    num_target = target_points.shape[0]
    
    # Generate distinct colors for each source point
    source_colors = get_distinct_colors(num_source)
    
    # Rotate source if needed (90 degrees around X axis: Y -> Z, Z -> -Y)
    if rotate_source:
        print("Rotating source skeleton to match target orientation...")
        rotated_source = source_points.copy()
        # Rotation matrix for 90 degrees around X axis
        rotated_source[:, 1], rotated_source[:, 2] = -source_points[:, 2].copy(), source_points[:, 1].copy()
        source_points = rotated_source
    
    # Center both point clouds first
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # Normalize both to same scale based on bounding box diagonal
    source_bbox = source_centered.max(axis=0) - source_centered.min(axis=0)
    target_bbox = target_centered.max(axis=0) - target_centered.min(axis=0)
    source_extent = np.linalg.norm(source_bbox)
    target_extent = np.linalg.norm(target_bbox)
    
    print(f"Source extent: {source_extent:.4f}")
    print(f"Target extent: {target_extent:.4f}")
    
    # Scale both to unit size, then scale to display size
    if source_extent > 0:
        source_centered = source_centered / source_extent * display_scale
    if target_extent > 0:
        target_centered = target_centered / target_extent * display_scale
    
    print(f"Both skeletons normalized to size: {display_scale}")
    
    # Offset source to the left, target to the right
    source_offset = np.array([-separation, 0, 0])
    target_offset = np.array([separation, 0, 0])
    
    # Create source markers (each with unique color)
    print("Creating source markers...")
    for i in range(num_source):
        loc = source_centered[i] + source_offset
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=marker_radius,
            location=tuple(loc),
        )
        marker = bpy.context.active_object
        marker.name = f"Source_{i:03d}"
        apply_color_to_object(marker, source_colors[i])
    
    # Create target markers (color from corresponding source)
    print("Creating target markers...")
    for i in range(num_target):
        loc = target_centered[i] + target_offset
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=marker_radius,
            location=tuple(loc),
        )
        marker = bpy.context.active_object
        marker.name = f"Target_{i:03d}"
        # Get color from corresponding source point
        source_idx = correspondence[i]
        apply_color_to_object(marker, source_colors[source_idx])
    
    # Add text labels
    # Source label
    bpy.ops.object.text_add(location=(-separation, 0, display_scale * 0.7))
    source_text = bpy.context.active_object
    source_text.data.body = f"Source ({num_source} pts)"
    source_text.data.size = 0.05
    source_text.data.align_x = 'CENTER'
    source_text.rotation_euler = (math.pi/2, 0, 0)
    
    # Target label
    bpy.ops.object.text_add(location=(separation, 0, display_scale * 0.7))
    target_text = bpy.context.active_object
    target_text.data.body = f"Target ({num_target} pts)"
    target_text.data.size = 0.05
    target_text.data.align_x = 'CENTER'
    target_text.rotation_euler = (math.pi/2, 0, 0)
    
    # Set camera
    set_camera_front(camera_dist=camera_dist)
    
    # Render
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved correspondence visualization to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True,
                        help="Path to source points .npy file")
    parser.add_argument("--target_file", type=str, required=True,
                        help="Path to target points .npy file")
    parser.add_argument("--correspondence_file", type=str, required=True,
                        help="Path to correspondence .npy file")
    parser.add_argument("--output", type=str, default="correspondence_vis.png",
                        help="Output image path")
    parser.add_argument("--marker_radius", type=float, default=0.015,
                        help="Radius of joint markers")
    parser.add_argument("--separation", type=float, default=0.5,
                        help="Distance between source and target")
    parser.add_argument("--rotate_source", action="store_true",
                        help="Rotate source 90 degrees around X axis to match target orientation")
    parser.add_argument("--display_scale", type=float, default=0.5,
                        help="Size of normalized skeletons (larger = bigger)")
    parser.add_argument("--camera_dist", type=float, default=2.5,
                        help="Camera distance (smaller = more zoomed in)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["CYCLES", "BLENDER_EEVEE"])

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    # Setup render settings
    scene = bpy.context.scene
    render = scene.render
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100
    scene.render.film_transparent = True

    if render.engine == 'CYCLES':
        scene.cycles.device = "CPU"
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True

    create_correspondence_visualization(
        source_file=args.source_file,
        target_file=args.target_file,
        correspondence_file=args.correspondence_file,
        output_path=args.output,
        marker_radius=args.marker_radius,
        separation=args.separation,
        rotate_source=args.rotate_source,
        display_scale=args.display_scale,
        camera_dist=args.camera_dist,
    )
