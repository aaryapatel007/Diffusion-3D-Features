#!/usr/bin/env python3
"""
ply_frames_to_mp4.py

Usage:
  python ply_frames_to_mp4.py --in_dir ./mesh_to_ply/test --out video.mp4 --fps 24 --width 1280 --height 720

Dependencies:
  pip install open3d plyfile tqdm

Requires ffmpeg available on PATH to assemble PNGs into MP4.
"""

import os
import sys
import argparse
import re
import tempfile
import subprocess
from plyfile import PlyData
import numpy as np
import open3d as o3d
from tqdm import tqdm

NUMERIC = re.compile(r'(\d+)')

def numeric_key(s):
    # extract last numeric group for sorting; fallback to lexicographic
    nums = NUMERIC.findall(s)
    if nums:
        return int(nums[-1])
    return s

def list_ply_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.ply')]
    files.sort(key=lambda p: numeric_key(os.path.basename(p)))
    return files

def read_ply_points_and_colors(path):
    ply = PlyData.read(path)
    v = ply['vertex'].data
    # positions
    x = v['x'].astype(np.float32)
    y = v['y'].astype(np.float32)
    z = v['z'].astype(np.float32)
    pts = np.vstack([x, y, z]).T
    # colors if present (common property names)
    color_names = None
    for candidate in ('red','green','blue','r','g','b'):
        if candidate in v.dtype.names:
            color_names = ('red','green','blue') if 'red' in v.dtype.names else ('r','g','b')
            break
    if color_names and all(c in v.dtype.names for c in color_names):
        cols = np.vstack([v[c].astype(np.uint8) for c in color_names]).T
    else:
        cols = None
    return pts, cols

def make_o3d_pointcloud(pts, cols=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        # normalize colors to 0..1 float
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32) / 255.0)
    else:
        # default small gray
        pcd.paint_uniform_color([0.8,0.8,0.8])
    return pcd

def fit_camera_to_geometry(geometry, center=None, scale=1.0):
    # geometry is an open3d.geometry.PointCloud
    bbox = geometry.get_axis_aligned_bounding_box()
    if center is None:
        center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())
    cam_dist = extent * (1.8 * scale + 0.2)
    # position camera slightly behind -Y and above +Z
    eye = center + np.array([0.0, -cam_dist, cam_dist * 0.6])
    up = np.array([0.0, 0.0, 1.0])
    return eye, center, up

def render_frames_to_pngs(ply_files, out_dir, width=1280, height=720, bgcolor=(0,0,0,1)):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(bgcolor)

    cam_set = False
    png_paths = []

    for i, ply in enumerate(tqdm(ply_files, desc="Rendering")):
        pts, cols = read_ply_points_and_colors(ply)
        pcd = make_o3d_pointcloud(pts, cols)

        # remove previous geometry if present (safe no-op with try/except)
        try:
            renderer.scene.remove_geometry("pcd")
        except Exception:
            pass

        # add geometry with same id every frame
        mat = o3d.visualization.rendering.MaterialRecord()
        # use unlit if we have vertex colors (keeps colors accurate)
        mat.shader = "defaultUnlit" if cols is not None else "defaultLit"
        renderer.scene.add_geometry("pcd", pcd, mat)

        if not cam_set:
            # determine camera from the pointcloud bbox (not renderer.scene)
            eye, center, up = fit_camera_to_geometry(pcd)
            fov = 60.0
            renderer.setup_camera(fov, center, eye, up)
            cam_set = True

        img = renderer.render_to_image()
        out_png = os.path.join(out_dir, f"frame_{i:06d}.png")
        o3d.io.write_image(out_png, img)
        png_paths.append(out_png)

        # cleanup renderer (safe across Open3D versions)
    try:
        renderer.release()
    except AttributeError:
        # older/newer builds may not expose release(); try alternatives or just delete
        try:
            renderer.close()
        except Exception:
            pass
    # ensure object is removed
    try:
        del renderer
    except Exception:
        pass
    return png_paths

def assemble_pngs_to_mp4(png_dir, out_mp4, fps=24, codec='libx264', crf=18):
    # ffmpeg expects sequentially numbered files; we'll use frame_%06d.png
    pattern = os.path.join(png_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        out_mp4
    ]
    print("Running ffmpeg:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="Directory containing per-frame .ply files")
    p.add_argument("--out", required=True, help="Output mp4 path")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--tmp_dir", default=None, help="Optional temp directory for PNGs (defaults to a tempdir)")
    args = p.parse_args()

    ply_files = list_ply_files(args.in_dir)
    if len(ply_files) == 0:
        print("No .ply files found in", args.in_dir)
        sys.exit(1)
    print(f"Found {len(ply_files)} PLYs (first: {os.path.basename(ply_files[0])}, last: {os.path.basename(ply_files[-1])})")

    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="ply_frames_")
    os.makedirs(tmp_dir, exist_ok=True)
    print("Rendering PNGs to", tmp_dir)

    # Render
    pngs = render_frames_to_pngs(ply_files, tmp_dir, width=args.width, height=args.height)

    # Assemble with ffmpeg
    assemble_pngs_to_mp4(tmp_dir, args.out, fps=args.fps)
    print("Wrote MP4:", args.out)
    print("Temp PNGs are in", tmp_dir, "(delete when happy)")

if __name__ == "__main__":
    main()
