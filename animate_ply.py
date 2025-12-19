# bake_gltf_apply_displacement.py
# Usage (example):
# blender --background --python bake_gltf_apply_displacement.py -- \
#   --gltf my_anim.glb --corr mesh_to_ply_corr.npy --ply_in first_frame.ply \
#   --out_dir out_frames --start 0 --end 120 --step 1 --save_npz baked_anim.npz

import bpy, sys, os, argparse, numpy as np

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--gltf", required=True)
    p.add_argument("--corr", required=True, help="numpy array: mapping indices (ply->mesh or mesh->ply).")
    p.add_argument("--ply_in", required=True, help="input PLY for first frame (to get base ply order/colors).")
    p.add_argument("--out_dir", default="out_frames")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--save_npz", default=None)
    return p.parse_args()

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_gltf(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError("Unsupported format: " + ext)

def get_first_mesh_object():
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH':
            return ob
    return None

def eval_mesh_vertices(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    mesh = eval_obj.to_mesh()
    verts = np.array([ (v.co.x, v.co.y, v.co.z) for v in mesh.vertices ], dtype=np.float32)
    eval_obj.to_mesh_clear()
    return verts

from plyfile import PlyData

def read_ply_positions(ply_path):
    plydata = PlyData.read(ply_path)
    verts = np.stack([plydata['vertex'][axis] for axis in ('x','y','z')], axis=1).astype(np.float32)

    colors = None
    if all(c in plydata['vertex'].data.dtype.names for c in ('red','green','blue')):
        colors = np.stack([plydata['vertex'][c] for c in ('red','green','blue')], axis=1).astype(np.uint8)
    else:
        colors = np.full((verts.shape[0], 3), 255, dtype=np.uint8)

    return verts, colors

def write_ply(path, points, colors=None):
    n = points.shape[0]
    header = ["ply","format ascii 1.0", f"element vertex {n}",
              "property float x","property float y","property float z"]
    if colors is not None:
        header += ["property uchar red","property uchar green","property uchar blue"]
    header += ["end_header"]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if colors is None:
            for i in range(n):
                x,y,z = points[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for i in range(n):
                x,y,z = points[i]
                r,g,b = colors[i].astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def invert_mapping_if_needed(corr, n_ply, n_mesh, base_ply, base_mesh):
    # Return ply2mesh mapping (length n_ply) always.
    corr = np.asarray(corr, dtype=np.int64)
    if corr.size == n_ply:
        # ply -> mesh already
        return corr
    if corr.size == n_mesh:
        # assume mesh -> ply ; invert
        inv = np.full(n_ply, -1, dtype=np.int64)
        # corr[mesh_idx] = ply_idx
        for mesh_idx, ply_idx in enumerate(corr):
            if 0 <= ply_idx < n_ply:
                inv[ply_idx] = mesh_idx
        # fill any unassigned ply indices by nearest mesh vertex (brute force; may be slow)
        missing = np.where(inv == -1)[0]
        if missing.size > 0:
            print(f"Warning: {missing.size} ply points had no mapping; filling by nearest mesh vertex (brute-force).")
            # compute distances chunked if memory heavy
            mesh_pts = base_mesh
            for mi in missing:
                p = base_ply[mi]
                d = np.linalg.norm(mesh_pts - p[None,:], axis=1)
                inv[mi] = int(d.argmin())
        return inv
    raise ValueError("Corr length doesn't match either ply or mesh vertex count: corr_len=%d, n_ply=%d, n_mesh=%d" %
                     (corr.size, n_ply, n_mesh))

def main():
    args = parse_args()
    clear_scene()
    import_gltf(args.gltf)
    obj = get_first_mesh_object()
    if obj is None:
        raise RuntimeError("No mesh object found in glTF import.")

    # load ply and colors (base ply)
    base_ply, ply_colors = read_ply_positions(args.ply_in)
    n_ply = base_ply.shape[0]

    # load corr
    corr_raw = np.load(args.corr, allow_pickle=True)
    # position scene at start and get base mesh verts
    bpy.context.scene.frame_set(args.start)
    base_mesh_verts = eval_mesh_vertices(obj)
    n_mesh = base_mesh_verts.shape[0]

    # produce ply->mesh mapping
    ply2mesh = invert_mapping_if_needed(corr_raw, n_ply, n_mesh, base_ply, base_mesh_verts)

    # sanity check
    if ply2mesh.max() >= n_mesh or ply2mesh.min() < 0:
        raise RuntimeError("ply2mesh mapping indices out of range after processing.")

    os.makedirs(args.out_dir, exist_ok=True)
    frames_list = []

    for f in range(args.start, args.end + 1, args.step):
        bpy.context.scene.frame_set(f)
        verts_t = eval_mesh_vertices(obj)  # (n_mesh,3)
        if verts_t.shape[0] != n_mesh:
            raise RuntimeError(f"Mesh vertex count changed at frame {f}: was {n_mesh}, now {verts_t.shape[0]}")
        displacement = verts_t - base_mesh_verts        # (n_mesh,3)
        warped_ply = base_ply + displacement[ply2mesh] # (n_ply,3)
        out_ply = os.path.join(args.out_dir, f"ply_frame_{f:04d}.ply")
        write_ply(out_ply, warped_ply, colors=ply_colors)
        frames_list.append(warped_ply)
        print(f"[frame {f}] wrote {out_ply}")

    if args.save_npz:
        np.savez_compressed(args.save_npz, frames=np.stack(frames_list, axis=0))
        print("Saved npz:", args.save_npz)

if __name__ == "__main__":
    main()
