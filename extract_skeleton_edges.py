"""Extract skeleton edges from FBX armature."""
import bpy
import numpy as np
import argparse
import sys
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Extract skeleton edges from FBX armature")
parser.add_argument("--fbx_file", type=str, required=True, help="Path to input FBX file")
parser.add_argument("--output_file", type=str, required=True, help="Path to output .npy file for edges")

# Get arguments after '--'
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = []

args = parser.parse_args(argv)

# Clear scene and load FBX
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=args.fbx_file)

# Get ordered joint list (same as get_keypoints.py)
def get_ordered_joint_list(scene):
    joint_list = []
    for arm in [o for o in scene.objects if o.type == "ARMATURE"]:
        adj_list = {}
        all_bones = {}
        roots = []
        for pbone in arm.pose.bones:
            all_bones[pbone.name] = pbone
            if pbone.parent is None:
                roots.append(pbone)
            else:
                parent_name = pbone.parent.name
                if parent_name not in adj_list:
                    adj_list[parent_name] = []
                adj_list[parent_name].append(pbone.name)
        roots = sorted(roots, key=lambda b: b.name)
        stack = list(reversed(roots))
        while stack:
            pbone = stack.pop()
            joint_list.append((arm, pbone, False))  # head
            children_names = sorted(adj_list.get(pbone.name, []))
            if not children_names:
                joint_list.append((arm, pbone, True))  # tail (leaf only)
            else:
                for child_name in reversed(children_names):
                    stack.append(all_bones[child_name])
    return joint_list

scene = bpy.context.scene
joint_list = get_ordered_joint_list(scene)

# Build bone edges: for each bone, connect head to tail or to first child's head
edges = []
joint_to_index = {}
for idx, (arm, pbone, is_tail) in enumerate(joint_list):
    key = (pbone.name, is_tail)
    joint_to_index[key] = idx

for idx, (arm, pbone, is_tail) in enumerate(joint_list):
    if is_tail:
        # Leaf bone: connect head to tail
        head_key = (pbone.name, False)
        if head_key in joint_to_index:
            head_idx = joint_to_index[head_key]
            edges.append((head_idx, idx))
    else:
        # Non-leaf bone: connect to parent's head
        if pbone.parent:
            parent_key = (pbone.parent.name, False)
            if parent_key in joint_to_index:
                parent_idx = joint_to_index[parent_key]
                edges.append((parent_idx, idx))

print(f"Total joints: {len(joint_list)}")
print(f"Total edges: {len(edges)}")
print(f"First 20 edges: {edges[:20]}")

# Create output directory if needed
os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

# Save edges
np.save(args.output_file, np.array(edges, dtype=np.int64))
print(f"Saved skeleton edges to {args.output_file}")
