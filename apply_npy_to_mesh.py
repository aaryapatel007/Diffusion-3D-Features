import bpy
import numpy as np
import os
import math
from mathutils import Vector, Quaternion, Matrix
from typing import Tuple, Generator, List, Dict, Any

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
NPY_PATH = "C:\\Users\\aarya\\Downloads\\joint_displacements_abs_dog.npy"
DEBUG = True
TARGET_COLLECTION = "NPY_Targets"
PROXY_ARMATURE_NAME = "NPY_Proxy_Rig"

RECONSTRUCT_ROTATIONS = False
# When True, rotations are reconstructed via a Proxy Rig approach:
#   1. Create proxy armature (duplicate of original, at rest pose)
#   2. Proxy gets COPY_LOCATION + DAMPED_TRACK to match NPY exactly (stretches)
#   3. Bake proxy
#   4. Original gets COPY_ROTATION from proxy (all bones) + COPY_LOCATION (root)
#   5. Bake original -> mesh keeps its proportions, only rotation transferred
# When False, only COPY_LOCATION is applied (position-only, bones stretch).
# ==============================================================================


def log(msg):
    if DEBUG:
        print(msg)


# ==============================================================================
# SCENE NORMALIZATION (same as get_keypoints.py)
# ==============================================================================

def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def get_armature_bbox(arm_obj):
    min_v = Vector((math.inf, math.inf, math.inf))
    max_v = Vector((-math.inf, -math.inf, -math.inf))
    for pbone in arm_obj.pose.bones:
        head_world = arm_obj.matrix_world @ pbone.head
        tail_world = arm_obj.matrix_world @ pbone.tail
        for v in [head_world, tail_world]:
            min_v = Vector((min(min_v.x, v.x), min(min_v.y, v.y), min(min_v.z, v.z)))
            max_v = Vector((max(max_v.x, v.x), max(max_v.y, v.y), max(max_v.z, v.z)))
    return min_v, max_v


def scene_bbox_inclusive(ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in get_scene_meshes():
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = Vector(tuple(min(x, y) for x, y in zip(bbox_min, coord)))
            bbox_max = Vector(tuple(max(x, y) for x, y in zip(bbox_max, coord)))
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            found = True
            arm_min, arm_max = get_armature_bbox(obj)
            bbox_min = Vector(tuple(min(x, y) for x, y in zip(bbox_min, arm_min)))
            bbox_max = Vector(tuple(max(x, y) for x, y in zip(bbox_max, arm_max)))
    if not found:
        raise RuntimeError("no objects in scene")
    return bbox_min, bbox_max


def normalize_scene() -> None:
    log("Normalizing scene...")
    if len(list(get_scene_root_objects())) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox_inclusive()
    max_dim = max(bbox_max - bbox_min)
    if max_dim == 0:
        max_dim = 1.0
    scale = 1 / max_dim
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox_inclusive()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.update()
    log("Scene normalized.")


# ==============================================================================
# JOINT LIST (same as get_keypoints.py)
# ==============================================================================

def get_ordered_joint_list(scene: bpy.types.Scene) -> List[Tuple[Any, Any, bool]]:
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
            joint_list.append((arm, pbone, False))
            children_names = sorted(adj_list.get(pbone.name, []))
            if not children_names:
                joint_list.append((arm, pbone, True))
            else:
                for child_name in reversed(children_names):
                    stack.append(all_bones[child_name])
    return joint_list


# ==============================================================================
# CLEAR & SETUP
# ==============================================================================

def clear_armature(arm_obj):
    log("Clearing armature...")
    if arm_obj.animation_data:
        arm_obj.animation_data.action = None
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    for pbone in arm_obj.pose.bones:
        for c in list(pbone.constraints):
            pbone.constraints.remove(c)
        pbone.location = Vector()
        pbone.rotation_quaternion = Quaternion()
        pbone.rotation_euler = (0, 0, 0)
        pbone.scale = Vector((1, 1, 1))
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()


def get_or_create_collection(name: str):
    if name in bpy.data.collections:
        col = bpy.data.collections[name]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


# ==============================================================================
# CREATE EMPTIES & SETUP CONSTRAINTS
# ==============================================================================

def create_empties_and_constraints(arm_obj, joint_list, npy_data):
    """
    Create animated empties at NPY positions, then set up:
    - COPY_LOCATION on EVERY bone -> positions each bone independently
    - DAMPED_TRACK on EVERY bone -> rotates each bone independently
    """
    num_frames = npy_data.shape[0]
    collection = get_or_create_collection(TARGET_COLLECTION)
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = num_frames - 1

    # --- Create empties ---
    log("Creating target empties...")
    empty_map = {}  # (bone_name, is_tail) -> empty object

    for i, (arm, pbone, is_tail) in enumerate(joint_list):
        if i >= npy_data.shape[1]:
            continue
        suffix = "tail" if is_tail else "head"
        name = f"TGT_{pbone.name}_{suffix}"

        empty = bpy.data.objects.new(name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.01
        collection.objects.link(empty)

        for f in range(num_frames):
            empty.location = Vector(npy_data[f, i])
            empty.keyframe_insert(data_path="location", frame=f)

        empty_map[(pbone.name, is_tail)] = empty

    log(f"Created {len(empty_map)} empties")

    # --- Setup constraints ---
    log("Setting up constraints...")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    for pbone in arm_obj.pose.bones:
        # COPY_LOCATION only: position bone head at NPY position
        head_empty = empty_map.get((pbone.name, False))
        if head_empty:
            c = pbone.constraints.new('COPY_LOCATION')
            c.target = head_empty
            c.owner_space = 'WORLD'
            c.target_space = 'WORLD'

        log(f"  {pbone.name}: LOC={head_empty is not None}")

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    return empty_map


def bake_and_cleanup(arm_obj, num_frames):
    log("Baking to keyframes...")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')

    bpy.ops.nla.bake(
        frame_start=0,
        frame_end=num_frames - 1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        bake_types={'POSE'}
    )

    bpy.ops.object.mode_set(mode='OBJECT')
    log("Baking complete.")

    # Hide empties
    if TARGET_COLLECTION in bpy.data.collections:
        col = bpy.data.collections[TARGET_COLLECTION]
        for obj in col.objects:
            obj.hide_set(True)
            obj.hide_render = True


# ==============================================================================
# PROXY RIG (used when RECONSTRUCT_ROTATIONS = True)
# ==============================================================================

def create_proxy_armature(original_arm_obj):
    """Create a duplicate armature to serve as the Proxy Rig."""
    if PROXY_ARMATURE_NAME in bpy.data.objects:
        obj = bpy.data.objects[PROXY_ARMATURE_NAME]
        bpy.data.objects.remove(obj, do_unlink=True)
        if PROXY_ARMATURE_NAME in bpy.data.armatures:
            bpy.data.armatures.remove(bpy.data.armatures[PROXY_ARMATURE_NAME])

    new_obj = original_arm_obj.copy()
    new_obj.data = original_arm_obj.data.copy()
    new_obj.name = PROXY_ARMATURE_NAME
    new_obj.data.name = PROXY_ARMATURE_NAME
    bpy.context.collection.objects.link(new_obj)

    if new_obj.animation_data:
        new_obj.animation_data.action = None

    bpy.context.view_layer.objects.active = new_obj
    bpy.ops.object.mode_set(mode='POSE')
    for pb in new_obj.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        pb.location = Vector()
        pb.rotation_quaternion = Quaternion()
        pb.scale = Vector((1, 1, 1))
    bpy.ops.object.mode_set(mode='OBJECT')

    new_obj.display_type = 'WIRE'
    new_obj.show_in_front = True
    return new_obj


def setup_proxy_constraints(proxy_arm, empty_map):
    """Proxy gets COPY_LOCATION (head) + DAMPED_TRACK (tail) to match NPY exactly."""
    bpy.context.view_layer.objects.active = proxy_arm
    bpy.ops.object.mode_set(mode='POSE')

    for pb in proxy_arm.pose.bones:
        he = empty_map.get((pb.name, False))
        if he:
            c = pb.constraints.new('COPY_LOCATION')
            c.target = he
            c.owner_space = 'WORLD'
            c.target_space = 'WORLD'

        te = None
        if len(pb.children) == 0:
            te = empty_map.get((pb.name, True))
        elif len(pb.children) == 1:
            te = empty_map.get((pb.children[0].name, False))
        if te:
            c = pb.constraints.new('DAMPED_TRACK')
            c.target = te
            c.track_axis = 'TRACK_Y'

    bpy.ops.object.mode_set(mode='OBJECT')


def setup_original_rotation_constraints(original_arm, proxy_arm):
    """Original gets COPY_ROTATION from Proxy (all bones) + COPY_LOCATION (root only)."""
    bpy.context.view_layer.objects.active = original_arm
    bpy.ops.object.mode_set(mode='POSE')

    for pb in original_arm.pose.bones:
        for c in list(pb.constraints):
            pb.constraints.remove(c)

        target_bone = proxy_arm.pose.bones.get(pb.name)
        if not target_bone:
            continue

        if pb.parent is None:
            c = pb.constraints.new('COPY_LOCATION')
            c.target = proxy_arm
            c.subtarget = target_bone.name
            c.owner_space = 'WORLD'
            c.target_space = 'WORLD'

        c = pb.constraints.new('COPY_ROTATION')
        c.target = proxy_arm
        c.subtarget = target_bone.name
        c.owner_space = 'WORLD'
        c.target_space = 'WORLD'

    bpy.ops.object.mode_set(mode='OBJECT')


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 70)
    print("MOTION TRANSFER")
    print("=" * 70)

    if not os.path.exists(NPY_PATH):
        print(f"ERROR: File not found: {NPY_PATH}")
        return

    log(f"Loading: {NPY_PATH}")
    npy_data = np.load(NPY_PATH)
    log(f"Shape: {npy_data.shape}")
    num_frames = npy_data.shape[0]

    scene = bpy.context.scene
    arm_obj = None
    for o in scene.objects:
        if o.type == 'ARMATURE':
            arm_obj = o
            break

    if not arm_obj:
        print("ERROR: No armature!")
        return

    log(f"Armature: {arm_obj.name}")

    # Step 1: Clear
    clear_armature(arm_obj)

    # Step 2: Normalize
    normalize_scene()

    # Step 3: Get joint list & create empties
    joint_list = get_ordered_joint_list(scene)
    log(f"Joints: {len(joint_list)}")
    empty_map = create_empties_and_constraints(arm_obj, joint_list, npy_data)

    if RECONSTRUCT_ROTATIONS:
        # ---- Proxy Rig path (rotation-preserving) ----
        # Create proxy from CLEAN rest-pose armature (before any baking!)
        log("--- RECONSTRUCT_ROTATIONS: Proxy Rig path ---")
        proxy_arm = create_proxy_armature(arm_obj)

        # Remove COPY_LOCATION from original (it was set by create_empties_and_constraints)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        for pb in arm_obj.pose.bones:
            for c in list(pb.constraints):
                pb.constraints.remove(c)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Proxy stretches to match NPY
        setup_proxy_constraints(proxy_arm, empty_map)

        log("Baking Proxy...")
        scene.frame_start = 0
        scene.frame_end = num_frames - 1
        bpy.context.view_layer.objects.active = proxy_arm
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.nla.bake(
            frame_start=0, frame_end=num_frames - 1,
            only_selected=True, visual_keying=True,
            clear_constraints=True, bake_types={'POSE'}
        )
        bpy.ops.object.mode_set(mode='OBJECT')

        # Original copies ROTATION from Proxy (+ root LOCATION)
        log("Constraining Original to Proxy (rotation only)...")
        setup_original_rotation_constraints(arm_obj, proxy_arm)

        log("Baking Original...")
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.nla.bake(
            frame_start=0, frame_end=num_frames - 1,
            only_selected=True, visual_keying=True,
            clear_constraints=True, bake_types={'POSE'}
        )
        bpy.ops.object.mode_set(mode='OBJECT')

        proxy_arm.hide_viewport = True
        proxy_arm.hide_render = True
    else:
        # ---- Position-only path (bones stretch to NPY positions) ----
        log("--- Position-only path ---")
        bake_and_cleanup(arm_obj, num_frames)

    # Hide empties
    if TARGET_COLLECTION in bpy.data.collections:
        for obj in bpy.data.collections[TARGET_COLLECTION].objects:
            obj.hide_set(True)
            obj.hide_render = True

    scene.frame_set(0)

    print("\n" + "=" * 70)
    print("DONE! Press SPACEBAR to play.")
    print("=" * 70)


if __name__ == "__main__":
    main()
