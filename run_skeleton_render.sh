#!/bin/bash

# Script to render skeleton animation with bones
# Usage: ./run_skeleton_render.sh <input_file> <output_dir> <frame_count> <camera_dist>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_PATH="${SCRIPT_DIR}/../Diffusion4D/rendering/blender-3.2.2-linux-x64/blender"
RENDER_SCRIPT="${SCRIPT_DIR}/render_transferred_motion.py"
VIDEO_SCRIPT="${SCRIPT_DIR}/create_video_from_frames.py"

INPUT_FILE="${1:-data/animated_target_from_source.npy}"
OUTPUT_DIR="${2:-skeleton_output}"
FRAME_NUM="${3:-24}"
CAMERA_DIST="${4:-3.0}"
SKELETON_EDGES="${5:-data/skeleton_edges_78.npy}"

echo "=== Rendering Skeleton Animation ==="
echo "Input file: ${INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Frame count: ${FRAME_NUM}"
echo "Camera distance: ${CAMERA_DIST}"
echo "Skeleton edges: ${SKELETON_EDGES}"
echo ""

# Check if skeleton edges file exists
if [ ! -f "${SKELETON_EDGES}" ]; then
    echo "ERROR: Skeleton edges file not found: ${SKELETON_EDGES}"
    echo "Please create it first using extract_skeleton_edges.py"
    exit 1
fi

${BLENDER_PATH} --background --python ${RENDER_SCRIPT} -- \
    --joint_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --frame_num ${FRAME_NUM} \
    --mode_front 1 \
    --abs \
    --marker_radius 0.01 \
    --skeleton \
    --skeleton_edges_file "${SKELETON_EDGES}" \
    --bone_radius 0.004 \
    --camera_dist ${CAMERA_DIST} \
    --resolution 512

echo ""
echo "=== Creating Video ==="

python3 ${VIDEO_SCRIPT} \
    --input_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/skeleton_animation.mp4" \
    --fps 24 \
    --prefix "front_frame"

echo ""
echo "=== Complete ==="
echo "Frames saved to: ${OUTPUT_DIR}/"
echo "Video saved to: ${OUTPUT_DIR}/skeleton_animation.mp4"
