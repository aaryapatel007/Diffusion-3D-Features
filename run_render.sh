#!/bin/bash
# Render transferred motion animation using Blender and create video
#
# Usage:
#   ./run_render.sh [input_file] [output_dir] [frame_count]
#
# Example:
#   ./run_render.sh data/animated_target_from_source.npy output_renders 24

BLENDER_PATH="/home/aaryap/research_paper_implementations/Diffusion4D/rendering/blender-3.2.2-linux-x64/blender"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SCRIPT="${SCRIPT_DIR}/render_transferred_motion.py"
VIDEO_SCRIPT="${SCRIPT_DIR}/create_video_from_frames.py"

INPUT_FILE="${1:-data/animated_target_from_source.npy}"
OUTPUT_DIR="${2:-output_renders}"
FRAME_NUM="${3:-24}"
CAMERA_DIST="${4:-4.0}"

echo "=== Rendering Transferred Motion ==="
echo "Input file: ${INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Frame count: ${FRAME_NUM}"
echo "Camera distance: ${CAMERA_DIST}"
echo ""

${BLENDER_PATH} --background --python ${RENDER_SCRIPT} -- \
    --joint_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --frame_num ${FRAME_NUM} \
    --mode_front 1 \
    --abs \
    --marker_radius 0.02 \
    --camera_dist ${CAMERA_DIST} \
    --resolution 512

echo ""
echo "=== Creating Video ==="

# Activate conda environment and run video creation
source /home/aaryap/miniconda3/etc/profile.d/conda.sh
conda activate diff3f

python ${VIDEO_SCRIPT} \
    --input_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/transferred_motion.mp4" \
    --fps 24 \
    --prefix "front_frame"

echo ""
echo "=== Complete ==="
echo "Frames saved to: ${OUTPUT_DIR}/"
echo "Video saved to: ${OUTPUT_DIR}/transferred_motion.mp4"
