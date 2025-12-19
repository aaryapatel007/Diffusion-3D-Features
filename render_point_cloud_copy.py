from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
import torch
import math
import sys
import trimesh
import numpy as np
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
)

def render_pointcloud_with_normals(device, points, num_views, H, W, add_angle_azi=0, add_angle_ele=0):
    # 1. Estimate normals in object space
    pointcloud = Pointclouds(points=[points])
    normals = estimate_pointcloud_normals(pointcloud, neighborhood_size=16)[0]  # (N, 3)

    # 2. Build cameras
    steps = int(math.sqrt(num_views))
    end = 360 - 360/steps
    elev = torch.linspace(0, end, steps).repeat(steps) + add_angle_ele
    azim = torch.linspace(0, end, steps)
    azim = torch.repeat_interleave(azim, steps) + add_angle_azi

    rotation, translation = look_at_view_transform(dist=2.7, azim=azim, elev=elev, device=device)
    cameras = PerspectiveCameras(R=rotation, T=translation, device=device)

    # 3. Transform normals into camera space
    # normals: (N, 3)
    N = normals.shape[0]
    B = rotation.shape[0]

    # Expand normals to (B, N, 3)
    normals_exp = normals.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)

    # Transform into camera space: (B, N, 3)
    normals_cam = torch.bmm(normals_exp, rotation.transpose(1, 2))  

    # Map to [0,1]
    normals_cam = (normals_cam + 1) / 2

    # 4. Assign normals as per-point features
    pc_with_normals = Pointclouds(points=[points], features=[normals_cam])

    # 5. Render
    raster_settings = PointsRasterizationSettings(
        image_size=H,
        radius=0.01,
        points_per_pixel=1,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    batch_pc = pc_with_normals.extend(num_views)
    normal_batched_renderings = renderer(batch_pc)

    return normal_batched_renderings

def get_colored_depth_maps(raw_depths,H,W):
    import matplotlib
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Greys')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    depth_images = []
    for i in range(raw_depths.size()[0]):
        d = raw_depths[i]
        dmax = torch.max(d) ; dmin = torch.min(d)
        d = (d-dmin)/(dmax-dmin)
        flat_d = d.view(1,-1).cpu().detach().numpy()
        flat_colors = mapper.to_rgba(flat_d)
        depth_colors = np.reshape(flat_colors,(H,W,4))[:,:,:3]
        np_image = depth_colors*255
        np_image = np_image.astype('uint8')
        depth_images.append(np_image)

    return depth_images


@torch.no_grad()
def run_rendering(device, points, num_views, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False,return_images=False):
    if use_normal_map:
        # Estimate normals for the point cloud
        # You might need to adjust the neighborhood size for your specific data
        pointclouds = Pointclouds(points=[points])
        normals = estimate_pointcloud_normals(pointclouds)[0]
        # Convert normals to colors by mapping from [-1, 1] to [0, 1]
        normal_colors = (normals + 1) / 2
        features = normal_colors
    else:
        # Use default white color features
        features = torch.ones_like(points)

    pointclouds = Pointclouds(points=[points], features=[features])

    # pointclouds = Pointclouds(points=[points], features=[torch.ones(points.size()).float().cuda()])
    bbox = pointclouds.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    steps = int(math.sqrt(num_views))
    end = 360 - 360/steps
    elevation = torch.linspace(start = 0 , end = end , steps = steps).repeat(steps) + add_angle_ele
    azimuth = torch.linspace(start = 0 , end = end , steps = steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    #rasterizer
    rasterization_settings = PointsRasterizationSettings(
        image_size=H,
        radius = 0.01,
        points_per_pixel = 1,
        bin_size = 0,
        max_points_per_bin = 0
    )

    #render pipeline
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    batch_renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    batch_points = pointclouds.extend(num_views)
    fragments = rasterizer(batch_points)
    raw_depth = fragments.zbuf

    batched_renderings = batch_renderer(batch_points)
    # normal_batched_renderings = render_pointcloud_with_normals(device,points,num_views,H,W,add_angle_azi,add_angle_ele) if use_normal_map else None
    normal_batched_renderings = batched_renderings if use_normal_map else None

    if not return_images:
        return batched_renderings,normal_batched_renderings,camera,raw_depth
    else:
        list_depth_images_np = get_colored_depth_maps(raw_depth,H,W)
        return batched_renderings,normal_batched_renderings,camera,raw_depth,list_depth_images_np


def batch_render(device, points, num_views, H, W, use_normal_map=False,return_images=False):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(device, points, num_views, H, W, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele, use_normal_map=use_normal_map,return_images=return_images)
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue