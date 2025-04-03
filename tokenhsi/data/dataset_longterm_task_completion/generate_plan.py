import sys
sys.path.append("./")

import os
import yaml
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import os.path as osp
import shutil

all_task_plans = os.listdir(osp.join(osp.dirname(__file__), "task_plans"))

for plan_name in all_task_plans:
    print("Processing Task Plan: [{}]".format(plan_name))

    curr_root_dir = osp.join(osp.dirname(__file__), "task_plans", plan_name)
    curr_save_dir = osp.join(osp.dirname(__file__), "task_plans", plan_name, "vis")
    if os.path.exists(curr_save_dir):
        shutil.rmtree(curr_save_dir)
    os.makedirs(curr_save_dir, exist_ok=True)

    # read plan from the config file
    with open(osp.join(curr_root_dir, "cfg.yaml"), "r") as f:
        plan = yaml.load(f, Loader=yaml.SafeLoader)

    # ground plane
    ground = trimesh.creation.box([20, 20, 0.001])
    ground.export(osp.join(curr_save_dir, "ground.obj"))
    
    # load objects and transform them
    for i, v in enumerate(plan["scene"]):
        mesh = trimesh.load(
            os.path.join(
                curr_root_dir, 
                "objects",
                v["category"],
                v["model_id"],
                "geom/mesh.obj"), process=False)

        rot_angle_radian = v["z_rot_angle"]
        rotation = R.from_euler("xyz", [0, 0, rot_angle_radian], degrees=False).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        mesh.apply_transform(transform)

        mesh.apply_translation(v["pos3d"])

        mesh_export_dir = osp.join(curr_save_dir, "obj_{}".format(i))
        os.makedirs(mesh_export_dir, exist_ok=True)
        mesh.export(osp.join(mesh_export_dir, "mesh.obj"))
        
    # load markers indicating the target location
    if "tar_pos" in list(plan.keys()):
        for i, tar_pos in enumerate(plan["tar_pos"]):
            marker = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
            marker.apply_translation(tar_pos)
            marker.visual.vertex_colors[:, :3] = [255, 0, 0]

            marker.export(osp.join(curr_save_dir, "tar_pos_{}.obj").format(i))
    
    if "traj" in list(plan.keys()):
        for k, v in plan["traj"].items():

            components = []
            
            for tar_pos in v:
        
                marker = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
                marker.apply_translation([tar_pos[0], tar_pos[1], 1.0])
                marker.visual.vertex_colors[:, :3] = [255, 0, 0]

                components.append(marker)
            
            components = trimesh.util.concatenate(components)
            components.export(osp.join(curr_save_dir, "traj_{}.obj").format(k))

    # load a platform indicating the initial location of the humanoid
    platform = trimesh.creation.box([0.5, 0.5, 0.05])
    platform.visual.vertex_colors[:, :3] = [0, 255, 0]
    platform.apply_translation([plan["humanoid_init_pos2d"][0], plan["humanoid_init_pos2d"][1], 0.0])
    platform.export(osp.join(curr_save_dir, "humanoid_init_pos2d.obj"))
