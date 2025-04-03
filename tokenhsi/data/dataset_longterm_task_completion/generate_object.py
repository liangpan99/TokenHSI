import sys
sys.path.append("./")

import os
import os.path as osp
import numpy as np

import trimesh

import PIL
import json

from tokenhsi.data.dataset_sit.generate_object import get_heightmap_raytrace

all_task_plans = os.listdir(osp.join(osp.dirname(__file__), "task_plans"))

for plan_name in all_task_plans:
    print("Processing Task Plan: [{}]".format(plan_name))

    obj_dir = osp.join(osp.dirname(__file__), "task_plans", plan_name, "objects")
    obj_categories = os.listdir(obj_dir)

    print(obj_dir)
    print(obj_categories)

    # generate object config

    for categeory in obj_categories:

        obj_ids = os.listdir(osp.join(obj_dir, categeory))
        for id in obj_ids:
            curr_dir = osp.join(obj_dir, categeory, id)
            mesh = trimesh.load(osp.join(curr_dir, "geom/mesh.obj"), process=False)

            cfg = {}

            # generate bbox points
            bbox_length = np.max(mesh.vertices, axis=0) - np.min(mesh.vertices, axis=0)
            bbox_center = (np.max(mesh.vertices, axis=0) + np.min(mesh.vertices, axis=0)) / 2
            assert bbox_center.sum() == 0.0
            cfg["bbox"] = bbox_length.tolist()
            cfg["center"] = bbox_center.tolist()
            cfg["facing"] = [0, 1, 0]
            cfg["up_facing"] = [0, 0, 1]

            # generate height map
            height_map = get_heightmap_raytrace(mesh.copy(), HEIGHT_MAP_DIM=128)
            curr_output_dir = osp.join(curr_dir, "height_map")
            os.makedirs(curr_output_dir, exist_ok=True)
            np.save(osp.join(curr_output_dir, "map2d.npy"), height_map)

            height_map2d_int = (height_map * 255).round().astype(np.uint8) # height map image
            img = PIL.Image.fromarray(height_map2d_int)
            # img.show()
            img.save(osp.join(curr_output_dir, "map2d.png"))

            # generate target sit position
            offset = 0.1
            sit_height = height_map[128 // 2, 128 // 2] + offset
            tar_sit_pos = [0.0, 0.0, sit_height - bbox_length[-1] / 2] # always place the point on the (0, 0)

            cfg["tarSitPos"] = tar_sit_pos

            # generate target climb position
            tar_root_height = height_map[128 // 2, 128 // 2]
            tar_climb_pos = [0.0, 0.0, tar_root_height - bbox_length[-1] / 2] # always place the point on the (0, 0)

            cfg["tarClimbPos"] = tar_climb_pos # still use the name of tarSitPos to simplify the env code

            # generate config file
            with open(os.path.join(curr_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            # generate visualization
            components = []

            marker = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
            marker.apply_translation(cfg["tarSitPos"])
            marker.visual.vertex_colors[:, :3] = [255, 0, 0] # red
            components.append(marker)

            marker = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
            marker.apply_translation(cfg["tarClimbPos"])
            marker.visual.vertex_colors[:, :3] = [0, 0, 255] # blue
            components.append(marker)

            components.append(mesh)

            bbox = cfg["bbox"] # bbox
            line_x = trimesh.creation.box([bbox[0], 0.02, 0.02])
            line_y = trimesh.creation.box([0.02, bbox[1], 0.02])
            line_z = trimesh.creation.box([0.02, 0.02, bbox[2]])

            line_x.visual.vertex_colors[:, :3] = [0, 255, 0]
            line_y.visual.vertex_colors[:, :3] = [0, 255, 0]
            line_z.visual.vertex_colors[:, :3] = [0, 255, 0]

            components.append(line_x.copy().apply_translation([0,      bbox[1]/2, -1 * bbox[2]/2]))
            components.append(line_x.copy().apply_translation([0,      bbox[1]/2,      bbox[2]/2]))
            components.append(line_x.copy().apply_translation([0, -1 * bbox[1]/2,      bbox[2]/2]))
            components.append(line_x.copy().apply_translation([0, -1 * bbox[1]/2, -1 * bbox[2]/2]))
            components.append(line_y.copy().apply_translation([     bbox[0]/2, 0, -1 * bbox[2]/2]))
            components.append(line_y.copy().apply_translation([-1 * bbox[0]/2, 0, -1 * bbox[2]/2]))
            components.append(line_y.copy().apply_translation([-1 * bbox[0]/2, 0,      bbox[2]/2]))
            components.append(line_y.copy().apply_translation([     bbox[0]/2, 0,      bbox[2]/2]))
            components.append(line_z.copy().apply_translation([     bbox[0]/2,      bbox[1]/2, 0]))
            components.append(line_z.copy().apply_translation([     bbox[0]/2, -1 * bbox[1]/2, 0]))
            components.append(line_z.copy().apply_translation([-1 * bbox[0]/2, -1 * bbox[1]/2, 0]))
            components.append(line_z.copy().apply_translation([-1 * bbox[0]/2,      bbox[1]/2, 0]))

            # ground plane
            ground = trimesh.creation.box([2, 2, 0.001])
            ground.apply_translation([0, 0, -1 * bbox[2] / 2])
            components.append(ground)

            merged_mesh = trimesh.util.concatenate(components)

            # save
            save_dir = osp.join(obj_dir, categeory, id, "vis")
            os.makedirs(save_dir, exist_ok=True)
            merged_mesh.export(osp.join(save_dir, "mesh.obj"))

            print("Saving processed object mesh at {}".format(save_dir))
