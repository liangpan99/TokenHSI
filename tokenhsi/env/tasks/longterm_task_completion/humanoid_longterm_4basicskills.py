# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from enum import Enum
import numpy as np
import torch
import torch.utils.data as torch_data
import yaml
import json
import time
import pickle
import trimesh

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils
from utils import traj_generator_custom

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch, Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    FoVOrthographicCameras,
    MeshRasterizer,
    RasterizationSettings,
    rasterize_meshes,
)

class LightweightMeshRasterizer(MeshRasterizer):
    def __init__(self, cameras=None, raster_settings=None) -> None:
        super().__init__(cameras, raster_settings)

    def forward(self, meshes_world, **kwargs):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()
        if raster_settings.z_clip_value is not None:
            z_clip = raster_settings.z_clip_value
        else:
            znear = cameras.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().item()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

        _, zbuf, _, _ = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )

        return zbuf # only return this value

class ObjectLib():
    def __init__(self, plan, dataset_root, device):
        self.device = device

        # values from obj config
        self._obj_urdfs = []
        self._obj_meshs = []
        self._obj_bbox_centers = []
        self._obj_bbox_lengths = []
        self._obj_facings = []
        self._obj_up_facings = []
        self._obj_on_ground_trans = []
        self._obj_height_maps = []

        self._obj_tar_sit_pos = []
        self._obj_tar_climb_pos = []

        # values from plan config
        self._obj_is_static = []
        self._obj_init_pos3ds = []
        self._obj_init_z_rot_angs = []

        for obj_info in plan["scene"]:
            self._obj_is_static.append(obj_info["static"])

            curr_dir = os.path.join(dataset_root, obj_info["category"], obj_info["model_id"])
            self._obj_urdfs.append(os.path.join(curr_dir, "asset.urdf"))
            if os.path.exists(os.path.join(curr_dir, "geom_downsampled")):
                self._obj_meshs.append(os.path.join(curr_dir, "geom_downsampled/mesh.obj"))
            else:
                self._obj_meshs.append(os.path.join(curr_dir, "geom/mesh.obj"))

            with open(os.path.join(os.getcwd(), curr_dir, "config.json"), "r") as f:
                object_cfg = json.load(f)
                assert not np.sum(np.abs(object_cfg["center"])) > 0.0 
                self._obj_bbox_centers.append(object_cfg["center"])
                self._obj_bbox_lengths.append(object_cfg["bbox"])
                self._obj_facings.append(object_cfg["facing"])
                self._obj_up_facings.append(object_cfg["up_facing"])
                self._obj_tar_sit_pos.append(object_cfg["tarSitPos"])
                self._obj_tar_climb_pos.append(object_cfg["tarClimbPos"])
                self._obj_on_ground_trans.append(-1 * (self._obj_bbox_centers[-1][2] - self._obj_bbox_lengths[-1][2] / 2))

            map2d = np.load(os.path.join(os.getcwd(), curr_dir, "height_map/map2d.npy"))
            self._obj_height_maps.append(map2d)

            self._obj_init_pos3ds.append(obj_info["pos3d"])
            self._obj_init_z_rot_angs.append(obj_info["z_rot_angle"])

        self._num_objects = len(self._obj_urdfs)

        self._obj_is_static = to_torch(self._obj_is_static, dtype=torch.bool, device=self.device)
        self._obj_bbox_centers = to_torch(self._obj_bbox_centers, dtype=torch.float, device=self.device)
        self._obj_bbox_lengths = to_torch(self._obj_bbox_lengths, dtype=torch.float, device=self.device)
        self._obj_facings = to_torch(self._obj_facings, dtype=torch.float, device=self.device)
        self._obj_up_facings = to_torch(self._obj_up_facings, dtype=torch.float, device=self.device)
        self._obj_on_ground_trans = to_torch(self._obj_on_ground_trans, dtype=torch.float, device=self.device)
        self._obj_height_maps = to_torch(np.array(self._obj_height_maps), dtype=torch.float, device=self.device)

        self._obj_tar_sit_pos = to_torch(self._obj_tar_sit_pos, dtype=torch.float, device=self.device)
        self._obj_tar_climb_pos = to_torch(self._obj_tar_climb_pos, dtype=torch.float, device=self.device)

        self._obj_init_pos3ds = to_torch(self._obj_init_pos3ds, dtype=torch.float, device=self.device)
        self._obj_init_z_rot_angs = to_torch(self._obj_init_z_rot_angs, dtype=torch.float, device=self.device)

        self._build_object_bps()

        return
    
    def _build_object_bps(self):

        bps_0 = torch.cat([     self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_1 = torch.cat([-1 * self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_2 = torch.cat([-1 * self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_3 = torch.cat([     self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_4 = torch.cat([     self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_5 = torch.cat([-1 * self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_6 = torch.cat([-1 * self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_7 = torch.cat([     self._obj_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._obj_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._obj_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        
        self._obj_bps = torch.cat([
            bps_0.unsqueeze(1),
            bps_1.unsqueeze(1),
            bps_2.unsqueeze(1),
            bps_3.unsqueeze(1),
            bps_4.unsqueeze(1),
            bps_5.unsqueeze(1),
            bps_6.unsqueeze(1),
            bps_7.unsqueeze(1)]
        , dim=1)
            
        self._obj_bps += self._obj_bbox_centers.unsqueeze(1) # (num_envs, 8, 3)

        return

class HumanoidLongTerm4BasicSkills(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    class TaskUID(Enum):
        traj = 0
        sit = 1
        carry = 2
        climb = 3
    
    class SampleFromUID(Enum):
        traj = 0
        tarpos = 1
        scene = 2

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # read task plan cfg
        assert cfg["env"]["task_plan"] != ""
        with open(cfg["env"]["task_plan"], "r") as f:
            self._plan = yaml.load(f, Loader=yaml.SafeLoader)
            self._plan_name = cfg["env"]["task_plan"].split("/")[-2]
            self._plan_supported_task = self._plan["task_plan"]

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._enable_task_mask_obs = cfg["env"]["enableTaskMaskObs"]
        self._enable_apply_mask_on_task_obs = cfg["env"]["enableApplyMaskOnTaskObs"]

        self._num_tasks = 0
        self._each_subtask_obs_size = []
        self._multiple_task_names = []

        self.register_task_traj_pre_init(cfg)
        self.register_task_sit_pre_init(cfg)
        self.register_task_carry_pre_init(cfg)
        self.register_task_climb_pre_init(cfg)

        self._supported_tasks = cfg["env"]["supportedTasks"] # list of task names
        assert self._multiple_task_names == self._supported_tasks
        
        # height map obs in a dynamic env (static objects + dynamic objects)

        self._use_height_map = cfg["env"]["heightmap"]["use"]
        if self._use_height_map:
            self._cube_side_length = cfg["env"]["heightmap"]["cubeHeightMapSideLength"]
            self._cube_side_num_points = cfg["env"]["heightmap"]["cubeHeightMapSideNumPoints"]
            self._height_map_sensor_num_grid_points = self._cube_side_num_points ** 2
            self._viz_height_map = cfg["env"]["heightmap"]["vizHeightMap"]
            if not cfg["args"].test:
                self._viz_height_map = False

            self._extra_each_subtask_obs_size = self._height_map_sensor_num_grid_points # observations on env

        # task-specific conditional disc
        self._enable_task_specific_disc = cfg["env"]["enableTaskSpecificDisc"]

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidLongTerm4BasicSkills.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {} # to enable multi-skill reference init, use dict instead of list
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._skill = cfg["env"]["skill"]
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch
        assert self._skill_disc_prob.sum() == 1.0
        self._common_skill = cfg["env"].get("commonSkill", "")

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # pre-defined task execution order
        self._task_exec_order_task_uid = to_torch([HumanoidLongTerm4BasicSkills.TaskUID[task_name].value for task_name in self._plan["task_plan"]], dtype=torch.long, device=self.device)
        self._task_exec_order_tar_obj = to_torch([obj_id for obj_id in self._plan["tar_object"]], dtype=torch.long, device=self.device)
        self._task_exec_order_tar_traj = to_torch([obj_id for obj_id in self._plan["tar_traj"]], dtype=torch.long, device=self.device)
        self._task_exec_pointer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # Points to the number of tasks currently being executed. The default is to start from the 0th task.

        
        self._carry_tar_pos = to_torch([tar_pos for tar_pos in self._plan["tar_pos"]], dtype=torch.float, device=self.device)

        self._sample_target_from = self._plan["sample_target_from"]
        self._sample_target_from_source = to_torch(
            [HumanoidLongTerm4BasicSkills.SampleFromUID[x.split("_")[0]].value for x in self._sample_target_from], dtype=torch.long, device=self.device)
        self._sample_target_from_id = to_torch(
            [int(x.split("_")[1]) for x in self._sample_target_from], dtype=torch.long, device=self.device)

        # tensors for task transition
        self._task_transition_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._task_transition_max_steps = cfg["env"]["maxTransitionSteps"]
        if cfg["args"].test:
            self._task_transition_max_steps = cfg["env"]["maxTransitionStepsDemo"]
        self._task_transition_dist_threshold = cfg["env"]["successThreshold"]
        self._enable_IET = cfg["env"]["enableIET"]
        self._IET_max_steps = cfg["env"]["maxIETSteps"]
        self._IET_triggered_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self._enable_dyn_obj_bug_reset = cfg["env"]["enableDynObjBUGReset"]
        self._enable_dyn_obj_fall_termination = cfg["env"]["enableDynObjFallTermination"]
        self._enable_dyn_obj_up_facing_termination = cfg["env"]["enableDynObjUpfacingTermination"]
        self._enable_dyn_obj_up_facing_rwd = cfg["env"]["enableDynObjUpfacingRwd"]

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self._prev_rigid_body_pos = torch.zeros_like(self._rigid_body_pos)
        self._task_mask = torch.zeros([self.num_envs, self._num_tasks], device=self.device, dtype=torch.bool) # indicate which task is being performed via the format of one-hot

        self.register_task_traj_post_init(cfg)
        self.register_task_sit_post_init(cfg)
        self.register_task_carry_post_init(cfg)
        self.register_task_climb_post_init(cfg)
        self.post_process_disc_dataset_collection(cfg)

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        ###########################################
        ###########################################

        num_actors = self.get_num_actors_per_env()

        if (not self.headless):
            idx = self._marker_handles[0]
            self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., idx, :]
            self._marker_pos = self._marker_states[..., :3]
            
            self._marker_actor_ids = self._humanoid_actor_ids + idx

        s_idx = self._object_handles[0][0]
        e_idx = self._object_handles[0][-1] + 1
        self._object_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., s_idx:e_idx, :]
        
        self._object_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._object_handles, dtype=torch.int32, device=self.device)
        self._object_actor_ids = self._object_actor_ids.flatten()

        self._initial_object_states = self._object_states.clone()
        self._initial_object_states[..., 7:13] = 0

        self._prev_object_pos = torch.zeros([self.num_envs, self._obj_lib._num_objects, 3], device=self.device, dtype=torch.float)

        ###########################################
        ###########################################

        # tensors for heightmap
        if self._use_height_map:
            self._build_heightmap_tensors()

        self._enable_climb_human_fall_termination = cfg["env"]["enableClimbHumanFallTermination"]
        if self._enable_climb_human_fall_termination:
            self._termination_heights_backup = self._termination_heights.clone()
            self._termination_heights = torch.broadcast_to(self._termination_heights.unsqueeze(0), (self.num_envs, self._termination_heights.shape[-1])).clone()

        ###### evaluation!!!
        self._is_eval = cfg["args"].eval
        if self._is_eval:

            self._enable_IET = False # as default, we disable this feature

            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)

        return
    
    def _build_heightmap_tensors(self):

        # configs for rendering height map of the dynamic scene
        self._heightmap_spacing = self.cfg["env"]["heightmap"]["FoVSpacing"]
        
        self._heightmap_minx = self._heightmap_miny = -self._heightmap_spacing
        self._heightmap_maxx = self._heightmap_maxy = +self._heightmap_spacing

        self._heightmap_dim = self.cfg["env"]["heightmap"]["dim"]
        self._heightmap_interval = (self._heightmap_maxx - self._heightmap_minx) / self._heightmap_dim

        self._render_device = self.cfg["env"].get("render_device", "")
        if self._render_device == "":
            self._render_device = self.device
        self._render_batch_size = self.cfg["env"]["heightmap"]["batch_size"]

        # load meshes
        filenames = []
        for is_static, path in zip(self._obj_lib._obj_is_static, self._obj_lib._obj_meshs):
            if not is_static:
                filenames.append(path)
            
        if len(filenames) > 0:

            meshes = load_objs_as_meshes(filenames, device=self._render_device, load_textures=False)
            self._dynamic_scene_num_verts_per_mesh = meshes.num_verts_per_mesh()

            self._dynamic_scene = join_meshes_as_scene(meshes, include_textures=False) # merge all individual meshes into a complete scene
            self._dynamic_scene = self._dynamic_scene.extend(self._render_batch_size)
            self._batched_scene_verts = self._dynamic_scene.verts_padded()
            self._batched_scene_faces = self._dynamic_scene.faces_padded()

            # create cameras
            self._dynamic_scene_cam_height = self.cfg["env"]["heightmap"]["camHeight"]
            R, T = look_at_view_transform(eye=[[0, 0, self._dynamic_scene_cam_height]], at=[[0, 0, 0]]) # R: (num_envs, 3, 3), T: (num_envs, 3)
            fl = 1.0 / self._heightmap_spacing
            self._dynamic_scene_camera = OrthographicCameras(device=self._render_device, R=R, T=T, focal_length=fl)

            raster_settings = RasterizationSettings(
                image_size=self._heightmap_dim,
                blur_radius=0.0, 
                faces_per_pixel=1, 
                bin_size=0,
            )
            self._dynamic_scene_rasterizer = LightweightMeshRasterizer(raster_settings=raster_settings, cameras=self._dynamic_scene_camera)

        # build height sensors centered at the humanoid root joint
        assert self._heightmap_spacing * 2 >= self._cube_side_length

        x = torch.linspace(-self._cube_side_length / 2, self._cube_side_length / 2, self._cube_side_num_points).flip(0)
        y = torch.linspace(-self._cube_side_length / 2, self._cube_side_length / 2, self._cube_side_num_points)
        x, y = torch.meshgrid(x, y, indexing="ij")
        z = torch.zeros_like(x)

        self._humanoid_height_sensors = torch.cat(
            [x.reshape(-1).unsqueeze(-1), y.reshape(-1).unsqueeze(-1), z.reshape(-1).unsqueeze(-1)], dim=-1)
        self._humanoid_height_sensors = torch.broadcast_to(
            self._humanoid_height_sensors.unsqueeze(0), 
            (self.num_envs, self._humanoid_height_sensors.shape[0], self._humanoid_height_sensors.shape[1])
        ).to(self.device)

        # build markers
        if self._viz_height_map and (not self.headless):
            num_actors = self._root_states.shape[0] // self.num_envs

            s_idx = self._height_map_marker_handles[0][0]
            e_idx = self._height_map_marker_handles[0][-1] + 1
            self._height_map_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., s_idx:e_idx, :]
            self._height_map_marker_pos = self._height_map_marker_states[..., :3]

            self._height_map_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._height_map_marker_handles, dtype=torch.int32, device=self.device)
            self._height_map_marker_actor_ids = self._height_map_marker_actor_ids.flatten()

            # this tensor is on the sim_device
            self._humanoid_height_values = torch.zeros((self.num_envs, self._height_map_sensor_num_grid_points), dtype=torch.float, device=self.device)

        return
    
    def register_task_traj_pre_init(self, cfg):
        k = "traj"
        assert HumanoidLongTerm4BasicSkills.TaskUID[k].value >= 0

        self._num_traj_samples = cfg["env"][k]["numTrajSamples"]
        self._traj_sample_timestep = cfg["env"][k]["trajSampleTimestep"]
        self._speed_min = cfg["env"][k]["speedMin"]
        self._speed_max = cfg["env"][k]["speedMax"]
        self._accel_max = cfg["env"][k]["accelMax"]
        # self._sharp_turn_prob = cfg["env"][k]["sharpTurnProb"] # The trajectory is specified in advance, not randomly generated, so these two parameters are useless.
        # self._sharp_turn_angle = cfg["env"][k]["sharpTurnAngle"]
        self._fail_dist = cfg["env"][k]["failDist"]

        self._enable_extended_traj = cfg["env"][k]["enableExtendedTraj"]
        self._extend_dist = cfg["env"][k]["extendDist"]

        self._num_tasks += 1
        self._each_subtask_obs_size.append(2 * self._num_traj_samples)
        self._multiple_task_names.append(k)

        return
    
    def register_task_traj_post_init(self, cfg):
        k = "traj"

        self._traj_skill = cfg["env"][k]["skill"]
        # self._traj_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        self._traj_begin_progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # record the progress value when traj task beginning

        self._traj_gen = traj_generator_custom.CustomTrajGenerator(
            self._plan["traj"],
            self.device, self.dt,
            self._speed_min, self._speed_max, self._accel_max,
            self._enable_extended_traj, self._extend_dist,
        )

        if (not self.headless):
            num_actors = self._root_states.shape[0] // self.num_envs

            s_idx = self._traj_marker_handles[0][0]
            e_idx = self._traj_marker_handles[0][-1] + 1

            self._traj_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., s_idx:e_idx, :]
            self._traj_marker_pos = self._traj_marker_states[..., :3]

            self._traj_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._traj_marker_handles, dtype=torch.int32, device=self.device)
            self._traj_marker_actor_ids = self._traj_marker_actor_ids.flatten()

        return
    
    def register_task_sit_pre_init(self, cfg):
        k = "sit"
        assert HumanoidLongTerm4BasicSkills.TaskUID[k].value >= 0

        self._num_tasks += 1
        self._each_subtask_obs_size.append(3 + 3 + 6 + 2 + 8 * 3) # target sit position, object pos + 6d rot, object 2d facing dir, object bps
        self._multiple_task_names.append(k)

        return
    
    def register_task_sit_post_init(self, cfg):
        k = "sit"

        self._sit_skill = cfg["env"][k]["skill"]
        # self._sit_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        return
    
    def register_task_carry_pre_init(self, cfg):
        k = "carry"
        assert HumanoidLongTerm4BasicSkills.TaskUID[k].value >= 0

        self._carry_rwd_only_vel_reward = cfg["env"][k]["onlyVelReward"]
        self._carry_rwd_only_height_handheld_reward = cfg["env"][k]["onlyHeightHandHeldReward"]

        self._carry_rwd_box_vel_penalty = cfg["env"][k]["box_vel_penalty"]
        self._carry_rwd_box_vel_pen_coeff = cfg["env"][k]["box_vel_pen_coeff"]
        self._carry_rwd_box_vel_pen_thre = cfg["env"][k]["box_vel_pen_threshold"]

        self._num_tasks += 1
        self._each_subtask_obs_size.append(3 + 3 + 6 + 3 + 3 + 8 * 3) # target box location, box state: pos (3) + rot (6) + lin vel (3) + ang vel (6), bps
        self._multiple_task_names.append(k)

        return
    
    def register_task_carry_post_init(self, cfg):
        k = "carry"

        self._carry_skill = cfg["env"][k]["skill"]
        # self._carry_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        return
    
    def register_task_climb_pre_init(self, cfg):
        k = "climb"
        assert HumanoidLongTerm4BasicSkills.TaskUID[k].value >= 0

        self._num_tasks += 1
        self._each_subtask_obs_size.append(3 + 8 * 3) # target root position, object bps
        self._multiple_task_names.append(k)

        return
    
    def register_task_climb_post_init(self, cfg):
        k = "climb"

        self._climb_skill = cfg["env"][k]["skill"]
        # self._climb_skill_init_prob = torch.tensor(cfg["env"][k]["skillInitProb"], device=self.device, dtype=torch.float)

        return
    
    def post_process_disc_dataset_collection(self, cfg):
        skill_idx = {sk_name: i for i, sk_name in enumerate(self._skill)}

        task_names = self._multiple_task_names
        for i, n in enumerate(task_names):
            print("checking whether task {} is enabled".format(n))
            if n not in self._plan["task_plan"]:
                print("task {} is not enabled".format(n))

                its_skill_set = cfg["env"][n]["skill"]
                for k in its_skill_set:
                    if k != self._common_skill:
                        self._skill_disc_prob[skill_idx[k]] = 0.0 # clear its skill init prob for training disc
        
        print("disc skill: ", self._skill)
        print("disc prob: ", self._skill_disc_prob)

        scale_coeff = 1.0 / sum(self._skill_disc_prob)
        self._skill_disc_prob *= scale_coeff

        assert torch.abs(self._skill_disc_prob.sum() - 1.0 ) < 1e-5

        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = sum(self._each_subtask_obs_size)
            if (self._enable_task_mask_obs):
                obs_size += self._num_tasks
            if self._use_height_map:
                obs_size += self._extra_each_subtask_obs_size # The only additional obs (height map) for this environment
        return obs_size

    def get_multi_task_info(self):

        num_subtasks = self._num_tasks
        each_subtask_obs_size = self._each_subtask_obs_size

        each_subtask_obs_mask = torch.zeros(num_subtasks, sum(each_subtask_obs_size), dtype=torch.bool, device=self.device)

        index = torch.cumsum(torch.tensor([0] + each_subtask_obs_size), dim=0).to(self.device)
        for i in range(num_subtasks):
            each_subtask_obs_mask[i, index[i]:index[i + 1]] = True

        onehot_index = torch.zeros(2, dtype=torch.long, device=self.device)
        onehot_index[0] = index[-1]
        onehot_index[1] = index[-1] + num_subtasks

        if self._use_height_map:
            # Currently only supports 1 additional obs so we can hack like this
            extra_index = torch.cumsum(torch.tensor([0, self._extra_each_subtask_obs_size]), dim=0).to(self.device)
            extra_index[:] += onehot_index[-1]

        info = {
            "onehot_size": num_subtasks,
            "onehot_indx": onehot_index,
            "tota_subtask_obs_size": sum(each_subtask_obs_size),
            "each_subtask_obs_size": each_subtask_obs_size,
            "each_subtask_obs_mask": each_subtask_obs_mask,
            "each_subtask_obs_indx": index,
            "enable_task_mask_obs": self._enable_task_mask_obs,

            "each_subtask_name": self._multiple_task_names,
            "plan_supported_task": self._plan_supported_task,
        }

        if self._use_height_map:
            info["has_extra"] = True
            info["extra_each_subtask_obs_size"] = self._extra_each_subtask_obs_size
            info["extra_each_subtask_obs_indx"] = extra_index
        else:
            info["has_extra"] = False
        return info

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_root_rot[:] = self._humanoid_root_states[..., 3:7]
        self._prev_object_pos[:] = self._object_states[..., 0:3]
        self._prev_rigid_body_pos[:] = self._rigid_body_pos.clone()
        return

    def _update_task(self):
            
            step_tensor = torch.zeros_like(self.progress_buf)
            step_tensor[:] = self._task_transition_max_steps

            traj_mask = (self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["traj"].value)
            step_tensor[traj_mask] = 5

            # climb_mask = (self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["climb"].value)
            # step_tensor[climb_mask] = 60
            
            subgoal_finished_mask = (self._task_transition_step_buf == (step_tensor - 1)) # Update only once

            num_total_subgoals = len(self._task_exec_order_task_uid) # how many subgoals we need to complete in an episode
            still_in_process_mask = self._task_exec_pointer < num_total_subgoals - 1

            # task transition
            update_mask = torch.logical_and(subgoal_finished_mask, still_in_process_mask)
            if update_mask.sum() > 0:
                
                if self._enable_climb_human_fall_termination:

                    executing_climb = torch.logical_and(
                        update_mask,
                        self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["climb"].value
                    )
                    self._termination_heights[executing_climb] += (self._humanoid_root_states[executing_climb, 2] - self._char_h).unsqueeze(-1)

                # clear
                self._task_transition_step_buf[update_mask] = 0

                # push forward
                self._task_exec_pointer[update_mask] += 1

                # update
                self._task_mask[update_mask, :] = False
                self._task_mask[update_mask, self._task_exec_order_task_uid[self._task_exec_pointer[update_mask]]] = True
                assert self._task_mask.sum() == self.num_envs

                traj_mask = (self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["traj"].value) & update_mask
                if traj_mask.sum() > 0:
                    self._traj_begin_progress_buf[traj_mask] = self.progress_buf[traj_mask] + 1

            return

    def _update_marker(self):
        traj_samples = self._fetch_traj_samples()
        self._traj_marker_pos[:] = traj_samples
        self._traj_marker_pos[..., 2] = self._char_h

        ####### get tar pos

        env_ids = torch.arange(0, self.num_envs, device=self.device)
        tar_pos = torch.zeros_like(self._humanoid_root_states[..., 0:3])

        sample_tar_from_sources = self._sample_target_from_source[self._task_exec_pointer[env_ids]]
        sample_tar_from_ids = self._sample_target_from_id[self._task_exec_pointer[env_ids]]

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.traj.value)
        if mask.sum() > 0:
            traj_ids_flat = sample_tar_from_ids[mask]
            traj_timesteps = torch.ones_like(traj_ids_flat) * torch.inf
            tar_pos[mask] = self._traj_gen.calc_pos(traj_ids_flat, traj_timesteps.flatten())
            tar_pos[mask, -1] = self._char_h

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.tarpos.value)
        if mask.sum() > 0:
            tar_pos[mask] = self._carry_tar_pos[sample_tar_from_ids[mask]]

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.scene.value)
        if mask.sum() > 0:
            oid = sample_tar_from_ids[mask]
            task = self._task_exec_order_task_uid[self._task_exec_pointer[env_ids[mask]]]
            masked_ids = env_ids[mask]

            task_equal_sit = (task == HumanoidLongTerm4BasicSkills.TaskUID.sit.value)
            if task_equal_sit.sum() > 0:
                curr_env_ids = masked_ids[task_equal_sit]
                tar_pos[curr_env_ids] = self._obj_lib._obj_tar_sit_pos[oid[task_equal_sit]] + self._object_states[curr_env_ids, oid[task_equal_sit], 0:3]
            
            task_equal_climb = (task == HumanoidLongTerm4BasicSkills.TaskUID.climb.value)
            if task_equal_climb.sum() > 0:
                curr_env_ids = masked_ids[task_equal_climb]
                tar_pos[curr_env_ids] = self._obj_lib._obj_tar_climb_pos[oid[task_equal_climb]] + self._object_states[curr_env_ids, oid[task_equal_climb], 0:3]
                tar_pos[curr_env_ids, 2] += self._char_h
        
        self._marker_pos[:] = tar_pos # 3d xyz

        actor_ids = torch.cat([self._traj_marker_actor_ids, self._marker_actor_ids, self._object_actor_ids,], dim=0)

        if self._use_height_map:
            if self._viz_height_map and (not self.headless):

                # draw humanoid_height_sensors

                # humanoid local space to world space
                num_markers = self._height_map_sensor_num_grid_points
                root_rot_xy = torch_utils.calc_heading_quat(self._humanoid_root_states[..., 3:7])
                root_rot_xy_exp = torch.broadcast_to(root_rot_xy.unsqueeze(1), (root_rot_xy.shape[0], num_markers, root_rot_xy.shape[1]))
                grid = quat_rotate(root_rot_xy_exp.reshape(-1, 4), self._humanoid_height_sensors.reshape(-1, 3))
                grid = grid.reshape(-1, num_markers, 3)
                grid += self._humanoid_root_states[..., 0:3].unsqueeze(1)
                grid[..., -1] = self._humanoid_height_values

                self._height_map_marker_pos[:] = grid

                actor_ids = torch.cat([actor_ids, self._height_map_marker_actor_ids], dim=0)
       
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._traj_marker_handles = [[] for _ in range(self.num_envs)]
            if self._use_height_map:
                if self._viz_height_map:
                    self._height_map_marker_handles = [[] for _ in range(num_envs)]
            self._marker_handles = []
            self._load_marker_asset()
        
        self._obj_lib = ObjectLib(
            plan=self._plan,
            dataset_root=os.path.join(os.path.dirname(self.cfg['env']['task_plan']), "objects"),
            device=self.device,
        )

        # load physical assets
        self._object_handles = [[] for _ in range(num_envs)]
        self._object_assets = self._load_object_asset(self._obj_lib._obj_urdfs, self._obj_lib._obj_is_static)

        super()._create_envs(num_envs, spacing, num_per_row)
        return
    
    def _load_marker_asset(self):
        asset_root = "tokenhsi/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _load_object_asset(self, object_urdfs, object_is_static):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Load materials from meshes
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        # use default convex decomposition params
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.vhacd_params.max_convex_hulls = 128
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        asset_options.replace_cylinder_with_capsule = False # support cylinder

        asset_root = "./"
        object_assets = []
        for is_static, urdf in zip(object_is_static, object_urdfs):
            asset_options.fix_base_link = is_static
            object_assets.append(self.gym.load_asset(self.sim, asset_root, urdf, asset_options))

        return object_assets

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_object(env_id, env_ptr)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return
    
    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs + 10
        col_filter = 1
        segmentation_id = 0
        default_pose = gymapi.Transform()

        # traj markers
        for i in range(self._num_traj_samples):

            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
            self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          gymapi.Vec3(1.0, 0.0, 0.0))
            self._traj_marker_handles[env_id].append(marker_handle)
        
        if self._use_height_map:
            if self._viz_height_map and (not self.headless):
                ##### marker to indicate circle height map
                for i in range(self._height_map_sensor_num_grid_points):
                    
                    num_points_each_line = self._cube_side_num_points

                    if i in [n for n in range(0, num_points_each_line)]:
                        color = gymapi.Vec3(0.0, 0.0, 1.0)
                    else:
                        color = gymapi.Vec3(0.0, 1.0, 0.0)

                    marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
                    self.gym.set_actor_scale(env_ptr, marker_handle, 0.2)
                    self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                                gymapi.MESH_VISUAL,
                                                color)
                    self._height_map_marker_handles[env_id].append(marker_handle)
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
        self._marker_handles.append(marker_handle)

        return
    
    def _build_object(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i in range(len(self._object_assets)):
            default_pose = gymapi.Transform()
            default_pose.p.x = self._obj_lib._obj_init_pos3ds[i][0]
            default_pose.p.y = self._obj_lib._obj_init_pos3ds[i][1]
            default_pose.p.z = self._obj_lib._obj_init_pos3ds[i][2]

            z_rot_angle = self._obj_lib._obj_init_z_rot_angs[i].reshape(1,)
            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3)
            quat = quat_from_angle_axis(z_rot_angle, axis)
            default_pose.r.x = quat[0, 0]
            default_pose.r.y = quat[0, 1]
            default_pose.r.z = quat[0, 2]
            default_pose.r.w = quat[0, 3]

            object_handle = self.gym.create_actor(env_ptr, self._object_assets[i], default_pose, "object_{}".format(i), col_group, col_filter, segmentation_id)
            self._object_handles[env_id].append(object_handle)

        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return
    
    def _fetch_traj_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = (self.progress_buf[env_ids] - self._traj_begin_progress_buf[env_ids]) * self.dt
        timesteps = torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape).flatten()
        
        traj_ids_flat = self._task_exec_order_tar_traj[self._task_exec_pointer[env_ids_tiled]]

        traj_samples_flat = self._traj_gen.calc_pos(traj_ids_flat, traj_timesteps.flatten())
        traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], self._num_traj_samples, traj_samples_flat.shape[-1]))

        return traj_samples

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        num_envs = len(env_ids)

        root_states = self._humanoid_root_states[env_ids]
        object_states = self._object_states[env_ids]

        if self._use_height_map:
            height_sensors = self._humanoid_height_sensors[env_ids]

        tar_obj_ids = self._task_exec_order_tar_obj[self._task_exec_pointer[env_ids]]

        tar_obj_states = self._object_states[env_ids, tar_obj_ids]
        tar_obj_bps = self._obj_lib._obj_bps[tar_obj_ids]
        tar_obj_facings = self._obj_lib._obj_facings[tar_obj_ids]
        
        ####### get tar pos

        tar_pos = torch.zeros_like(self._humanoid_root_states[env_ids, 0:3])

        sample_tar_from_sources = self._sample_target_from_source[self._task_exec_pointer[env_ids]]
        sample_tar_from_ids = self._sample_target_from_id[self._task_exec_pointer[env_ids]]

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.traj.value)
        if mask.sum() > 0:
            traj_ids_flat = sample_tar_from_ids[mask]
            traj_timesteps = torch.ones_like(traj_ids_flat) * torch.inf
            tar_pos[mask] = self._traj_gen.calc_pos(traj_ids_flat, traj_timesteps.flatten())
            tar_pos[mask, -1] = self._char_h

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.tarpos.value)
        if mask.sum() > 0:
            tar_pos[mask] = self._carry_tar_pos[sample_tar_from_ids[mask]]

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.scene.value)
        if mask.sum() > 0:
            oid = sample_tar_from_ids[mask]
            task = self._task_exec_order_task_uid[self._task_exec_pointer[env_ids[mask]]]
            masked_ids = env_ids[mask]

            task_equal_sit = (task == HumanoidLongTerm4BasicSkills.TaskUID.sit.value)
            if task_equal_sit.sum() > 0:
                curr_env_ids = masked_ids[task_equal_sit]
                tar_pos[curr_env_ids] = self._obj_lib._obj_tar_sit_pos[oid[task_equal_sit]] + self._object_states[curr_env_ids, oid[task_equal_sit], 0:3]
            
            task_equal_climb = (task == HumanoidLongTerm4BasicSkills.TaskUID.climb.value)
            if task_equal_climb.sum() > 0:
                curr_env_ids = masked_ids[task_equal_climb]
                tar_pos[curr_env_ids] = self._obj_lib._obj_tar_climb_pos[oid[task_equal_climb]] + self._object_states[curr_env_ids, oid[task_equal_climb], 0:3]
                tar_pos[curr_env_ids, 2] += self._char_h

        sit_object_states = tar_obj_states
        sit_object_bps = tar_obj_bps
        sit_object_facings = tar_obj_facings
        sit_tar_pos = tar_pos

        box_states = tar_obj_states
        box_bps = tar_obj_bps
        box_tar_pos = tar_pos

        climb_object_states = tar_obj_states
        climb_object_bps = tar_obj_bps
        climb_tar_pos = tar_pos
        
        task_mask = self._task_mask[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)
        obs = compute_location_observations(root_states, traj_samples,
                                            sit_tar_pos, sit_object_states, sit_object_bps, sit_object_facings,
                                            box_states, box_bps, box_tar_pos,
                                            climb_object_states, climb_object_bps, climb_tar_pos,
                                            task_mask, self.get_multi_task_info()["each_subtask_obs_mask"], self._enable_apply_mask_on_task_obs)

        if (self._enable_task_mask_obs):
            obs = torch.cat([obs, task_mask.float()], dim=-1)

        ###### surrounding heights

        if self._use_height_map:

            ## static scene
            mask = self._obj_lib._obj_is_static.clone()
            heights_of_statc_objs = sample_surrounding_heights(
                root_states[..., 0:3],
                root_states[..., 3:7],
                object_states[..., 0:3][:, mask],
                object_states[..., 3:7][:, mask],
                self._obj_lib._obj_bbox_lengths[mask],
                self._obj_lib._obj_height_maps[mask],
                self._obj_lib._obj_on_ground_trans[mask],
                height_sensors,
            )

            ## dynamic scene
            mask = ~self._obj_lib._obj_is_static.clone()
            num_dyn_objs = mask.sum()
            if num_dyn_objs > 0:
                dyn_obj_states = object_states[:, mask]
                dyn_obj_pos_exp = []
                dyn_obj_rot_exp = []
                root_pos_exp = []
                for i in range(num_dyn_objs):
                    root_pos_exp.append(torch.broadcast_to(root_states[..., 0:3].unsqueeze(1), (num_envs, self._dynamic_scene_num_verts_per_mesh[i], 3)))
                    dyn_obj_pos_exp.append(torch.broadcast_to(dyn_obj_states[:, i, 0:3].unsqueeze(1), (num_envs, self._dynamic_scene_num_verts_per_mesh[i], 3)))
                    dyn_obj_rot_exp.append(torch.broadcast_to(dyn_obj_states[:, i, 3:7].unsqueeze(1), (num_envs, self._dynamic_scene_num_verts_per_mesh[i], 4)))
                dyn_obj_pos_exp = torch.cat(dyn_obj_pos_exp, dim=1).to(self._render_device)
                dyn_obj_rot_exp = torch.cat(dyn_obj_rot_exp, dim=1).to(self._render_device)
                root_pos_exp = torch.cat(root_pos_exp, dim=1).to(self._render_device)
                root_pos_exp[..., -1] = 0.0 # zero height

                total_data_ids = torch.arange(0, num_envs, device=self._render_device, dtype=torch.long)
                data_loader = torch_data.DataLoader(
                    dataset=torch_data.TensorDataset(total_data_ids, dyn_obj_pos_exp, dyn_obj_rot_exp, root_pos_exp),
                    batch_size=self._render_batch_size,
                    shuffle=False,
                )

                heights_of_dynamic_objs = torch.zeros(
                    (num_envs, self._heightmap_dim, self._heightmap_dim), device=self._render_device, dtype=torch.float)

                # t0 = time.time()

                with torch.no_grad():

                    for step, data in enumerate(data_loader):

                        curr_data_ids, curr_trans, curr_rot, curr_humanoid_root_trans = data
                        num_samples = len(curr_data_ids)

                        curr_batched_scene_verts = self._batched_scene_verts[:num_samples]
                        curr_batched_scene_faces = self._batched_scene_faces[:num_samples]

                        transformed_verts = (
                            quat_rotate(curr_rot.reshape(-1, 4), curr_batched_scene_verts.reshape(-1, 3)) \
                            + curr_trans.reshape(-1, 3) \
                            - curr_humanoid_root_trans.reshape(-1, 3) # convert to humanoid local space
                        ).reshape(num_samples, -1, 3)

                        # t = time.time()
                        scene = Meshes(verts=transformed_verts, faces=curr_batched_scene_faces)
                        # print(f"Time for creating scene: {time.time() - t}, step: {step}, num_samples: {num_samples}")

                        # t = time.time()
                        zbuf = self._dynamic_scene_rasterizer(scene)
                        # print(f"Time for rasterizing: {time.time() - t}")

                        depth = zbuf[..., 0]
                        negative_mask = (depth < 0.0)
                        height = self._dynamic_scene_cam_height - depth
                        height[negative_mask] = 0.0

                        heights_of_dynamic_objs[curr_data_ids] = height.clone()

                # t = time.time()
                # print(f"Time for computing heights: {t - t0}")

                ######## sample height values around the character
                sampled_heights = sample_surrounding_heights_v_dynamic(
                    root_states[..., 0:3],
                    root_states[..., 3:7],
                    height_sensors,
                    heights_of_dynamic_objs.to(self.device), # transfer tensor from render_device to sim_device
                    self._heightmap_spacing
                )

                heights_of_statc_objs = torch.cat([heights_of_statc_objs, sampled_heights.unsqueeze(1)], dim=1)

            heights = heights_of_statc_objs.max(dim=1)[0]

            if self._viz_height_map and (not self.headless):
                self._humanoid_height_values[env_ids] = heights.clone()

            if self.cfg["env"]["heightmap"]["localHeightObs"]:
                heights = heights - root_states[..., 2:3]

            obs = torch.cat([obs, heights], dim=-1)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        object_pos = self._object_states[..., 0:3]
        object_rot = self._object_states[..., 3:7]

        rigid_body_pos = self._rigid_body_pos
        hands_ids = self._key_body_ids[[0, 1]]
        feet_ids = self._key_body_ids[[2, 3]]

        traj_tar_pos = self._fetch_traj_samples()[:, 0]

        reward = self.rew_buf.clone()

        # speicify task rwd func
        if "0" in self._plan_name or "1" in self._plan_name or "2" in self._plan_name:

            climb_tar_pos = object_pos[:, 0] + self._obj_lib._obj_tar_climb_pos[0].unsqueeze(0)
            climb_tar_pos[:, 2] += self._char_h

            sit_tar_pos = object_pos[:, 1] + self._obj_lib._obj_tar_sit_pos[1].unsqueeze(0)

            reward = compute_0_reward(
                root_pos, self._prev_root_pos, 
                traj_tar_pos, HumanoidLongTerm4BasicSkills.TaskUID.traj.value,
                object_pos[:, 0], self._prev_object_pos[:, 0], object_rot[:, 0], self.dt, self._carry_tar_pos[0], self._obj_lib._obj_bbox_lengths[0],
                self._enable_dyn_obj_up_facing_rwd,  self._obj_lib._obj_up_facings[0], HumanoidLongTerm4BasicSkills.TaskUID.carry.value,
                object_pos[:, 0], climb_tar_pos, rigid_body_pos, feet_ids, self._char_h, HumanoidLongTerm4BasicSkills.TaskUID.climb.value,
                object_pos[:, 1], sit_tar_pos, HumanoidLongTerm4BasicSkills.TaskUID.sit.value,
                self._task_exec_order_task_uid[self._task_exec_pointer],
            )

            humanoid_vel_penalty = compute_box_vel_penalty(root_pos, self._prev_root_pos, self.dt, 1.0, 1.5)
            reward += humanoid_vel_penalty

            right_foot_penalty = compute_box_vel_penalty_rf(rigid_body_pos[:, 11], self._prev_rigid_body_pos[:, 11], self.dt, 0.5, 3.0)
            left_foot_penalty = compute_box_vel_penalty_lf(rigid_body_pos[:, 14], self._prev_rigid_body_pos[:, 14], self.dt, 0.5, 3.0)

            reward += left_foot_penalty + right_foot_penalty

        else:
            raise NotImplementedError

        
        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self._power_coefficient * power

        if self._power_reward:
            self.rew_buf[:] = reward + power_reward
        else:
            self.rew_buf[:] = reward

        # task transition

        traj_env_ids = self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["traj"].value
        sit_env_ids = self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["sit"].value
        carry_env_ids = self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["carry"].value
        climb_env_ids = self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["climb"].value

        env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)

        tar_obj_ids = self._task_exec_order_tar_obj[self._task_exec_pointer]
        tar_obj_pos = self._object_states[env_ids, tar_obj_ids, 0:3]

        anchor_pos = root_pos.clone()
        anchor_pos[carry_env_ids] = tar_obj_pos[carry_env_ids]
        
        ####### get tar pos

        env_ids = torch.arange(0, self.num_envs, device=self.device)
        tar_pos = torch.zeros_like(self._humanoid_root_states[..., 0:3])

        sample_tar_from_sources = self._sample_target_from_source[self._task_exec_pointer[env_ids]]
        sample_tar_from_ids = self._sample_target_from_id[self._task_exec_pointer[env_ids]]

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.traj.value)
        if mask.sum() > 0:
            traj_ids_flat = sample_tar_from_ids[mask]
            traj_timesteps = torch.ones_like(traj_ids_flat) * torch.inf
            tar_pos[mask] = self._traj_gen.calc_pos(traj_ids_flat, traj_timesteps.flatten())
            tar_pos[mask, -1] = self._char_h

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.tarpos.value)
        if mask.sum() > 0:
            tar_pos[mask] = self._carry_tar_pos[sample_tar_from_ids[mask]]

        mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.scene.value)
        if mask.sum() > 0:
            oid = sample_tar_from_ids[mask]
            task = self._task_exec_order_task_uid[self._task_exec_pointer[env_ids[mask]]]
            masked_ids = env_ids[mask]

            task_equal_sit = (task == HumanoidLongTerm4BasicSkills.TaskUID.sit.value)
            if task_equal_sit.sum() > 0:
                curr_env_ids = masked_ids[task_equal_sit]
                tar_pos[curr_env_ids] = self._obj_lib._obj_tar_sit_pos[oid[task_equal_sit]] + self._object_states[curr_env_ids, oid[task_equal_sit], 0:3]
            
            task_equal_climb = (task == HumanoidLongTerm4BasicSkills.TaskUID.climb.value)
            if task_equal_climb.sum() > 0:
                curr_env_ids = masked_ids[task_equal_climb]
                tar_pos[curr_env_ids] = self._obj_lib._obj_tar_climb_pos[oid[task_equal_climb]] + self._object_states[curr_env_ids, oid[task_equal_climb], 0:3]
                tar_pos[curr_env_ids, 2] += self._char_h

        mask = compute_finish_state(anchor_pos, tar_pos, self._task_transition_dist_threshold)
        self._task_transition_step_buf[mask] += 1
        self._task_transition_step_buf[mask & (traj_env_ids)] -= 1

        traj_success_mask = torch.norm(anchor_pos[..., 0:2] - tar_pos[..., 0:2], p=2, dim=-1) <= 0.5
        traj_success_mask = torch.logical_and(traj_success_mask, traj_env_ids)
        self._task_transition_step_buf[traj_success_mask] += 1

        if self._enable_IET:
            IET_triggered = torch.logical_and(
                self._task_exec_pointer == (len(self._task_exec_order_task_uid) - 1), # performing the last subtask
                self._task_transition_step_buf == self._IET_max_steps - 1,
            )
            if IET_triggered.sum() > 0:
                self._IET_triggered_buf[IET_triggered] = True

        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _draw_task(self):
        self._update_marker()

        traj_cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        if self._show_lines_flag:

            # draw lines of the bbox
            cols = np.zeros((12, 3), dtype=np.float32) # 24 lines
            cols[:, :] = [1.0, 0.0, 0.0] # red

            tar_obj_ids = self._task_exec_order_tar_obj[self._task_exec_pointer]

            # transform bps from object local space to world space
            env_ids = to_torch(np.arange(self.num_envs), dtype=torch.long, device=self.device)
            object_bps = self._obj_lib._obj_bps[tar_obj_ids].clone()
            object_pos = self._object_states[env_ids, tar_obj_ids, 0:3]
            object_rot = self._object_states[env_ids, tar_obj_ids, 3:7]
            object_pos_exp = torch.broadcast_to(object_pos.unsqueeze(-2), (object_pos.shape[0], object_bps.shape[1], object_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
            object_rot_exp = torch.broadcast_to(object_rot.unsqueeze(-2), (object_rot.shape[0], object_bps.shape[1], object_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
            object_bps_world_space = (quat_rotate(object_rot_exp.reshape(-1, 4), object_bps.reshape(-1, 3)) + object_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

            verts_tar_obj_bbox = torch.cat([
                object_bps_world_space[:, 0, :], object_bps_world_space[:, 1, :],
                object_bps_world_space[:, 1, :], object_bps_world_space[:, 2, :],
                object_bps_world_space[:, 2, :], object_bps_world_space[:, 3, :],
                object_bps_world_space[:, 3, :], object_bps_world_space[:, 0, :],

                object_bps_world_space[:, 4, :], object_bps_world_space[:, 5, :],
                object_bps_world_space[:, 5, :], object_bps_world_space[:, 6, :],
                object_bps_world_space[:, 6, :], object_bps_world_space[:, 7, :],
                object_bps_world_space[:, 7, :], object_bps_world_space[:, 4, :],

                object_bps_world_space[:, 0, :], object_bps_world_space[:, 4, :],
                object_bps_world_space[:, 1, :], object_bps_world_space[:, 5, :],
                object_bps_world_space[:, 2, :], object_bps_world_space[:, 6, :],
                object_bps_world_space[:, 3, :], object_bps_world_space[:, 7, :],
            ], dim=-1).cpu()

            for i, env_ptr in enumerate(self.envs):

                # traj
                verts = self._traj_gen.get_traj_verts(self._task_exec_order_tar_traj[self._task_exec_pointer[i]])
                verts[..., 2] = self._char_h
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                curr_cols = np.broadcast_to(traj_cols, [lines.shape[0], traj_cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

                curr_verts = verts_tar_obj_bbox[i].numpy()
                curr_verts = curr_verts.reshape([12, 6])
                self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

            ####### get tar pos

            tar_pos = torch.zeros_like(self._humanoid_root_states[..., 0:3])

            sample_tar_from_sources = self._sample_target_from_source[self._task_exec_pointer[env_ids]]
            sample_tar_from_ids = self._sample_target_from_id[self._task_exec_pointer[env_ids]]

            mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.traj.value)
            if mask.sum() > 0:
                traj_ids_flat = sample_tar_from_ids[mask]
                traj_timesteps = torch.ones_like(traj_ids_flat) * torch.inf
                tar_pos[mask] = self._traj_gen.calc_pos(traj_ids_flat, traj_timesteps.flatten())
                tar_pos[mask, -1] = self._char_h

            mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.tarpos.value)
            if mask.sum() > 0:
                tar_pos[mask] = self._carry_tar_pos[sample_tar_from_ids[mask]]

            mask = (sample_tar_from_sources == HumanoidLongTerm4BasicSkills.SampleFromUID.scene.value)
            if mask.sum() > 0:
                oid = sample_tar_from_ids[mask]
                task = self._task_exec_order_task_uid[self._task_exec_pointer[env_ids[mask]]]
                masked_ids = env_ids[mask]

                task_equal_sit = (task == HumanoidLongTerm4BasicSkills.TaskUID.sit.value)
                if task_equal_sit.sum() > 0:
                    curr_env_ids = masked_ids[task_equal_sit]
                    tar_pos[curr_env_ids] = self._obj_lib._obj_tar_sit_pos[oid[task_equal_sit]] + self._object_states[curr_env_ids, oid[task_equal_sit], 0:3]
                
                task_equal_climb = (task == HumanoidLongTerm4BasicSkills.TaskUID.climb.value)
                if task_equal_climb.sum() > 0:
                    curr_env_ids = masked_ids[task_equal_climb]
                    tar_pos[curr_env_ids] = self._obj_lib._obj_tar_climb_pos[oid[task_equal_climb]] + self._object_states[curr_env_ids, oid[task_equal_climb], 0:3]
                    tar_pos[curr_env_ids, 2] += self._char_h

            sphere_geom_1 = gymutil.WireframeSphereGeometry(self._task_transition_dist_threshold, 16, 16, None, color=(0, 0, 1))
            for i, env_ptr in enumerate(self.envs):

                pose = gymapi.Transform(gymapi.Vec3(tar_pos[i, 0], tar_pos[i, 1], tar_pos[i, 2]), r=None)
                gymutil.draw_lines(sphere_geom_1, self.gym, self.viewer, env_ptr, pose)

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["policy_obs"] = self.obs_buf.clone()
        
        for task_name in self._multiple_task_names:
            curr_task_uid = HumanoidLongTerm4BasicSkills.TaskUID[task_name].value
            curr_env_mask = self._task_exec_order_task_uid[self._task_exec_pointer] == curr_task_uid

            self.extras[task_name] = curr_env_mask
        
        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = torch.ones_like(self._success_buf) # Assuming that all tasks can be completed, calculate the average number of subtasks completed using precision
            self.extras["precision"] = self._success_buf

        return
    
    def _compute_metrics_evaluation(self):
        self._success_buf = self._task_exec_pointer.clone()

        num_total_subgoals = len(self._task_exec_order_task_uid) # how many subgoals we need to complete in an episode

        final_task_finished = (self._task_exec_pointer == num_total_subgoals - 1) & (self._task_transition_step_buf > 0)

        self._success_buf[final_task_finished] = num_total_subgoals # All 4 tasks are completed, success_buf is equal to 4

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        # random sample a skill
        sk_id = torch.multinomial(self._skill_disc_prob, num_samples=1, replacement=True)
        sk_name = self._skill[sk_id]
        curr_motion_lib = self._motion_lib[sk_name]

        # assign a task label
        task_onehot = torch.zeros(num_samples * self._num_amp_obs_steps, self._num_tasks, device=self.device, dtype=torch.float32)
        if sk_name != self._common_skill:
            if sk_name in self._traj_skill:
                task_onehot[..., HumanoidLongTerm4BasicSkills.TaskUID["traj"].value] = 1.0
            elif sk_name in self._sit_skill:
                task_onehot[..., HumanoidLongTerm4BasicSkills.TaskUID["sit"].value] = 1.0
            elif sk_name in self._carry_skill:
                task_onehot[..., HumanoidLongTerm4BasicSkills.TaskUID["carry"].value] = 1.0
            elif sk_name in self._climb_skill:
                task_onehot[..., HumanoidLongTerm4BasicSkills.TaskUID["climb"].value] = 1.0
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = curr_motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = curr_motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0, curr_motion_lib)

        if self._enable_task_specific_disc:
            amp_obs_demo = torch.cat([amp_obs_demo, task_onehot], dim=-1)

        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        if self._enable_task_specific_disc:
            self._num_amp_obs_per_step += self._num_tasks

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            self._skill_categories = list(motion_config['motions'].keys()) # all skill names stored in the yaml file
            self._motion_lib = {}
            for skill in self._skill_categories:
                self._motion_lib[skill] = MotionLib(motion_file=motion_file,
                                                    skill=skill,
                                                    dof_body_ids=self._dof_body_ids,
                                                    dof_offsets=self._dof_offsets,
                                                    key_body_ids=self._key_body_ids.cpu().numpy(), 
                                                    device=self.device)
        else:
            raise NotImplementedError

        return
    
    def _reset_task_traj(self, env_ids):
        self._traj_begin_progress_buf[env_ids] = 0
        return

    def _reset_task(self, env_ids):
        self._task_exec_pointer[env_ids] = 0 # As default, we always start from the first task

        self._task_mask[env_ids, :] = False
        self._task_mask[env_ids, self._task_exec_order_task_uid[self._task_exec_pointer[env_ids]]] = True
        assert self._task_mask[env_ids].sum() == len(env_ids)

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_objects(env_ids)
            self._reset_task(env_ids)
            self._reset_task_traj(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._init_amp_obs(env_ids)

        return

    def _reset_objects(self, env_ids):

        assert self._reset_ref_env_ids == {}
        assert self._reset_ref_motion_ids == {}
        assert  self._reset_ref_motion_times == {}

        self._object_states[env_ids] = self._initial_object_states[env_ids].clone()

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        
        self._task_transition_step_buf[env_ids] = 0
        self._IET_triggered_buf[env_ids] = 0

        if self._enable_climb_human_fall_termination:
            self._termination_heights[env_ids] = self._termination_heights_backup.clone()

        if self._is_eval:
            self._success_buf[env_ids] = 0

        env_ids_int32 = self._object_actor_ids.reshape(self.num_envs, self._obj_lib._num_objects)[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidLongTerm4BasicSkills.StateInit.Default):
            self._reset_default(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        # self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        # self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        # self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        # self._reset_default_env_ids = env_ids

        # if (len(self._reset_default_env_ids) > 0):
        #     self._kinematic_humanoid_rigid_body_states[self._reset_default_env_ids] = self._initial_humanoid_rigid_body_states[self._reset_default_env_ids]

        # self._every_env_init_dof_pos[self._reset_default_env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        root_pos = self._initial_humanoid_root_states[env_ids, 0:3].clone()
        root_vel = self._initial_humanoid_root_states[env_ids, 7:10].clone()
        root_ang_vel = self._initial_humanoid_root_states[env_ids, 10:13].clone()
        dof_pos = self._initial_dof_pos[env_ids].clone()
        dof_vel = self._initial_dof_vel[env_ids].clone()

        root_pos[:, 0:2] = torch.tensor(self._plan["humanoid_init_pos2d"], device=self.device)
        root_pos[:, -1] = self._char_h

        axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([env_ids.shape[0], -1])
        ang = torch.rand((len(env_ids),), device=self.device) * 2 * np.pi
        root_rot = quat_from_angle_axis(ang, axis)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_default_env_ids = env_ids
        self._kinematic_humanoid_rigid_body_states[env_ids] = compute_kinematic_rigid_body_states(root_pos, root_rot, self._initial_humanoid_rigid_body_states[env_ids])
        self._every_env_init_dof_pos[env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        for i, sk_name in enumerate(self._skill):

            curr_env_ids = torch.tensor([], dtype=torch.int32, device=self.device)
            for k, v in self._reset_ref_env_ids.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_env_ids = torch.hstack([curr_env_ids, vv])
            
            curr_motion_ids = torch.tensor([], dtype=torch.int32, device=self.device)
            for k, v in self._reset_ref_motion_ids.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_motion_ids = torch.hstack([curr_motion_ids, vv])
            
            curr_motion_times = torch.tensor([], dtype=torch.float32, device=self.device)
            for k, v in self._reset_ref_motion_times.items():
                for kk, vv in v.items():
                    if kk == sk_name:
                        curr_motion_times = torch.hstack([curr_motion_times, vv])

            if (len(curr_env_ids) > 0):
                self._init_amp_obs_ref(curr_env_ids, curr_motion_ids, curr_motion_times, sk_name)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times, skill_name):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib[skill_name].get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        

        if self._enable_task_specific_disc and self._enable_task_mask_obs:
            motion_labels = self._task_mask[env_ids]
            motion_labels = torch.broadcast_to(motion_labels.unsqueeze(-2), [motion_labels.shape[0], self._num_amp_obs_steps - 1, motion_labels.shape[1]])

            amp_obs_demo = torch.cat([amp_obs_demo, motion_labels.reshape(-1, motion_labels.shape[-1]).float()], dim=-1)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            amp_obs = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
            
            if self._enable_task_specific_disc:
                self._curr_amp_obs_buf[:] = torch.cat([amp_obs, self._task_mask.float()], dim=-1)
            else:
                self._curr_amp_obs_buf[:] = amp_obs

        else:
            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            amp_obs = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
            
            if self._enable_task_specific_disc:
                self._curr_amp_obs_buf[env_ids] = torch.cat([amp_obs, self._task_mask[env_ids].float()], dim=-1)
            else:
                self._curr_amp_obs_buf[env_ids] = amp_obs

        return
    
    def _compute_reset(self):
        time = (self.progress_buf - self._traj_begin_progress_buf) * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        traj_ids_flat = self._task_exec_order_tar_traj[self._task_exec_pointer[env_ids]]
        tar_pos = self._traj_gen.calc_pos(traj_ids_flat, time)

        traj_env_ids = self._task_exec_order_task_uid[self._task_exec_pointer] == HumanoidLongTerm4BasicSkills.TaskUID["traj"].value
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, tar_pos,
                                                           self.max_episode_length, self._fail_dist, traj_env_ids,
                                                           self._enable_IET, self._IET_triggered_buf,
                                                           self._enable_early_termination, self._termination_heights,
                                                           self._enable_dyn_obj_bug_reset, self._object_states, self._obj_lib._obj_is_static,
                                                           self._enable_dyn_obj_fall_termination, self._obj_lib._obj_bbox_lengths,
                                                           self._enable_dyn_obj_up_facing_termination, self._obj_lib._obj_up_facings,)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_finish_state(root_pos, tar_pos, success_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    pos_diff = tar_pos - root_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold # dist
    return dist_mask

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_location_observations(root_states, traj_samples, sit_tar_pos, sit_object_states, sit_object_bps, sit_object_facings, box_states, box_bps, box_tar_pos, climb_object_states, climb_object_bps, climb_tar_pos, task_mask, each_subtask_obs_mask, enable_apply_mask):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    B = root_pos.shape[0]
    NM = traj_samples.shape[1]

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (B, NM, 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (B, NM, 3))

    # traj
    N = traj_samples.shape[1]
    local_traj_sampls = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), traj_samples.reshape(-1, 3) - root_pos_exp[:, :N].reshape(-1, 3))[..., 0:2]

    # sit
    local_sit_tar_pos = quat_rotate(heading_rot, sit_tar_pos - root_pos) # 3d xyz

    sit_obj_root_pos = sit_object_states[:, 0:3]
    sit_obj_root_rot = sit_object_states[:, 3:7]

    local_sit_obj_root_pos = quat_rotate(heading_rot, sit_obj_root_pos - root_pos)
    local_sit_obj_root_rot = quat_mul(heading_rot, sit_obj_root_rot)
    local_sit_obj_root_rot = torch_utils.quat_to_tan_norm(local_sit_obj_root_rot)

    N = sit_object_bps.shape[1]
    sit_obj_root_pos_exp = torch.broadcast_to(sit_obj_root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    sit_obj_root_rot_exp = torch.broadcast_to(sit_obj_root_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

    sit_obj_bps_world_space = quat_rotate(sit_obj_root_rot_exp, sit_object_bps.reshape(-1, 3)) + sit_obj_root_pos_exp
    sit_obj_bps_local_space = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), sit_obj_bps_world_space - root_pos_exp[:, :N].reshape(-1, 3)).reshape(-1, N * 3)

    sit_face_vec_world_space = quat_rotate(sit_obj_root_rot, sit_object_facings)
    sit_face_vec_local_space = quat_rotate(heading_rot, sit_face_vec_world_space)[..., 0:2]

    # carry
    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]
    
    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)

    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    # compute observations for bounding points of the box
    ## transform from object local space to world space
    box_pos_exp = torch.broadcast_to(box_pos.unsqueeze(-2), (box_pos.shape[0], box_bps.shape[1], box_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
    box_rot_exp = torch.broadcast_to(box_rot.unsqueeze(-2), (box_rot.shape[0], box_bps.shape[1], box_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
    box_bps_world_space = quat_rotate(box_rot_exp.reshape(-1, 4), box_bps.reshape(-1, 3)) + box_pos_exp.reshape(-1, 3) # (num_envs*8, 3)

    ## transform from world space to humanoid local space
    box_bps_local_space = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), box_bps_world_space - root_pos_exp[:, :N].reshape(-1, 3)) # (num_envs*8, 3)

    # task obs
    local_box_tar_pos = quat_rotate(heading_rot, box_tar_pos - root_pos) # 3d xyz

    # climb
    local_climb_tar_pos = quat_rotate(heading_rot, climb_tar_pos - root_pos) # 3d xyz

    climb_obj_root_pos = climb_object_states[:, 0:3]
    climb_obj_root_rot = climb_object_states[:, 3:7]

    N = climb_object_bps.shape[1]
    climb_obj_root_pos_exp = torch.broadcast_to(climb_obj_root_pos.unsqueeze(1), (B, N, 3)).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    climb_obj_root_rot_exp = torch.broadcast_to(climb_obj_root_rot.unsqueeze(1), (B, N, 4)).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

    climb_obj_bps_world_space = quat_rotate(climb_obj_root_rot_exp, climb_object_bps.reshape(-1, 3)) + climb_obj_root_pos_exp
    climb_obj_bps_local_space = quat_rotate(heading_rot_exp[:, :N].reshape(-1, 4), climb_obj_bps_world_space - root_pos_exp[:, :N].reshape(-1, 3)).reshape(-1, N * 3)

    obs = torch.cat([
        local_traj_sampls.reshape(root_pos.shape[0], -1),
        local_sit_tar_pos, sit_obj_bps_local_space, sit_face_vec_local_space, local_sit_obj_root_pos, local_sit_obj_root_rot, 
        local_box_vel, local_box_ang_vel, local_box_pos, local_box_rot_obs, box_bps_local_space.reshape(root_pos.shape[0], -1), local_box_tar_pos,
        local_climb_tar_pos, climb_obj_bps_local_space,
        ], dim=-1)

    if enable_apply_mask:
        mask = task_mask[:, None, :].float() @ torch.broadcast_to(each_subtask_obs_mask[None, :, :].float(), (root_states.shape[0], each_subtask_obs_mask.shape[0], each_subtask_obs_mask.shape[1]))
        obs *= mask.squeeze(1)
    
    return obs

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist, traj_env_ids,
                           enable_IET, IET_triggered,
                           enable_early_termination, termination_heights,
                           enable_dyn_obj_bug_reset, object_states, object_is_static,
                           enable_dyn_obj_fall_termination, obj_bbox_lengths,
                           enable_dyn_obj_up_facing_termination, obj_up_facings):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, bool, Tensor, bool, Tensor, bool, Tensor, Tensor, bool, Tensor, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        # masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0
        # fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = fall_height
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[..., 0:2]
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        tar_fail[~traj_env_ids] = False

        has_failed = torch.logical_or(has_fallen, tar_fail)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

        if enable_dyn_obj_fall_termination:
            mask = ~object_is_static.clone()
            num_dyn_objs = mask.sum()
            if num_dyn_objs > 0:
                dyn_obj_states = object_states[:, mask]
                obj_bbox_lengths_exp = torch.broadcast_to(obj_bbox_lengths.unsqueeze(0), (len(reset_buf), obj_bbox_lengths.shape[0], obj_bbox_lengths.shape[1]))
                obj_bbox_lengths_exp = obj_bbox_lengths_exp[:, mask]

                dyn_obj_has_fallen = torch.any(dyn_obj_states[..., 2] < (obj_bbox_lengths_exp[..., 2] / 2.0 + 0.2),
                                               dim=-1)
                terminated = torch.where(dyn_obj_has_fallen, torch.ones_like(reset_buf), terminated)

        if enable_dyn_obj_up_facing_termination:
            mask = ~object_is_static.clone()
            num_dyn_objs = mask.sum()
            if num_dyn_objs > 0:
                dyn_obj_states = object_states[:, mask]
                obj_up_facings_exp = torch.broadcast_to(obj_up_facings.unsqueeze(0), (len(reset_buf), obj_up_facings.shape[0], obj_up_facings.shape[1]))
                obj_up_facings_exp = obj_up_facings_exp[:, mask]

                obj_curr_up_facing_dir = quat_rotate(dyn_obj_states[..., 3:7].view(-1, 4), obj_up_facings_exp.view(-1, 3)).reshape(-1, num_dyn_objs, 3)
                obj_targ_up_facing_dir = obj_up_facings_exp
                dir_err = torch.clamp_min(torch.sum(obj_curr_up_facing_dir * obj_targ_up_facing_dir, dim=-1), 0.0) # xyz

                dyn_obj_has_flipped = torch.any(dir_err < 0.5, dim=-1)
                terminated = torch.where(dyn_obj_has_flipped, torch.ones_like(reset_buf), terminated)

    if enable_IET:
        reset = torch.logical_or(IET_triggered, terminated)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    else:
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    if enable_dyn_obj_bug_reset:
        mask = ~object_is_static.clone()
        num_dyn_objs = mask.sum()
        if num_dyn_objs > 0:
            dyn_obj_states = object_states[:, mask]
            dyn_obj_lin_vel = dyn_obj_states[..., 7:10]
            triggered_tensor = torch.logical_and(
                progress_buf <= 5,
                torch.any(torch.sum(dyn_obj_lin_vel ** 2, dim=-1) >= 3 ** 2, dim=-1),
            )
            # if triggered_tensor.sum() > 0:
            #     print(triggered_tensor.nonzero())
            reset = torch.where(triggered_tensor, torch.ones_like(reset_buf), reset)

    return reset, terminated

@torch.jit.script
def compute_kinematic_rigid_body_states(root_pos, root_rot, initial_humanoid_rigid_body_states):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    num_envs = initial_humanoid_rigid_body_states.shape[0]
    num_bodies = initial_humanoid_rigid_body_states.shape[1] # 15

    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (root_pos.shape[0], num_bodies, root_pos.shape[1])).reshape(-1, 3) # (num_envs, 3) >> (num_envs, 15, 3) >> (num_envs*15, 3)
    root_rot_exp = torch.broadcast_to(root_rot.unsqueeze(1), (root_rot.shape[0], num_bodies, root_rot.shape[1])).reshape(-1, 4) # (num_envs, 4) >> (num_envs, 15, 4) >> (num_envs*15, 4)

    init_body_pos = initial_humanoid_rigid_body_states[..., 0:3] # (num_envs, 15, 3)
    init_body_rot = initial_humanoid_rigid_body_states[..., 3:7] # (num_envs, 15, 4)
    init_body_vel = initial_humanoid_rigid_body_states[..., 7:10] # (num_envs, 15, 3)
    init_body_ang_vel = initial_humanoid_rigid_body_states[..., 10:13] # (num_envs, 15, 3)

    init_root_pos = init_body_pos[:, 0:1, :] # (num_envs, 1, 3)
    init_body_pos_canonical = (init_body_pos - init_root_pos).reshape(-1, 3) # (num_envs, 15, 3) >> (num_envs*15, 3)
    init_body_rot = init_body_rot.reshape(-1, 4)
    init_body_vel = init_body_vel.reshape(-1, 3)
    init_body_ang_vel = init_body_ang_vel.reshape(-1, 3)
    
    curr_body_pos = (quat_rotate(root_rot_exp, init_body_pos_canonical) + root_pos_exp).reshape(-1, num_bodies, 3)
    curr_body_rot = (quat_mul(root_rot_exp, init_body_rot)).reshape(-1, num_bodies, 4)
    curr_body_vel = (quat_rotate(root_rot_exp, init_body_vel)).reshape(-1, num_bodies, 3)
    curr_body_ang_vel = (quat_rotate(root_rot_exp, init_body_ang_vel)).reshape(-1, num_bodies, 3)
    curr_humanoid_rigid_body_states = torch.cat((curr_body_pos, curr_body_rot, curr_body_vel, curr_body_ang_vel), dim=-1)
    
    return curr_humanoid_rigid_body_states

@torch.jit.script
def sample_surrounding_heights(root_pos, root_rot, object_pos, object_rot, object_bbox, object_height_maps, object_on_ground_trans, humanoid_height_sensors):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    num_envs = root_pos.shape[0]
    num_sensor_points = humanoid_height_sensors.shape[1]
    num_objects = object_pos.shape[1]

    # humanoid local space to world space
    root_rot_xy = torch_utils.calc_heading_quat(root_rot)
    root_rot_xy_expend = torch.broadcast_to(root_rot_xy.unsqueeze(1), (root_rot_xy.shape[0], num_sensor_points, root_rot_xy.shape[1]))
    grid = quat_rotate(root_rot_xy_expend.reshape(-1, 4), humanoid_height_sensors.reshape(-1, 3)) # (num_envs * num_sensor_points, 3)
    grid = grid.reshape(-1, num_sensor_points, 3)
    grid += root_pos.unsqueeze(1)
    grid[..., -1] = 0.0

    # world space to object local space
    grid = torch.broadcast_to(grid.unsqueeze(1), (num_envs, num_objects, num_sensor_points, 3))
    grid = grid - object_pos.unsqueeze(2)
    grid[..., -1] = 0.0

    object_rot_xy_inv = torch_utils.calc_heading_quat_inv(object_rot.reshape(-1, 4)).reshape(num_envs, num_objects, 4)
    object_rot_xy_inv_expend = torch.broadcast_to(object_rot_xy_inv.unsqueeze(2), (num_envs, num_objects, num_sensor_points, 4))
    grid = quat_rotate(object_rot_xy_inv_expend.reshape(-1, 4), grid.reshape(-1, 3)) # (num_envs * num_sensor_points, 3)
    grid = grid.reshape(num_envs, num_objects, num_sensor_points, 3)
    
    # normalize to [-1, +1]
    object_bbox_exp = torch.broadcast_to(object_bbox.unsqueeze(0), (num_envs, num_objects, 3))
    grid[..., 0] /= (object_bbox_exp[..., 0].unsqueeze(-1) / 2)
    grid[..., 1] /= -1 * (object_bbox_exp[..., 1].unsqueeze(-1) / 2) # Fuck

    object_height_maps_exp = torch.broadcast_to(object_height_maps.unsqueeze(0), (num_envs, num_objects, object_height_maps.shape[1], object_height_maps.shape[2]))

    # process obj not on the ground plane
    object_pos_z = object_pos[..., -1] - object_on_ground_trans.unsqueeze(0)
    object_height_maps_exp = object_height_maps_exp + object_pos_z.unsqueeze(-1).unsqueeze(-1)

    object_height_maps_exp = object_height_maps_exp.reshape(-1, object_height_maps.shape[1], object_height_maps.shape[2])
    grid = grid.reshape(-1, num_sensor_points, 3)

    heights = torch.nn.functional.grid_sample(object_height_maps_exp.unsqueeze(1), grid[..., 0:2].unsqueeze(1), padding_mode="zeros", align_corners=False, mode="nearest")
    return heights.squeeze(1).squeeze(1).reshape(num_envs, num_objects, num_sensor_points)

@torch.jit.script
def sample_surrounding_heights_v_dynamic(root_pos, root_rot, humanoid_height_sensors, heightmap, heightmap_spacing):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor

    num_envs = root_pos.shape[0]
    num_sensor_points = humanoid_height_sensors.shape[1]

    # humanoid local space to world space
    root_rot_xy = torch_utils.calc_heading_quat(root_rot)
    root_rot_xy_expend = torch.broadcast_to(root_rot_xy.unsqueeze(1), (root_rot_xy.shape[0], num_sensor_points, root_rot_xy.shape[1]))
    grid = quat_rotate(root_rot_xy_expend.reshape(-1, 4), humanoid_height_sensors.reshape(-1, 3)) # (num_envs * num_sensor_points, 3)
    grid = grid.reshape(-1, num_sensor_points, 3)
    # grid += root_pos.unsqueeze(1) # height map is in humanoid local space
    
    # normalize to [-1, +1]
    grid[..., 0] /= heightmap_spacing
    grid[..., 1] /= -1 * heightmap_spacing # Fuck

    heights = torch.nn.functional.grid_sample(heightmap.unsqueeze(1), grid[..., 0:2].unsqueeze(1), padding_mode="zeros", align_corners=False, mode="nearest")
    return heights.squeeze(1).squeeze(1)

@torch.jit.script
def compute_sit_reward(root_pos, prev_root_pos, object_root_pos, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor

    d_obj2human_xy = torch.sum((object_root_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
    reward_far_pos = torch.exp(-0.5 * d_obj2human_xy)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir = object_root_pos[..., 0:2] - root_pos[..., 0:2] # d* is a horizontal unit vector pointing from the root to the object's location
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)

    reward_far_final = 0.0 * reward_far_pos + 1.0 * reward_far_vel
    dist_mask = (d_obj2human_xy <= 0.5 ** 2)
    reward_far_final[dist_mask] = 1.0

    # when humanoid is close to the object
    reward_near = torch.exp(-10.0 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    reward = 0.7 * reward_near + 0.3 * reward_far_final

    return reward

@torch.jit.script
def compute_walk_reward(root_pos, prev_root_pos, box_pos, dt, tar_vel, only_vel_reward):
    # type: (Tensor, Tensor, Tensor, float, float, bool) -> Tensor

    # this reward encourages the character to walk towards the box and stay close to it

    pos_diff = box_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = tar_vel

    tar_dir = box_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = pos_err < 0.5 ** 2
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 1.0 * vel_reward
    else:
        reward = 0.5 * pos_reward + 0.5 * vel_reward
    return reward

@torch.jit.script
def compute_handheld_reward(humanoid_rigid_body_pos, box_pos, hands_ids, tar_pos, only_height, apply_near_constrain=True):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    if only_height:
        left_hand_height = humanoid_rigid_body_pos[:, hands_ids[0], 2]
        left_hand_diff = left_hand_height - box_pos[:, 2]
        left_hand_diff = torch.clamp_max(left_hand_diff, 0.0)
        left_hands2box_pos_err = left_hand_diff ** 2 # height
        left_hands2box = torch.exp(-20.0 * left_hands2box_pos_err)

        right_hand_height = humanoid_rigid_body_pos[:, hands_ids[1], 2]
        right_hand_diff = right_hand_height - box_pos[:, 2]
        right_hand_diff = torch.clamp_max(right_hand_diff, 0.0)
        right_hands2box_pos_err = right_hand_diff ** 2 # height
        right_hands2box = torch.exp(-20.0 * right_hands2box_pos_err)

        hands2box = left_hands2box * right_hands2box
    else:
        hands2box_pos_err = torch.sum((humanoid_rigid_body_pos[:, hands_ids].mean(dim=1) - box_pos) ** 2, dim=-1) # xyz
        hands2box = torch.exp(-10.0 * hands2box_pos_err)

    # box2tar = torch.sum((box_pos[..., 0:2] - tar_pos[..., 0:2]) ** 2, dim=-1) # 2d
    # hands2box[box2tar < 0.7 ** 2] = 1.0 # assume this reward is max when the box is close enough to its target location

    if apply_near_constrain:
        root_pos = humanoid_rigid_body_pos[:, 0, :]
        box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
        hands2box[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    return hands2box

@torch.jit.script
def compute_handheld_reward_climb2move_0(humanoid_rigid_body_pos, box_pos, hands_ids, tar_pos, only_height, apply_near_constrain=True):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    left_hand_height = humanoid_rigid_body_pos[:, hands_ids[0], 2]
    left_hand_diff = left_hand_height - box_pos[:, 2]
    left_hand_diff = torch.clamp_max(left_hand_diff, 0.0)
    left_hand_2box_pos_err_z = left_hand_diff ** 2 # height

    right_hand_height = humanoid_rigid_body_pos[:, hands_ids[1], 2]
    right_hand_diff = right_hand_height - box_pos[:, 2]
    right_hand_diff = torch.clamp_max(right_hand_diff, 0.0)
    right_hand_2box_pos_err_z = right_hand_diff ** 2 # height
    height_mask = torch.logical_and(left_hand_2box_pos_err_z < 0.1 ** 2, right_hand_2box_pos_err_z < 0.1 ** 2)
    
    hands2box_pos_err_xy = torch.sum((humanoid_rigid_body_pos[:, hands_ids].mean(dim=1)[..., 0:2] - box_pos[..., 0:2]) ** 2, dim=-1) # xy
    hands2box = torch.exp(-10.0 * hands2box_pos_err_xy)

    if apply_near_constrain:
        hands2box[~height_mask] = 0 # disable this reward when the box is not close to the humanoid

    return hands2box

@torch.jit.script
def compute_carry_far_reward(box_pos, prev_box_pos, tar_box_pos, dt, tar_vel, box_size, only_vel_reward):
    # type: (Tensor, Tensor, Tensor, float, float, Tensor, bool) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_xy = torch.sum(pos_diff[..., 0:2] ** 2, dim=-1)
    pos_reward_far = torch.exp(-0.5 * pos_err_xy)

    min_speed = tar_vel

    tar_dir = tar_box_pos[..., 0:2] - box_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    # tar_vel_err = torch.clamp_min(tar_vel_err, 0.0) # constrain vel around the peak value
    vel_reward = torch.exp(-5.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    height_mask = box_pos[..., 2] <= (box_size[..., 2] / 2 + 0.2) # avoid learning to kick the box
    pos_reward_far[height_mask] = 0.0
    vel_reward[height_mask] = 0.0

    dist_mask = pos_err_xy < 0.5 ** 2
    pos_reward_far[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    if only_vel_reward:
        reward = 1.0 * vel_reward 
    else:
        reward = 0.5 * pos_reward_far + 0.5 * vel_reward

    return reward

@torch.jit.script
def compute_carry_far_reward_pos_xyz(box_pos, tar_box_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_xyz = torch.sum(pos_diff ** 2, dim=-1) # consider height
    pos_reward_far = torch.exp(-0.5 * pos_err_xyz)

    dist_mask = pos_err_xyz < 0.5 ** 2
    pos_reward_far[dist_mask] = 1.0

    return pos_reward_far

@torch.jit.script
def compute_carry_far_reward_pos_z(box_pos, tar_box_pos, root_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_z = pos_diff[..., -1] ** 2
    pos_reward_far = torch.exp(-2.0 * pos_err_z)

    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    pos_reward_far[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    return pos_reward_far

@torch.jit.script
def compute_carry_far_reward_pos_x(box_pos, tar_box_pos, root_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    
    # this reward encourages the character to carry the box to a target position

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_x = pos_diff[..., 0] ** 2
    pos_reward_far = torch.exp(-1.0 * pos_err_x)

    box2human = torch.sum((box_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) # 2d
    pos_reward_far[box2human > 0.7 ** 2] = 0 # disable this reward when the box is not close to the humanoid

    dist_mask = torch.sum(pos_diff ** 2, dim=-1) < 0.5 ** 2
    pos_reward_far[dist_mask] = 1.0

    return pos_reward_far

@torch.jit.script
def compute_carry_near_reward(box_pos, box_rot, default_box_up_facings, tar_box_pos, enable_dyn_obj_up_facing_rwd):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    pos_diff = tar_box_pos - box_pos # xyz
    pos_err_xyz = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward_near = torch.exp(-10.0 * pos_err_xyz)

    if enable_dyn_obj_up_facing_rwd:
        obj_curr_up_facing_dir = quat_rotate(box_rot, default_box_up_facings)
        obj_targ_up_facing_dir = default_box_up_facings
        dir_err = torch.clamp_min(torch.sum(obj_curr_up_facing_dir * obj_targ_up_facing_dir, dim=-1), 0.0) # xyz

        pos_reward_near = pos_reward_near * (dir_err)

        # dir_mask = (dir_err >= 0.95)
        # pos_reward_near[~dir_mask] = 0.0

    return pos_reward_near

@torch.jit.script
def compute_putdown_reward(box_pos, box_rot, default_box_up_facings, tar_pos, enable_dyn_obj_up_facing_rwd):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    reward = torch.exp(-500.0 * (box_pos[:, -1] - tar_pos[:, -1]) ** 2)
    
    pos_err_xy = torch.sum((tar_pos[..., :2] - box_pos[..., :2]) ** 2, dim=-1)
    reward[(pos_err_xy > 0.1 ** 2)] = 0.0

    if enable_dyn_obj_up_facing_rwd:
        obj_curr_up_facing_dir = quat_rotate(box_rot, default_box_up_facings)
        obj_targ_up_facing_dir = default_box_up_facings
        dir_err = torch.sum(obj_curr_up_facing_dir * obj_targ_up_facing_dir, dim=-1) # xyz
        reward[dir_err < 0.95] = 0.0
    
    return reward

@torch.jit.script
def compute_box_vel_penalty(box_pos, prev_box_pos, dt, box_vel_pen_coeff, box_vel_penalty_thre):
    # type: (Tensor, Tensor, float, float, float) -> Tensor
    
    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt

    min_speed_penalty = box_vel_penalty_thre
    root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
    root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
    root_vel_err = min_speed_penalty - root_vel_norm
    root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))

    return root_vel_penalty

@torch.jit.script
def compute_box_vel_penalty_rf(box_pos, prev_box_pos, dt, box_vel_pen_coeff, box_vel_penalty_thre):
    # type: (Tensor, Tensor, float, float, float) -> Tensor
    
    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt

    min_speed_penalty = box_vel_penalty_thre
    root_vel_norm = torch.abs(root_vel[..., 2]) # torch.norm(root_vel, p=2, dim=-1)
    root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
    root_vel_err = min_speed_penalty - root_vel_norm
    root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))

    return root_vel_penalty

@torch.jit.script
def compute_box_vel_penalty_lf(box_pos, prev_box_pos, dt, box_vel_pen_coeff, box_vel_penalty_thre):
    # type: (Tensor, Tensor, float, float, float) -> Tensor
    
    delta_root_pos = box_pos - prev_box_pos
    root_vel = delta_root_pos / dt

    min_speed_penalty = box_vel_penalty_thre
    root_vel_norm = torch.abs(root_vel[..., 2]) # torch.norm(root_vel, p=2, dim=-1)
    root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
    root_vel_err = min_speed_penalty - root_vel_norm
    root_vel_penalty = -1 * box_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))

    return root_vel_penalty

@torch.jit.script
def compute_climb_reward(root_pos, prev_root_pos, object_pos, dt, tar_pos, rigid_body_pos, feet_ids, char_h):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float) -> Tensor

    pos_diff = object_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-0.5 * pos_err)

    min_speed = 1.5

    tar_dir = object_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = min_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-2.0 * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    dist_mask = (pos_err <= 1.0 ** 2)
    pos_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    pos_reward_near = torch.exp(-10 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    feet_height_err = (rigid_body_pos[:, feet_ids, -1].mean(dim=1) - (tar_pos[..., 2] - char_h)) ** 2 # height
    feet_height_reward = torch.exp(-50.0 * feet_height_err)
    feet_height_reward[~dist_mask] = 0.0

    reward = 0.0 * pos_reward + 0.2 * vel_reward + 0.5 * pos_reward_near + 0.3 * feet_height_reward
    return reward

@torch.jit.script
def compute_climb_reward_near(root_pos, prev_root_pos, object_pos, dt, tar_pos, rigid_body_pos, feet_ids, char_h):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float) -> Tensor

    pos_diff = object_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    dist_mask = (pos_err <= 1.0 ** 2)

    pos_reward_near = torch.exp(-5 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    feet_height_err = (rigid_body_pos[:, feet_ids, -1].mean(dim=1) - (tar_pos[..., 2] - char_h)) ** 2 # height
    feet_height_reward = torch.exp(-50.0 * feet_height_err)
    feet_height_reward[~dist_mask] = 0.0

    reward = 0.5 * pos_reward_near + 0.5 * feet_height_reward
    return reward

@torch.jit.script
def compute_0_reward(root_pos, prev_root_pos,
                     traj_tar_pos, traj_uid,
                     carry_obj_pos, prev_carry_obj_pos, carry_obj_rot, dt, carry_tar_pos, carry_obj_size,
                     enable_dyn_obj_up_facing_rwd, carry_obj_up_facings, carry_uid,
                     climb_obj_pos, climb_tar_pos, rigid_body_pos, feet_ids, char_h, climb_uid,
                     sit_obj_pos, sit_tar_pos, sit_uid,
                     task_phase,):
    # type: (Tensor, Tensor, Tensor, int, Tensor, Tensor, Tensor, float, Tensor, Tensor, bool, Tensor, int, Tensor, Tensor, Tensor, Tensor, float, int, Tensor, Tensor, int, Tensor) -> Tensor

    num_envs = root_pos.shape[0]

    traj_ongoing = (task_phase == traj_uid)
    carry_ongoing = (task_phase == carry_uid)
    climb_ongoing = (task_phase == climb_uid)
    sit_ongoing = (task_phase == sit_uid)

    # 1. traj

    traj_reward = torch.exp(-2.0 * torch.sum((traj_tar_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1))
    traj_reward[~traj_ongoing] = 1.0
    traj_reward *= 0.1

    # 2. carry
    carry_tar_pos_exp = torch.broadcast_to(carry_tar_pos.unsqueeze(0), (num_envs, 3))
    carry_obj_size_exp = torch.broadcast_to(carry_obj_size.unsqueeze(0), (num_envs, 3))
    carry_obj_up_facings_exp = torch.broadcast_to(carry_obj_up_facings.unsqueeze(0), (num_envs, 3))
    
    walk2box_r = compute_walk_reward(root_pos, prev_root_pos, carry_obj_pos, dt, 1.5, only_vel_reward=True)
    box2tar_far_r = compute_carry_far_reward(carry_obj_pos, prev_carry_obj_pos, 
                                             carry_tar_pos_exp, dt, 1.5, carry_obj_size_exp, only_vel_reward=True)
    box2tar_near_r = compute_carry_near_reward(carry_obj_pos, carry_obj_rot, carry_obj_up_facings_exp, carry_tar_pos_exp, enable_dyn_obj_up_facing_rwd)

    walk2box_r[traj_ongoing] = 0.0 # When the first traj task is executed, it cannot interfere with the traj task
    box2tar_far_r[traj_ongoing] = 0.0
    box2tar_near_r[traj_ongoing] = 0.0

    walk2box_r[torch.logical_or(climb_ongoing, sit_ongoing)] = 1.0 # When executing the 3rd and 4th tasks, this item is the maximum
    # box2tar_far_r[torch.logical_or(climb_ongoing, sit_ongoing)] = 1.0
    # box2tar_near_r[torch.logical_or(climb_ongoing, sit_ongoing)] = 1.0

    carry_reward = 0.1 * walk2box_r + 0.1 * box2tar_far_r + 0.3 * box2tar_near_r

    # 3. climb
    climb_near_r = compute_climb_reward_near(root_pos, prev_root_pos, climb_obj_pos, dt, climb_tar_pos, rigid_body_pos, feet_ids, char_h)
    climb_near_r[torch.logical_or(traj_ongoing, carry_ongoing)] = 0.0 # When executing the first two tasks, this item is set to 0
    climb_near_r[sit_ongoing] = 1.0 # When executing the 4th task, this item is max
    climb_near_r *= 1.0

    # 4. sit
    sit_r = compute_sit_reward(root_pos, prev_root_pos, sit_obj_pos, sit_tar_pos, 1.5, dt)
    sit_r[~sit_ongoing] = 0.0 # When sit is not executed, it is always 0.0. When it is executed, it is calculated normally.
    sit_r *= 1.0

    reward = traj_reward + carry_reward + climb_near_r + sit_r
  
    return reward
