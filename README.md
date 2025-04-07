<p align="center">
<h1 align="center"<strong>TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization</strong></h1>
  <p align="center">
    <a href='https://liangpan99.github.io/' target='_blank'>Liang Pan</a><sup>1,2</sup>
    ·
    <a href='https://zeshiyang.github.io/' target='_blank'>Zeshi Yang</a> <sup>3</sup>
    ·
    <a href='https://frank-zy-dou.github.io/' target='_blank'>Zhiyang Dou</a><sup>2</sup>
    ·
    <a href='https://wenjiawang0312.github.io/' target='_blank'>Wenjia Wang</a><sup>2</sup>
    ·
    <a href='https://www.buzhenhuang.com/about/' target='_blank'>Buzhen Huang</a><sup>4</sup>
    ·
    <a href='https://scholar.google.com/citations?user=KNWTvgEAAAAJ&hl=en' target='_blank'>Bo Dai</a><sup>2,5</sup>
    ·
    <a href='https://i.cs.hku.hk/~taku/' target='_blank'>Taku Komura</a><sup>2</sup>
    ·
    <a href='https://scholar.google.com/citations?user=GStTsxAAAAAJ&hl=en&oi=ao' target='_blank'>Jingbo Wang</a><sup>1</sup>
    <br>
    <sup>1</sup>Shanghai AI Lab <sup>2</sup>The University of Hong Kong <sup>3</sup>Independent Researcher <sup>4</sup>Southeast University <sup>5</sup>Feeling AI
    <br>
    <strong>CVPR 2025</strong>
  </p>
</p>
<p align="center">
  <a href='https://arxiv.org/abs/2503.19901'>
    <img src='https://img.shields.io/badge/Arxiv-2502.20390-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2503.19901'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://liangpan99.github.io/TokenHSI/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>

## 🏠 About
<div style="text-align: center;">
    <img src="https://github.com/liangpan99/TokenHSI/blob/page/static/images/teaser.png" width=100% >
</div>
Introducing TokenHSI, a unified model that enables physics-based characters to perform diverse human-scene interaction tasks. It excels at seamlessly unifying multiple <b>foundational HSI skills</b> within a single transformer network and flexibly adapting learned skills to <b>challenging new tasks</b>, including skill composition, object/terrain shape variation, and long-horizon task completion.
</br>

## 📹 Demo
<p align="center">
    <img src="assets/longterm_demo_isaacgym.gif" align="center" width=60% >
    <br>
    Long-horizon Task Completion in a Complex Dynamic Environment
</p>

<!-- ## 🕹 Pipeline
<div style="text-align: center;">
    <img src="https://github.com/liangpan99/TokenHSI/blob/page/static/images/pipeline.jpg" width=100% >
</div> -->

## 🔥 News  
- **[2025-04-03]** Released long-horizon task completion with a pre-trained model.
- **[2025-04-01]** We just updated the Getting Started section. You can play TokenHSI now!
- **[2025-03-31]** We've released the codebase and checkpoint for the foundational skill learning part.

## 📝 TODO List  
- [x] Release foundational skill learning 
- [ ] Release policy adaptation - skill composition  
- [ ] Release policy adaptation - object/terrain shape variation
- [x] Release policy adaptation - long-horizon task completion

## 📖 Getting Started

### Dependencies

Follow the following instructions: 

1. Create new conda environment and install pytroch

    ```
    conda create -n tokenhsi python=3.8
    conda activate tokenhsi
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

2. Install [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) 

    ```
    cd IsaacGym_Preview_4_Package/isaacgym/python
    pip install -e .

    # add your conda env path to ~/.bashrc
    conda env list
    export LD_LIBRARY_PATH="your_conda_env_path/lib:$LD_LIBRARY_PATH"

    # e.g., in my Slurm sbatch job, add:
    export LD_LIBRARY_PATH="/home/wenleyan/.conda/envs/tokenhsi/lib:$LD_LIBRARY_PATH"
    ```

3. Install pytorch3d (optional, if you want to run the long-horizon task completion demo)

    **We use pytorch3d to rapidly render height maps of dynamic objects for thousands of simulation environments.**

    ```
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7
    ```

4. Download [SMPL body models](https://smpl.is.tue.mpg.de/) and organize them as follows:

    ```
    |-- assets
    |-- body_models
        |-- smpl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_MALE.pkl
            |-- SMPL_NEUTRAL.pkl
            |-- ...
    |-- lpanlib
    |-- tokenhsi
    ```

### Motion & Object Data

We provide two methods to generate the motion and object data.

* Download pre-processed data from [Huggingface](https://huggingface.co/datasets/lianganimation/TokenHSI). Please follow the instruction in the dataset page.

* Generate data from source:

  1. Download [AMASS (SMPL-X Neutral)](https://amass.is.tue.mpg.de/), [SAMP](https://samp.is.tue.mpg.de/), and [OMOMO](https://github.com/lijiaman/omomo_release).

  2. Modify dataset paths in ```tokenhsi/data/dataset_cfg.yaml``` file.

      ```
      # Motion datasets, please use your own paths
      amass_dir: "/YOUR_PATH/datasets/AMASS"
      samp_pkl_dir: "/YOUR_PATH/datasets/samp"
      omomo_dir: "/YOUR_PATH/datasets/OMOMO/data"
      ```

  3. We still need to download the pre-processed data from [Huggingface](https://huggingface.co/datasets/lianganimation/TokenHSI). But now we only require the object data.

  4. Run the following script:

      ```
      bash tokenhsi/scripts/gen_data.sh
      ```

### Checkpoints

Download checkpoints from [Huggingface](https://huggingface.co/lianganimation/TokenHSI). Please follow the instruction in the model page.

### Play TokenHSI!

* Single task policy trained with AMP

  * Path-following

      ```
      # test
      sh tokenhsi/scripts/single_task/traj_test.sh
      # train
      sh tokenhsi/scripts/single_task/traj_train.sh
      ```

  * Sitting

      ```
      # test
      sh tokenhsi/scripts/single_task/sit_test.sh
      # train
      sh tokenhsi/scripts/single_task/sit_train.sh
      ```
  * Climbing

      ```
      # test
      sh tokenhsi/scripts/single_task/climb_test.sh
      # train
      sh tokenhsi/scripts/single_task/climb_train.sh
      ```

  * Carrying

      ```
      # test
      sh tokenhsi/scripts/single_task/carry_test.sh
      # train
      sh tokenhsi/scripts/single_task/carry_train.sh
      ```

* TokenHSI's unified transformer policy

  * Foundational Skill Learning

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage1_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage1_eval.sh carry # we need to specify a task to eval, e.g., traj, sit, climb, or carry.
      # train
      sh tokenhsi/scripts/tokenhsi/stage1_train.sh
      ```

      If you successfully run the test command, you will see:
      <p align="center">
        <img src="assets/stage1_demo.gif" align="center" width=60% >
      </p>


  * Policy Adaptation - Skill Composition

  * Policy Adaptation - Object Shape Variation

  * Policy Adaptation - Terrain Shape Variation

  * Policy Adaptation - Long-horizon Task Completion

      ```
      # test
      sh tokenhsi/scripts/tokenhsi/stage2_longterm_test.sh
      # eval
      sh tokenhsi/scripts/tokenhsi/stage2_longterm_eval.sh
      # train
      sh tokenhsi/scripts/tokenhsi/stage2_longterm_train.sh
      ```

### Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| F | focus on humanoid |
| Right Click + WASD | change view port |
| Shift + Right Click + WASD | change view port fast |
| K | visualize lines |
| L | record screenshot, press again to stop recording|

The recorded screenshots are saved in ``` output/imgs/ ```. You can use ``` lpanlib/others/video.py ``` to generate mp4 video from the recorded images.

```
python lpanlib/others/video.py --imgs_dir output/imgs/example_path --delete_imgs
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{pan2025tokenhsi,
  title={TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization},
  author={Pan, Liang and Yang, Zeshi and Dou, Zhiyang and Wang, Wenjia and Huang, Buzhen and Dai, Bo and Komura, Taku and Wang, Jingbo},
  booktitle={CVPR},
  year={2025},
}

@inproceedings{pan2024synthesizing,
  title={Synthesizing physically plausible human motions in 3d scenes},
  author={Pan, Liang and Wang, Jingbo and Huang, Buzhen and Zhang, Junyu and Wang, Haofan and Tang, Xu and Wang, Yangang},
  booktitle={2024 International Conference on 3D Vision (3DV)},
  pages={1498--1507},
  year={2024},
  organization={IEEE}
}
```

Please also consider citing the following papers that inspired TokenHSI.

```bibtex
@article{tessler2024maskedmimic,
  title={Maskedmimic: Unified physics-based character control through masked motion inpainting},
  author={Tessler, Chen and Guo, Yunrong and Nabati, Ofir and Chechik, Gal and Peng, Xue Bin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--21},
  year={2024},
  publisher={ACM New York, NY, USA}
}

@article{he2024hover,
  title={Hover: Versatile neural whole-body controller for humanoid robots},
  author={He, Tairan and Xiao, Wenli and Lin, Toru and Luo, Zhengyi and Xu, Zhenjia and Jiang, Zhenyu and Kautz, Jan and Liu, Changliu and Shi, Guanya and Wang, Xiaolong and others},
  journal={arXiv preprint arXiv:2410.21229},
  year={2024}
}
```

## 👏 Acknowledgements and 📚 License

This repository builds upon the following awesome open-source projects:

- [ASE](https://github.com/nv-tlabs/ASE): Contributes to the physics-based character control codebase  
- [Pacer](https://github.com/nv-tlabs/pacer): Contributes to the procedural terrain generation and trajectory following task
- [rl_games](https://github.com/Denys88/rl_games): Contributes to the reinforcement learning code  
- [OMOMO](https://github.com/lijiaman/omomo_release)/[SAMP](https://samp.is.tue.mpg.de/)/[AMASS](https://amass.is.tue.mpg.de/)/[3D-Front](https://arxiv.org/abs/2011.09127): Used for the reference dataset construction
- [InterMimic](https://github.com/Sirui-Xu/InterMimic): Used for the github repo readme design 

This codebase is released under the [MIT License](LICENSE).  
Please note that it also relies on external libraries and datasets, each of which may be subject to their own licenses and terms of use.

## 🌟 Star History

<p align="center">
    <a href="https://www.star-history.com/#liangpan99/TokenHSI&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=liangpan99/TokenHSI&type=Date" alt="Star History Chart">
    </a>
<p>



# Notes:

## Goal:
- Carry task + specify parameters
  - Start/end location
  - Size
  - Weight
  - Shape
  - Obstacles in env

- Make it ergo correct
  - Ergo loss



## Debug notes

### 2025-04-03

#### On Windows
- issacgym dont support, can't import w. correct version error

#### On Linux
- no kernel image
  ```
  reward: 0.27504950761795044 steps: 2.0
  [Error] [carb.gym.plugin] Gym cuda error: no kernel image is available for execution on the device: ../../../source/plugins/carb/gym/impl/Gym/GymPhysXCuda.cu: 948
  [Error] [carb.gym.plugin] Gym cuda error: no kernel image is available for execution on the device: ../../../source/plugins/carb/gym/impl/Gym/GymPhysXCuda.cu: 1001

  ```
  - GPU too old (Office linux) changing to Vicon lab
  - Try: `export CUDA_LAUNCH_BLOCKING=1`  
    ```
    /buildAgent/work/99bede84aa0a52c2/source/gpucommon/include/PxgCudaUtils.h (80) : internal error : SynchronizeStreams failed


    /buildAgent/work/99bede84aa0a52c2/source/physx/src/NpScene.cpp (3509) : internal error : PhysX Internal CUDA error. Simulation can not continue!

    PxgCudaDeviceMemoryAllocator fail to allocate memory 675282944 bytes!! Result = 700
    /buildAgent/work/99bede84aa0a52c2/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp (11310) : internal error : GPU compressContactStage1 fail to launch kernel stage 1!!


    /buildAgent/work/99bede84aa0a52c2/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp (11347) : internal error : GPU compressContactStage2 fail to launch kernel stage 1!!


    [Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 4202
    [Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 4210
    [Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 3480
    [Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 3535
    [Error] [carb.gym.plugin] Gym cuda error: an illegal memory access was encountered: ../../../source/plugins/carb/gym/impl/Gym/GymPhysX.cpp: 5993
    [Error] [carb.gym.plugin] Gym cuda error: no kernel image is available for execution on the device: ../../../source/plugins/carb/gym/impl/Gym/GymPhysXCuda.cu: 937
    [Error] [carb.gym.plugin] Failed to fill root state tensor
    Traceback (most recent call last):
      File "./tokenhsi/run.py", line 229, in <module>
        main()
      File "./tokenhsi/run.py", line 224, in main
        runner.run(vargs)
      File "/home/leyang/anaconda3/envs/tokenhsi/lib/python3.8/site-packages/rl_games/torch_runner.py", line 144, in run
        player.run()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/learning/amp_players.py", line 160, in run
        super().run()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/learning/common_player.py", line 101, in run
        obs_dict, r, done, info =  self.env_step(self.env, action)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/learning/common_player.py", line 171, in env_step
        obs, rewards, dones, infos = env.step(actions)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/vec_task.py", line 129, in step
        self.task.step(actions_tensor)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/base_task.py", line 155, in step
        self._physics_step()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/base_task.py", line 543, in _physics_step
        self.render()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py", line 504, in render
        super().render(sync_frame_time)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/humanoid.py", line 539, in render
        self._update_camera()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/humanoid.py", line 589, in _update_camera
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
    RuntimeError: CUDA error: an illegal memory access was encountered
    CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

    ```

    - Try `CUDA_LAUNCH_BLOCKING=1 python`
    ```
    Traceback (most recent call last):
      File "./tokenhsi/run.py", line 229, in <module>
        main()
      File "./tokenhsi/run.py", line 224, in main
        runner.run(vargs)
      File "/home/leyang/anaconda3/envs/tokenhsi/lib/python3.8/site-packages/rl_games/torch_runner.py", line 144, in run
        player.run()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/learning/amp_players.py", line 160, in run
        super().run()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/learning/common_player.py", line 101, in run
        obs_dict, r, done, info =  self.env_step(self.env, action)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/learning/common_player.py", line 171, in env_step
        obs, rewards, dones, infos = env.step(actions)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/vec_task.py", line 129, in step
        self.task.step(actions_tensor)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/base_task.py", line 155, in step
        self._physics_step()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/base_task.py", line 543, in _physics_step
        self.render()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/basic_interaction_skills/humanoid_carry.py", line 504, in render
        super().render(sync_frame_time)
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/humanoid.py", line 539, in render
        self._update_camera()
      File "/home/leyang/Documents/TokenHSI/tokenhsi/env/tasks/humanoid.py", line 589, in _update_camera
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
    RuntimeError: CUDA error: an illegal memory access was encountered
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

    ```

#### On slurm
- `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`
  - Fixed: just run in conda `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`

- libmem_filesys.so
  ```
  [Error] [carb] [Plugin: libcarb.gym.plugin.so] Could not load the dynamic library from /home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64/libcarb.gym.plugin.so. Error: libmem_filesys.so: cannot open shared object file: No such file or directory
  Using /home/wenleyan/.cache/torch_extensions/py38_cu118 as PyTorch extensions root...
  ...
  [Error] [carb] Failed to acquire interface: [carb::gym::Gym v0.1], by client: carb.gym.python.gym_38 (plugin name: (null))
  Traceback (most recent call last):
  ...
  RuntimeError: Failed to acquire interface: carb::gym::Gym (pluginName: nullptr)
  ```
  - https://github.com/isaac-sim/IsaacGymEnvs/issues/62#issuecomment-1722505385
  - this works
  - `export LD_LIBRARY_PATH="/home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"`

- seg fault
  ```
  Using /home/wenleyan/.cache/torch_extensions/py38_cu118 as PyTorch extensions root...
  Emitting ninja build file /home/wenleyan/.cache/torch_extensions/py38_cu118/gymtorch/build.ninja...
  Building extension module gymtorch...
  Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
  Loading extension module gymtorch...
  2025-04-03 15:43:34,676 - INFO - logger - logger initialized
  /var/spool/slurmd.spool/job23440867/slurm_script: line 33: 2704004 Segmentation fault      (core dumped) python ./tokenhsi/run.py --task HumanoidCarry --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry.yaml --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml --checkpoint output/single_task/ckpt_carry.pth --test --num_envs 16
  ```
  - Using remote desktop GUI w. `spgpu` does not help, same seg fault for this & isaac gym examples
  - Changing partition from `gpu` or `spgpu` to `gpu_mig40` seem to help, lead to new error `[Error] [carb.gym.plugin] Failed to create Nvf device in createNvfGraphics. Please make sure Vulkan is correctly installed.`
  - Others says it should be headless: https://github.com/isaac-sim/IsaacGymEnvs/issues/52#issuecomment-2187660362
    - add `--headless` in python
    - works for `spgpu`
    - also `self.graphics_device_id` need to stay `-1` in `base_task.py` for `create_sim`. https://forums.developer.nvidia.com/t/not-sure-what-to-set-for-graphics-device-id/193625/2


- Runs okay wihout error, but no output anywhere
  - There is a `--record` arg input, but dont seem to be used in this repo or do anything, maybe passed into packages via `**kargs` but can't find
  - Make it render and save even if headless, check out example py files mentioned here https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901
  - `tokenhsi/env/tasks/base_task.py` --> `render` modify to save
    - need camera handle locations
  - also for `render` in the child classes, `humanoid.py`, then `humanoid-xxxtask.py`

- `SetCameraLocation: Error: could not find camera with handle -1 in environment 15`
    - https://forums.developer.nvidia.com/t/why-it-returns-1-when-i-tried-to-create-camera-sensor/218083/3
    - Caused by `self.graphics_device_id=-1` when `create_sim`
    - If remove, lead to segfault (gpu, spgpu) or Vulkan error (mig40)
    - Tried VNC interactive sessions, same issue


-gpu render instead of cpu

