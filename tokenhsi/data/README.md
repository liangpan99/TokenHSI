---
license: mit
task_categories:
- robotics
---

# TokenHSI Dataset Instruction

Please download the datasets and organize them as the following file structure:

```
|-- tokenhsi
    |-- data
        |-- dataset_amass_loco
            |-- motions
            |-- smpl_params
            |-- ...
        |-- dataset_amass_climb
            |-- motions
            |-- objects
            |-- smpl_params
            |-- ...
        |-- dataset_sit
            |-- motions
            |-- objects
            |-- ...
        |-- dataset_carry
            |-- motions
            |-- ...
|-- body_models
|-- output
```

* dataset_stage1.zip, which is used in the foundational skill learning stage, contains four folders:
  *   dataset_amass_loco
  *   dataset_amass_climb
  *   dataset_sit
  *   dataset_carry

GitHub repo: https://github.com/liangpan99/TokenHSI  
arXiv paper: https://huggingface.co/papers/2503.19901  
Project page: https://liangpan99.github.io/TokenHSI/