<div align="center">

<img src="assets/omnivggt.png" alt="OmniVGGT Logo" width="120"/>

<h2 style="color: red; font-weight: bold;">CVPR 2026 Highlight</h2>

<h1>OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer</h1>


<a href="https://arxiv.org/abs/2511.10560" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-OmniVGGT-red" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2511.10560"><img src="https://img.shields.io/badge/arXiv-2510.22706-b31b1b" alt="arXiv"></a>
<a href="https://livioni.github.io/OmniVGGT-official"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/Livioni/OmniVGGT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-OmniVGGT-yellow" alt="Hugging Face"></a>

---

Haosong Peng*, Hao Li*, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi, <br>
Zhengshen Zhang, Yufeng Zhan†, Junfei Zhang†, Wenchao Xu†, Ziwei Liu

 \* Equal Contribution, † Corresponding Author

</div>

<div align="center">
  <img src="assets/teaser.png" alt="OmniVGGT Overview" width="800"/>
</div>

## 🔍 Overview

OmniVGGT is a spatial foundation model that can effectively benefit from an arbitrary number of auxiliary geometric modalities (depth, camera intrinsics and pose) to obtain high-quality 3D geometric results. Experimental results show that OmniVGGT achieves state-of-the-art performance across various downstream tasks and further improves performance on robot manipulation tasks.

## 🔧 Installation

### Setup Environment

```bash
conda create -n omnivggt python=3.10

conda activate omnivggt

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```


## 🚀 Quick Start

You can use OmniVGGT directly in your Python code:

```python
import torch
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import load_images_and_cameras

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = OmniVGGT().to(device)
from safetensors.torch import load_file
state_dict = load_file("checkpoints/OmniVGGT.safetensors")
model.load_state_dict(state_dict, strict=True)
model.eval()

# Load and preprocess images
images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
    load_images_and_cameras(
        image_folder="example/office/images/",
        camera_folder=None,  # Optional
        depth_folder=None,   # Optional
        target_size=518
    )

# Prepare inputs
inputs = {
    'images': images.to(device),
    'extrinsics': extrinsics.to(device),
    'intrinsics': intrinsics.to(device),
    'depth': depthmaps.to(device),
    'mask': masks.to(device),
    'depth_gt_index': depth_indices,
    'camera_gt_index': camera_indices
}

# Run inference
with torch.no_grad():
    predictions = model(**inputs)
```

### Advanced Options

```bash
# Basic usage - only images required
python inference.py --image_folder example/office/images/

# With auxiliary camera and depth (optional)
python inference.py \
    --image_folder example/office/images/ \
    --camera_folder example/office/cameras/ \  # optional: auxiliary camera parameters
    --depth_folder example/office/depths/ \    # optional: auxiliary depth maps
    --target_size 518 \                        # target image size (default: 518)
    \
    # Processing options
    --use_point_map \                          # use point map instead of depth-based points
    --mask_sky \                               # apply sky segmentation to filter out sky points
    --mask_black_bg \                          # mask out black background pixels (RGB sum < 16)
    --mask_white_bg \                          # mask out white background pixels (RGB > 240)
    \
    # Visualization options
    --conf_threshold 25.0 \                    # initial confidence threshold percentage (default: 25.0)
    --port 8080 \                              # viser server port (default: 8080)
    --background_mode \                        # run server in background mode
    \
    # Export options
    --save_glb                                 # save output as GLB file (saved as scene.glb)
```

## 📊 Input Description

- The *image_folder* contains all the images to be processed for reconstruction. The *camera_folder* and *depth_folder* are optional and may include any combination. For example, all the following combinations are ok.

<details>
<summary>📁 Click to see example folder structure combinations</summary>

```plaintext
example/infinigen
├── cameras
│   ├── 26_0_0001_0.txt
│   ├── 33_0_0001_0.txt
│   ├── 81_0_0001_0.txt
│   └── 91_0_0001_0.txt
├── depths
│   ├── 26_0_0001_0.npy
│   ├── 33_0_0001_0.npy
│   ├── 81_0_0001_0.npy
│   └── 91_0_0001_0.npy
└── images
    ├── 26_0_0001_0.png
    ├── 33_0_0001_0.png
    ├── 81_0_0001_0.png
    └── 91_0_0001_0.png
```

```plaintext
example/infinigen
├── cameras
│   ├── 26_0_0001_0.txt
│   └── 91_0_0001_0.txt
├── depths
│   ├── 33_0_0001_0.npy
│   └── 81_0_0001_0.npy
└── images
    ├── 26_0_0001_0.png
    ├── 33_0_0001_0.png
    ├── 81_0_0001_0.png
    └── 91_0_0001_0.png
```

```plaintext
example/infinigen
├── cameras
│   ├── 26_0_0001_0.txt
│   └── 33_0_0001_0.txt
├── depths
│   └── 91_0_0001_0.npy
└── images
    ├── 26_0_0001_0.png
    ├── 33_0_0001_0.png
    ├── 81_0_0001_0.png
    └── 91_0_0001_0.png
```

</details>

- If one or more images have auxiliary camera information, please ensure that the first image always includes camera information.
- Camera poses and intrinsics are provided in **.txt** files. Please refer to [frame-000002.txt](example/office/cameras/frame-000002.txt) for specific examples. Depth maps can be loaded from either **.png** or **.npy** files.
- Camera poses are expected to follow the OpenCV `camera-to-world` convention, Depth maps should be aligned with their corresponding camera poses.

## 📸 Example

### Comparison: Without vs. With Camera Parameters

<div align="center">
  <img src="assets/officewoc.png" alt="Without Camera" width="400"/>
  <img src="assets/officewc.png" alt="With Camera" width="400"/>
</div>

**Left**: Results without auxiliary camera parameters

```bash
python inference.py --image_folder example/office/images
```

**Right**: Results with auxiliary camera parameters

```bash
python inference.py --image_folder example/office/images --camera_folder example/office/cameras
```

## Training

### Prepare Datasets

Follow [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) to download and preprocess the datasets.

In general, a preprocessed dataset should contain at least **RGB images** and the corresponding **depth** and **camera parameters**, including **extrinsics** and **intrinsics**, and some may contain additional sky masks.

Take [dl3dv.py](omnivggt/datasets/dl3dv.py) as an example: a complete scene is organized as follows:

```bash
dl3dv
├── 1K
│   ├── 001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f
│   │   └── dense
│   │       ├── cam           # camera parameters (extrinsics + intrinsics), frame_xxxxx.npz
│   │       ├── depth         # depth maps, frame_xxxxx.npz
│   │       ├── outlier_mask  # depth outlier masks (invalid depth regions), frame_xxxxx.png
│   │       ├── rgb           # original RGB image sequence, frame_xxxxx.png
│   │       └── sky_mask      # sky segmentation masks, frame_xxxxx.png 
│   ├── <scene_id_2>
│   │   └── dense
│   │       └── ...
│   └── ...
├── 2K
│   ├── <scene_id_1>
│   │   └── dense
│   │       └── ...
│   └── ...
└── ...
```

We have the following important configs in the script.

1. **dataset_location**: Dataset storage location
2. **use_cache**: Whether to use cached annotations. Set `use_cache = False` for the first run to traverse the dataset and cache data addresses, `use_cache = True` for training to load cached data directly for faster startup.
3. **dset**: Used when datasets have subsets, such as distinguishing between Train and Test.
4. **specify**: Used for testing to fix the images extracted by get_item for easier comparison.
5. **top_k**: Number of cameras closest to each anchor frame camera, used for sequence sampling range per scene during training.
6. **z_far**: Maximum scene depth, pixels above z_far will be masked out.
7. **quick**: When `use_cache = False`, quickly load the first a few scenes of the dataset.
8. **verbose**: Print detailed information.

```python
    dataset = Dl3dv(
        dataset_location="/mnt/disk3.8-4/datasets/dl3dv",
        dset='1K',
        use_cache=False,
        top_k=50,
        quick=False,
        verbose=True,
        resolution=(512, 224),
        seed=777,
        aug_crop=16,
        z_far=200)
```

Modify lines 34, 393 to the dataset_location.

Modify lines 160-167 to save the cached data paths. 

Modify line 87 to the annoataions locations (cached data paths).

Set `use_cache = False` and `quick = False` and run [dl3dv.py](omnivggt/datasets/dl3dv.py) with the above settings for the first time to generate cache files.  

```bash
python omnivggt/datasets/dl3dv.py
```

You can also use `visualize_scene((100, 0, num_views))` to visualize the saved scene to make sure the dataloader is correct.

### Training Config

This section explains the configuration parameters in `configs/train.py`:

#### Common Configuration
- **output_dir**: Output directory for saving model checkpoints and logs (default: "outputs")
- **exp_name**: Experiment name (default: "omnivggt")
- **logging_dir**: Directory for logging files (default: "logs")

#### Logging Configuration
- **wandb**: Enable Weights & Biases logging (default: False)
- **tensorboard**: Enable TensorBoard logging (default: True)
- **num_save_log**: Number of recent log files to keep (default: 10)
- **num_save_visual**: Frequency of saving visualization results to the output_dir. (every N steps, default: 5000)
- **checkpointing_steps**: Save checkpoint every N steps (default: 10000)

#### Model Configuration
- **model_url**: URL to load pretrained model weights (default: VGGT-1B model)
- **model_load_strict**: Whether to strictly load model weights (default: False)
- **model_requires_grad**: Whether model parameters require gradients during training (default: True)
- **enable_point**: Enable point prediction head (default: True)
- **enable_depth**: Enable depth prediction head (default: True)
- **enable_camera**: Enable camera parameter prediction head (default: True)

#### Training Configuration
- **mixed_precision**: Mixed precision training mode, options: "no", "fp16", "bf16" (default: "bf16")
- **seed**: Random seed for reproducibility (default: 42)
- **num_train_epochs**: Number of training epochs (default: 10)
- **gradient_accumulation_steps**: Gradient accumulation steps (default: 2)
- **max_grad_norm**: Maximum gradient norm for clipping (default: 1.0)
- **cam_drop_prob**: Camera dropout probability during training (default: 0.1)
- **depth_drop_prob**: Depth dropout probability during training (default: 0.3)
- **save_each_epoch**: Whether to save checkpoint after each epoch (default: False)

#### Dataset Configuration
- **train_batch_images**: Number of images per training batch (default: 24)
- **num_workers**: Number of data loading workers (default: 8)
- **resolution**: List of image resolutions for multi-resolution training
- **train_dataset**: Dataset composition string defining training datasets and their configurations, make sure set use_cache = True, quick = False here to accelerate loading speed.


#### Resume Configuration
- **resume_model_path**: Path to resume training from a checkpoint (default: None)

### Start Training

#### Single GPU Training
```bash
python train_omnivggt.py --config configs/train.py
```

#### Multi-GPU Training (One Node 8x GPUs)
```bash
accelerate launch --num_processes=8 train_omnivggt.py --config configs/train.py
```

## 📝 To-Do List

- [X] Release project paper.
- [X] Release pretrained models.
- [X] Release training code.

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
{omnivggt2025,
  title={OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer},
  author={Haosong Peng and Hao Li and Yalun Dai and Yushi Lan and Yihang Luo and Tianyu Qi and Zhengshen Zhang and Yufeng Zhan and Junfei Zhang and Wenchao Xu and Ziwei Liu}
  journal={arXiv preprint arXiv:2511.10560},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon [VGGT](https://github.com/facebookresearch/vggt) by Meta AI
- Uses [viser](https://github.com/nerfstudio-project/viser) for 3D visualization
