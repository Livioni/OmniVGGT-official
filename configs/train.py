# ======================================================
# OmniVGGT Training Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs"
exp_name = "omnivggt"
logging_dir = "logs"

# == Logging Configuration ==
wandb = False
tensorboard = True
report_to = "tensorboard"
num_save_log = 10
num_save_visual = 5000
checkpointing_steps = 10000

# == Model Configuration ==
model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model_load_strict = False
model_requires_grad = True
enable_point = True
enable_depth = True
enable_camera = True

# == Training Configuration ==
mixed_precision = "bf16"  # Options: "no", "fp16", "bf16"
seed = 42
num_train_epochs = 10
gradient_accumulation_steps = 2
max_grad_norm = 1.0
cam_drop_prob = 0.1
depth_drop_prob = 0.3
save_each_epoch = False


# == Dataset Configuration ==
train_batch_images = 24
num_workers = 8

# == Optimizer Configuration ==
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.01

# == Learning Rate Configuration ==
lr = 2e-5
lr_patch_embed = 1e-5
lr_camera_head = 2e-5
lr_depth_head = 2e-5
lr_point_head = 2e-5

# == Learning Rate Scheduler Configuration ==
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 8000
eta_min_factor = 0.1  # Minimum learning rate factor for cosine decay

# == Loss Configuration ==
# Camera loss
camera_loss_weight = 5.0
camera_loss_type = "l1"  # Options: "l1", "l2", "smooth_l1"

# Depth loss
depth_loss_weight = 1.0
depth_gradient_loss_fn = "grad"
depth_valid_range = 0.98

# Point loss
point_loss_weight = 1.0
point_loss_type = "normal"
point_valid_range = 0.98

# == Visualization Configuration ==
save_glb_visualization = False
vis_conf_threshold = 0.2
vis_filter_by_frames = "All"
vis_mask_black_bg = False
vis_mask_white_bg = False
vis_show_cam = True
vis_mask_sky = False
vis_prediction_mode = "Predicted Depth"

# == Resume Configuration ==
resume_model_path = None

# == Dataset Configuration ==
resolution = [(518, 518), (518, 490), (518, 462), 
              (518, 434), (518, 406), (518, 378), 
              (518, 350), (518, 336), (518, 322), 
              (518, 294), (518, 266), (518, 252), 
              (518, 238), (518, 210), (518, 182), 
              (518, 168)]

train_dataset = f"22_400 @ ARKitScenesHigh(use_cache = True, quick = False, top_k = 64, dset='Training', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
                + 3_600  @ Bedlam(use_cache = True, quick = False, top_k = 64, dset='', z_far = 200, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
                + 24_800 @ Co3d(use_cache = True, quick = False, top_k = 64, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  9_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='1K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 10_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='2K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  9_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='3K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  6_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='4K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 10_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='5K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 10_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='6K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 10_000 @ Dl3dv(use_cache = True, quick = False, top_k = 64, dset='7K', z_far = 500, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 11_200 @ Hypersim(use_cache = True, quick = False, top_k = 64, dset='', z_far = 200, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 10_000 @ Kubric(use_cache = True, quick = False, top_k = 128, dset='trackings', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 84_000 @ MapFree(use_cache = True, quick = False, top_k = 256, dset='', z_far = 400, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 22_400 @ MegaDepth(use_cache = True, quick = False, top_k = 64, dset='', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 28_800 @ Mp3d(use_cache = True, quick = False, top_k = 32, dset='', z_far = 100, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  1_400 @ Mvs_Synth(use_cache = True, quick = False, top_k = 64, dset='', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 38_400 @ Scannet(use_cache = True, quick = False, top_k = 64, dset='scans', z_far = 100, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 16_000 @ ScannetppV2(use_cache = True, quick = False, top_k = 64, dset='', z_far = 100, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  9_400 @ Spring(use_cache = True, quick = False, top_k = 128, dset='', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  7_200 @ PointOdysseyDUSt3R(use_cache = True, quick = False, top_k = 128, dset='train', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 14_400 @ Uasol(use_cache = True, quick = False, top_k = 64, dset='', z_far = 100, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 20_000 @ Waymo(use_cache = True, quick = False, top_k = 64, dset='', z_far = 655, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +    168 @ Unreal4k(use_cache = True, quick = False, top_k =64, dset='', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 56_000 @ TarTanAirDUSt3R(use_cache = True, quick = False, top_k =64, dset='', z_far = 1000, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               +  5_600 @ Vkitti(use_cache = True, quick = False, top_k = 64, dset='', z_far = 655, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 36_000 @ Dynamic_Replica(use_cache = True, quick = False, top_k =36, dset='train', z_far = 100, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
               + 26_000 @ Wildrgb(use_cache = True, quick = False, top_k = 128, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985)"

                    
                    
              

