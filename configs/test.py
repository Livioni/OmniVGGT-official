# ======================================================
# OmniVGGT Training Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs"
exp_name = "omnivggt-test"
logging_dir = "logs"

# == Logging Configuration ==
wandb = False
tensorboard = True
report_to = "tensorboard"
num_save_log = 10
num_save_visual = 500
checkpointing_steps = 100

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
num_train_epochs = 2
gradient_accumulation_steps = 1
max_grad_norm = 1.0
cam_drop_prob = 0.1
depth_drop_prob = 0.3

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
warmup_steps = 50
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
point_gradient_loss_fn = "normal"
point_valid_range = 0.98

# == Visualization Configuration ==
vis_conf_threshold = 0.2
vis_filter_by_frames = "All"
vis_mask_black_bg = False
vis_mask_white_bg = False
vis_show_cam = True
vis_mask_sky = False
vis_prediction_mode = "Predicted Depth"

# == Resume Configuration ==
resume_model_path = "outputs/omnivggt-test/checkpoint-1-250"

# == Dataset Configuration ==
resolution = [(518, 518), (518, 490), (518, 462), 
              (518, 434), (518, 406), (518, 378), 
              (518, 350), (518, 336), (518, 322), 
              (518, 294), (518, 266), (518, 252), 
              (518, 238), (518, 210), (518, 182), 
              (518, 168)]

train_dataset = f"1000 @ ARKitScenesHigh(use_cache = True, quick = False, top_k = 64, dset='Training', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985)"



