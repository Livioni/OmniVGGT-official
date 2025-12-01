#!/usr/bin/env python3
"""OmniVGGT Training Script"""

import os
import gc
from pathlib import Path

import torch
import wandb
import accelerate
from tqdm import tqdm

from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs

from omnivggt.utils.configs import parse_configs
from omnivggt.datasets.utils.misc import merge_dicts
from omnivggt.utils.misc import select_first_batch
from omnivggt.utils.normalization import normalize_camera_extrinsics_and_points_batch
from visual_util import (
    predictions_to_glb,
    get_world_points_from_depth,
)
from train_utils import (
    build_dataset,
    build_cosine_warmup_scheduler,
    setup_logging,
    setup_directories,
    setup_wandb,
    setup_tensorboard,
    load_model,
    build_optimizer,
    build_loss_criterion,
)

logger = get_logger(__name__, log_level="INFO")


if __name__ == '__main__':
    # ======================================================
    # 1. Configuration and Initialization
    # ======================================================
    # Parse configuration
    cfg = parse_configs()
    save_dir, logging_dir = setup_directories(cfg)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=save_dir,
        logging_dir=logging_dir
    )
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        gradient_as_bucket_view=False,
    )

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.get("mixed_precision", "no "),
        log_with=cfg.get("report_to", "tensorboard"),
        project_config=accelerator_project_config,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp_kwargs],
    )
    
    setup_logging(accelerator)
    set_seed(cfg.get("seed", 42))
    logger.info(f"Random seed set to {cfg.get('seed', 42)}")
    
    writer = None
    if accelerator.is_main_process:
        setup_wandb(cfg, save_dir)
        writer = setup_tensorboard(cfg, save_dir)
    
    # Load model
    model, weight_dtype = load_model(cfg, accelerator.device)

    # ======================================================
    # 2. Dataset and DataLoader
    # ======================================================
    logger.info("Building datasets...")
    
    # Training dataset
    train_dataloader = build_dataset(
        dataset=cfg.train_dataset,
        batch_size=cfg.get("train_batch_images", 24),
        num_workers=cfg.get("num_workers", 8),
        test=False
    )
    
    # ======================================================
    # 3. Optimizer, Scheduler, and Loss
    # ======================================================
    
    # Build optimizer
    optimizer = build_optimizer(model, cfg)
    
    # Calculate training steps
    world_size = accelerator.num_processes
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 2)
    
    total_batches_before_sharding = len(train_dataloader)
    batches_per_process = total_batches_before_sharding // world_size
    local_steps_per_epoch = batches_per_process // gradient_accumulation_steps
    total_training_steps = cfg.get('num_train_epochs') * local_steps_per_epoch
    
    logger.info("Training steps calculation (BEFORE Accelerate sharding):")
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Total batches (full dataset): {total_batches_before_sharding}")
    logger.info(f"  Batches per process (estimated): {batches_per_process}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Steps per epoch (per process, estimated): {local_steps_per_epoch}")
    logger.info(f"  Total training steps (estimated): {total_training_steps}")
    
    # Build learning rate scheduler
    lr_scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_steps=cfg.get("warmup_steps", 5000),
        total_steps=total_training_steps,
        eta_min_factor=cfg.get("eta_min_factor", 0.1)
    )
    
    # Build loss criterion
    train_criterion = build_loss_criterion(cfg)
    
    # ======================================================
    # 4. Prepare for Distributed Training
    # ======================================================
    logger.info("Preparing model, optimizer, and dataloaders for distributed training...")
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    # Verify actual steps after sharding
    actual_local_batches = len(train_dataloader)
    actual_local_steps = actual_local_batches // gradient_accumulation_steps
    logger.info(f"After sharding - Local dataloader length: {actual_local_batches}")
    logger.info(f"After sharding - Actual steps per epoch (this process): {actual_local_steps}")
    
    if actual_local_steps != local_steps_per_epoch:
        logger.warning(f"Steps mismatch! Estimated: {local_steps_per_epoch}, Actual: {actual_local_steps}")
        logger.warning(f"This may happen when total batches is not evenly divisible by world_size.")
        logger.info(f"Using actual steps ({actual_local_steps}) for progress tracking.")
        local_steps_per_epoch = actual_local_steps
    
    # ======================================================
    # 5. Resume from Checkpoint (if specified)
    # ======================================================
    initial_step = 0
    initial_epoch = 0
    
    if cfg.get("resume_model_path"):
        resume_path = cfg.get("resume_model_path")
        if os.path.exists(resume_path):
            logger.info(f"Resuming from checkpoint: {resume_path}")
            accelerator.load_state(resume_path)
            
            checkpoint_dir = resume_path.rstrip('/')
            if os.path.isdir(checkpoint_dir):
                checkpoint_name = os.path.basename(checkpoint_dir)
            else:
                checkpoint_name = os.path.basename(os.path.dirname(checkpoint_dir))
            
            if checkpoint_name.startswith('checkpoint-'):
                parts = checkpoint_name.replace('checkpoint-', '').split('-')
                initial_epoch = int(parts[0])
                initial_step = int(parts[1])
                logger.info(f"Resumed at epoch {initial_epoch}, step {initial_step}")
            else:
                logger.warning(f"Checkpoint name does not match expected format: {checkpoint_name}")
        else:
            logger.warning(f"Resume path does not exist: {resume_path}")
            logger.warning("Starting training from scratch...")
    
    # ======================================================
    # 6. Training Information
    # ======================================================
    logger.info("=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"  Number of epochs: {cfg.get('num_train_epochs')}")
    logger.info(f"  Examples per epoch (this process): {len(train_dataloader)}")
    logger.info(f"  Steps per epoch (this process): {local_steps_per_epoch}")
    logger.info(f"  Total training steps (this process): {total_training_steps}")
    logger.info("---")
    logger.info(f"  Batch size per device: {cfg.get('train_batch_images')}")
    logger.info(f"  Number of GPUs: {world_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Effective global batch size: {cfg.get('train_batch_images') * world_size * gradient_accumulation_steps}")
    logger.info("---")
    logger.info(f"  Max gradient norm: {cfg.get('max_grad_norm', 1.0)}")
    logger.info(f"  Mixed precision: {cfg.get('mixed_precision', 'no')}")
    logger.info(f"  Checkpointing frequency: every {cfg.get('checkpointing_steps', 10000)} steps")
    logger.info(f"  Logging frequency: every {cfg.get('num_save_log', 10)} steps")
    logger.info("=" * 60)

    # ======================================================
    # 7. Training Loop
    # ======================================================
    global_step = initial_step
    accumulation_steps = cfg.get("gradient_accumulation_steps", 2)
    
    for epoch in range(initial_epoch, cfg.get('num_train_epochs')):
        logger.info("=" * 60)
        logger.info(f"Starting Epoch {epoch + 1}/{cfg.get('num_train_epochs')}")
        logger.info("=" * 60)
        
        model.train()
        
        # Set epoch for proper shuffling in distributed training
        if hasattr(train_dataloader, 'dataset') and hasattr(train_dataloader.dataset, 'set_epoch'):
            train_dataloader.dataset.set_epoch(epoch)
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        if epoch == initial_epoch:
            step_in_epoch = global_step % local_steps_per_epoch
        else:
            step_in_epoch = 0
        
        progress_bar = tqdm(
            total=local_steps_per_epoch,
            initial=step_in_epoch,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )
        
        # Training loop for this epoch
        for step, batch in enumerate(train_dataloader):
            batch = merge_dicts(batch)
            
            # Normalize camera extrinsics and points for loss computation
            new_extrinsics, _, new_world_points, new_depths = normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch['extrinsic'],
                cam_points=None,
                world_points=batch['world_points'],
                depths=batch['depth'],
                point_masks=batch['valid_mask'],
            )
            
            # Store original inputs for model
            input_extrinsics = batch['extrinsic'].clone()
            input_depths = batch['depth'].clone()
            input_mask = batch['valid_mask'].clone()
            
            # Update batch with normalized values for loss computation
            batch['extrinsic'] = new_extrinsics
            batch['world_points'] = new_world_points
            batch['depth'] = new_depths
            
            # Forward pass
            inputs = {
                'images': batch['images'],
                'extrinsics': input_extrinsics,
                'intrinsics': batch['intrinsic'],
                'depth': input_depths,
                'mask': input_mask
            }
            predictions = model(**inputs)
            
            # Compute loss
            loss_details = {}
            with torch.amp.autocast('cuda', enabled=False):
                loss_dict = train_criterion(predictions, batch)
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        loss_details[key] = value.detach().item()
                    else:
                        loss_details[key] = value
            
            accelerator.backward(loss_dict['objective'])
            progress_bar.set_postfix(**loss_details)
            
            # Optimizer step with gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.get('max_grad_norm', 1.0))
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                accelerator.log(loss_details, step=global_step)
                
                # Logging
                if accelerator.is_main_process and global_step % cfg.get("num_save_log", 10) == 0:
                    if cfg.get("wandb", False):
                        wandb_dict = {**loss_details, "epoch": epoch}
                        for i, param_group in enumerate(optimizer.param_groups):
                            wandb_dict[f"lr/group_{i}"] = param_group['lr']
                        wandb.log(wandb_dict, step=global_step)
                    
                    if writer is not None:
                        for k, v in loss_details.items():
                            if isinstance(v, (int, float)):
                                writer.add_scalar(f"train/{k}", v, global_step)
                        for i, param_group in enumerate(optimizer.param_groups):
                            writer.add_scalar(f"lr/group_{i}", param_group['lr'], global_step)
                        writer.add_scalar("train/epoch", epoch, global_step)

                # Visualization
                if accelerator.is_main_process and cfg.get("save_glb_visualization", False) and global_step % cfg.get("num_save_visual", 5000) == 0:
                    logger.info(f"Generating visualization at step {global_step}...")
                    save_pts_dir = os.path.join(save_dir, f'epoch-{epoch}')
                    Path(save_pts_dir).mkdir(parents=True, exist_ok=True)
                    
                    try:
                        with torch.no_grad():
                            predictions_0 = select_first_batch(predictions)
                            get_world_points_from_depth(predictions_0)
                            
                            glbscene = predictions_to_glb(
                                predictions_0,
                                conf_thres=cfg.get("vis_conf_threshold", 0.2),
                                filter_by_frames=cfg.get("vis_filter_by_frames", "All"),
                                mask_black_bg=cfg.get("vis_mask_black_bg", False),
                                mask_white_bg=cfg.get("vis_mask_white_bg", False),
                                show_cam=cfg.get("vis_show_cam", True),
                                mask_sky=cfg.get("vis_mask_sky", False),
                                target_dir=save_pts_dir,
                                prediction_mode=cfg.get("vis_prediction_mode", "Predicted Depth"),
                            )
                            
                            glb_path = os.path.join(save_pts_dir, f'glbscene_{global_step}.glb')
                            glbscene.export(file_obj=glb_path)
                            logger.info(f"Visualization saved to {glb_path}")
                            
                            if cfg.get("wandb", False):
                                wandb.log({"visualization": wandb.Object3D(glb_path)}, step=global_step)
                            
                            del glbscene, predictions_0
                            gc.collect()
                    except Exception as e:
                        logger.warning(f"Failed to generate visualization: {e}")
                
                # Checkpointing
                if accelerator.is_main_process and global_step % cfg.get("checkpointing_steps", 10000) == 0:
                    save_path = os.path.join(save_dir, f"checkpoint-{epoch}-{global_step}")
                    logger.info(f"Saving checkpoint to {save_path}...")
                    accelerator.save_state(save_path)
                    logger.info(f"Checkpoint saved successfully")
        
        progress_bar.close()
        
        # Apply remaining gradients at epoch end
        if (step + 1) % accumulation_steps != 0:
            logger.info(f"Applying remaining gradients at end of epoch {epoch + 1}")
            accelerator.clip_grad_norm_(model.parameters(), cfg.get('max_grad_norm', 1.0))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch + 1} completed - Global step: {global_step}")
        logger.info("=" * 60)
        
        if accelerator.is_main_process and cfg.get("save_each_epoch", True):
            epoch_save_path = os.path.join(save_dir, f"checkpoint-epoch-{epoch + 1}")
            logger.info(f"Saving end-of-epoch checkpoint to {epoch_save_path}...")
            accelerator.save_state(epoch_save_path)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Training completed
    logger.info("=" * 60)
    logger.info("Training Completed!")
    logger.info("=" * 60)
    
    if accelerator.is_main_process:
        final_save_path = os.path.join(save_dir, "final_checkpoint")
        logger.info(f"Saving final checkpoint to {final_save_path}...")
        accelerator.save_state(final_save_path)
        logger.info("Final checkpoint saved successfully")
        
        if cfg.get("wandb", False):
            wandb.finish()
            logger.info("WandB logging finished")
        
        if writer is not None:
            writer.close()
            logger.info("TensorBoard logging finished")
    
    logger.info("All done!")