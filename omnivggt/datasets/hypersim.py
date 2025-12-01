# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed hypersim dataset
# See datasets_preprocess/preprocess_hypersim.py
# --------------------------------------------------------
import os.path as osp
import cv2, os
import numpy as np
import sys
sys.path.append('.')
import torch
import numpy as np
import glob, math
import random
from PIL import Image
import json
import joblib

from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.datasets.utils.image_ranking import compute_ranking
from omnivggt.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
from omnivggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from omnivggt.datasets.utils.misc import threshold_depth_map
from omnivggt.utils.image import imread_cv2

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

broken_scenes = [
    'ai_003_001',
    'ai_004_009',
    'ai_015_006',
    'ai_038_007',
    'ai_046_001',
    'ai_046_009',
    'ai_048_004',
    'ai_053_005',
    'ai_012_007',
    'ai_013_001',
    'ai_023_008',
    'ai_026_020',
    'ai_023_009',
    'ai_038_007',
    'ai_023_004',
    'ai_023_006',
    'ai_026_013',
    'ai_026_018',
]

class Hypersim(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/mnt/disk3.8-4/datasets/hypersim',
                 dset='',
                 use_cache=False,
                 use_augs=False,
                 top_k=256,
                 z_far=200,
                 quick=False,
                 verbose=False,
                 specify=False,
                 *args,
                 **kwargs
                 ):

        print('loading Hypersim dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'Hypersim'
        self.dset = dset
        self.top_k = top_k
        self.z_far = z_far
        self.verbose = verbose
        self.specify = specify
        self.use_augs = use_augs
        self.use_cache = use_cache

        # Initialize data containers
        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.rank = dict()

        # Find sequences
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))

        if quick:
           self.sequences = self.sequences[0:1]

        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if self.use_cache:
            dataset_location = '/mnt/disk3.8-4/annotations/hypersim_annotations'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))

            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

        else:

            for seq in self.sequences:
                if self.verbose:
                    print('seq', seq)

                if seq.split('/')[-2] in broken_scenes:
                    print(f"Skipping broken scene: {seq.split('/')[-2]}")
                    continue

                sub_scenes = os.listdir(seq)
                for sub_seq in sub_scenes:

                    rgb_path = os.path.join(seq, sub_seq)
                    depth_path = os.path.join(seq, sub_seq)
                    annotaions_file_path = os.path.join(seq, sub_seq)
                    num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                    if num_frames < 24:
                        print('skipping %s, too few images' % (seq))
                        continue

                    new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                    old_sequence_length = len(self.full_idxs)
                    self.full_idxs.extend(new_sequence)
                    self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, '*.png'))))
                    self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, '*.npy'))))
                    seq_annotaions_path = sorted(glob.glob(os.path.join(annotaions_file_path, '*.npz')))
                    self.all_annotation_paths.extend(seq_annotaions_path)

                    N = len(self.full_idxs)
                    assert len(self.all_rgb_paths) == N and \
                           len(self.all_annotation_paths) == N and \
                           len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."

                    # load annotations
                    extrinsics_seq = []
                    #load intrinsics and extrinsics
                    for anno in seq_annotaions_path:
                        camera_info = np.load(anno)
                        pose = np.array(camera_info['pose'], dtype=np.float32)
                        intrinsics = np.array(camera_info['intrinsics'], dtype=np.float32)
                        assert pose.shape == (4, 4), f"Pose shape mismatch in {anno}: {pose.shape}"
                        assert intrinsics.shape == (3, 3), f"Intrinsics shape mismatch in {anno}: {intrinsics.shape}"
                        self.all_extrinsic.extend([pose])
                        self.all_intrinsic.extend([intrinsics])
                        extrinsics_seq.append(pose)
                    all_extrinsic_numpy = np.array(extrinsics_seq)

                    assert len(all_extrinsic_numpy) != 0
                    ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                    ranking = np.array(ranking, dtype=np.int32)
                    ranking += old_sequence_length
                    for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                        self.rank[i] = ranking[ind]

            os.makedirs(f'annotations/hypersim_annotations/{dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/hypersim_annotations/{dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/hypersim_annotations/{dset}/depth_paths.json')
            joblib.dump(self.all_extrinsic, f'annotations/hypersim_annotations/{dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/hypersim_annotations/{dset}/intrinsics.joblib')
            joblib.dump(self.rank, f'annotations/hypersim_annotations/{dset}/rankings.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)
    
    def __len__(self):
        return len(self.full_idxs)
    
    def _get_views(self, index, num, resolution, rng):
        # Get frame indices based on number of views needed
        if num != 1:
            anchor_frame = self.full_idxs[index]
            top_k = min(self.top_k, len(self.rank[anchor_frame]))
            rest_frame = self.rank[anchor_frame][:top_k]

            if self.specify:
                L = len(rest_frame)
                step = max(1, math.floor(L / (num)))
                idxs = list(range(step - 1, L, step))[:(num - 1)]
                rest_frame_indexs = [rest_frame[i] for i in idxs]
                if len(rest_frame_indexs) < (num - 1):
                    rest_frame_indexs.append(rest_frame[-1])
            else:
                rest_frame_indexs = np.random.choice(rest_frame, size=num-1, replace=True).tolist()

            full_idx = [anchor_frame] + rest_frame_indexs
        else:
            full_idx = [self.full_idxs[index]]

        # Extract paths and camera parameters for selected frames
        rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
        depth_paths = [self.all_depth_paths[i] for i in full_idx]
        camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
        intrinsics_list = [self.all_intrinsic[i] for i in full_idx]

        views = []
        for impath, depthpath, camera_pose, intrinsics in zip(rgb_paths, depth_paths, camera_pose_list, intrinsics_list):
            # Load and preprocess images
            rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)
            depthmap = np.load(depthpath).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # Replace invalid depths

            depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng, info=impath)

            # Create view dictionary
            views.append({
                'img': rgb_image,
                'depthmap': depthmap,
                'camera_pose': camera_pose,  # cam2world
                'camera_intrinsics': intrinsics,
                'dataset': self.dataset_label,
                'label': impath.split('/')[-3],
                'instance': osp.basename(impath),
            })

        return views
    
    def __getitem__(self, idx):
        # Parse index tuple: (idx, ar_idx[, num])
        if isinstance(idx, tuple):
            idx, ar_idx, *num_args = idx
            num = num_args[0] if num_args else 1
        else:
            assert len(self._resolutions) == 1
            ar_idx, num = 0, 1

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, num, resolution, self._rng)
        assert len(views) == num

        # Process each view
        for v, view in enumerate(views):
            # Basic assertions
            assert 'pts3d' not in view and 'valid_mask' not in view, \
                f"pts3d/valid_mask should not be present in view {view_name(view)}"
            assert 'camera_intrinsics' in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'

            # Set view metadata
            view['idx'] = (idx, ar_idx, v)
            view['z_far'] = self.z_far
            view['true_shape'] = np.int32(view['img'].size[::-1])  # (height, width)
            view['img'] = self.transform(view['img'])

            # Handle camera pose
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'

            # Validate data types
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

            # Compute 3D coordinates
            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
            world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                view['depthmap'], view['camera_pose'], view['camera_intrinsics'], z_far=self.z_far
            )
            view['world_coords_points'] = world_coords_points
            view['cam_coords_points'] = cam_coords_points
            view['point_mask'] = point_mask

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')

        # Define field mappings for data collection and stacking
        field_config = {
            'img': ('images', torch.stack),
            'depthmap': ('depth', lambda x: np.stack([d[:, :, np.newaxis] for d in x]), 'depthmap'),
            'camera_pose': ('extrinsic', lambda x: np.stack([p[:3] for p in x]), 'camera_pose'),
            'camera_intrinsics': ('intrinsic', np.stack),
            'world_coords_points': ('world_points', np.stack),
            'true_shape': ('true_shape', np.array),
            'point_mask': ('valid_mask', np.stack),
            'label': ('label', lambda x: x),  # Keep as list
            'instance': ('instance', lambda x: x),  # Keep as list
        }

        # Collect and stack data using list comprehensions and field config
        result = {}
        for field_key, (output_key, stack_func, *input_keys) in field_config.items():
            input_key = input_keys[0] if input_keys else field_key
            data_list = [view[input_key] for view in views]
            result[output_key] = stack_func(data_list)

        # Add dataset label
        result['dataset'] = self.dataset_label

        return result

if __name__ == "__main__":
    from omnivggt.viz import SceneViz, auto_cam_size
    from omnivggt.utils.image import rgb

    num_views = 10
    use_augs = False
    n_views_list = range(num_views)

    dataset = Hypersim(
        dataset_location="/mnt/disk3.8-4/datasets/hypersim",
        dset='',
        use_cache=True,
        use_augs=use_augs,
        top_k=50,
        quick=False,
        verbose=True,
        resolution=(518, 378),
        aug_crop=16,
        aug_focal=1,
        z_far=100,
        seed=985)

    def visualize_scene(idx):
        views = dataset[idx]
        # assert len(views['images']) == num_views, f"Expected {num_views} views, got {len(views)}"
        viz = SceneViz()
        poses = views['extrinsic']
        views['extrinsic'] = closed_form_inverse_se3(poses)
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = views['world_points'][view_idx]
            valid_mask = views['valid_mask'][view_idx]
            colors = rgb(views['images'][view_idx])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views['extrinsic'][view_idx],
                        focal=views['intrinsic'][view_idx][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        # return viz.show()
        return viz.save_glb('hypersim_scene.glb')

    dataset[(0, 0, num_views)]
    # visualize_scene((200, 0, num_views))
    print('dataset loaded')
