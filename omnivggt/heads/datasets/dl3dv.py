# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed DL3DV dataset
# See datasets_preprocess/preprocess_dl3dv.py
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

import omnivggt.datasets.utils.cropping as cropping
from omnivggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from omnivggt.datasets.utils.image_ranking import compute_ranking
from omnivggt.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
from omnivggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from omnivggt.datasets.utils.misc import threshold_depth_map

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class Dl3dv(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/mnt/disk3.8-4/datasets/dl3dv',
                 dset='',
                 use_cache=True,
                 use_augs=False,
                 specify=False,
                 top_k=256,
                 z_far=500,
                 quick=False,
                 verbose=False,
                 load_mask=True,
                 *args,
                 **kwargs
                 ):

        print('loading DL3DV dataset...')
        super().__init__(*args, **kwargs)

        # Initialize instance attributes
        self.dataset_label = 'DL3DV'
        self.dset = dset
        self.top_k = top_k
        self.z_far = z_far
        self.verbose = verbose
        self.specify = specify
        self.use_cache = use_cache
        self.load_mask = load_mask
        self.use_augs = use_augs

        # Initialize data containers
        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_outlier_mask_paths = []
        self.all_sky_mask_paths = []
        self.rank = dict()

        # Find sequences
        self.sequences = sorted(glob.glob(os.path.join(dataset_location, dset, "*/")))

        if quick:
           self.sequences = self.sequences[0:1]

        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if self.use_cache:
            dataset_location = '/mnt/disk3.8-4/annotations/dl3dv_annotations'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            all_outlier_mask_paths_file = os.path.join(dataset_location, dset, 'outlier_mask_paths.json')
            all_sky_mask_paths_file = os.path.join(dataset_location, dset, 'sky_mask_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)
            with open(all_outlier_mask_paths_file, 'r', encoding='utf-8') as file:
                self.all_outlier_mask_paths = json.load(file)
            with open(all_sky_mask_paths_file, 'r', encoding='utf-8') as file:
                self.all_sky_mask_paths = json.load(file)

            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.all_outlier_mask_paths = [self.all_outlier_mask_paths[str(i)] for i in range(len(self.all_outlier_mask_paths))]
            self.all_sky_mask_paths = [self.all_sky_mask_paths[str(i)] for i in range(len(self.all_sky_mask_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))

            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))

        else:

            for seq in self.sequences:
                if self.verbose:
                    print('seq', seq)

                rgb_path = os.path.join(seq, "dense", "rgb")
                depth_path = os.path.join(seq, "dense", 'depth')
                caminfo_path = os.path.join(seq,'dense','cam')
                outlier_mask_path = os.path.join(seq, 'dense', 'outlier_mask')
                sky_mask_path = os.path.join(seq, 'dense', 'sky_mask')
                num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                if num_frames < 30:
                    print('skipping %s, too few images' % (seq))
                    continue

                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, 'frame_*.png'))))
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, 'frame_*.npy'))))
                self.all_outlier_mask_paths.extend(sorted(glob.glob(os.path.join(outlier_mask_path, 'frame_*.png'))))
                self.all_sky_mask_paths.extend(sorted(glob.glob(os.path.join(sky_mask_path, 'frame_*.png'))))
                seq_annotaions_path = sorted(glob.glob(os.path.join(caminfo_path, 'frame_*.npz')))
                self.all_annotation_paths.extend(seq_annotaions_path)

                N = len(self.full_idxs)

                assert len(self.all_rgb_paths) == N and \
                       len(self.all_outlier_mask_paths) == N and \
                       len(self.all_sky_mask_paths) == N and \
                       len(self.all_annotation_paths) == N and \
                       len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."

                extrinsics_seq = []
                #load intrinsics and extrinsics
                for anno in seq_annotaions_path:
                    camera_info = np.load(anno)
                    pose = np.array(camera_info['pose'], dtype=np.float32)
                    intrinsics = np.array(camera_info['intrinsic'], dtype=np.float32)
                    self.all_extrinsic.extend([pose])
                    self.all_intrinsic.extend([intrinsics])
                    extrinsics_seq.append(pose)

                all_extrinsic_numpy = np.array(extrinsics_seq)
                #compute ranking
                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind]

            os.makedirs(f'annotations/dl3dv_annotations/{dset}', exist_ok=True)
            self._save_paths_to_json(self.all_rgb_paths, f'annotations/dl3dv_annotations/{dset}/rgb_paths.json')
            self._save_paths_to_json(self.all_depth_paths, f'annotations/dl3dv_annotations/{dset}/depth_paths.json')
            self._save_paths_to_json(self.all_outlier_mask_paths, f'annotations/dl3dv_annotations/{dset}/outlier_mask_paths.json')
            self._save_paths_to_json(self.all_sky_mask_paths, f'annotations/dl3dv_annotations/{dset}/sky_mask_paths.json')
            joblib.dump(self.rank, f'annotations/dl3dv_annotations/{dset}/rankings.joblib')
            joblib.dump(self.all_extrinsic, f'annotations/dl3dv_annotations/{dset}/extrinsics.joblib')
            joblib.dump(self.all_intrinsic, f'annotations/dl3dv_annotations/{dset}/intrinsics.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
        
    
    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def __len__(self):
        return len(self.full_idxs)
    
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        if W != 480 and H != 270:
            cx, cy = W//2, H//2
        else:
            cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            # resolution = resolution[::-1]
            pass
            
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                # resolution = resolution[::-1]
                pass

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2
    
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
        sky_mask_paths = [self.all_sky_mask_paths[i] for i in full_idx]
        outlier_mask_paths = [self.all_outlier_mask_paths[i] for i in full_idx]

        views = []
        for impath, depthpath, camera_pose, intrinsics, sky_mask_path, outlier_mask_path in zip(
                rgb_paths, depth_paths, camera_pose_list, intrinsics_list, sky_mask_paths, outlier_mask_paths):

            # Load and preprocess images
            rgb_image = Image.open(impath).convert("RGB")
            depthmap = np.load(depthpath)

            # Apply sky and outlier masks
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE) == 0
            outlier_mask = cv2.imread(outlier_mask_path, cv2.IMREAD_GRAYSCALE) == 0
            depthmap[sky_mask & outlier_mask] = depthmap[sky_mask & outlier_mask]  # Keep valid values
            depthmap[~(sky_mask & outlier_mask)] = 0  # Set invalid values to 0

            depthmap = threshold_depth_map(depthmap, max_percentile=98, min_percentile=-1)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng, info=impath)

            # Create view dictionary
            views.append({
                'img': rgb_image,
                'ori_image_path': impath,
                'ori_depth_path': depthpath,
                'ori_camera_pose': camera_pose,
                'ori_intrinsics': intrinsics,
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
            'ori_image_path': ('ori_image_path', lambda x: x),  # Keep as list
            'ori_depth_path': ('ori_depth_path', lambda x: x),  # Keep as list
            'ori_camera_pose': ('ori_camera_pose', np.array),
            'ori_intrinsics': ('ori_intrinsics', np.array),
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

    dataset = Dl3dv(
        dataset_location="/mnt/disk3.8-4/datasets/dl3dv",
        dset='7K',
        use_cache=True,
        use_augs=use_augs,
        top_k=50,
        quick=False,
        verbose=True,
        resolution=(512, 224),
        seed=777,
        load_mask=False,
        aug_crop=16,
        z_far=200)

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
        viz.save_glb(f'dl3dv_{dataset.dset}_views_{num_views}.glb')
        return

    dataset[(101, 0, 16)]
    # visualize_scene((100, 0, num_views))
    print('dataset loaded')
