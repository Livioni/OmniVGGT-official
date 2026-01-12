# --------------------------------------------------------
# utilitary functions for DUSt3R
# --------------------------------------------------------
import torch
import cv2
import numpy as np
from omnivggt.utils.vo_eval import save_trajectory_tum_format
from PIL import Image

def get_stride_distribution(strides, dist_type='uniform'):

    # input strides sorted by descreasing order by default
    
    if dist_type == 'uniform':
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == 'exponential':
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith('linear'): # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split('_')[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f'Invalid linear distribution format: {dist_type}')
    else:
        raise ValueError('Unknown distribution type %s' % dist_type)

    # normalize to sum to 1
    return dist / np.sum(dist)


def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1['instance']
    y = gt2['instance']
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    """ flip so that tensor[0::2] <=> tensor[1::2] """
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

def save_tum_poses(traj, path):
    # traj = self.get_tum_poses()
    save_trajectory_tum_format(traj, path)
    return traj[0] # return the poses

def save_focals(focals, path):
    # convert focal to txt
    # focals = self.get_focals()
    np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
    return focals

def save_intrinsics(K_raw, path):
    # K_raw = self.get_intrinsics()
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
    return K_raw

def save_conf_maps(conf, path):
    # conf = self.get_conf()
    for i, c in enumerate(conf):
        np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
    return conf

def save_rgb_imgs(imgs, path):
    # imgs = self.imgs
    for i, img in enumerate(imgs):
        # convert from rgb to bgr
        img = img[..., ::-1]
        cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
    return imgs

def save_dynamic_masks(dynamic_masks, path):
    # dynamic_masks = self.dynamic_masks
    for i, dynamic_mask in enumerate(dynamic_masks):
        cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))
    return dynamic_masks

def save_depth_maps(depth_maps, path):
    images = []
    for i, depth_map in enumerate(depth_maps):
        depth_map_colored = cv2.applyColorMap((depth_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
        img_path = f'{path}/frame_{(i):04d}.png'
        cv2.imwrite(img_path, depth_map_colored)
        images.append(Image.open(img_path))
        # Save npy file
        np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
    
    # Save gif using Pillow
    images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    return depth_maps

def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, list):
        return [to_cpu(xx) for xx in x]


def select_first_batch(inputs, weight_dtype=None):
    """
    Remove batch dimension from all tensors in the inputs dictionary and optionally change data type.
    
    Args:
        inputs (dict): Dictionary containing tensors with batch dimension.
        keys (list, optional): Keys to process. If None, processes all tensors.
        weight_dtype (torch.dtype, optional): If specified, converts all tensors to this data type.
    
    Returns:
        dict: Dictionary with batch dimension removed and data type converted.
    """
    new_dicts = {}
    keys = ["pose_enc", "depth", "world_points", "images", "extrinsic", "intrinsic", "world_points_from_depth", 'depth_conf', 'world_points_conf']
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and (keys is None or key in keys):
            try:
                value = value[:1, ...].detach()
            except:
                value = value[:1, ...]
            
            if weight_dtype:
                value = value.to(weight_dtype)
        elif isinstance(value, list) and key == "pose_enc":
            value = value[-1]
            value = value[:1, ...].detach()
        
        new_dicts[key] = value
    
    return new_dicts


def save_images_from_tensor(tensor, save_dir="frames", prefix="frame"):
    """
    Save tensor frames as images.

    Args:
        tensor (torch.Tensor): Image tensor with shape (1, 4, H, W, 3), values in [0, 1] or [0, 255].
        save_dir (str): Directory to save images.
        prefix (str): Filename prefix.
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    tensor = tensor.squeeze(0)

    for i, img_tensor in enumerate(tensor):
        img_array = img_tensor.permute(1,2,0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        img.save(f"{save_dir}/{prefix}_{i}.png")
        print(f"Saved: {save_dir}/{prefix}_{i}.png")
