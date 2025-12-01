from .arkitscenes_high import ARKitScenesHigh
from .bedlam import Bedlam
from .blendedmvs import BlendedMVS
from .co3d import Co3d
from .dl3dv import Dl3dv
from .dynamic_replica import Dynamic_Replica
from .hypersim import Hypersim
from .kubric import Kubric
from .mapfree import MapFree
from .megadepth import MegaDepth
from .mp3d import Mp3d
from .mvs_synth import Mvs_Synth
from .scannet import Scannet
from .scannetppv2 import Scannetppv2
from .spring import Spring
from .tartanair import TarTanAirDUSt3R
from .uasol import Uasol
from .unreal4k import Unreal4k
from .vkitti import Vkitti
from .waymo import Waymo
from .wildrgb import Wildrgb

from omnivggt.datasets.utils.transforms import ImgNorm, ColorJitter

def get_data_loader(dataset, batch_size, num_workers=8,
                    shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from omnivggt.datasets.utils.misc import get_world_size, get_rank
    
    world_size = get_world_size()
    rank = get_rank()
    if isinstance(dataset, str):
        dataset = eval(dataset)
    
    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
            
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1, #Do not modify this
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
        drop_last=drop_last,
        )   
    
    return data_loader
