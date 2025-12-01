# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
from omnivggt.utils.image import ImgNorm
from omnivggt.datasets.utils.augmentation import get_image_augmentation

ColorJitter = get_image_augmentation()