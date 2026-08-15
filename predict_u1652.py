# -*- coding: utf-8 -*-
# @Author : Gan
# @Time : 2024/5/7 21:30

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 设置使用的GPU编号，可改为'1','2','3'等
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random

import os
import torch
import torchvision
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

import torch

from models.modelycbcr_b import GeoModel
from utils.trainer import evaluate
from Datasets.university1 import U1652DatasetEval, get_transforms


@dataclass
class Configuration:
    # Model
    backbone: str = 'ConvNeXt'
    attention: str = 'GSRA'
    aggregation: str = 'SALAD'

    num_channels: int = 1024
    img_size: int = 384
    num_clusters: int = 128
    cluster_dim: int = 64

    seed = 1
    verbose: bool = True

    # Eval
    batch_size_eval: int = 32
    eval_gallery_n: int = -1  # -1 for all or int
    normalize_features: bool = True

    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = r"/data3/limian1/Documents/LPN-main/data/University-Release"

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 0
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True
    # make cudnn deterministic/data3/limian1/Downloads/LRFR-main/weights/best_score.pth
    cudnn_deterministic: bool = False
    model_path = '/data3/limian1/Downloads/LRFR-main/result/U1652-D2S_ConvNeXt_GSRA_SALAD/08-02-13-06_lr-0.0001_loss-SemiTriplet/best_score.pth'


# -----------------------------------------------------------------------------#
# Config                                                                      #
# -----------------------------------------------------------------------------#

config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_test = '/data3/limian1/Documents/LPN-main/data/University-Release/test/query_drone'
    config.gallery_folder_test = '/data3/limian1/Documents/LPN-main/data/University-Release/test/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_test = '/data3/limian1/Documents/LPN-main/data/University-Release/test/query_satellite'
    config.gallery_folder_test = '/data3/limian1/Documents/LPN-main/data/University-Release/test/gallery_drone'

if __name__ == '__main__':
    val_transforms, _, _ = get_transforms((config.img_size, config.img_size))
    model = GeoModel(config)

    # 加载权重并自动处理DataParallel前缀
    state_dict = torch.load(config.model_path, map_location=config.device)
    # 如果权重带有 'module.' 前缀（来自DataParallel训练），则去除
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        print("检测到DataParallel前缀，已自动去除 'module.' 前缀")
    model.load_state_dict(state_dict)
    print(f"权重加载成功: {config.model_path}")

    model = model.to(config.device)
    model.eval()

    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test, mode="query",
                                          transforms=val_transforms)

    query_dataloader_test = DataLoader(query_dataset_test, batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers, shuffle=False, pin_memory=True)

    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test, mode="gallery",
                                            transforms=val_transforms, sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n, )

    gallery_dataloader_test = DataLoader(gallery_dataset_test, batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers, shuffle=False, pin_memory=True)

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
