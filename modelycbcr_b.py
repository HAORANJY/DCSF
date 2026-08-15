import os
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone1 import get_backbone
from .attention import get_attention
from .aggregation import get_aggregation


# RGB转YCbCr颜色空间（PyTorch实现，适配0~1输入，符合JPEG标准）
class RGB2YCbCr(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB转YCbCr的标准转换矩阵（BT.601，JPEG/视频编码标准）
        self.transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],  # Y分量（亮度）
            [-0.1687, -0.3313, 0.5],  # Cb分量（蓝色差）
            [0.5, -0.4187, -0.0813]  # Cr分量（红色差）
        ]).float()
        # Cb/Cr分量的偏移量（将[-0.5, 0.5]映射到[0, 1]）
        self.offset = torch.tensor([0.0, 0.5, 0.5]).float().reshape(1, 3, 1, 1)

    def forward(self, x):
        """
        输入: x [B, 3, H, W] RGB图像，取值范围0~1
        输出: ycbcr [B, 3, H, W] YCbCr图像，各通道范围：
             Y(Luminance): 0~1（亮度）
             Cb(Chrominance Blue): 0~1（蓝色差）
             Cr(Chrominance Red): 0~1（红色差）
        严格遵循JPEG标准转换公式，计算效率远高于Lab/HSV
        """
        b, c, h, w = x.shape
        device = x.device

        # 转换矩阵移到对应设备
        transform = self.transform_matrix.to(device)
        offset = self.offset.to(device)

        # RGB转YCbCr（矩阵乘法实现，高效）
        # x形状调整: [B, 3, H, W] → [B, H, W, 3]
        x_perm = x.permute(0, 2, 3, 1)
        ycbcr = torch.matmul(x_perm, transform.t())  # [B, H, W, 3]
        ycbcr = ycbcr.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # 偏移校正，将Cb/Cr从[-0.5, 0.5]映射到[0, 1]
        ycbcr = ycbcr + offset

        # 裁剪到0~1范围，保证数值稳定性
        ycbcr = torch.clamp(ycbcr, 0.0, 1.0)

        return ycbcr


# YCbCr颜色空间处理分支（无需通道适配，3通道直接输入LRFR）
class YCbCrBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rgb2ycbcr = RGB2YCbCr()  # RGB转YCbCr（3通道输出）
        self.lrfr = LRFR(config)  # 复用LRFR模块

    def forward(self, x):
        # Step 1: RGB转YCbCr（3通道，符合JPEG标准，无增强）
        ycbcr = self.rgb2ycbcr(x)

        # Step 2: 输入LRFR模块提取特征（直接输入，无需额外处理）
        ycbcr_feat = self.lrfr(ycbcr)

        return ycbcr_feat


# 双分支特征融合模块（可学习权重）
class FeatureFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # 可学习的分支权重（通过softmax归一化，保证权重和为1）
        self.rgb_weight = nn.Parameter(torch.ones(1))
        self.ycbcr_weight = nn.Parameter(torch.ones(1))

    def forward(self, rgb_feat, ycbcr_feat):
        # 权重归一化
        weights = F.softmax(torch.cat([self.rgb_weight, self.ycbcr_weight]), dim=0)
        # 加权融合特征
        fused_feat = weights[0] * rgb_feat + weights[1] * ycbcr_feat
        # 保持特征L2归一化（与原模型输出格式完全一致）
        return F.normalize(fused_feat, p=2, dim=1)


class LRFR(nn.Module):
    def __init__(self, config):
        super(LRFR, self).__init__()
        self.backbone = get_backbone(backbone=config.backbone)

        # 动态获取骨干网络的输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.img_size, config.img_size)
            backbone_output = self.backbone(dummy)
            in_channels = backbone_output.shape[1]

        # 添加通道数转换模块，将骨干网络的输出通道数转换为config.num_channels
        self.channel_adapter = nn.Conv2d(in_channels, config.num_channels, kernel_size=1, stride=1, padding=0)
        self.attention = get_attention(attention=config.attention, channel=config.num_channels,
                                       spatial=((config.img_size // 16) * (config.img_size // 16)))
        self.aggregation = get_aggregation(aggregation=config.aggregation, num_channels=config.num_channels,
                                           num_clusters=config.num_clusters, cluster_dim=config.cluster_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.channel_adapter(x)
        x = self.attention(x)
        x = self.aggregation(x)
        return F.normalize(x.flatten(1), p=2, dim=1)

savepath = r'/data3/limian1/Documents/LPN-main/data/University-Release/relitu/ycbcr'
os.makedirs(savepath, exist_ok=True)

def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))

class GeoModel(nn.Module):
    def __init__(self, config):
        super(GeoModel, self).__init__()
        self.config = config
        # 原始RGB分支（保持完全不变）
        self.rgb_branch = LRFR(config)
        # 新增YCbCr分支（替换原HSV分支）
        self.ycbcr_branch = YCbCrBranch(config)

        # 自动推导特征维度（适配不同backbone，无需手动指定）
        with torch.no_grad():
            dummy = torch.randn(1, 3, config.img_size, config.img_size)
            feat_dim = self.rgb_branch(dummy).shape[1]
        self.feature_fusion = FeatureFusion(feat_dim)

    def forward(self, img1, img2=None):
        if img2 is not None:
            # RGB分支特征提取
            rgb_feat1 = self.rgb_branch(img1)
            rgb_feat2 = self.rgb_branch(img2)

            # YCbCr分支特征提取
            ycbcr_feat1 = self.ycbcr_branch(img1)
            ycbcr_feat2 = self.ycbcr_branch(img2)

            # 特征融合
            fused_feat1 = self.feature_fusion(rgb_feat1, ycbcr_feat1)
            fused_feat2 = self.feature_fusion(rgb_feat2, ycbcr_feat2)

            # 生成融合特征热力图
            # 将1D特征向量reshape为2D热力图: [B, D] -> [1, 1, num_clusters, cluster_dim]
            nc, cd = self.config.num_clusters, self.config.cluster_dim
            feat1_map = fused_feat1[0].reshape(1, 1, nc, cd).detach().cpu().numpy()
            draw_features(1, 1, feat1_map, "{}/fused_feat1".format(savepath))
            feat2_map = fused_feat2[0].reshape(1, 1, nc, cd).detach().cpu().numpy()
            draw_features(1, 1, feat2_map, "{}/fused_feat2".format(savepath))

            return fused_feat1, fused_feat2
        else:
            # 单图像输入
            rgb_feat = self.rgb_branch(img1)
            ycbcr_feat = self.ycbcr_branch(img1)
            fused_feat = self.feature_fusion(rgb_feat, ycbcr_feat)



            return fused_feat