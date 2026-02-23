"""
model_v2.py - LSS v2 모델
=========================
백본: EfficientNet-B0 (5.3M params) → MobileNetV3-Small (2.5M) fallback
개선: SE-Attention BEV Compressor, 분리된 Depth/Context Head
목적: 실시간 3D Semantic Occupancy 추론 (PC → STM32 CAN 자율주행)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    _BACKBONE = 'efficientnet_b0'
except ImportError:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    _BACKBONE = 'mobilenet_v3_small'

from splat import VoxelPooling


# ──────────────────────────────────────────────
# Squeeze-and-Excitation Block (채널 어텐션)
# ──────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


# ──────────────────────────────────────────────
# CamEncoder v2 (EfficientNet-B0 백본)
# ──────────────────────────────────────────────
class CamEncoderV2(nn.Module):
    """
    경량 카메라 인코더
    EfficientNet-B0: ImageNet pretrained, 1280ch @ 1/32
    출력: depth_probs (B, D, H, W)  context (B, C, H, W)
    """

    def __init__(self, D, C):
        super().__init__()
        self.D = D
        self.C = C

        if _BACKBONE == 'efficientnet_b0':
            net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = net.features          # (B, 1280, H/32, W/32)
            in_ch = 1280
            print("  백본: EfficientNet-B0 (5.3M params)")
        else:
            net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.backbone = net.features          # (B, 576, H/32, W/32)
            in_ch = 576
            print("  백본: MobileNetV3-Small (2.5M params)")

        MID = 256
        # Neck: 채널 축소 + SE 어텐션
        self.neck = nn.Sequential(
            nn.Conv2d(in_ch, MID, 1, bias=False),
            nn.BatchNorm2d(MID),
            nn.SiLU(inplace=True),
            SEBlock(MID, reduction=8),
            nn.Conv2d(MID, MID, 3, padding=1, bias=False),
            nn.BatchNorm2d(MID),
            nn.SiLU(inplace=True),
        )
        # 분리된 헤드 (공유 neck → 각자 학습)
        self.depth_head   = nn.Conv2d(MID, D, 1)
        self.context_head = nn.Conv2d(MID, C, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        depth_probs = self.depth_head(x).softmax(dim=1)
        context     = self.context_head(x)
        return depth_probs, context


# ──────────────────────────────────────────────
# BEV Compressor v2 (Residual + SE)
# ──────────────────────────────────────────────
class BEVCompressorV2(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.se    = SEBlock(in_ch, reduction=16)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block2(self.se(self.block1(x)))   # Residual


# ──────────────────────────────────────────────
# LSSModelV2
# ──────────────────────────────────────────────
class LSSModelV2(nn.Module):
    """
    LSS v2: EfficientNet-B0 + SE-BEV + 개선 Decoder
    동일한 NuScenesDataset, VoxelPooling 재사용
    """

    def __init__(self, xbound, ybound, zbound, dbound,
                 num_classes=4, C=64):
        super().__init__()
        self.C           = C
        self.dbound      = dbound
        self.xbound      = xbound
        self.ybound      = ybound
        self.zbound      = zbound
        self.num_classes = num_classes

        self.img_H,  self.img_W  = 384, 1056
        self.feat_H, self.feat_W = self.img_H // 32, self.img_W // 32

        self.D  = int((dbound[1] - dbound[0]) / dbound[2])
        self.nz = int((zbound[1] - zbound[0]) / zbound[2])
        nz_C    = self.nz * C                         # 4 × 64 = 256

        self.frustum      = self._make_frustum()
        self.cam_encoder  = CamEncoderV2(D=self.D, C=C)
        self.voxel_pool   = VoxelPooling(xbound, ybound, zbound, dbound)
        self.bev_compress = BEVCompressorV2(nz_C)

        # Decoder: nz_C → nz × num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(nz_C, nz_C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(nz_C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nz_C // 2, self.nz * num_classes, 1),
        )

    def _make_frustum(self):
        H, W = self.feat_H, self.feat_W
        ds = torch.arange(*self.dbound).view(-1, 1, 1).expand(-1, H, W)
        D  = ds.shape[0]
        xs = torch.linspace(0, self.img_W - 1, W).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, self.img_H - 1, H).view(1, H, 1).expand(D, H, W)
        return nn.Parameter(torch.stack((xs, ys, ds), -1), requires_grad=False)

    def _get_geom(self, rots, trans, intrinsics):
        B  = rots.shape[0]
        pts = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1).view(B, -1, 3)
        d   = pts[..., 2]
        pts[..., 0] = (pts[..., 0] - intrinsics[:, 0, 2:3]) * d / intrinsics[:, 0, 0:1]
        pts[..., 1] = (pts[..., 1] - intrinsics[:, 1, 2:3]) * d / intrinsics[:, 1, 1:2]
        pts = torch.bmm(rots, pts.permute(0, 2, 1)).permute(0, 2, 1) + trans.unsqueeze(1)
        return pts.view(B, self.D, self.feat_H, self.feat_W, 3)

    def forward(self, imgs, rots, trans, intrinsics):
        B, N, _, H, W = imgs.shape
        imgs       = imgs.view(B * N, 3, H, W)
        rots       = rots.view(B * N, 3, 3)
        trans      = trans.view(B * N, 3)
        intrinsics = intrinsics.view(B * N, 3, 3)

        depth, ctx = self.cam_encoder(imgs)
        geom       = self._get_geom(rots, trans, intrinsics)

        # Frustum lift: (B*N, D, C, H, W)
        frustum_feats = (ctx.unsqueeze(1) * depth.unsqueeze(2))

        geom         = geom.reshape(B, -1, self.feat_H, self.feat_W, 3)
        frustum_feats = frustum_feats.reshape(B, -1, self.C, self.feat_H, self.feat_W)

        bev = self.voxel_pool(geom, frustum_feats)   # (B, nz*C, nx, ny)
        bev = self.bev_compress(bev)
        out = self.decoder(bev)                       # (B, nz*ncls, nx, ny)
        return out.view(B, self.num_classes, self.nz,
                        out.shape[2], out.shape[3])
