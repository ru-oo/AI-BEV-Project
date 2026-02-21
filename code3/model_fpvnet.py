"""
model_fpvnet.py — FPVNet (Front-Projection-Voxel Network)
==========================================================
LSS와 완전히 다른 접근:
  LSS  : 학습된 깊이 분포(D bins) → frustum feature volume → voxel pooling(splat)
  FPVNet: 명시적 깊이 예측 → 기하학적 3D 투영 → 복셀 그리드 → 3D CNN 정제

구조:
  EfficientNet-B2(ImageNet pretrained)
    └─ FPN Neck (P3, P4, P5 multi-scale)
         ├─ Depth Head  → 2D metric depth map (H/2 × W/2)
         ├─ Sem Head    → 2D semantic map     (H/2 × W/2)
         └─ Geometric Projection → 3D Voxel Grid
              └─ 3D Refine CNN → 최종 예측 (C, nZ, nX, nY)

목표: 5클래스 mIoU ≥ 50%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────
# EfficientNet-B2 백본 (ImageNet pretrained)
# ─────────────────────────────────────────────────────
def _make_backbone():
    try:
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        # features[0..8] : stem + MBConv stages
        # out channels: 24, 48, 120, 208, 1408
        return net.features, [24, 48, 120, 208, 1408]
    except Exception:
        from torchvision.models import resnet50
        # fallback: ResNet-50
        net = resnet50(pretrained=True)
        return net, [256, 512, 1024, 2048]


# ─────────────────────────────────────────────────────
# FPN Neck
# ─────────────────────────────────────────────────────
class FPN(nn.Module):
    """3-scale Feature Pyramid Network (P3, P4, P5)"""

    def __init__(self, in_channels, out_ch=128):
        super().__init__()
        c3, c4, c5 = in_channels[-3], in_channels[-2], in_channels[-1]

        self.lat5 = nn.Conv2d(c5, out_ch, 1)
        self.lat4 = nn.Conv2d(c4, out_ch, 1)
        self.lat3 = nn.Conv2d(c3, out_ch, 1)

        self.out5 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.out4 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.out3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, scale_factor=2,
                                             mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2,
                                             mode='nearest')
        return self.out3(p3), self.out4(p4), self.out5(p5)


# ─────────────────────────────────────────────────────
# ASPP Head (깊이 / 의미 공통)
# ─────────────────────────────────────────────────────
class ASPPHead(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch//2, 3, padding=r,
                          dilation=r, bias=False),
                nn.BatchNorm2d(in_ch//2), nn.ReLU(inplace=True))
            for r in rates
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch//2, 1, bias=False),
            nn.ReLU(inplace=True))
        fused = in_ch // 2 * (len(rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(fused, in_ch//2, 1, bias=False),
            nn.BatchNorm2d(in_ch//2), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, out_ch, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        branches = [b(x) for b in self.branches]
        pool_out = F.interpolate(self.pool(x), size=(h, w), mode='bilinear',
                                 align_corners=False)
        return self.project(torch.cat(branches + [pool_out], dim=1))


# ─────────────────────────────────────────────────────
# 3D 정제 CNN (3D-UNet lightweight)
# ─────────────────────────────────────────────────────
class Refine3D(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        mid = max(in_ch // 2, 16)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=True),
            nn.Conv3d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=True),
            nn.Conv3d(mid, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────
# FPVNet 메인 모델
# ─────────────────────────────────────────────────────
class FPVNet(nn.Module):
    """
    FPVNet — LSS가 아닌 기하학적 투영 기반 3D Semantic Occupancy

    Parameters
    ----------
    xbound : [xmin, xmax, xstep]   BEV x 범위 [m]
    ybound : [ymin, ymax, ystep]   BEV y 범위 [m]
    zbound : [zmin, zmax, zstep]   높이 범위   [m]
    dbound : [dmin, dmax]          유효 깊이 범위 [m]
    num_classes : 의미 클래스 수
    fpn_ch : FPN 채널 수
    """

    def __init__(self,
                 xbound=(-50, 50, 0.5),
                 ybound=(-50, 50, 0.5),
                 zbound=(-2.0, 6.0, 1.0),
                 dbound=(1.0, 50.0),
                 num_classes=5,
                 fpn_ch=128):
        super().__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.num_classes = num_classes

        # 복셀 그리드 크기 계산
        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])

        # ── EfficientNet-B2 백본 ──
        self._build_backbone(fpn_ch)

        # ── 깊이 헤드 (metric depth) ──
        # softplus 활성화로 항상 양수 깊이
        self.depth_head = ASPPHead(fpn_ch, 1)

        # ── 의미 헤드 (2D semantic) ──
        self.sem_head = ASPPHead(fpn_ch, num_classes)

        # ── 3D 복셀 정제 ──
        self.refine3d = Refine3D(num_classes, num_classes)

        print(f"[FPVNet] 복셀 그리드: {self.nZ}×{self.nX}×{self.nY}")
        print(f"[FPVNet] 클래스: {num_classes}, FPN ch: {fpn_ch}")

    def _build_backbone(self, fpn_ch):
        """EfficientNet-B2 → stage별 feature 추출"""
        try:
            from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
            net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            feats = net.features

            # Stage별로 분리
            # EfficientNet-B2 stages: 0=stem, 1-7=MBConv blocks
            self.stage3 = feats[:4]    # -> ch=48
            self.stage4 = feats[4:6]   # -> ch=120
            self.stage5 = feats[6:]    # -> ch=1408

            self.fpn = FPN([48, 120, 1408], fpn_ch)
            print("[FPVNet] 백본: EfficientNet-B2 (9.1M)")
            self._backbone_type = 'efficientnet'

        except Exception as e:
            print(f"[FPVNet] EfficientNet 로드 실패 ({e}), ResNet-50 사용")
            from torchvision.models import resnet50
            net = resnet50(pretrained=True)
            self.stage3 = nn.Sequential(net.conv1, net.bn1, net.relu,
                                        net.maxpool, net.layer1)  # 256
            self.stage4 = net.layer2  # 512
            self.stage5 = nn.Sequential(net.layer3, net.layer4)   # 2048
            self.fpn = FPN([256, 512, 2048], fpn_ch)
            print("[FPVNet] 백본: ResNet-50 (25.5M, fallback)")
            self._backbone_type = 'resnet'

    # ── 특징 추출 ──────────────────────────────
    def _extract_features(self, img):
        c3 = self.stage3(img)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        return p3  # 가장 높은 해상도 사용

    # ── 기하학적 3D 투영 (핵심 — LSS 아님) ───
    def _project_to_voxel(self,
                           depth: torch.Tensor,
                           sem_logits: torch.Tensor,
                           K: torch.Tensor) -> torch.Tensor:
        """
        depth      : (B, 1, H, W)  — metric depth [m]
        sem_logits : (B, C, H, W)  — 2D semantic logits
        K          : (B, 3, 3)     — camera intrinsic matrix

        Returns
        -------
        voxel : (B, C, nZ, nX, nY)
        """
        B, C, H, W = sem_logits.shape
        device = depth.device
        dmin, dmax = self.dbound

        # 픽셀 좌표 그리드 생성
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H, W)
        uu = uu.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        vv = vv.unsqueeze(0).expand(B, -1, -1)

        # 카메라 내부 파라미터
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        d = depth[:, 0]  # (B, H, W)
        # 깊이 클램프
        d = d.clamp(dmin, dmax)

        # 역투영: 픽셀 → 카메라 좌표
        x_cam = (uu - cx) * d / fx   # (B, H, W)
        y_cam = (vv - cy) * d / fy
        z_cam = d                     # depth = z in camera frame

        # 복셀 인덱스로 변환
        xmin, xmax, xstep = self.xbound
        ymin, ymax, ystep = self.ybound
        zmin, zmax, zstep = self.zbound

        xi = ((x_cam - xmin) / xstep).long()
        yi = ((z_cam - ymin) / ystep).long()   # z_cam → BEV y축
        zi = ((y_cam - zmin) / zstep).long()   # y_cam → voxel z (높이)

        valid = (xi >= 0) & (xi < self.nX) & \
                (yi >= 0) & (yi < self.nY) & \
                (zi >= 0) & (zi < self.nZ)

        # 의미 확률 계산
        sem_prob = sem_logits.softmax(dim=1)  # (B, C, H, W)

        voxel = torch.zeros(B, C, self.nZ, self.nX, self.nY,
                            device=device)

        # 배치별 산포(scatter)
        for b in range(B):
            v_mask = valid[b].reshape(-1)         # (H*W,)
            xi_b   = xi[b].reshape(-1)[v_mask]    # (N_valid,)
            yi_b   = yi[b].reshape(-1)[v_mask]
            zi_b   = zi[b].reshape(-1)[v_mask]
            prob_b = sem_prob[b].reshape(C, -1)[:, v_mask]  # (C, N_valid)

            # 선형 인덱스
            flat_idx = zi_b * self.nX * self.nY + xi_b * self.nY + yi_b
            flat_idx = flat_idx.clamp(0, self.nZ * self.nX * self.nY - 1)

            # scatter_add (채널별)
            voxel_flat = voxel[b].reshape(C, -1)
            voxel_flat.scatter_add_(1,
                                    flat_idx.unsqueeze(0).expand(C, -1),
                                    prob_b)

        return voxel

    # ── Forward ───────────────────────────────
    def forward(self,
                img: torch.Tensor,
                K:   torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        img : (B, 3, H, W)       카메라 이미지
        K   : (B, 3, 3)          카메라 내부 파라미터

        Returns
        -------
        voxel_logits : (B, C, nZ, nX, nY)   3D 의미 점유 (loss용)
        depth        : (B, 1, H/2, W/2)      예측 깊이    (depth loss용)
        sem2d        : (B, C, H/2, W/2)      2D 의미      (2D loss용)
        """
        # 이미지 절반 크기로 다운샘플 (메모리 절약)
        H2, W2 = img.shape[2] // 2, img.shape[3] // 2
        img_small = F.interpolate(img, size=(H2, W2),
                                  mode='bilinear', align_corners=False)

        feat = self._extract_features(img_small)  # (B, fpn_ch, H', W')

        # 2D 예측 (깊이·의미)
        depth  = F.softplus(self.depth_head(feat)) + self.dbound[0]
        sem2d  = self.sem_head(feat)

        # 깊이를 입력 절반 크기에 맞춤
        depth_up = F.interpolate(depth, size=(H2, W2),
                                 mode='bilinear', align_corners=False)
        sem_up   = F.interpolate(sem2d, size=(H2, W2),
                                 mode='bilinear', align_corners=False)

        # K 축소 보정 (이미지가 절반 크기이므로)
        K_scaled = K.clone().float()
        K_scaled[:, 0, :] /= (img.shape[3] / W2)
        K_scaled[:, 1, :] /= (img.shape[2] / H2)

        # 기하학적 3D 투영
        voxel = self._project_to_voxel(depth_up, sem_up, K_scaled)

        # 3D 정제
        voxel_logits = self.refine3d(voxel)

        return voxel_logits, depth_up, sem2d


# ─────────────────────────────────────────────────────
# 파라미터 수 확인
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FPVNet(
        xbound=(-25, 25, 0.5),
        ybound=(-25, 25, 0.5),
        zbound=(-2.0, 6.0, 1.0),
        num_classes=5,
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n총 파라미터: {total:.2f}M")

    dummy_img = torch.randn(1, 3, 224, 400)
    dummy_K   = torch.eye(3).unsqueeze(0)
    dummy_K[0, 0, 0] = 500; dummy_K[0, 1, 1] = 500
    dummy_K[0, 0, 2] = 200; dummy_K[0, 1, 2] = 112

    with torch.no_grad():
        v, d, s = model(dummy_img, dummy_K)
    print(f"voxel: {v.shape}  depth: {d.shape}  sem2d: {s.shape}")
