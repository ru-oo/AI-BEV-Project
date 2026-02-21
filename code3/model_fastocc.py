"""
model_fastocc.py — FastOcc 6-Camera Surround (v3)
==================================================

★ 6카메라 서라운드뷰 기반 3D Semantic Occupancy ★

LSS와의 근본적 차이:
  LSS    : 카메라별 학습된 깊이 분포(D bins) → frustum 피처 → voxel pooling(splat)
  FastOcc: 복셀 중심 → 각 카메라 기하학적 투영 → grid_sample → 다중카메라 집계
           D-bin 깊이 분포 없음, frustum pooling 없음

6카메라 처리 흐름:
  (B, 6, 3, H, W)  ← 전방/전방좌/전방우/후방/후방좌/후방우
        │
  [공유 EfficientNet-B2 + FPN]  — 6장 동시 인코딩
        │ (B, 6, C, H/8, W/8)
        │
  [Multi-Cam Voxel Query Sampler]  ★ 핵심 — LSS 아님 ★
    복셀 중심(x,y,z) → ego→cam 변환 → K 투영
    → 유효 카메라 grid_sample → 카메라간 평균 집계
        │ (B, C, nZ, nX, nY)
        │
  [Channel-to-Height (C2H) Refiner]  — 2D conv으로 3D 표현
        │ (B, c2h_ch, nZ, nX, nY)
        │
  [3D Classifier]
        │
  (B, num_classes, nZ, nX, nY)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# EfficientNet-B2 + FPN 백본
# ══════════════════════════════════════════════════════════════
class EfficientFPN(nn.Module):
    """EfficientNet-B2 + 3-scale FPN → P3 (H/8, W/8, out_ch)"""

    def __init__(self, out_ch=128, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        w = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        f = efficientnet_b2(weights=w).features

        # EfficientNet-B2 stage 출력 채널 (실측):
        # features[:4]  → 48ch  @ H/8
        # features[4:6] → 120ch @ H/16
        # features[6:]  → 1408ch@ H/32
        self.s3 = f[:4]    # 48ch
        self.s4 = f[4:6]   # 120ch
        self.s5 = f[6:]    # 1408ch

        self.lat5 = nn.Sequential(
            nn.Conv2d(1408, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.lat4 = nn.Sequential(
            nn.Conv2d(120,  out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.lat3 = nn.Sequential(
            nn.Conv2d(48,   out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.out3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))

        print(f"[FastOcc] 백본: EfficientNet-B2 + FPN ({out_ch}ch)")

    def forward(self, x):
        # x: (B*N, 3, H, W) — 6카메라를 배치로 처리
        c3 = self.s3(x)   # 48ch,  H/8
        c4 = self.s4(c3)  # 120ch, H/16
        c5 = self.s5(c4)  # 1408ch,H/32
        p5 = self.lat5(c5)
        # 홀수 해상도 불일치 방지 → size= 사용
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        return self.out3(p3)   # (B*N, out_ch, H/8, W/8)


# ══════════════════════════════════════════════════════════════
# Multi-Camera Voxel Query Sampler (핵심 — LSS 아님)
# ══════════════════════════════════════════════════════════════
class MultiCamVoxelSampler(nn.Module):
    """
    6카메라 서라운드뷰에서 3D 복셀 피처 추출

    LSS와의 차이:
      LSS  : 픽셀마다 D개 깊이 빈 확률 → outer product → frustum pool
      FastOcc: 복셀 중심(x,y,z)을 각 카메라에 투영 → bilinear_grid_sample
               → 유효 카메라 평균 (깊이 분포 학습 없음)

    ego 좌표계:
      x = 오른쪽, y = 전방(앞), z = 위

    Parameters
    ----------
    xbound, ybound, zbound : (min, max, step)
    feat_h, feat_w : 피처맵 크기 (H/8, W/8)
    """

    def __init__(self, xbound, ybound, zbound, feat_h, feat_w):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w

        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])

        # 복셀 중심 좌표 (ego 프레임): (nZ, nX, nY, 3) = [x, y, z]
        xs = torch.linspace(xbound[0] + xbound[2]/2, xbound[1] - xbound[2]/2, self.nX)
        ys = torch.linspace(ybound[0] + ybound[2]/2, ybound[1] - ybound[2]/2, self.nY)
        zs = torch.linspace(zbound[0] + zbound[2]/2, zbound[1] - zbound[2]/2, self.nZ)

        gz, gx, gy = torch.meshgrid(zs, xs, ys, indexing='ij')
        # (nZ, nX, nY, 4) homogeneous  [x, y, z, 1]
        ones = torch.ones_like(gz)
        vox_h = torch.stack([gx, gy, gz, ones], dim=-1)
        self.register_buffer('vox_centers_h', vox_h)  # (nZ, nX, nY, 4)

    def forward(self, feats: torch.Tensor,
                Ks: torch.Tensor,
                sensor2ego: torch.Tensor) -> torch.Tensor:
        """
        feats      : (B, N, C, Hf, Wf)  — N=6 카메라 피처맵
        Ks         : (B, N, 3, 3)        — 각 카메라 내부 파라미터
        sensor2ego : (B, N, 4, 4)        — cam→ego 변환행렬

        Returns
        -------
        voxel : (B, C, nZ, nX, nY)
        """
        B, N, C, Hf, Wf = feats.shape
        device = feats.device
        nZ, nX, nY = self.nZ, self.nX, self.nY

        # 복셀 중심: (nZ*nX*nY, 4) homogeneous
        centers = self.vox_centers_h.to(device)          # (nZ, nX, nY, 4)
        pts_h   = centers.reshape(-1, 4)                 # (M, 4)  M=nZ*nX*nY

        # ego→cam 역변환: sensor2ego는 cam→ego
        # ego→cam = inv(sensor2ego)
        # (B, N, 4, 4)
        s2e = sensor2ego                                  # (B, N, 4, 4)
        R   = s2e[:, :, :3, :3]   # (B, N, 3, 3)  cam rotation
        t   = s2e[:, :, :3,  3]   # (B, N, 3)     cam translation
        # ego→cam: R^T @ (pt - t)
        Rt  = R.transpose(2, 3)   # (B, N, 3, 3)

        # pts_ego: (M, 3)
        pts_ego = pts_h[:, :3]    # (M, 3)

        # → (B, N, M, 3)
        pts_ego_bn = pts_ego.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        t_bn       = t.unsqueeze(2)                      # (B, N, 1, 3)
        pts_cam    = torch.matmul(pts_ego_bn - t_bn, Rt) # (B, N, M, 3)
        # (X_cam, Y_cam, Z_cam): Z_cam = 깊이 (전방)

        # K 투영: (B, N, 3, 3) × (B, N, M, 3)^T
        Xc = pts_cam[..., 0]   # (B, N, M)
        Yc = pts_cam[..., 1]
        Zc = pts_cam[..., 2]   # depth

        fx = Ks[:, :, 0, 0].unsqueeze(-1)   # (B, N, 1)
        fy = Ks[:, :, 1, 1].unsqueeze(-1)
        cx = Ks[:, :, 0, 2].unsqueeze(-1)
        cy = Ks[:, :, 1, 2].unsqueeze(-1)

        # 양수 깊이만 유효
        eps   = 1e-4
        valid = Zc > eps                                  # (B, N, M)

        u = Xc * fx / Zc.clamp(min=eps) + cx             # (B, N, M) pixel u
        v = Yc * fy / Zc.clamp(min=eps) + cy             # (B, N, M) pixel v

        # 정규화 [-1, 1] (grid_sample 형식)
        u_n = (u / (self.feat_w - 1)) * 2 - 1            # (B, N, M)
        v_n = (v / (self.feat_h - 1)) * 2 - 1

        # 이미지 범위 내 유효 마스크
        valid = valid & (u_n >= -1) & (u_n <= 1) & (v_n >= -1) & (v_n <= 1)

        # grid_sample: feats (B,N,C,Hf,Wf) → sample at M points
        # feats를 (B*N, C, Hf, Wf)로 reshape
        feats_flat = feats.reshape(B * N, C, Hf, Wf)

        # grid: (B*N, 1, M, 2)
        grid = torch.stack([u_n, v_n], dim=-1)            # (B, N, M, 2)
        grid = grid.reshape(B * N, 1, -1, 2)              # (B*N, 1, M, 2)

        sampled = F.grid_sample(
            feats_flat, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)                            # (B*N, C, 1, M)
        sampled = sampled.reshape(B, N, C, -1)             # (B, N, C, M)

        # 유효하지 않은 위치 마스킹
        valid_mask = valid.float().unsqueeze(2)             # (B, N, 1, M)
        sampled    = sampled * valid_mask

        # 카메라간 집계: 유효 카메라 수로 나눠 평균
        n_valid = valid_mask.sum(dim=1).clamp(min=1)       # (B, 1, M)
        vox_feat = sampled.sum(dim=1) / n_valid            # (B, C, M)

        # (B, C, M) → (B, C, nZ, nX, nY)
        voxel = vox_feat.reshape(B, C, nZ, nX, nY)

        return voxel


# ══════════════════════════════════════════════════════════════
# Channel-to-Height (C2H) Refiner
# ══════════════════════════════════════════════════════════════
class C2HRefiner(nn.Module):
    """
    Channel-to-Height: 3D 연산 없이 2D conv으로 높이 차원 표현

    (B, C, nZ, nX, nY)
      → flatten: (B, C*nZ, nX, nY)
      → depthwise + pointwise 2D conv
      → reshape: (B, mid_ch, nZ, nX, nY)

    FlashOcc/FastBEV 에서 제안된 핵심 아이디어
    """

    def __init__(self, in_ch, nZ, mid_ch=64):
        super().__init__()
        flat_in  = in_ch  * nZ
        flat_out = mid_ch * nZ
        self.net = nn.Sequential(
            # Depthwise: 높이 슬라이스 내 공간 정보
            nn.Conv2d(flat_in, flat_in, 3, 1, 1,
                      groups=nZ, bias=False),
            nn.BatchNorm2d(flat_in), nn.ReLU(True),
            # Pointwise: 채널 간 믹싱 (높이 간 상호작용)
            nn.Conv2d(flat_in, flat_out, 1, bias=False),
            nn.BatchNorm2d(flat_out), nn.ReLU(True),
            nn.Conv2d(flat_out, flat_out, 3, 1, 1,
                      groups=nZ, bias=False),
            nn.BatchNorm2d(flat_out), nn.ReLU(True),
        )
        self.mid_ch = mid_ch
        self.nZ     = nZ

    def forward(self, x):
        # x: (B, C, nZ, nX, nY)
        B, C, nZ, nX, nY = x.shape
        flat = x.reshape(B, C * nZ, nX, nY)
        flat = self.net(flat)
        return flat.reshape(B, self.mid_ch, nZ, nX, nY)


# ══════════════════════════════════════════════════════════════
# FastOcc 메인 모델 (6-Camera Surround)
# ══════════════════════════════════════════════════════════════
class FastOcc(nn.Module):
    """
    FastOcc — 6카메라 서라운드뷰 3D Semantic Occupancy

    입력: 6방향 카메라 (전방·전방좌·전방우·후방·후방좌·후방우)
    출력: 3D Semantic Occupancy (nZ × nX × nY)

    vs LSS:
      - 카메라당 D-bin 깊이 확률 학습 없음
      - frustum voxel pooling(splat) 없음
      - 순수 기하학적 복셀 투영 + 다중카메라 평균
      - Channel-to-Height로 3D conv 최소화
    """

    def __init__(self,
                 xbound=(-50., 50., .5),
                 ybound=(-50., 50., .5),
                 zbound=(-2.,  6.,  .5),
                 num_classes=5,
                 fpn_ch=128,
                 c2h_ch=64,
                 img_h=256,
                 img_w=704,
                 num_cams=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_cams    = num_cams

        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])
        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])

        # EfficientNet-B2 P3 기준 피처맵 크기
        feat_h = img_h // 8
        feat_w = img_w // 8

        # ── 모듈 ──────────────────────────────────────
        self.backbone = EfficientFPN(out_ch=fpn_ch)

        self.sampler = MultiCamVoxelSampler(
            xbound, ybound, zbound,
            feat_h=feat_h, feat_w=feat_w)

        self.c2h = C2HRefiner(
            in_ch=fpn_ch, nZ=self.nZ, mid_ch=c2h_ch)

        self.classifier = nn.Sequential(
            nn.Conv3d(c2h_ch, c2h_ch, 3, 1, 1, bias=False),
            nn.BatchNorm3d(c2h_ch), nn.ReLU(True),
            nn.Conv3d(c2h_ch, num_classes, 1),
        )

        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[FastOcc] 6-Cam Surround | 복셀 {self.nZ}×{self.nX}×{self.nY}"
              f" | 파라미터 {total:.1f}M")

    def forward(self,
                imgs: torch.Tensor,
                Ks:   torch.Tensor,
                sensor2ego: torch.Tensor) -> torch.Tensor:
        """
        imgs       : (B, N, 3, H, W)      6카메라 이미지
        Ks         : (B, N, 3, 3)          각 카메라 내부 파라미터
        sensor2ego : (B, N, 4, 4)          cam→ego 변환행렬

        Returns
        -------
        logits : (B, num_classes, nZ, nX, nY)
        """
        B, N, _, H, W = imgs.shape

        # 1. 6카메라 동시 인코딩 (배치로 처리)
        imgs_flat = imgs.reshape(B * N, 3, H, W)
        feats_flat = self.backbone(imgs_flat)             # (B*N, C, H/8, W/8)
        _, C, Hf, Wf = feats_flat.shape
        feats = feats_flat.reshape(B, N, C, Hf, Wf)      # (B, N, C, Hf, Wf)

        # 2. 다중카메라 기하학적 복셀 샘플링 (LSS 아님)
        vox = self.sampler(feats, Ks, sensor2ego)         # (B, C, nZ, nX, nY)

        # 3. Channel-to-Height 정제 (2D conv만 사용)
        vox = self.c2h(vox)                               # (B, c2h_ch, nZ, nX, nY)

        # 4. 3D 분류
        return self.classifier(vox)                       # (B, nc, nZ, nX, nY)


# ══════════════════════════════════════════════════════════════
# 동작 확인
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    B, N = 1, 6
    model = FastOcc(
        xbound=(-50., 50., .5),
        ybound=(-50., 50., .5),
        zbound=(-2.,  6.,  .5),
        num_classes=5, fpn_ch=128, c2h_ch=64,
        img_h=256, img_w=704, num_cams=6)

    imgs = torch.randn(B, N, 3, 256, 704)
    Ks   = torch.eye(3).view(1,1,3,3).repeat(B, N, 1, 1)
    Ks[:,:,0,0] = 800; Ks[:,:,1,1] = 800
    Ks[:,:,0,2] = 352; Ks[:,:,1,2] = 128
    s2e  = torch.eye(4).view(1,1,4,4).repeat(B, N, 1, 1)

    with torch.no_grad():
        out = model(imgs, Ks, s2e)
    print(f"출력: {out.shape}")   # (1, 5, 16, 200, 200)
    print("FastOcc 6-Cam 동작 확인 ✅")
