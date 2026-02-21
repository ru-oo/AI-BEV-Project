"""
model_fastocc.py — FastOcc (Fast 3D Semantic Occupancy)
========================================================

LSS와의 근본적 차이:
  LSS    : 학습된 깊이 분포(D bins) → frustum 피처 볼륨 → voxel pooling(splat)
  FastOcc: 복셀 중심 → 카메라 기하학적 투영 → grid_sample + Channel-to-Height(C2H)

구조 (FlashOcc/FastBEV 개념 기반):
  EfficientNet-B2 → FPN Neck (P3, P4, P5)
      │
  [Voxel Query Sampler]  ← 복셀 중심을 이미지에 투영, bilinear sample
      │  기하학적 투영: world(x,y,z) → cam → K → image (u,v)
      │  ★ D-bin 깊이 분포 없음 — LSS와 완전히 다름
      │
  (B, C, nZ, nX, nY)  ← 3D feature volume
      │
  [Channel-to-Height (C2H) Refiner]  ← 채널↔높이 치환으로 2D 연산만 사용
      │  (B, C, nZ, nX, nY) → (B, C*nZ, nX, nY) → 2D conv → (B, C*nZ, nX, nY) → reshape
      │
  [Classifier]  → (B, num_classes, nZ, nX, nY)

장점:
  - LSS frustum pooling 없음 → 빠름
  - Channel-to-Height → 3D conv 불필요 → VRAM 절약
  - 단일 카메라로 동작
  - 실시간 추론 가능

클래스 (5):
  0=Free, 1=Road, 2=Vehicle, 3=Pedestrian, 4=StaticObstacle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════
# 백본: EfficientNet-B2 + FPN
# ══════════════════════════════════════════════════════
class EfficientFPN(nn.Module):
    """EfficientNet-B2 → FPN → P3/P4/P5 (모두 out_ch 채널)"""

    def __init__(self, out_ch=128, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b2(weights=weights)
        f = net.features

        # Stage 분리 (EfficientNet-B2 stage별 출력 채널)
        self.s3 = f[:4]    # out: 48ch
        self.s4 = f[4:6]   # out: 120ch
        self.s5 = f[6:]    # out: 1408ch

        self.lat5 = nn.Sequential(nn.Conv2d(1408, out_ch, 1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.lat4 = nn.Sequential(nn.Conv2d(120,  out_ch, 1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.lat3 = nn.Sequential(nn.Conv2d(48,   out_ch, 1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(True))

        self.out3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(True))

        print(f"[FastOcc] 백본: EfficientNet-B2 + FPN (out={out_ch}ch)")

    def forward(self, x):
        c3 = self.s3(x)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        return self.out3(p3)   # (B, out_ch, H/8, W/8)


# ══════════════════════════════════════════════════════
# 핵심 1: Voxel Query Sampler (LSS 아님)
# ══════════════════════════════════════════════════════
class VoxelQuerySampler(nn.Module):
    """
    각 3D 복셀 중심을 이미지 평면에 투영하여 feature를 샘플링.

    LSS의 차이:
      LSS  : 픽셀마다 D개 깊이 빈의 확률 → outer product → frustum 풀링
      FastOcc: 복셀 중심(x,y,z)을 K로 직접 투영 → bilinear grid_sample
              → D-bin 깊이 분포 없음, learnable depth 없음

    Parameters
    ----------
    xbound, ybound, zbound : (min, max, step) 복셀 범위
    feat_ch  : 이미지 피처 채널 수
    img_h/w  : 이미지 피처맵 크기 (다운샘플 후)
    """

    def __init__(self, xbound, ybound, zbound, feat_ch, img_h, img_w):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        xmin, xmax, xstep = xbound
        ymin, ymax, ystep = ybound
        zmin, zmax, zstep = zbound

        self.nX = int((xmax - xmin) / xstep)
        self.nY = int((ymax - ymin) / ystep)
        self.nZ = int((zmax - zmin) / zstep)

        # 복셀 중심 좌표 생성 (ego 좌표계): (nZ, nX, nY, 3)
        xs = torch.arange(self.nX) * xstep + xmin + xstep/2
        ys = torch.arange(self.nY) * ystep + ymin + ystep/2
        zs = torch.arange(self.nZ) * zstep + zmin + zstep/2

        # meshgrid → (nZ, nX, nY, 3) [x, y, z] in ego/world frame
        gz, gx, gy = torch.meshgrid(zs, xs, ys, indexing='ij')
        # nuScenes convention: x=right, y=forward, z=up
        # camera convention: X=right, Y=down, Z=forward
        # ego→cam: x_cam=x, y_cam=-z, z_cam=y  (대략, 실제는 extrinsic 사용)
        voxel_centers = torch.stack([gx, gy, gz], dim=-1)  # (nZ, nX, nY, 3)
        self.register_buffer('voxel_centers', voxel_centers)

    def forward(self, feat: torch.Tensor, K: torch.Tensor,
                sensor2ego: torch.Tensor = None) -> torch.Tensor:
        """
        feat       : (B, C, H', W')  이미지 피처맵
        K          : (B, 3, 3)       카메라 내부 파라미터
        sensor2ego : (B, 4, 4)       카메라→ego 변환행렬 (선택)

        Returns
        -------
        voxel_feat : (B, C, nZ, nX, nY)
        """
        B, C, Hf, Wf = feat.shape
        device = feat.device
        nZ, nX, nY = self.nZ, self.nX, self.nY

        # 복셀 중심 (nZ*nX*nY, 3) → ego XYZ
        centers = self.voxel_centers.to(device)       # (nZ, nX, nY, 3)
        pts = centers.reshape(-1, 3)                   # (N, 3)  N=nZ*nX*nY
        N = pts.shape[0]

        # ── ego → camera 변환 ──────────────────────
        if sensor2ego is not None:
            # sensor2ego: camera→ego, 역변환 = ego→camera
            # (B, 4, 4)
            ego2cam_R = sensor2ego[:, :3, :3].transpose(1, 2)   # (B,3,3)
            ego2cam_t = -torch.bmm(ego2cam_R,
                         sensor2ego[:, :3, 3:])                  # (B,3,1)
            # pts: (N,3) → (B,N,3)
            pts_b = pts.unsqueeze(0).expand(B, -1, -1)           # (B,N,3)
            pts_cam = torch.bmm(pts_b, ego2cam_R.transpose(1,2)) \
                      + ego2cam_t.transpose(1, 2)                 # (B,N,3)
        else:
            # 카메라가 ego 원점에 있다고 가정 (단순화)
            # nuScenes front cam: 대략 x=0, y=1.5, z=1.5m
            t_default = torch.tensor([0.0, 1.5, 1.5], device=device)
            pts_cam = pts.unsqueeze(0).expand(B, -1, -1).clone()
            pts_cam[..., 1] = pts_cam[..., 1] - t_default[1]
            pts_cam[..., 2] = pts_cam[..., 2] - t_default[2]
            # ego(x,y,z) → cam(X=x, Y=-z+h, Z=y-d)
            x_c =  pts_cam[..., 0]
            y_c = -pts_cam[..., 2]          # ego z → cam -Y
            z_c =  pts_cam[..., 1]          # ego y → cam Z (depth)
            pts_cam = torch.stack([x_c, y_c, z_c], dim=-1)

        # ── K 투영: cam_3d → image (u, v) ─────────
        # K: (B,3,3), pts_cam: (B,N,3)
        fx = K[:, 0, 0].view(B, 1)   # (B,1)
        fy = K[:, 1, 1].view(B, 1)
        cx = K[:, 0, 2].view(B, 1)
        cy = K[:, 1, 2].view(B, 1)

        Z  = pts_cam[..., 2].clamp(min=0.1)   # (B,N) depth > 0
        u  = pts_cam[..., 0] * fx / Z + cx    # (B,N) pixel u
        v  = pts_cam[..., 1] * fy / Z + cy    # (B,N) pixel v

        # ── grid_sample 좌표 정규화 [-1, 1] ────────
        # feat은 (B, C, Hf, Wf) 피처맵
        u_norm = (u / (self.img_w - 1)) * 2.0 - 1.0   # (B,N)
        v_norm = (v / (self.img_h - 1)) * 2.0 - 1.0

        # 유효한 투영 마스크 (이미지 안, 양수 depth)
        valid = (Z > 0.5) & (u_norm >= -1) & (u_norm <= 1) \
                         & (v_norm >= -1) & (v_norm <= 1)  # (B,N)

        # grid_sample: (B, C, 1, N) grid (B, 1, N, 2)
        grid = torch.stack([u_norm, v_norm], dim=-1)   # (B,N,2)
        grid = grid.unsqueeze(1)                        # (B,1,N,2)

        # feat: (B,C,Hf,Wf) → sample at grid → (B,C,1,N)
        sampled = F.grid_sample(feat, grid,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True)     # (B,C,1,N)
        sampled = sampled.squeeze(2)                    # (B,C,N)

        # 유효하지 않은 포인트 마스킹
        valid_mask = valid.unsqueeze(1).float()         # (B,1,N)
        sampled = sampled * valid_mask

        # (B,C,N) → (B,C,nZ,nX,nY)
        voxel_feat = sampled.reshape(B, C, nZ, nX, nY)

        return voxel_feat


# ══════════════════════════════════════════════════════
# 핵심 2: Channel-to-Height (C2H) Refiner
# ══════════════════════════════════════════════════════
class C2HRefiner(nn.Module):
    """
    Channel-to-Height (C2H) — FastOcc/FlashOcc 핵심 아이디어

    3D conv 없이 2D conv만으로 3D 피처를 정제:
      입력 : (B, C, nZ, nX, nY)
      → flatten: (B, C*nZ, nX, nY)
      → 2D conv (depth-wise + point-wise)
      → reshape: (B, C, nZ, nX, nY)

    3D conv 대비 장점:
      - VRAM 절약 (CUDA 2D 최적화)
      - 빠른 연산
      - 채널 간 상호작용으로 높이 맥락 포착
    """

    def __init__(self, feat_ch, nZ, mid_ch=64):
        super().__init__()
        in_ch = feat_ch * nZ

        self.net = nn.Sequential(
            # Depth-wise: 높이 정보 혼합
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=nZ, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(True),
            # Point-wise: 채널 정제
            nn.Conv2d(in_ch, mid_ch * nZ, 1, bias=False),
            nn.BatchNorm2d(mid_ch * nZ), nn.ReLU(True),
            nn.Conv2d(mid_ch * nZ, mid_ch * nZ, 3, 1, 1,
                      groups=nZ, bias=False),
            nn.BatchNorm2d(mid_ch * nZ), nn.ReLU(True),
        )
        self.mid_ch = mid_ch
        self.nZ     = nZ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, nZ, nX, nY) → (B, mid_ch, nZ, nX, nY)"""
        B, C, nZ, nX, nY = x.shape
        x_flat = x.reshape(B, C * nZ, nX, nY)      # flatten Z into ch
        x_flat = self.net(x_flat)                    # 2D conv
        x_out  = x_flat.reshape(B, self.mid_ch, nZ, nX, nY)
        return x_out


# ══════════════════════════════════════════════════════
# FastOcc 메인 모델
# ══════════════════════════════════════════════════════
class FastOcc(nn.Module):
    """
    FastOcc: 기하학적 복셀 샘플링 + Channel-to-Height 3D Occupancy

    Parameters
    ----------
    xbound, ybound, zbound : (min, max, step)  복셀 범위 [m]
    num_classes : 의미 클래스 수
    fpn_ch      : FPN 출력 채널
    c2h_ch      : C2H 중간 채널 (메모리/속도 트레이드오프)
    img_h, img_w: 입력 이미지 크기
    """

    def __init__(self,
                 xbound=(-25., 25., .5),
                 ybound=(-25., 25., .5),
                 zbound=(-2.,  6.,  .5),
                 num_classes=5,
                 fpn_ch=128,
                 c2h_ch=64,
                 img_h=224,
                 img_w=400):
        super().__init__()
        self.num_classes = num_classes

        nZ = int((zbound[1] - zbound[0]) / zbound[2])
        nX = int((xbound[1] - xbound[0]) / xbound[2])
        nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = nZ
        self.nX = nX
        self.nY = nY

        # 피처맵 크기 (EfficientNet-B2 FPN P3: 1/8 다운샘플)
        feat_h = img_h // 8
        feat_w = img_w // 8

        # ── 모듈 ──────────────────────────────────
        self.backbone = EfficientFPN(out_ch=fpn_ch)

        self.sampler  = VoxelQuerySampler(
            xbound, ybound, zbound,
            feat_ch=fpn_ch,
            img_h=feat_h, img_w=feat_w)

        self.c2h      = C2HRefiner(feat_ch=fpn_ch, nZ=nZ, mid_ch=c2h_ch)

        self.classifier = nn.Sequential(
            nn.Conv3d(c2h_ch, c2h_ch, 3, 1, 1, bias=False),
            nn.BatchNorm3d(c2h_ch), nn.ReLU(True),
            nn.Conv3d(c2h_ch, num_classes, 1),
        )

        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[FastOcc] 복셀: {nZ}×{nX}×{nY}  |  파라미터: {total:.1f}M")

    def forward(self,
                img: torch.Tensor,
                K:   torch.Tensor,
                sensor2ego: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        img        : (B, 3, H, W)
        K          : (B, 3, 3)   카메라 내부 파라미터
        sensor2ego : (B, 4, 4)   카메라→ego 변환 (없으면 기본값)

        Returns
        -------
        logits : (B, num_classes, nZ, nX, nY)
        """
        # 1. 이미지 피처 추출
        feat = self.backbone(img)                    # (B, fpn_ch, H/8, W/8)

        # 2. 복셀 쿼리 샘플링 (기하학적 투영 — LSS 아님)
        vox  = self.sampler(feat, K, sensor2ego)     # (B, fpn_ch, nZ, nX, nY)

        # 3. Channel-to-Height 정제 (2D conv만 사용 — 빠름)
        vox  = self.c2h(vox)                         # (B, c2h_ch, nZ, nX, nY)

        # 4. 분류
        logits = self.classifier(vox)                # (B, num_classes, nZ, nX, nY)

        return logits


# ══════════════════════════════════════════════════════
# 동작 확인
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    model = FastOcc(
        xbound=(-25., 25., .5),
        ybound=(-25., 25., .5),
        zbound=(-2.,  6.,  .5),
        num_classes=5,
        fpn_ch=128,
        c2h_ch=64,
        img_h=224,
        img_w=400,
    )

    B = 1
    img = torch.randn(B, 3, 224, 400)
    K   = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    K[:, 0, 0] = 500; K[:, 1, 1] = 500
    K[:, 0, 2] = 200; K[:, 1, 2] = 112

    with torch.no_grad():
        out = model(img, K)
    print(f"출력 shape: {out.shape}")   # (1, 5, 16, 100, 100)
    print("FastOcc 동작 확인 완료 ✅")
