import torch
import torch.nn as nn

class VoxelPooling(nn.Module):
    def __init__(self, xbound, ybound, zbound, dbound):
        super(VoxelPooling, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        self.dx = xbound[2]
        self.dy = ybound[2]
        self.dz = zbound[2]

        self.nx = int((xbound[1] - xbound[0]) / self.dx)
        self.ny = int((ybound[1] - ybound[0]) / self.dy)
        self.nz = int((zbound[1] - zbound[0]) / self.dz)

    def forward(self, geom_feats, x):
        """
        geom_feats: (B, D, H, W, 3) -> 각 frustum 점의 3D ego 좌표
        x:          (B, D, C, H, W) -> 각 점의 특징 벡터
        반환:       (B, nz*C, nx, ny) -> Z 정보를 채널에 포함한 BEV 특징 맵
        """
        B, D, H, W, _ = geom_feats.shape
        C = x.shape[2]

        # 배치 인덱스 생성: 각 점이 몇 번째 배치인지
        batch_indices = (
            torch.arange(B, device=geom_feats.device)
            .view(B, 1, 1, 1)
            .expand(B, D, H, W)
            .reshape(-1)
        )

        # Flatten
        geom_feats = geom_feats.reshape(-1, 3)
        x = x.permute(0, 1, 3, 4, 2).reshape(-1, C)  # (B*D*H*W, C)

        # 범위 밖 점 제거
        keep = (
            (geom_feats[:, 0] >= self.xbound[0]) & (geom_feats[:, 0] < self.xbound[1]) &
            (geom_feats[:, 1] >= self.ybound[0]) & (geom_feats[:, 1] < self.ybound[1]) &
            (geom_feats[:, 2] >= self.zbound[0]) & (geom_feats[:, 2] < self.zbound[1])
        )

        geom_feats = geom_feats[keep]
        x = x[keep]
        batch_indices = batch_indices[keep]

        # 실수 좌표 → 복셀 인덱스 변환
        coords = (
            (geom_feats - torch.tensor(
                [self.xbound[0], self.ybound[0], self.zbound[0]],
                device=geom_feats.device
            )) /
            torch.tensor([self.dx, self.dy, self.dz], device=geom_feats.device)
        ).long()

        # 각 축 인덱스 분리 (clamp로 경계 보호)
        ix = coords[:, 0].clamp(0, self.nx - 1)
        iy = coords[:, 1].clamp(0, self.ny - 1)
        iz = coords[:, 2].clamp(0, self.nz - 1)

        # Z-aware BEV 텐서: (B, nz, nx, ny, C)
        final_bev = torch.zeros(
            (B, self.nz, self.nx, self.ny, C),
            device=x.device
        )

        # 각 점을 해당 Z 층, X, Y 위치에 누적 (높이 정보 보존)
        final_bev.index_put_(
            (batch_indices, iz, ix, iy),
            x,
            accumulate=True
        )

        # (B, nz, nx, ny, C) → (B, nz, C, nx, ny) → (B, nz*C, nx, ny)
        final_bev = final_bev.permute(0, 1, 4, 2, 3).reshape(B, self.nz * C, self.nx, self.ny)

        return final_bev


# ------------------------------------------------
# 테스트 코드
# ------------------------------------------------
if __name__ == "__main__":
    xbound = [-50.0, 50.0, 0.5]
    ybound = [-50.0, 50.0, 0.5]
    zbound = [-2.0,  6.0,  2.0]   # nz = 4
    dbound = [4.0, 45.0,  1.0]

    pooler = VoxelPooling(xbound, ybound, zbound, dbound)

    B, D, H, W, C = 1, 41, 8, 22, 64
    dummy_geom = torch.rand(B, D, H, W, 3) * 100 - 50
    dummy_features = torch.randn(B, D, C, H, W)

    print("[테스트] Z-aware VoxelPooling 수행 중...")
    bev_map = pooler(dummy_geom, dummy_features)

    print(f"입력 특징: {dummy_features.shape}")
    print(f"출력 BEV : {bev_map.shape}")  # 기대: (1, nz*C, 200, 200) = (1, 256, 200, 200)

    expected_ch = pooler.nz * C
    if bev_map.shape == (B, expected_ch, pooler.nx, pooler.ny):
        print(f"성공! Z-aware BEV: (B={B}, nz*C={expected_ch}, nx={pooler.nx}, ny={pooler.ny})")
    else:
        print(f"실패! 기대 크기: ({B}, {expected_ch}, {pooler.nx}, {pooler.ny})")
