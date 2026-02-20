import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from torchvision.models import resnet18
from PIL import Image

# =========================================================
# 1. 옛날 모델 구조 복원 (ResNet18 + 저해상도 + Binary)
# =========================================================
class OldCamEncoder(nn.Module):
    def __init__(self, D, C):
        super(OldCamEncoder, self).__init__()
        self.D = D
        self.C = C
        self.trunk = resnet18(pretrained=False) # 평가만 할 거라 pretrained 상관없음
        self.trunk.fc = nn.Identity()
        self.trunk.avgpool = nn.Identity()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, D + C, kernel_size=1, padding=0)
        )

    def get_cam_feats(self, x):
        x = self.trunk.conv1(x)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)
        x = self.trunk.layer1(x)
        x = self.trunk.layer2(x)
        x = self.trunk.layer3(x)
        x = self.trunk.layer4(x) 
        x = self.layer1(x)
        return x 

    def forward(self, x):
        x = self.get_cam_feats(x)
        depth_logits = x[:, :self.D]
        context = x[:, self.D:]
        depth_probs = depth_logits.softmax(dim=1)
        return depth_probs, context

class OldLSSModel(nn.Module):
    def __init__(self, device):
        super(OldLSSModel, self).__init__()
        # 옛날 설정 (ResNet18, 704x256, 1채널 출력)
        self.xbound = [-50, 50, 0.5]
        self.ybound = [-50, 50, 0.5]
        self.zbound = [-10, 10, 20] # 높이 1개 층으로 가정 (Binary)
        self.dbound = [4, 45, 1]
        self.C = 64
        
        self.nx = int((self.xbound[1] - self.xbound[0]) / self.xbound[2]) # 200
        self.ny = int((self.ybound[1] - self.ybound[0]) / self.ybound[2]) # 200
        self.D = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])  # 41
        
        # Frustum (8x22)
        H, W = 8, 22
        ds = torch.arange(self.dbound[0], self.dbound[1], self.dbound[2]).view(-1, 1, 1).expand(-1, H, W)
        D = ds.shape[0]
        xs = torch.linspace(0, 703, W).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, 255, H).view(1, H, 1).expand(D, H, W)
        self.frustum = nn.Parameter(torch.stack((xs, ys, ds), -1), requires_grad=False)
        
        self.cam_encoder = OldCamEncoder(D=self.D, C=self.C)
        
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )
        
        # Decoder 출력 1채널 (Binary)
        self.decoder = nn.Conv2d(self.C, 1, kernel_size=1)
        self.device = device

    def get_geometry(self, rots, trans, intrinsics):
        B = rots.shape[0]
        points = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        points = points.view(B, -1, 3)
        points_d = points[:, :, 2]
        points[:, :, 0] = (points[:, :, 0] - intrinsics[:, 0, 2].unsqueeze(1)) * points_d / intrinsics[:, 0, 0].unsqueeze(1)
        points[:, :, 1] = (points[:, :, 1] - intrinsics[:, 1, 2].unsqueeze(1)) * points_d / intrinsics[:, 1, 1].unsqueeze(1)
        points = torch.bmm(rots, points.permute(0, 2, 1)).permute(0, 2, 1) + trans.unsqueeze(1)
        return points.view(B, self.D, 8, 22, 3)

    def voxel_pooling(self, geom_feats, x):
        B, D, H, W, _ = geom_feats.shape
        geom_feats = geom_feats.reshape(-1, 3)
        x = x.permute(0, 1, 3, 4, 2).reshape(-1, x.shape[2])
        
        keep = ((geom_feats[:, 0] >= self.xbound[0]) & (geom_feats[:, 0] < self.xbound[1]) &
                (geom_feats[:, 1] >= self.ybound[0]) & (geom_feats[:, 1] < self.ybound[1]))
        geom_feats = geom_feats[keep]
        x = x[keep]
        
        # Batch Indices (간소화)
        # 배치 1개씩만 처리한다고 가정하면 간단함
        coords = ((geom_feats - torch.tensor([self.xbound[0], self.ybound[0], self.zbound[0]]).to(x.device)) / 
                  torch.tensor([self.xbound[2], self.ybound[2], self.zbound[2]]).to(x.device)).long()
        
        final_bev = torch.zeros((self.nx, self.ny, self.C), device=x.device)
        final_bev.index_put_((coords[:, 0], coords[:, 1]), x, accumulate=True)
        return final_bev.permute(2, 0, 1).unsqueeze(0)

    def forward(self, imgs, rots, trans, intrinsics):
        # 이미지 크기 강제 조절 (1056x384 -> 704x256)
        # 구형 모델은 작은 이미지를 원하므로 리사이즈 필요
        # (하지만 여기서는 Dataset에서 이미 704x256으로 줄여서 온다고 가정하거나, 
        #  모델 내부에서 처리해야 함. 보통 Dataset 설정을 따름)
        
        B, N, _, H, W = imgs.shape
        imgs = imgs.view(B * N, 3, H, W)
        rots = rots.view(B * N, 3, 3)
        trans = trans.view(B * N, 3)
        intrinsics = intrinsics.view(B * N, 3, 3)

        depth_probs, context = self.cam_encoder(imgs)
        geom = self.get_geometry(rots, trans, intrinsics)
        
        context = context.unsqueeze(1)
        depth_probs = depth_probs.unsqueeze(2)
        frustum_features = context * depth_probs 
        
        geom = geom.reshape(B, -1, 8, 22, 3)
        frustum_features = frustum_features.reshape(B, -1, self.C, 8, 22)

        # Splat (Batch 1일 때만 작동하는 간이 버전)
        bev_map = self.voxel_pooling(geom, frustum_features)
        bev_map = self.bev_compressor(bev_map)
        out = self.decoder(bev_map)
        return out

# =========================================================
# 2. 실행 코드
# =========================================================
def evaluate_old_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 NVIDIA GPU (CUDA) 가속 활성화됨! 사용 장치: {torch.cuda.get_device_name(0)}")
        
    # 2순위: MPS (Apple Silicon Mac) 확인
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple M1/M2/M3 MPS 가속 활성화됨!")
        
    # 3순위: CPU (모두 없을 경우)
    else:
        device = torch.device("cpu")
        print("⚠️ GPU를 찾을 수 없습니다. CPU로 실행합니다.")
    
    # [설정] 구버전 모델 파일명
    # (아까 에러 났던 그 파일 이름을 여기에 적으세요)
    model_path = "best_lss_multicam.pth" 

    # 1. 모델 준비
    model = OldLSSModel(device).to(device)
    
    try:
        # strict=False로 하면 불필요한 키(running_mean 등) 무시하고 로드
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"✅ 구버전 모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    model.eval()

    # 2. 데이터셋 (해상도 704x256으로 강제 설정 필요)
    # 현재 nuscenes_dataset.py가 고해상도(1056x384)로 되어 있다면, 
    # 일시적으로 704x256으로 동작하도록 하는 것이 좋지만,
    # 여기서는 코드를 수정하지 않고 Dataset을 불러온 뒤, 이미지 Resize를 다시 해주는 방식 사용
    
    dataset = NuScenesDataset(dataroot='../data/sets/nuscenesmini', is_train=False) # Validation set
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("🚀 구버전 모델 정확도 측정 시작...")
    
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for i, (imgs, intrinsics, sensor2ego, gt_bev) in enumerate(tqdm(loader)):
            # [중요] 이미지를 704x256으로 리사이즈 (모델이 옛날 거라)
            imgs = torch.nn.functional.interpolate(imgs.view(-1, 3, 384, 1056), size=(256, 704), mode='bilinear')
            imgs = imgs.view(1, 6, 3, 256, 704)
            
            # Intrinsics 스케일 보정 (384 -> 256)
            scale_x = 704 / 1056
            scale_y = 256 / 384
            intrinsics = intrinsics.clone()
            intrinsics[..., 0] *= scale_x
            intrinsics[..., 1] *= scale_y

            imgs = imgs.to(device)
            intrinsics = intrinsics.float().to(device)
            rots = sensor2ego[:, :, :3, :3].float().to(device)
            trans = sensor2ego[:, :, :3, 3].float().to(device)
            gt_bev = gt_bev.to(device)
            
            # 예측
            preds = model(imgs, rots, trans, intrinsics) # (1, 1, 200, 200)
            preds_prob = torch.sigmoid(preds)
            
            # GT 처리 (Binary)
            # Semantic GT (0,1,2,3) -> Binary GT (0,1)
            gt_map = (torch.max(gt_bev[0], dim=0)[0] > 0).float().cpu().numpy()
            pred_map = (preds_prob[0, 0] > 0.4).cpu().numpy().astype(float)
            
            # IoU 계산
            intersection = (pred_map * gt_map).sum()
            union = (pred_map + gt_map).sum() - intersection
            
            if union > 0:
                total_iou += intersection / union
                count += 1
                
    print("\n" + "="*40)
    print(f"📊 구버전 모델(Binary) 최종 결과")
    print(f" - 평균 IoU: {total_iou / count * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_old_model()