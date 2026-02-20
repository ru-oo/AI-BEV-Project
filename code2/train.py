import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from model import CamEncoder
from splat import VoxelPooling
from tqdm import tqdm
import os
import csv
import json
import time
import matplotlib
matplotlib.use('Agg')  # 디스플레이 없이 PNG 저장
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LSS Model (유지)
# --------------------------------------------------
class LSSModel(nn.Module):
    def __init__(self, xbound, ybound, zbound, dbound, num_classes=4, C=64):
        super(LSSModel, self).__init__()
        self.dbound = dbound
        self.C = C
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.num_classes = num_classes
        
        # 해상도 설정 (nuscenes_dataset.py와 일치해야 함: 1056x384)
        self.img_H, self.img_W = 384, 1056
        self.feat_H, self.feat_W = self.img_H // 32, self.img_W // 32
        
        self.frustum = self.create_frustum()
        self.D = int((dbound[1] - dbound[0]) / dbound[2])
        
        self.cam_encoder = CamEncoder(D=self.D, C=C)
        self.voxel_pooling = VoxelPooling(xbound, ybound, zbound, dbound)
        
        self.nz = int((zbound[1] - zbound[0]) / zbound[2])
        nz_C = self.nz * C  # Z층별 채널 묶음: 4 * 64 = 256

        # VoxelPooling이 (B, nz*C, nx, ny)를 반환하므로 입력 채널을 nz_C로 설정
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(nz_C, nz_C, kernel_size=3, padding=1),
            nn.BatchNorm2d(nz_C),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Conv2d(nz_C, self.nz * self.num_classes, kernel_size=1)

    def create_frustum(self):
        H, W = self.feat_H, self.feat_W
        ds = torch.arange(self.dbound[0], self.dbound[1], self.dbound[2]).view(-1, 1, 1).expand(-1, H, W)
        D = ds.shape[0]
        xs = torch.linspace(0, self.img_W - 1, W).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, self.img_H - 1, H).view(1, H, 1).expand(D, H, W)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrinsics):
        B = rots.shape[0]
        points = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        points = points.view(B, -1, 3)
        points_d = points[:, :, 2]
        
        points[:, :, 0] = (points[:, :, 0] - intrinsics[:, 0, 2].unsqueeze(1)) * points_d / intrinsics[:, 0, 0].unsqueeze(1)
        points[:, :, 1] = (points[:, :, 1] - intrinsics[:, 1, 2].unsqueeze(1)) * points_d / intrinsics[:, 1, 1].unsqueeze(1)
        
        points = torch.bmm(rots, points.permute(0, 2, 1)).permute(0, 2, 1) + trans.unsqueeze(1)
        return points.view(B, self.D, self.feat_H, self.feat_W, 3)

    def forward(self, imgs, rots, trans, intrinsics):
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
        
        geom = geom.reshape(B, -1, self.feat_H, self.feat_W, 3)
        frustum_features = frustum_features.reshape(B, -1, self.C, self.feat_H, self.feat_W)

        bev_map = self.voxel_pooling(geom, frustum_features)
        bev_map = self.bev_compressor(bev_map)
        
        out = self.decoder(bev_map)
        out = out.view(B, self.num_classes, self.nz, out.shape[2], out.shape[3])
        return out

# --------------------------------------------------
# 2. 학습 실행 (Early Stopping 추가)
# --------------------------------------------------
def main():
    # 1순위: CUDA (NVIDIA GPU) 확인 - 원격 접속 환경
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

    # === 설정 ===
    epochs = 200        # 넉넉하게 설정
    
    # [Early Stopping 설정]
    patience = 15       # 15번 연속으로 성능 향상이 없으면 멈춤
    counter = 0         # 참은 횟수 카운터
    
# === [수정 1] 메모리 터짐 방지 설정 ===
    # 기존 batch_size=8 은 ResNet50 + 고해상도에서 너무 큽니다.
    # 4로 줄이고 accumulation_steps를 2로 늘리면, 실제 학습 효과는 8과 동일합니다.
    batch_size = 4          # 기존 8 -> 4 (또는 2) 로 감소
    accumulation_steps = 2  # 기존 1 -> 2 (배치 줄인 만큼 늘리기)
    learning_rate = 3e-4
    
    # '../'는 '현재 폴더의 한 단계 위로'라는 뜻입니다.
    # dataset = NuScenesDataset('../data/sets/nuscenes', version='v1.0-trainval', is_train=True)
    dataset = NuScenesDataset('../data/sets/nuscenesmini', version='v1.0-mini', is_train=True)
    
    loader = DataLoader(dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=8,      # 기존 2 -> 8 (CPU 코어 수에 따라 늘려주세요)
                    pin_memory=True,    # 기존 False -> True (NVIDIA GPU 필수 설정)
                    persistent_workers=True)

    model = LSSModel(xbound=[-50, 50, 0.5], 
                     ybound=[-50, 50, 0.5], 
                     zbound=[-2.0, 6.0, 2.0], 
                     dbound=[4, 45, 1],
                     num_classes=4).to(device)

    class_weights = torch.tensor([0.5, 5.0, 5.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                              total_steps=epochs * (len(loader) // accumulation_steps), 
                                              epochs=epochs)
    
    scaler = torch.amp.GradScaler('cuda') # 또는 device='cuda'

    # === 결과 저장 폴더 ===
    os.makedirs("results", exist_ok=True)
    log_path = "results/train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "lr", "best_loss"])

    train_info = {
        "model": "LSS (Lift-Splat-Shoot)",
        "backbone": "ResNet18",
        "dataset": "NuScenes mini (v1.0-mini)",
        "num_classes": 4,
        "classes": ["Empty", "Car", "Truck/Bus", "Pedestrian/Bike"],
        "xbound": [-50, 50, 0.5],
        "ybound": [-50, 50, 0.5],
        "zbound": [-2.0, 6.0, 2.0],
        "dbound": [4, 45, 1],
        "img_size": [384, 1056],
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch": batch_size * accumulation_steps,
        "learning_rate": learning_rate,
        "epochs_max": epochs,
        "patience": patience,
        "device": str(device),
    }

    loss_history = []
    print(f"학습 시작! (Max Epochs: {epochs}, Patience: {patience})")
    start_time = time.time()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        # [수정 2] loader 대신 위에서 만든 loop 객체 사용
        for i, (imgs, intrinsics, sensor2ego, gt_semantic) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            intrinsics = intrinsics.float().to(device)
            gt_semantic = gt_semantic.long().to(device)
            rots = sensor2ego[:, :, :3, :3].float().to(device)
            trans = sensor2ego[:, :, :3, 3].float().to(device)

            with torch.amp.autocast('cuda'):
                preds = model(imgs, rots, trans, intrinsics)
                loss = criterion(preds, gt_semantic)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # [추가] 진행 바 오른쪽에 실시간 Loss 표시 (선택사항)
                # 학습이 잘 되고 있는지 실시간으로 수치가 변하는걸 볼 수 있습니다.
                current_loss = loss.item() * accumulation_steps
                loop.set_postfix(loss=f"{current_loss:.4f}")

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(loader)
        
        try:
            current_lr = scheduler.get_last_lr()[0]
        except:
            current_lr = learning_rate

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (LR: {current_lr:.6f})")
        loss_history.append(avg_loss)

        # === [Early Stopping 로직] ===
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), "best_semantic_mini_model.pth")
            print(f"  [Best] 모델 저장 (Loss: {best_loss:.4f})")
        else:
            counter += 1
            print(f"  [No improve] ({counter}/{patience})")
            if counter >= patience:
                print(f"\nEarly Stopping: {patience} epoch 동안 향상 없음")
                break

        # CSV 로그 기록
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_loss:.6f}", f"{current_lr:.8f}", f"{best_loss:.6f}"])

    # === 학습 완료: 손실 그래프 저장 ===
    elapsed = time.time() - start_time
    train_info["epochs_trained"] = len(loss_history)
    train_info["best_loss"] = round(best_loss, 6)
    train_info["elapsed_sec"] = round(elapsed, 1)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='royalblue', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (best={best_loss:.4f}, epochs={len(loss_history)})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=120)
    plt.close()

    with open("results/train_info.json", "w") as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)

    print(f"\n학습 완료! ({elapsed/60:.1f}분)")
    print(f"  Best Loss : {best_loss:.4f}")
    print(f"  로그 저장 : results/train_log.csv")
    print(f"  그래프    : results/loss_curve.png")
    print(f"  설정 정보 : results/train_info.json")

if __name__ == "__main__":
    main()