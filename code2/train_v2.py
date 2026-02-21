"""
train_v2.py - LSS v2 학습 스크립트
====================================
개선:
  1. Dynamic Focal Loss (Cosine 보간 동적 가중치)
  2. EfficientNet-B0 백본 (model_v2.py)
  3. CosineAnnealingWarmRestarts LR 스케줄러 (OneCycleLR 불안정 제거)
  4. 학습 완료 즉시 결과·시각화 GitHub 자동 커밋
목표: 모든 클래스 mIoU ≥ 50% (동적 가중치로 배경↔전경 균형 복원)
"""

import math
import os
import csv
import json
import time
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nuscenes_dataset import NuScenesDataset
from model_v2 import LSSModelV2


# ══════════════════════════════════════════════════════
# Dynamic Focal Loss
# ══════════════════════════════════════════════════════
class DynamicFocalLoss(nn.Module):
    """
    에폭에 따라 클래스 가중치를 Cosine 보간으로 점진 변화
      Phase 1 (0 ~ warmup):   전경 극단 강조  → 전경 탐지 학습
      Phase 2 (warmup ~ end): 점진 균형 복원  → 배경 안정성 회복
    """

    def __init__(self, gamma: float = 2.0,
                 total_epochs: int = 300,
                 warmup_ratio: float = 0.45):
        super().__init__()
        self.gamma        = gamma
        self.total_epochs = total_epochs
        self.warmup_ratio = warmup_ratio

        # [Empty, Car, Truck/Bus, Pedestrian/Bike]
        self.w_start = torch.tensor([0.05, 60.0, 50.0, 100.0])
        self.w_end   = torch.tensor([2.0,  10.0,  8.0,  15.0])

    def get_weights(self, epoch: int) -> torch.Tensor:
        progress = min(epoch / (self.total_epochs * self.warmup_ratio), 1.0)
        alpha    = 0.5 * (1.0 - math.cos(math.pi * progress))   # 0 → 1 (cosine)
        w = self.w_start + (self.w_end - self.w_start) * alpha
        return w

    def forward(self, pred, target, epoch: int = 0):
        w  = self.get_weights(epoch).to(pred.device)
        ce = F.cross_entropy(pred, target, weight=w, reduction='none')
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# ══════════════════════════════════════════════════════
# 빠른 mIoU 평가 (학습 중 모니터링용)
# ══════════════════════════════════════════════════════
@torch.no_grad()
def quick_miou(model, loader, device, num_classes=4):
    model.eval()
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for imgs, intrinsics, sensor2ego, gt in loader:
        imgs       = imgs.to(device, non_blocking=True)
        intrinsics = intrinsics.float().to(device)
        gt         = gt.long().to(device)
        rots  = sensor2ego[:, :, :3, :3].float().to(device)
        trans = sensor2ego[:, :, :3, 3].float().to(device)

        pred  = model(imgs, rots, trans, intrinsics)
        pred_cls = pred.argmax(dim=1)       # (B, nz, nx, ny)

        for c in range(num_classes):
            p = (pred_cls == c)
            g = (gt == c)
            tp[c] += (p & g).sum().cpu()
            fp[c] += (p & ~g).sum().cpu()
            fn[c] += (~p & g).sum().cpu()

    iou = tp / (tp + fp + fn + 1e-6)
    model.train()
    return iou                              # (num_classes,)


# ══════════════════════════════════════════════════════
# GitHub 자동 커밋 헬퍼
# ══════════════════════════════════════════════════════
def git_commit_results(message: str, repo_root: str = ".."):
    try:
        subprocess.run(["git", "add", "code2/results_v2/", "README.md"],
                       cwd=repo_root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message,
                        "--author", "Claude Bot <noreply@anthropic.com>"],
                       cwd=repo_root, check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "feature/portfolio-3d-semantic"],
                       cwd=repo_root, check=True, capture_output=True)
        print(f"  ✅ GitHub 커밋 완료: {message}")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Git 오류 (계속 진행): {e}")


# ══════════════════════════════════════════════════════
# 학습 메인
# ══════════════════════════════════════════════════════
def main():
    # ── 장치 선택 ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 모드")

    # ── 하이퍼파라미터 ──
    epochs          = 300
    patience        = 25           # 이전 15 → 25 (더 오래 기다림)
    batch_size      = 4
    accum_steps     = 2            # effective batch = 8
    lr              = 3e-4
    eval_interval   = 10          # N 에폭마다 mIoU 평가

    # ── 데이터 ──
    train_ds = NuScenesDataset('../data/sets/nuscenesmini',
                               version='v1.0-mini', is_train=True)
    val_ds   = NuScenesDataset('../data/sets/nuscenesmini',
                               version='v1.0-mini', is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

    # ── 모델 ──
    print("\n[모델 초기화]")
    model = LSSModelV2(
        xbound=[-50, 50, 0.5],
        ybound=[-50, 50, 0.5],
        zbound=[-2.0, 6.0, 2.0],
        dbound=[4, 45, 1],
        num_classes=4,
        C=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  총 파라미터: {total_params:.2f}M")

    # ── 손실·옵티마이저·스케줄러 ──
    criterion = DynamicFocalLoss(gamma=2.0, total_epochs=epochs, warmup_ratio=0.45)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # CosineAnnealingWarmRestarts: 주기적 재시작으로 local minima 탈출
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')

    # ── 저장 폴더 ──
    os.makedirs("results_v2", exist_ok=True)
    log_path = "results_v2/train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "lr", "best_loss",
             "iou_empty", "iou_car", "iou_truck", "iou_ped", "miou_fg"])

    # ── 학습 루프 ──
    best_loss  = float('inf')
    best_miou  = 0.0
    counter    = 0
    loss_hist  = []
    miou_hist  = []

    class_names = ["Empty", "Car", "Truck/Bus", "Pedestrian"]
    print(f"\n학습 시작 (Max {epochs} epochs, patience={patience})\n")
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:03d}/{epochs}", leave=True)

        for i, (imgs, intrinsics, sensor2ego, gt) in enumerate(loop):
            imgs       = imgs.to(device, non_blocking=True)
            intrinsics = intrinsics.float().to(device)
            gt         = gt.long().to(device)
            rots  = sensor2ego[:, :, :3, :3].float().to(device)
            trans = sensor2ego[:, :, :3, 3].float().to(device)

            with torch.amp.autocast('cuda'):
                pred = model(imgs, rots, trans, intrinsics)
                loss = criterion(pred, gt, epoch=epoch) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                loop.set_postfix(
                    loss=f"{loss.item() * accum_steps:.4f}",
                    w_car=f"{criterion.get_weights(epoch)[1]:.1f}")

            total_loss += loss.item() * accum_steps

        scheduler.step(epoch)
        avg_loss = total_loss / len(train_loader)
        cur_lr   = optimizer.param_groups[0]['lr']
        loss_hist.append(avg_loss)

        # ── mIoU 평가 (N 에폭마다) ──
        iou_vals = torch.zeros(4)
        if (epoch + 1) % eval_interval == 0:
            iou_vals = quick_miou(model, val_loader, device)
            miou_fg  = iou_vals[1:].mean().item()
            miou_hist.append((epoch + 1, miou_fg))

            w_now = criterion.get_weights(epoch)
            print(f"\n  📊 mIoU @ Epoch {epoch+1}")
            for c, name in enumerate(class_names):
                print(f"     {name:<18}: {iou_vals[c]*100:5.1f}%")
            print(f"     전경 mIoU  : {miou_fg*100:5.1f}%")
            print(f"     가중치     : {[f'{w:.1f}' for w in w_now.tolist()]}")

            # 최고 전경 mIoU 갱신 시 별도 저장
            if miou_fg > best_miou:
                best_miou = miou_fg
                torch.save(model.state_dict(), "best_v2_miou_model.pth")
                print(f"  🏆 Best 전경 mIoU 갱신: {miou_fg*100:.1f}%")

        print(f"Epoch {epoch+1:03d} - Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")

        # ── Early Stopping (Loss 기반) ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter   = 0
            torch.save(model.state_dict(), "best_v2_model.pth")
            print(f"  ✅ Best Loss 저장 ({best_loss:.4f})")
        else:
            counter += 1
            print(f"  ⏳ No improve ({counter}/{patience})")
            if counter >= patience:
                print(f"\nEarly Stopping: {patience} epoch 향상 없음")
                break

        # CSV 기록
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{avg_loss:.6f}", f"{cur_lr:.8f}", f"{best_loss:.6f}",
                f"{iou_vals[0]:.4f}", f"{iou_vals[1]:.4f}",
                f"{iou_vals[2]:.4f}", f"{iou_vals[3]:.4f}",
                f"{iou_vals[1:].mean():.4f}",
            ])

    # ══ 학습 완료 ══
    elapsed = time.time() - t_start
    print(f"\n✅ 학습 완료 ({elapsed/60:.1f}분)")
    print(f"   Best Loss : {best_loss:.4f}")
    print(f"   Best 전경 mIoU: {best_miou*100:.1f}%")

    # ── 최종 전체 평가 ──
    print("\n[최종 평가 실행 중...]")
    model.load_state_dict(torch.load("best_v2_model.pth"))
    final_iou = quick_miou(model, val_loader, device, num_classes=4)
    print("\n══════════════════════════════")
    print("  최종 클래스별 IoU")
    for c, name in enumerate(class_names):
        print(f"  {name:<18}: {final_iou[c]*100:5.1f}%")
    print(f"  전경 mIoU      : {final_iou[1:].mean()*100:5.1f}%")
    print(f"  전체 mIoU      : {final_iou.mean()*100:5.1f}%")
    print("══════════════════════════════\n")

    # ── 손실 그래프 저장 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(loss_hist, color='royalblue', lw=1.5)
    axes[0].set(xlabel="Epoch", ylabel="Loss",
                title=f"Training Loss (best={best_loss:.4f})")
    axes[0].grid(alpha=0.3)

    if miou_hist:
        ep_arr, m_arr = zip(*miou_hist)
        axes[1].plot(ep_arr, [v*100 for v in m_arr],
                     color='tomato', lw=1.5, marker='o', ms=4)
        axes[1].axhline(50, color='green', ls='--', alpha=0.6, label='목표 50%')
        axes[1].set(xlabel="Epoch", ylabel="Foreground mIoU (%)",
                    title="Foreground mIoU (Car+Truck+Ped)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results_v2/loss_curve_v2.png", dpi=130)
    plt.close()

    # ── train_info 저장 ──
    info = {
        "model"       : "LSS v2 (EfficientNet-B0 + SE-BEV)",
        "backbone"    : "EfficientNet-B0",
        "dataset"     : "NuScenes mini",
        "epochs_trained": len(loss_hist),
        "best_loss"   : round(best_loss, 6),
        "best_fg_miou": round(best_miou, 4),
        "final_iou"   : {name: round(final_iou[c].item(), 4)
                         for c, name in enumerate(class_names)},
        "elapsed_min" : round(elapsed / 60, 1),
    }
    with open("results_v2/train_info_v2.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # ── GitHub 자동 커밋 ──
    fg_miou_pct = final_iou[1:].mean().item() * 100
    git_commit_results(
        f"v2 학습 완료: Loss={best_loss:.4f}, 전경mIoU={fg_miou_pct:.1f}%",
        repo_root=".."
    )


if __name__ == "__main__":
    main()
