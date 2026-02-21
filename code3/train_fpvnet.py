"""
train_fpvnet.py — FPVNet 메모리 효율 학습 스크립트
===================================================
특징:
  - FPVNet (LSS 아님): 깊이 예측 + 기하학적 투영
  - Mixed Precision (FP16) + Gradient Accumulation
  - 메모리 절약: batch=2, accum=4 → effective batch=8
  - 3중 손실: 3D 복셀 + 2D 의미 + 깊이 일관성
  - 결과 JPG 자동 생성 + GitHub 자동 push
  - mIoU ≥ 50% 달성 전략: 클래스 가중치 + Focal Loss

실행:
  cd code3
  python train_fpvnet.py
"""

import math
import os
import csv
import json
import time
import subprocess
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_fpvnet import FPVNet
from dataset_nuscenes_v3 import NuScenesV3Dataset, NUM_CLASSES, CLASS_NAMES

# ═══════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════
CFG = dict(
    # 데이터
    data_root   = '../data/sets/nuscenesmini',
    version     = 'v1.0-mini',
    # 복셀 그리드
    xbound      = (-25.0, 25.0, 0.5),
    ybound      = (-25.0, 25.0, 0.5),
    zbound      = (-2.0,  6.0,  1.0),
    dbound      = (1.0, 50.0),
    # 학습
    epochs      = 200,
    patience    = 30,
    batch_size  = 2,           # 메모리 절약 (VRAM 4GB↑ 권장)
    accum_steps = 4,           # effective batch = 8
    lr          = 2e-4,
    weight_decay= 1e-4,
    eval_every  = 5,           # N에폭마다 mIoU 평가
    num_workers = 0,           # Windows: 0 권장
    # 저장
    result_dir  = 'results_v3',
    save_best   = 'best_fpvnet.pth',
)


# ═══════════════════════════════════════════════════
# 손실 함수
# ═══════════════════════════════════════════════════
class FPVLoss(nn.Module):
    """
    3중 손실:
      L_vox  : 3D 복셀 Focal Loss (mIoU 최적화 핵심)
      L_sem  : 2D 의미 Cross-Entropy (보조)
      L_depth: 깊이 스무딩 정규화 (보조)

    클래스 가중치 (희소 클래스 강조):
      Free=1, Road=3, Vehicle=10, Ped=15, StaticObst=6
    """

    WEIGHTS = torch.tensor([1.0, 3.0, 10.0, 15.0, 6.0])

    def __init__(self, gamma=2.0, w_vox=1.0, w_sem=0.3, w_d=0.05):
        super().__init__()
        self.gamma = gamma
        self.w_vox = w_vox
        self.w_sem = w_sem
        self.w_d   = w_d

    def focal_ce(self, pred, target, weight):
        """Focal Loss = (1-pt)^γ × CE"""
        ce = F.cross_entropy(pred, target, weight=weight.to(pred.device),
                             reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()

    def forward(self, voxel_logits, depth, sem2d, gt_voxel, gt_sem2d=None):
        """
        voxel_logits : (B, C, nZ, nX, nY)
        depth        : (B, 1, H, W)
        sem2d        : (B, C, H, W)
        gt_voxel     : (B, nZ, nX, nY)  long
        gt_sem2d     : (B, H, W)        long  [선택]
        """
        # ── 3D 복셀 Focal Loss ──
        L_vox = self.focal_ce(voxel_logits, gt_voxel, self.WEIGHTS)

        # ── 2D 의미 보조 손실 ──
        if gt_sem2d is not None:
            L_sem = self.focal_ce(sem2d, gt_sem2d, self.WEIGHTS)
        else:
            # GT 없으면 3D → 2D z-max projection으로 대체
            with torch.no_grad():
                gt2d = gt_voxel.max(dim=1).values  # (B, nX, nY)
                gt2d = F.interpolate(
                    gt2d.float().unsqueeze(1),
                    size=sem2d.shape[-2:],
                    mode='nearest').long().squeeze(1)
            L_sem = self.focal_ce(sem2d, gt2d, self.WEIGHTS)

        # ── 깊이 스무딩 정규화 ──
        dy = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs().mean()
        dx = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs().mean()
        L_d = dy + dx

        total = (self.w_vox * L_vox +
                 self.w_sem * L_sem +
                 self.w_d   * L_d)

        return total, {'vox': L_vox.item(),
                       'sem': L_sem.item(),
                       'dep': L_d.item()}


# ═══════════════════════════════════════════════════
# mIoU 평가
# ═══════════════════════════════════════════════════
@torch.no_grad()
def compute_miou(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for imgs, Ks, gt in loader:
        imgs = imgs.to(device, non_blocking=True)
        Ks   = Ks.float().to(device)
        gt   = gt.long().to(device)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            vox, _, _ = model(imgs, Ks)

        pred = vox.argmax(dim=1)  # (B, nZ, nX, nY)

        for c in range(num_classes):
            p = (pred == c); g = (gt == c)
            tp[c] += (p & g).sum().cpu()
            fp[c] += (p & ~g).sum().cpu()
            fn[c] += (~p & g).sum().cpu()

    iou = tp / (tp + fp + fn + 1e-6)
    model.train()
    return iou  # (num_classes,)


# ═══════════════════════════════════════════════════
# GitHub 자동 커밋
# ═══════════════════════════════════════════════════
def git_push_results(msg: str, repo_root: str = '..'):
    """결과 JPG + 로그를 git에 자동 push"""
    try:
        subprocess.run(
            ['git', 'add',
             'code3/results_v3/',
             'code3/best_fpvnet.pth',
             'README.md'],
            cwd=repo_root, check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', msg,
             '--author', 'FPVNet Bot <noreply@github.com>'],
            cwd=repo_root, check=True, capture_output=True)
        subprocess.run(
            ['git', 'push', 'origin',
             'feature/portfolio-3d-semantic'],
            cwd=repo_root, check=True, capture_output=True)
        print(f'  ✅ Git push 완료: {msg}')
    except subprocess.CalledProcessError as e:
        print(f'  ⚠️  Git 오류 (학습 계속): {e.stderr.decode()[:200]}')


# ═══════════════════════════════════════════════════
# BEV 시각화 저장
# ═══════════════════════════════════════════════════
CMAP = {
    0: (0,   0,   0),    # Free  — 검정
    1: (80,  80,  80),   # Road  — 회색
    2: (0,   120, 255),  # Vehicle — 파랑
    3: (255, 50,  50),   # Pedestrian — 빨강
    4: (200, 200,  0),   # StaticObst — 노랑
}

def save_bev_vis(gt_voxel, pred_voxel, path, epoch):
    """BEV z-max projection 시각화 저장 (GT vs 예측 비교)"""
    import cv2

    def to_bev_rgb(voxel_np):
        """(nZ, nX, nY) → RGB (nX, nY, 3)"""
        bev = voxel_np.max(axis=0)                 # (nX, nY)
        rgb = np.zeros((*bev.shape, 3), dtype=np.uint8)
        for cid, color in CMAP.items():
            mask = bev == cid
            rgb[mask] = color
        return rgb

    gt_bev   = to_bev_rgb(gt_voxel)
    pred_bev = to_bev_rgb(pred_voxel)

    # 좌: GT  우: Pred
    vis = np.concatenate([gt_bev, pred_bev], axis=1)
    vis = cv2.resize(vis, (vis.shape[1]*2, vis.shape[0]*2),
                     interpolation=cv2.INTER_NEAREST)

    # 텍스트 추가
    cv2.putText(vis, f'GT  Epoch {epoch}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, 'Pred', (vis.shape[1]//2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ═══════════════════════════════════════════════════
# 메인 학습
# ═══════════════════════════════════════════════════
def main():
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f'🖥️  장치: {device}')
    if device.type == 'cuda':
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

    os.makedirs(CFG['result_dir'], exist_ok=True)

    # ── 데이터셋 ──
    print('\n[데이터 로드]')
    train_ds = NuScenesV3Dataset(
        CFG['data_root'], version=CFG['version'], is_train=True,
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'])
    val_ds = NuScenesV3Dataset(
        CFG['data_root'], version=CFG['version'], is_train=False,
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'])

    train_loader = DataLoader(
        train_ds, batch_size=CFG['batch_size'], shuffle=True,
        num_workers=CFG['num_workers'], pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(
        val_ds, batch_size=CFG['batch_size'], shuffle=False,
        num_workers=CFG['num_workers'], pin_memory=(device.type == 'cuda'))

    # ── 모델 ──
    print('\n[모델 초기화] FPVNet (LSS 아님 — 기하학적 투영 기반)')
    model = FPVNet(
        xbound=CFG['xbound'],
        ybound=CFG['ybound'],
        zbound=CFG['zbound'],
        dbound=CFG['dbound'],
        num_classes=NUM_CLASSES,
        fpn_ch=128,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  총 파라미터: {total_p:.2f}M')

    # ── 손실·옵티마이저·스케줄러 ──
    criterion = FPVLoss(gamma=2.0, w_vox=1.0, w_sem=0.3, w_d=0.05)
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda',
                                  enabled=(device.type == 'cuda'))

    # ── 학습 이력 ──
    log_path = os.path.join(CFG['result_dir'], 'train_log_v3.csv')
    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'lr', 'best_loss',
                    'iou_free', 'iou_road', 'iou_veh', 'iou_ped',
                    'iou_static', 'miou_all', 'miou_fg'])

    best_loss  = float('inf')
    best_miou  = 0.0
    counter    = 0
    loss_hist  = []
    miou_hist  = []

    print(f'\n학습 시작 (max {CFG["epochs"]} epochs, '
          f'patience={CFG["patience"]}, '
          f'effective batch={CFG["batch_size"]*CFG["accum_steps"]})\n')
    t_start = time.time()

    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        loop = tqdm(train_loader,
                    desc=f'Epoch {epoch:03d}/{CFG["epochs"]}',
                    leave=True)

        for step, (imgs, Ks, gt) in enumerate(loop, 1):
            imgs = imgs.to(device, non_blocking=True)
            Ks   = Ks.float().to(device)
            gt   = gt.long().to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                vox, depth, sem2d = model(imgs, Ks)
                loss, loss_dict   = criterion(vox, depth, sem2d, gt)
                loss = loss / CFG['accum_steps']

            scaler.scale(loss).backward()

            if step % CFG['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * CFG['accum_steps']
            loop.set_postfix(
                loss=f"{loss.item()*CFG['accum_steps']:.4f}",
                vox=f"{loss_dict['vox']:.3f}",
                sem=f"{loss_dict['sem']:.3f}")

        scheduler.step(epoch)
        avg_loss = total_loss / len(train_loader)
        cur_lr   = optimizer.param_groups[0]['lr']
        loss_hist.append(avg_loss)

        # ── mIoU 평가 ──────────────────────────
        iou_vals = torch.zeros(NUM_CLASSES)
        if epoch % CFG['eval_every'] == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            iou_vals = compute_miou(model, val_loader, device)
            miou_all = iou_vals.mean().item()
            miou_fg  = iou_vals[1:].mean().item()  # Road 포함 전경
            miou_hist.append((epoch, miou_fg * 100))

            print(f'\n  📊 mIoU @ Epoch {epoch}')
            for c, name in enumerate(CLASS_NAMES):
                print(f'     {name:<15}: {iou_vals[c]*100:5.1f}%')
            print(f'     전경 mIoU : {miou_fg*100:5.1f}%  '
                  f'전체: {miou_all*100:5.1f}%')

            # BEV 시각화 저장 (첫 배치 사용)
            with torch.no_grad():
                sample_imgs, sample_Ks, sample_gt = next(iter(val_loader))
                sample_imgs = sample_imgs.to(device)
                sample_Ks   = sample_Ks.float().to(device)
                pred_vox, _, _ = model(sample_imgs, sample_Ks)
            pred_np = pred_vox[0].argmax(dim=0).cpu().numpy()
            gt_np   = sample_gt[0].numpy()
            vis_path = os.path.join(
                CFG['result_dir'], f'bev_epoch{epoch:03d}.jpg')
            save_bev_vis(gt_np, pred_np, vis_path, epoch)

            # Best mIoU 갱신
            if miou_fg > best_miou:
                best_miou = miou_fg
                torch.save(model.state_dict(), 'best_fpvnet_miou.pth')
                print(f'  🏆 Best 전경 mIoU: {miou_fg*100:.1f}%')

                # mIoU 50%+ 달성 시 즉시 git push
                if miou_fg >= 0.50:
                    git_push_results(
                        f'FPVNet: mIoU {miou_fg*100:.1f}% 달성 (epoch {epoch})',
                        repo_root='..')

        print(f'Epoch {epoch:03d} | Loss: {avg_loss:.4f} | '
              f'LR: {cur_lr:.2e} | Best: {best_loss:.4f}')

        # ── Early Stopping ──────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter   = 0
            torch.save(model.state_dict(), CFG['save_best'])
            print(f'  ✅ Best Loss 저장 ({best_loss:.4f})')
        else:
            counter += 1
            if counter >= CFG['patience']:
                print(f'\nEarly Stopping at epoch {epoch}')
                break

        # ── CSV 기록 ────────────────────────────
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f'{avg_loss:.6f}',
                f'{cur_lr:.8f}',
                f'{best_loss:.6f}',
                *[f'{iou_vals[c]:.4f}' for c in range(NUM_CLASSES)],
                f'{iou_vals.mean():.4f}',
                f'{iou_vals[1:].mean():.4f}',
            ])

        # 메모리 정리
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ══ 학습 완료 ══
    elapsed = (time.time() - t_start) / 60
    print(f'\n✅ 학습 완료 ({elapsed:.1f}분)')
    print(f'   Best Loss      : {best_loss:.4f}')
    print(f'   Best 전경 mIoU : {best_miou*100:.1f}%')

    # ── 최종 평가 ──
    print('\n[최종 평가]')
    model.load_state_dict(torch.load(CFG['save_best'], map_location=device))
    final_iou = compute_miou(model, val_loader, device)
    print('\n' + '═'*40)
    print('  최종 클래스별 IoU')
    for c, name in enumerate(CLASS_NAMES):
        print(f'  {name:<15}: {final_iou[c]*100:5.1f}%')
    fg_miou = final_iou[1:].mean().item()
    print(f'  전경 mIoU      : {fg_miou*100:5.1f}%')
    print(f'  전체 mIoU      : {final_iou.mean()*100:5.1f}%')
    print('═'*40)

    # ── 학습 곡선 저장 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(loss_hist, color='royalblue', lw=1.5)
    axes[0].set(xlabel='Epoch', ylabel='Loss',
                title=f'FPVNet Training Loss (best={best_loss:.4f})')
    axes[0].grid(alpha=0.3)

    if miou_hist:
        ep_arr, m_arr = zip(*miou_hist)
        axes[1].plot(ep_arr, m_arr,
                     color='tomato', lw=1.5, marker='o', ms=4)
        axes[1].axhline(50, color='green', ls='--', alpha=0.6,
                        label='목표 50%')
        axes[1].set(xlabel='Epoch', ylabel='Foreground mIoU (%)',
                    title='FPVNet Foreground mIoU (Road+Veh+Ped+Static)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(CFG['result_dir'], 'loss_curve_v3.png')
    plt.savefig(curve_path, dpi=130)
    plt.close()
    print(f'  📈 학습 곡선: {curve_path}')

    # ── train_info 저장 ──
    info = {
        'model'          : 'FPVNet (EfficientNet-B2 + FPN + GeomProj)',
        'approach'       : 'Direct depth estimation + Geometric 3D projection (NOT LSS)',
        'dataset'        : 'NuScenes mini (front camera only)',
        'num_classes'    : NUM_CLASSES,
        'class_names'    : CLASS_NAMES,
        'epochs_trained' : len(loss_hist),
        'best_loss'      : round(best_loss, 6),
        'best_fg_miou'   : round(best_miou, 4),
        'final_iou'      : {name: round(final_iou[c].item(), 4)
                            for c, name in enumerate(CLASS_NAMES)},
        'final_fg_miou'  : round(fg_miou, 4),
        'elapsed_min'    : round(elapsed, 1),
        'voxel_grid'     : f'{CFG["xbound"]}, {CFG["ybound"]}, {CFG["zbound"]}',
    }
    info_path = os.path.join(CFG['result_dir'], 'train_info_v3.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # ── 최종 BEV 시각화 저장 ──
    with torch.no_grad():
        sample_imgs, sample_Ks, sample_gt = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        sample_Ks   = sample_Ks.float().to(device)
        pred_vox, _, _ = model(sample_imgs, sample_Ks)
    pred_np = pred_vox[0].argmax(dim=0).cpu().numpy()
    gt_np   = sample_gt[0].numpy()
    final_vis_path = os.path.join(CFG['result_dir'], 'bev_final.jpg')
    save_bev_vis(gt_np, pred_np, final_vis_path, epoch='Final')
    print(f'  🖼️  최종 BEV 시각화: {final_vis_path}')

    # ── GitHub 최종 push ──
    git_push_results(
        f'FPVNet 학습완료: Loss={best_loss:.4f}, fg_mIoU={fg_miou*100:.1f}%',
        repo_root='..')


if __name__ == '__main__':
    main()
