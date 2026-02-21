"""
train_fastocc.py — FastOcc 학습 스크립트
=========================================
목표: mIoU ≥ 50%  (5클래스: Free/Road/Vehicle/Pedestrian/StaticObst)

전략:
  1. EfficientNet-B2 (ImageNet pretrained) 백본 고정 없이 전체 fine-tune
  2. Focal Loss + 클래스 가중치 (희소 클래스 강조)
  3. CosineAnnealingWarmRestarts 스케줄러
  4. Mixed Precision (FP16) + Gradient Accumulation
  5. 5 epoch마다 BEV JPG 저장 + GitHub 자동 push
  6. mIoU 50% 달성 시 즉시 push

실행:
  cd C:/AI_Project/code3
  python train_fastocc.py
"""

import os, sys, csv, json, time, math, gc, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from model_fastocc import FastOcc
from dataset_nuscenes_v3 import NuScenesV3Dataset, NUM_CLASSES, CLASS_NAMES

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
CFG = dict(
    # 데이터
    data_root   = '../data/sets/nuscenesmini',
    version     = 'v1.0-mini',
    # 복셀 범위 (6카메라 360° → 전/후/좌/우 50m)
    xbound      = (-50., 50., .5),
    ybound      = (-50., 50., .5),
    zbound      = (-2.,  6.,  .5),   # nZ = 16
    # 이미지 (nuScenes 6-cam 표준 크기)
    img_h       = 256,
    img_w       = 704,
    # 모델
    fpn_ch      = 128,
    c2h_ch      = 64,
    num_cams    = 6,
    # 학습
    epochs      = 150,
    patience    = 30,
    batch_size  = 1,           # 6카메라 × 256×704 → VRAM 고려
    accum_steps = 8,           # effective batch = 8
    lr          = 2e-4,
    wd          = 1e-4,
    eval_every  = 5,
    num_workers = 0,           # Windows 안정성
    # 저장
    result_dir  = 'results_v3',
    ckpt_best   = 'best_fastocc.pth',
    ckpt_miou   = 'best_fastocc_miou.pth',
    # Git
    git_branch  = 'feature/portfolio-3d-semantic',
)

# ── 클래스 가중치 (희소 클래스 강조) ──────────────────
#   Free=1, Road=3, Vehicle=12, Ped=20, StaticObst=8
CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 12.0, 20.0, 8.0])

# ── BEV 시각화 색상 ───────────────────────────────────
BEV_COLORS = {
    0: (20,  20,  20),    # Free — 검정
    1: (100, 100, 100),   # Road — 회색
    2: (0,   120, 255),   # Vehicle — 파랑
    3: (220,  50,  50),   # Pedestrian — 빨강
    4: (0,   200, 200),   # StaticObst — 청록
}


# ══════════════════════════════════════════════════════
# Focal Loss
# ══════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=CLASS_WEIGHTS):
        super().__init__()
        self.gamma   = gamma
        self.weights = weights

    def forward(self, pred, target):
        # pred: (B, C, nZ, nX, nY)  target: (B, nZ, nX, nY) long
        w  = self.weights.to(pred.device)
        ce = F.cross_entropy(pred, target, weight=w, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ══════════════════════════════════════════════════════
# mIoU 계산
# ══════════════════════════════════════════════════════
@torch.no_grad()
def calc_miou(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for imgs, Ks, s2e, gt in loader:
        imgs = imgs.to(device, non_blocking=True)  # (B,6,3,H,W)
        Ks   = Ks.float().to(device)               # (B,6,3,3)
        s2e  = s2e.float().to(device)              # (B,6,4,4)
        gt   = gt.long().to(device)                # (B,nZ,nX,nY)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(imgs, Ks, s2e)

        pred = logits.argmax(1)      # (B, nZ, nX, nY)

        for c in range(num_classes):
            p = pred == c; g = gt == c
            tp[c] += (p & g).sum().cpu()
            fp[c] += (p & ~g).sum().cpu()
            fn[c] += (~p & g).sum().cpu()

    iou = tp / (tp + fp + fn + 1e-6)
    model.train()
    return iou


# ══════════════════════════════════════════════════════
# BEV 시각화
# ══════════════════════════════════════════════════════
def bev_vis(gt_np, pred_np, epoch, out_path):
    """z-max projection → GT|Pred 나란히 저장"""
    def to_rgb(vol):
        bev = vol.max(0)   # (nX, nY)
        rgb = np.zeros((*bev.shape, 3), np.uint8)
        for cid, col in BEV_COLORS.items():
            rgb[bev == cid] = col
        return rgb

    gt_img   = to_rgb(gt_np)
    pred_img = to_rgb(pred_np)
    vis      = np.concatenate([gt_img, pred_img], axis=1)
    scale    = 3
    vis      = cv2.resize(vis, (vis.shape[1]*scale, vis.shape[0]*scale),
                          interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis, f'GT  [epoch {epoch}]', (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.putText(vis, 'Pred', (vis.shape[1]//2+8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ══════════════════════════════════════════════════════
# GitHub 자동 push
# ══════════════════════════════════════════════════════
def git_push(msg, repo='..'):
    try:
        subprocess.run(['git', 'add',
                        'code3/results_v3/',
                        'code3/best_fastocc.pth',
                        'code3/best_fastocc_miou.pth',
                        'README.md'],
                       cwd=repo, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg,
                        '--author', 'FastOcc Bot <noreply@github.com>'],
                       cwd=repo, check=True, capture_output=True)
        subprocess.run(['git', 'push', 'origin', CFG['git_branch']],
                       cwd=repo, check=True, capture_output=True)
        print(f'  ✅ git push: {msg}')
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode()[:120] if e.stderr else ''
        print(f'  ⚠️  git 오류 (학습 계속): {err}')


# ══════════════════════════════════════════════════════
# 메인 학습
# ══════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  장치: {device}')
    if device.type == 'cuda':
        print(f'   GPU : {torch.cuda.get_device_name(0)}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    os.makedirs(CFG['result_dir'], exist_ok=True)

    # ── 데이터셋 ──────────────────────────────────────
    print('\n[데이터 로드]')
    tr_ds = NuScenesV3Dataset(CFG['data_root'], CFG['version'],
                               is_train=True,
                               xbound=CFG['xbound'],
                               ybound=CFG['ybound'],
                               zbound=CFG['zbound'],
                               img_h=CFG['img_h'], img_w=CFG['img_w'])
    va_ds = NuScenesV3Dataset(CFG['data_root'], CFG['version'],
                               is_train=False,
                               xbound=CFG['xbound'],
                               ybound=CFG['ybound'],
                               zbound=CFG['zbound'],
                               img_h=CFG['img_h'], img_w=CFG['img_w'])

    tr_loader = DataLoader(tr_ds, batch_size=CFG['batch_size'],
                            shuffle=True, num_workers=CFG['num_workers'],
                            pin_memory=(device.type == 'cuda'))
    va_loader = DataLoader(va_ds, batch_size=CFG['batch_size'],
                            shuffle=False, num_workers=CFG['num_workers'],
                            pin_memory=(device.type == 'cuda'))

    # ── 모델 ──────────────────────────────────────────
    print('\n[모델 초기화] FastOcc 6-Cam Surround (LSS 아님 — 기하학적 복셀 샘플링 + C2H)')
    model = FastOcc(
        xbound=CFG['xbound'],
        ybound=CFG['ybound'],
        zbound=CFG['zbound'],
        num_classes=NUM_CLASSES,
        fpn_ch=CFG['fpn_ch'],
        c2h_ch=CFG['c2h_ch'],
        img_h=CFG['img_h'],
        img_w=CFG['img_w'],
        num_cams=CFG['num_cams'],
    ).to(device)

    # ── 손실·옵티마이저·스케줄러 ──────────────────────
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(),
                             lr=CFG['lr'], weight_decay=CFG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── 로그 파일 ─────────────────────────────────────
    log_path = os.path.join(CFG['result_dir'], 'train_log_fastocc.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'loss', 'lr', 'best_loss'] +
            [f'iou_{n}' for n in CLASS_NAMES] +
            ['miou_all', 'miou_fg'])

    best_loss = float('inf')
    best_miou = 0.0
    no_improve = 0
    loss_hist = []
    miou_hist = []

    print(f'\n학습 시작 | max {CFG["epochs"]}ep | '
          f'eff_batch={CFG["batch_size"]*CFG["accum_steps"]} | '
          f'목표 mIoU ≥ 50%\n')
    t0 = time.time()

    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        epoch_loss = 0.
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(tr_loader,
                    desc=f'Epoch {epoch:03d}/{CFG["epochs"]}',
                    leave=True)

        for step, (imgs, Ks, s2e, gt) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)   # (B,6,3,H,W)
            Ks   = Ks.float().to(device)                # (B,6,3,3)
            s2e  = s2e.float().to(device)               # (B,6,4,4)
            gt   = gt.long().to(device)                 # (B,nZ,nX,nY)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs, Ks, s2e)
                loss   = criterion(logits, gt) / CFG['accum_steps']

            scaler.scale(loss).backward()

            if step % CFG['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * CFG['accum_steps']
            pbar.set_postfix(loss=f'{loss.item()*CFG["accum_steps"]:.4f}')

        scheduler.step(epoch)
        avg_loss = epoch_loss / len(tr_loader)
        cur_lr   = optimizer.param_groups[0]['lr']
        loss_hist.append(avg_loss)

        # ── 평가 (eval_every마다) ───────────────────
        iou_vals = torch.zeros(NUM_CLASSES)
        if epoch % CFG['eval_every'] == 0:
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

            iou_vals  = calc_miou(model, va_loader, device)
            miou_all  = iou_vals.mean().item()
            miou_fg   = iou_vals[1:].mean().item()
            miou_hist.append((epoch, miou_fg * 100))

            print(f'\n  📊 mIoU @ Epoch {epoch}')
            for c, nm in enumerate(CLASS_NAMES):
                mark = ' ✅' if iou_vals[c] >= 0.5 else ''
                print(f'     {nm:<16}: {iou_vals[c]*100:5.1f}%{mark}')
            print(f'     {"전경 mIoU":<16}: {miou_fg*100:5.1f}%'
                  f'  전체: {miou_all*100:5.1f}%')

            # BEV 시각화 저장
            with torch.no_grad():
                sv_imgs, sv_Ks, sv_s2e, sv_gt = next(iter(va_loader))
                sv_imgs = sv_imgs.to(device)
                sv_Ks   = sv_Ks.float().to(device)
                sv_s2e  = sv_s2e.float().to(device)
                pred_v  = model(sv_imgs, sv_Ks, sv_s2e)
            p_np = pred_v[0].argmax(0).cpu().numpy()
            g_np = sv_gt[0].numpy()
            vis_path = os.path.join(CFG['result_dir'],
                                     f'bev_epoch{epoch:03d}.jpg')
            bev_vis(g_np, p_np, epoch, vis_path)
            print(f'     📸 BEV 저장: {vis_path}')

            # Best mIoU 갱신
            if miou_fg > best_miou:
                best_miou = miou_fg
                torch.save(model.state_dict(), CFG['ckpt_miou'])
                print(f'  🏆 Best mIoU: {miou_fg*100:.1f}%')

                # mIoU 50%+ → 즉시 git push
                if miou_fg >= 0.50:
                    git_push(
                        f'🎯 FastOcc mIoU {miou_fg*100:.1f}% 달성 (epoch {epoch})',
                        repo='..')

            # 10 epoch마다 중간 push (BEV 이미지 업데이트)
            if epoch % 10 == 0:
                git_push(f'FastOcc 중간 결과 (epoch {epoch}, '
                          f'loss={avg_loss:.4f}, mIoU={miou_fg*100:.1f}%)',
                          repo='..')

        print(f'Epoch {epoch:03d} | Loss={avg_loss:.4f} | '
              f'LR={cur_lr:.2e} | BestLoss={best_loss:.4f}')

        # ── Early Stopping ──────────────────────────
        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), CFG['ckpt_best'])
            print(f'  ✅ Best Loss 저장: {best_loss:.4f}')
        else:
            no_improve += 1
            if no_improve >= CFG['patience']:
                print(f'\nEarly Stopping @ epoch {epoch}')
                break

        # CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, f'{avg_loss:.6f}', f'{cur_lr:.8f}',
                 f'{best_loss:.6f}'] +
                [f'{iou_vals[c]:.4f}' for c in range(NUM_CLASSES)] +
                [f'{iou_vals.mean():.4f}', f'{iou_vals[1:].mean():.4f}'])

        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    # ══ 학습 완료 ══════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print(f'\n✅ 학습 완료 ({elapsed:.1f}분)')
    print(f'   Best Loss : {best_loss:.4f}')
    print(f'   Best mIoU : {best_miou*100:.1f}%')

    # ── 최종 평가 ─────────────────────────────────────
    ckpt = CFG['ckpt_miou'] if os.path.exists(CFG['ckpt_miou']) else CFG['ckpt_best']
    model.load_state_dict(torch.load(ckpt, map_location=device))
    final_iou = calc_miou(model, va_loader, device)

    print('\n' + '═'*42)
    print('  최종 클래스별 IoU (FastOcc)')
    for c, nm in enumerate(CLASS_NAMES):
        mark = ' ✅' if final_iou[c] >= 0.5 else ''
        print(f'  {nm:<16}: {final_iou[c]*100:5.1f}%{mark}')
    fg = final_iou[1:].mean().item()
    print(f'  {"전경 mIoU":<16}: {fg*100:5.1f}%')
    print(f'  {"전체 mIoU":<16}: {final_iou.mean()*100:5.1f}%')
    print('═'*42)

    # ── 학습 곡선 ─────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(loss_hist, lw=1.5, color='royalblue')
    ax1.set(xlabel='Epoch', ylabel='Loss',
            title=f'FastOcc Training Loss (best={best_loss:.4f})')
    ax1.grid(alpha=.3)

    if miou_hist:
        ep, mi = zip(*miou_hist)
        ax2.plot(ep, mi, lw=1.5, color='tomato', marker='o', ms=4)
        ax2.axhline(50, ls='--', color='green', alpha=.6, label='목표 50%')
        ax2.set(xlabel='Epoch', ylabel='Foreground mIoU (%)',
                title='FastOcc Foreground mIoU')
        ax2.legend(); ax2.grid(alpha=.3)

    plt.tight_layout()
    curve_path = os.path.join(CFG['result_dir'], 'loss_curve_fastocc.png')
    plt.savefig(curve_path, dpi=130); plt.close()

    # ── 최종 BEV 시각화 ───────────────────────────────
    with torch.no_grad():
        fv_imgs, fv_Ks, fv_s2e, fv_gt = next(iter(va_loader))
        fv_imgs = fv_imgs.to(device)
        fv_Ks   = fv_Ks.float().to(device)
        fv_s2e  = fv_s2e.float().to(device)
        pred_v  = model(fv_imgs, fv_Ks, fv_s2e)
    p_np = pred_v[0].argmax(0).cpu().numpy()
    g_np = fv_gt[0].numpy()
    bev_vis(g_np, p_np, 'Final', os.path.join(CFG['result_dir'], 'bev_final.jpg'))

    # ── train_info 저장 ────────────────────────────────
    info = dict(
        model='FastOcc (EfficientNet-B2 + FPN + VoxelQuerySampler + C2H)',
        approach='Geometric voxel sampling + Channel-to-Height (NOT LSS)',
        epochs_trained=len(loss_hist),
        best_loss=round(best_loss, 6),
        best_fg_miou=round(best_miou, 4),
        final_iou={nm: round(final_iou[c].item(), 4)
                   for c, nm in enumerate(CLASS_NAMES)},
        final_fg_miou=round(fg, 4),
        elapsed_min=round(elapsed, 1),
    )
    with open(os.path.join(CFG['result_dir'], 'train_info_fastocc.json'),
              'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # ── 최종 git push ──────────────────────────────────
    git_push(f'FastOcc 학습 완료: Loss={best_loss:.4f}, '
              f'fg_mIoU={fg*100:.1f}%', repo='..')


if __name__ == '__main__':
    main()
