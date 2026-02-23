"""
train_fastocc_v2.py — FastOcc 개선 학습 (mIoU 50%+ 목표)
=========================================================
v1 대비 핵심 개선:
  1. Lovász-Softmax Loss (mIoU 직접 최적화)
  2. 극한 클래스 가중치 (불균형 해소: Ped×600, Veh×120)
  3. dataset_nuscenes_v4 (LiDAR 기반 GT + 박스 팽창)
  4. Two-phase 학습: Phase1=전체학습, Phase2=희소클래스 집중
  5. 최고 mIoU 시 즉시 git push (50%+ 달성 목표)
  6. .gitignore .pth 제외 처리

실행:
  cd C:/AI_Project/code3
  python train_fastocc_v2.py
"""

import os, sys, csv, json, time, gc, subprocess
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
from dataset_nuscenes_v4 import NuScenesV4Dataset, NUM_CLASSES, CLASS_NAMES
from lovasz_losses import LovaszSoftmax

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
CFG = dict(
    # 데이터
    data_root   = '../data/sets/nuscenesmini',
    version     = 'v1.0-mini',
    box_dilate  = 2,           # 박스 팽창 (희소 클래스 voxel 증가)
    # 복셀 범위
    xbound      = (-50., 50., .5),
    ybound      = (-50., 50., .5),
    zbound      = (-2.,  6.,  .5),
    # 이미지
    img_h       = 256,
    img_w       = 704,
    # 모델
    fpn_ch      = 128,
    c2h_ch      = 64,
    num_cams    = 6,
    # 학습
    epochs      = 200,
    patience    = 40,
    batch_size  = 1,
    accum_steps = 8,
    lr          = 2e-4,
    wd          = 1e-4,
    eval_every  = 5,
    num_workers = 0,
    # Phase2: 희소클래스 집중 (epoch > phase2_start 시 가중치 교체)
    phase2_start = 60,
    # 저장
    result_dir  = 'results_v4',
    ckpt_best   = 'best_fastocc_v2.pth',
    ckpt_miou   = 'best_fastocc_v2_miou.pth',
    git_branch  = 'feature/portfolio-3d-semantic',
)

# ── V4 GT 분포: Free(84%) Road(11.5%) Veh(1.65%) Ped(1.34%) Stat(1.33%)
# ── Phase1 가중치: 균형 학습 (비율 역수 기반)
#   Free=0.5, Road=5, Veh=84/1.65=51→60, Ped=84/1.34=63→80, Stat=80
WEIGHTS_PHASE1 = torch.tensor([0.5,  5.0,  60.0,  80.0,  70.0])

# ── Phase2 가중치: 희소 객체 클래스 집중 강조
WEIGHTS_PHASE2 = torch.tensor([0.3,  3.0, 100.0, 150.0, 120.0])

# ── BEV 색상 ──────────────────────────────────────
BEV_COLORS = {
    0: (20,  20,  20),
    1: (100, 100, 100),
    2: (0,   120, 255),
    3: (220,  50,  50),
    4: (0,   200, 200),
}


# ══════════════════════════════════════════════════════
# Loss (Lovász + CE 혼합)
# ══════════════════════════════════════════════════════
def build_criterion(weights, device):
    return LovaszSoftmax(
        alpha=0.7,           # 70% Lovász + 30% CE
        weights=weights.to(device),
        classes='present',
    )


# ══════════════════════════════════════════════════════
# mIoU
# ══════════════════════════════════════════════════════
@torch.no_grad()
def calc_miou(model, loader, device):
    model.eval()
    tp = torch.zeros(NUM_CLASSES)
    fp = torch.zeros(NUM_CLASSES)
    fn = torch.zeros(NUM_CLASSES)
    for imgs, Ks, s2e, gt in loader:
        imgs = imgs.to(device, non_blocking=True)
        Ks   = Ks.float().to(device)
        s2e  = s2e.float().to(device)
        gt   = gt.long().to(device)
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logits = model(imgs, Ks, s2e)
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            p = pred==c; g = gt==c
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
    def to_rgb(vol):
        bev = vol.max(0)
        rgb = np.zeros((*bev.shape, 3), np.uint8)
        for cid, col in BEV_COLORS.items():
            rgb[bev == cid] = col
        return rgb
    gt_img   = to_rgb(gt_np)
    pred_img = to_rgb(pred_np)
    vis  = np.concatenate([gt_img, pred_img], axis=1)
    vis  = cv2.resize(vis, (vis.shape[1]*3, vis.shape[0]*3),
                      interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis, f'GT  [ep {epoch}]',   (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.putText(vis, 'Pred', (vis.shape[1]//2+8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ══════════════════════════════════════════════════════
# git push
# ══════════════════════════════════════════════════════
def git_push(msg, repo='..'):
    try:
        subprocess.run(['git', 'add',
                        'code3/results_v4/',
                        'code3/best_fastocc_v2.pth',
                        'code3/best_fastocc_v2_miou.pth',
                        'README.md'],
                       cwd=repo, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg,
                        '--author', 'FastOcc Bot <noreply@github.com>'],
                       cwd=repo, check=True, capture_output=True)
        subprocess.run(['git', 'push', 'origin', CFG['git_branch']],
                       cwd=repo, check=True, capture_output=True)
        print(f'  ✅ git push: {msg}')
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b'').decode()[:120]
        print(f'  ⚠️  git 오류: {err}')


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
    print('\n[데이터 로드] V4 (LiDAR GT + 박스 팽창)')
    tr_ds = NuScenesV4Dataset(CFG['data_root'], CFG['version'],
                               is_train=True,
                               xbound=CFG['xbound'],
                               ybound=CFG['ybound'],
                               zbound=CFG['zbound'],
                               img_h=CFG['img_h'], img_w=CFG['img_w'],
                               box_dilate=CFG['box_dilate'])
    va_ds = NuScenesV4Dataset(CFG['data_root'], CFG['version'],
                               is_train=False,
                               xbound=CFG['xbound'],
                               ybound=CFG['ybound'],
                               zbound=CFG['zbound'],
                               img_h=CFG['img_h'], img_w=CFG['img_w'],
                               box_dilate=CFG['box_dilate'])

    tr_loader = DataLoader(tr_ds, batch_size=CFG['batch_size'],
                            shuffle=True, num_workers=CFG['num_workers'],
                            pin_memory=(device.type=='cuda'))
    va_loader = DataLoader(va_ds, batch_size=CFG['batch_size'],
                            shuffle=False, num_workers=CFG['num_workers'],
                            pin_memory=(device.type=='cuda'))

    # ── GT 클래스 분포 출력 ─────────────────────────
    print('\n[GT 분포 확인]')
    total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for i in range(min(10, len(tr_ds))):
        _, _, _, gt = tr_ds[i]
        for c in range(NUM_CLASSES): total[c] += (gt==c).sum()
    tot = total.sum().item()
    for c, nm in enumerate(CLASS_NAMES):
        n = total[c].item()
        print(f'  {nm:<16}: {n:>10,}  ({n/tot*100:.3f}%)')

    # ── 모델 ──────────────────────────────────────────
    print('\n[모델 초기화] FastOcc 6-Cam + 개선 학습')
    model = FastOcc(
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'],
        num_classes=NUM_CLASSES,
        fpn_ch=CFG['fpn_ch'], c2h_ch=CFG['c2h_ch'],
        img_h=CFG['img_h'], img_w=CFG['img_w'],
        num_cams=CFG['num_cams'],
    ).to(device)

    # v1 체크포인트 로드 (Fine-tune 시작)
    v1_ckpt = 'best_fastocc_miou.pth'
    if os.path.exists(v1_ckpt):
        model.load_state_dict(torch.load(v1_ckpt, map_location=device))
        print(f'  ✅ v1 체크포인트 로드: {v1_ckpt}')
    else:
        print(f'  ℹ️  v1 체크포인트 없음 — 새로 학습')

    # ── Loss / Optimizer / Scheduler ──────────────────
    criterion = build_criterion(WEIGHTS_PHASE1, device)
    optimizer = optim.AdamW(model.parameters(),
                             lr=CFG['lr'], weight_decay=CFG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    log_path = os.path.join(CFG['result_dir'], 'train_log_v2.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'phase', 'loss', 'lr', 'best_loss'] +
            [f'iou_{n}' for n in CLASS_NAMES] +
            ['miou_all', 'miou_fg'])

    best_loss = float('inf')
    best_miou = 0.0
    no_improve = 0
    loss_hist = []
    miou_hist = []

    print(f'\n학습 시작 v2 | max {CFG["epochs"]}ep | '
          f'Phase2 @ ep {CFG["phase2_start"]} | 목표 mIoU ≥ 50%\n')
    t0 = time.time()

    for epoch in range(1, CFG['epochs'] + 1):
        # ── Phase 전환 ─────────────────────────────
        phase = 1 if epoch <= CFG['phase2_start'] else 2
        if epoch == CFG['phase2_start'] + 1:
            criterion = build_criterion(WEIGHTS_PHASE2, device)
            # Phase2: 분류기 레이어만 높은 LR
            for pg in optimizer.param_groups:
                pg['lr'] = CFG['lr'] * 2.0
            print(f'\n  🔄 Phase2 시작 @ epoch {epoch} — '
                  f'극한 가중치 {WEIGHTS_PHASE2.tolist()} 적용\n')

        model.train()
        epoch_loss = 0.
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(tr_loader,
                    desc=f'Ep{epoch:03d}[P{phase}]/{CFG["epochs"]}',
                    leave=True)

        for step, (imgs, Ks, s2e, gt) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            Ks   = Ks.float().to(device)
            s2e  = s2e.float().to(device)
            gt   = gt.long().to(device)

            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
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
            pbar.set_postfix(loss=f'{loss.item()*CFG["accum_steps"]:.4f}',
                             phase=phase)

        scheduler.step(epoch)
        avg_loss = epoch_loss / len(tr_loader)
        cur_lr   = optimizer.param_groups[0]['lr']
        loss_hist.append(avg_loss)

        # ── 평가 ────────────────────────────────────
        iou_vals = torch.zeros(NUM_CLASSES)
        if epoch % CFG['eval_every'] == 0:
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

            iou_vals = calc_miou(model, va_loader, device)
            miou_all = iou_vals.mean().item()
            miou_fg  = iou_vals[1:].mean().item()
            miou_hist.append((epoch, miou_fg * 100))

            print(f'\n  📊 mIoU @ Epoch {epoch} [Phase {phase}]')
            for c, nm in enumerate(CLASS_NAMES):
                mark = ' ✅' if iou_vals[c] >= 0.5 else ''
                print(f'     {nm:<16}: {iou_vals[c]*100:5.1f}%{mark}')
            print(f'     {"전경 mIoU":<16}: {miou_fg*100:5.1f}%'
                  f'  전체: {miou_all*100:5.1f}%')

            # BEV 시각화
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

                if miou_fg >= 0.50:
                    git_push(
                        f'🎯 FastOcc v2 mIoU {miou_fg*100:.1f}% 달성! (epoch {epoch})',
                        repo='..')

            if epoch % 10 == 0:
                git_push(f'FastOcc v2 중간 (ep{epoch}, '
                          f'loss={avg_loss:.4f}, mIoU={miou_fg*100:.1f}%)',
                          repo='..')

        print(f'Ep{epoch:03d}[P{phase}] | Loss={avg_loss:.4f} | '
              f'LR={cur_lr:.2e} | BestLoss={best_loss:.4f}')

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), CFG['ckpt_best'])
        else:
            no_improve += 1
            if no_improve >= CFG['patience']:
                print(f'\nEarly Stopping @ epoch {epoch}')
                break

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, phase, f'{avg_loss:.6f}', f'{cur_lr:.8f}',
                 f'{best_loss:.6f}'] +
                [f'{iou_vals[c]:.4f}' for c in range(NUM_CLASSES)] +
                [f'{iou_vals.mean():.4f}', f'{iou_vals[1:].mean():.4f}'])

        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    # ══ 완료 ══════════════════════════════════════════
    elapsed = (time.time() - t0) / 60
    print(f'\n✅ 학습 완료 ({elapsed:.1f}분)')
    print(f'   Best Loss : {best_loss:.4f}')
    print(f'   Best mIoU : {best_miou*100:.1f}%')

    # 최종 평가
    ckpt = CFG['ckpt_miou'] if os.path.exists(CFG['ckpt_miou']) else CFG['ckpt_best']
    model.load_state_dict(torch.load(ckpt, map_location=device))
    final_iou = calc_miou(model, va_loader, device)

    print('\n' + '═'*44)
    print('  최종 클래스별 IoU (FastOcc v2)')
    for c, nm in enumerate(CLASS_NAMES):
        mark = ' ✅' if final_iou[c] >= 0.5 else ''
        print(f'  {nm:<16}: {final_iou[c]*100:5.1f}%{mark}')
    fg = final_iou[1:].mean().item()
    print(f'  {"전경 mIoU":<16}: {fg*100:5.1f}%')
    print('═'*44)

    # 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(loss_hist, lw=1.5, color='royalblue')
    if CFG['phase2_start'] < len(loss_hist):
        ax1.axvline(CFG['phase2_start'], ls='--', color='orange',
                    alpha=.7, label=f'Phase2 @{CFG["phase2_start"]}')
        ax1.legend()
    ax1.set(xlabel='Epoch', ylabel='Loss',
            title=f'FastOcc v2 Loss (best={best_loss:.4f})')
    ax1.grid(alpha=.3)

    if miou_hist:
        ep, mi = zip(*miou_hist)
        ax2.plot(ep, mi, lw=1.5, color='tomato', marker='o', ms=4)
        ax2.axhline(50, ls='--', color='green', alpha=.6, label='목표 50%')
        if CFG['phase2_start'] < max(ep):
            ax2.axvline(CFG['phase2_start'], ls='--', color='orange',
                        alpha=.7, label=f'Phase2')
        ax2.set(xlabel='Epoch', ylabel='Foreground mIoU (%)',
                title='FastOcc v2 Foreground mIoU')
        ax2.legend(); ax2.grid(alpha=.3)

    plt.tight_layout()
    curve_path = os.path.join(CFG['result_dir'], 'loss_curve_v2.png')
    plt.savefig(curve_path, dpi=130); plt.close()

    # 최종 BEV
    with torch.no_grad():
        fv_imgs, fv_Ks, fv_s2e, fv_gt = next(iter(va_loader))
        fv_imgs = fv_imgs.to(device)
        fv_Ks   = fv_Ks.float().to(device)
        fv_s2e  = fv_s2e.float().to(device)
        pred_v  = model(fv_imgs, fv_Ks, fv_s2e)
    bev_vis(fv_gt[0].numpy(), pred_v[0].argmax(0).cpu().numpy(),
            'Final', os.path.join(CFG['result_dir'], 'bev_final_v2.jpg'))

    # JSON
    info = dict(
        model='FastOcc v2 (LovászLoss + ExtremeWeights + LiDARGT)',
        epochs_trained=len(loss_hist),
        best_loss=round(best_loss, 6),
        best_fg_miou=round(best_miou, 4),
        final_iou={nm: round(final_iou[c].item(), 4)
                   for c, nm in enumerate(CLASS_NAMES)},
        final_fg_miou=round(fg, 4),
        elapsed_min=round(elapsed, 1),
    )
    with open(os.path.join(CFG['result_dir'], 'train_info_v2.json'),
              'w', encoding='utf-8') as fj:
        json.dump(info, fj, indent=2, ensure_ascii=False)

    git_push(f'FastOcc v2 완료: fg_mIoU={fg*100:.1f}%', repo='..')


if __name__ == '__main__':
    main()
