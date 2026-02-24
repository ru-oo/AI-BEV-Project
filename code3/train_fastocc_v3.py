"""
train_fastocc_v3.py — FastOcc 2단계 커리큘럼 학습 (mIoU 50%+ 최종 목표)
=========================================================================
v2 → v3 핵심 개선 (6가지 병목 완전 해소):

  Fix 2: num_workers=4 (GPU Starvation 제거, 병렬 데이터 로딩)
          gc.collect() 내부 루프 완전 제거 (CUDA 메모리 할당자 보호)

  Fix 3: model_fastocc.py grid_sample FP32 캐스팅 (별도 파일 수정)
          → NaN/언더플로우 방지, 희소 클래스 gradient 복원

  Fix 4: Gradient Accumulation 마지막 배치 처리 수정
          (step % accum == 0 or step == len(loader))

  Fix 5: box_dilate=0 (dataset_nuscenes_v5 기본값)
          → Lovász Loss FP 페널티 방지, 정밀 경계 학습

  Fix 6: 2단계 커리큘럼 학습
    Phase 1: Binary Pre-training (50에포크)
      - Free(0) vs Occupied(1) 이진 분류
      - 3D 기하학적 투영 구조 선학습
      - 가벼운 CE Loss
    Phase 2: 5-class Fine-tuning (150에포크)
      - Phase1 가중치(backbone+sampler+c2h) 로드
      - 5클래스 분류기 새로 초기화
      - Lovász + CE 혼합 손실, 극한 클래스 가중치

실행:
  cd C:/AI_Project/code3
  python train_fastocc_v3.py
"""

import os
import sys
import csv
import json
import time
import gc
import subprocess
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
from dataset_nuscenes_v5 import NuScenesV5Dataset, NUM_CLASSES, CLASS_NAMES
from lovasz_losses import LovaszSoftmax

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
CFG = dict(
    # 데이터
    data_root   = '../data/sets/nuscenesmini',
    version     = 'v1.0-mini',
    box_dilate  = 0,            # ★ Fix 5: Lovász 호환 (v4의 2 → 0)
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
    # 학습 공통
    batch_size  = 1,
    accum_steps = 8,
    wd          = 1e-4,
    # ★ Fix 2: num_workers=4 (GPU Starvation 제거)
    num_workers = 4,
    # Phase 1: Binary Pre-training
    ph1_epochs  = 50,
    ph1_lr      = 2e-4,
    ph1_ckpt    = 'binary_pretrain_v3.pth',
    # Phase 2: 5-class Fine-tuning
    ph2_epochs  = 150,
    ph2_lr      = 1e-4,
    ph2_lr_boost= 3e-4,    # 분류기 헤드 LR (나머지보다 3배 높게)
    ph2_patience= 40,
    # 저장
    result_dir  = 'results_v5',
    ckpt_best   = 'best_fastocc_v3.pth',
    ckpt_miou   = 'best_fastocc_v3_miou.pth',
    git_branch  = 'feature/portfolio-3d-semantic',
)

# ── Phase 1 Binary 가중치: Free(84%) vs Occupied(16%)
WEIGHTS_BINARY = torch.tensor([0.1, 0.9])

# ── Phase 2 5-class 가중치 (box_dilate=0 기준 역비율)
# Free(84%):0.2  Road(11.5%):3  Veh(~0.5%):60  Ped(~0.05%):300  Stat(~0.08%):200
WEIGHTS_5CLS = torch.tensor([0.2, 3.0, 60.0, 300.0, 200.0])

# ── BEV 색상 ──────────────────────────────────────────
BEV_COLORS = {
    0: (20,  20,  20),   # Free: 검정
    1: (100, 100, 100),  # Road: 회색
    2: (0,   120, 255),  # Vehicle: 파랑
    3: (220,  50,  50),  # Pedestrian: 빨강
    4: (0,   200, 200),  # StaticObst: 청록
}


# ══════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════
def build_5cls_criterion(device):
    """Lovász 70% + CE 30% (Phase 2 전용)"""
    return LovaszSoftmax(
        alpha=0.8,                          # 80% Lovász + 20% CE
        weights=WEIGHTS_5CLS.to(device),
        classes='present',
    )


@torch.no_grad()
def calc_miou_5cls(model, loader, device):
    """5-class mIoU 계산"""
    model.eval()
    tp = torch.zeros(NUM_CLASSES)
    fp = torch.zeros(NUM_CLASSES)
    fn = torch.zeros(NUM_CLASSES)
    for imgs, Ks, s2e, gt in loader:
        imgs = imgs.to(device, non_blocking=True)
        Ks   = Ks.float().to(device)
        s2e  = s2e.float().to(device)
        gt   = gt.long().to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(imgs, Ks, s2e)
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            p = pred == c; g = gt == c
            tp[c] += (p & g).sum().cpu()
            fp[c] += (p & ~g).sum().cpu()
            fn[c] += (~p & g).sum().cpu()
    iou = tp / (tp + fp + fn + 1e-6)
    model.train()
    return iou


@torch.no_grad()
def calc_acc_binary(model, loader, device):
    """Phase 1 Binary 정확도 (간단 검증)"""
    model.eval()
    correct = total = 0
    tp1 = fp1 = fn1 = 0
    for imgs, Ks, s2e, gt in loader:
        imgs = imgs.to(device, non_blocking=True)
        Ks   = Ks.float().to(device)
        s2e  = s2e.float().to(device)
        gt_bin = (gt > 0).long().to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(imgs, Ks, s2e)
        pred = logits.argmax(1)
        correct += (pred == gt_bin).sum().cpu().item()
        total   += gt_bin.numel()
        tp1 += ((pred == 1) & (gt_bin == 1)).sum().cpu().item()
        fp1 += ((pred == 1) & (gt_bin == 0)).sum().cpu().item()
        fn1 += ((pred == 0) & (gt_bin == 1)).sum().cpu().item()
    occ_iou = tp1 / (tp1 + fp1 + fn1 + 1e-6)
    model.train()
    return correct / (total + 1e-6), occ_iou


def bev_vis(gt_np, pred_np, label, out_path):
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
    cv2.putText(vis, f'GT [{label}]',  (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.putText(vis, 'Pred', (vis.shape[1]//2+8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def git_push(msg, repo='..'):
    try:
        subprocess.run(
            ['git', 'add',
             'code3/results_v5/',
             'code3/best_fastocc_v3.pth',
             'code3/best_fastocc_v3_miou.pth',
             'code3/binary_pretrain_v3.pth',
             'README.md'],
            cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', msg,
             '--author', 'FastOcc Bot <noreply@github.com>'],
            cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ['git', 'push', 'origin', CFG['git_branch']],
            cwd=repo, check=True, capture_output=True)
        print(f'  ✅ git push: {msg}')
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b'').decode()[:200]
        print(f'  ⚠️  git 오류: {err}')


# ══════════════════════════════════════════════════════
# Phase 1: Binary Pre-training
# ══════════════════════════════════════════════════════
def run_phase1(model, tr_loader, va_loader, device):
    """
    Free vs Occupied 이진 분류 50에포크
    → 3D 기하학적 투영 구조 사전 학습
    """
    print('\n' + '═'*50)
    print('  Phase 1: Binary Pre-training (Free vs Occupied)')
    print('  목표: 3D 기하학 투영 구조 사전 학습')
    print('═'*50)

    criterion = nn.CrossEntropyLoss(
        weight=WEIGHTS_BINARY.to(device))
    optimizer = optim.AdamW(model.parameters(),
                             lr=CFG['ph1_lr'], weight_decay=CFG['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_occ_iou = 0.0
    log_path = os.path.join(CFG['result_dir'], 'train_log_v3_ph1.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'lr', 'acc', 'occ_iou'])

    t0 = time.time()
    for epoch in range(1, CFG['ph1_epochs'] + 1):
        model.train()
        epoch_loss = 0.
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(tr_loader,
                    desc=f'Ph1 Ep{epoch:03d}/{CFG["ph1_epochs"]}',
                    leave=True)

        for step, (imgs, Ks, s2e, gt) in enumerate(pbar, 1):
            imgs   = imgs.to(device, non_blocking=True)
            Ks     = Ks.float().to(device)
            s2e    = s2e.float().to(device)
            gt_bin = (gt > 0).long().to(device)   # Binary GT

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs, Ks, s2e)
                loss   = criterion(logits, gt_bin) / CFG['accum_steps']

            scaler.scale(loss).backward()

            # ★ Fix 4: 마지막 배치도 반드시 optimizer.step() 수행
            if step % CFG['accum_steps'] == 0 or step == len(tr_loader):
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

        # 5에포크마다 검증
        acc = occ_iou = 0.0
        if epoch % 5 == 0:
            acc, occ_iou = calc_acc_binary(model, va_loader, device)
            print(f'\n  📊 Ph1 Ep{epoch} | Acc: {acc*100:.1f}% | Occ-IoU: {occ_iou*100:.1f}%')

            # BEV 시각화 (2클래스: Free=black, Occupied=gray)
            with torch.no_grad():
                sv = next(iter(va_loader))
                sv_imgs = sv[0].to(device)
                sv_Ks   = sv[1].float().to(device)
                sv_s2e  = sv[2].float().to(device)
                sv_gt   = sv[3]
                pred_v  = model(sv_imgs, sv_Ks, sv_s2e)
            # Binary 예측을 5클래스 색상으로 표시 (0→0, 1→4)
            p_np = pred_v[0].argmax(0).cpu().numpy() * 4   # 0 or 4
            g_np = (sv_gt[0].numpy() > 0).astype(np.int64) * 4
            bev_vis(g_np, p_np, f'Ph1-ep{epoch}',
                    os.path.join(CFG['result_dir'], f'bev_ph1_ep{epoch:03d}.jpg'))

            if occ_iou > best_occ_iou:
                best_occ_iou = occ_iou
                torch.save(model.state_dict(), CFG['ph1_ckpt'])
                print(f'  🏆 Best Occ-IoU: {occ_iou*100:.1f}% → {CFG["ph1_ckpt"]}')

        print(f'Ph1 Ep{epoch:03d} | Loss={avg_loss:.4f} | LR={cur_lr:.2e}')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, f'{avg_loss:.6f}', f'{cur_lr:.8f}',
                 f'{acc:.4f}', f'{occ_iou:.4f}'])

        # ★ Fix 2: gc.collect()는 에포크 끝에만 (루프 내부 금지)
        if epoch % 10 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    print(f'\n✅ Phase 1 완료 ({elapsed:.1f}분) | Best Occ-IoU: {best_occ_iou*100:.1f}%')

    # Best 가중치 로드
    if os.path.exists(CFG['ph1_ckpt']):
        model.load_state_dict(
            torch.load(CFG['ph1_ckpt'], map_location=device, weights_only=True))
        print(f'  최고 가중치 로드: {CFG["ph1_ckpt"]}')

    return model


# ══════════════════════════════════════════════════════
# Phase 2: 5-class Fine-tuning
# ══════════════════════════════════════════════════════
def run_phase2(model, tr_loader, va_loader, device, log_csv_path):
    """
    Phase 1 가중치(backbone+sampler+c2h) 기반 5-class Fine-tuning
    → Lovász + CE, 극한 클래스 가중치
    """
    print('\n' + '═'*50)
    print('  Phase 2: 5-class Fine-tuning')
    print('  가중치: Free=0.2 Road=3 Veh=60 Ped=300 Stat=200')
    print('  손실: Lovász(80%) + CE(20%)')
    print('═'*50)

    criterion = build_5cls_criterion(device)

    # 분류기 헤드에 더 높은 LR 적용 (특징 추출기와 분리)
    head_params  = list(model.classifier.parameters())
    body_params  = [p for n, p in model.named_parameters()
                    if not n.startswith('classifier')]
    optimizer = optim.AdamW([
        {'params': body_params,  'lr': CFG['ph2_lr']},
        {'params': head_params,  'lr': CFG['ph2_lr_boost']},
    ], weight_decay=CFG['wd'])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_loss = float('inf')
    best_miou = 0.0
    no_improve = 0
    loss_hist  = []
    miou_hist  = []

    t0 = time.time()
    with open(log_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'loss', 'lr', 'best_loss'] +
            [f'iou_{n}' for n in CLASS_NAMES] +
            ['miou_all', 'miou_fg'])

    for epoch in range(1, CFG['ph2_epochs'] + 1):
        model.train()
        epoch_loss = 0.
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(tr_loader,
                    desc=f'Ph2 Ep{epoch:03d}/{CFG["ph2_epochs"]}',
                    leave=True)

        for step, (imgs, Ks, s2e, gt) in enumerate(pbar, 1):
            imgs = imgs.to(device, non_blocking=True)
            Ks   = Ks.float().to(device)
            s2e  = s2e.float().to(device)
            gt   = gt.long().to(device)            # 5-class GT

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs, Ks, s2e)
                loss   = criterion(logits, gt) / CFG['accum_steps']

            scaler.scale(loss).backward()

            # ★ Fix 4: 마지막 배치 gradient 버리지 않음
            if step % CFG['accum_steps'] == 0 or step == len(tr_loader):
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

        # ── 5에포크마다 mIoU 평가 ─────────────────────
        iou_vals = torch.zeros(NUM_CLASSES)
        if epoch % 5 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            iou_vals = calc_miou_5cls(model, va_loader, device)
            miou_all = iou_vals.mean().item()
            miou_fg  = iou_vals[1:].mean().item()
            miou_hist.append((epoch, miou_fg * 100))

            print(f'\n  📊 mIoU @ Ph2 Ep{epoch}')
            for c, nm in enumerate(CLASS_NAMES):
                mark = ' ✅' if iou_vals[c] >= 0.5 else ''
                print(f'     {nm:<16}: {iou_vals[c]*100:5.1f}%{mark}')
            print(f'     {"전경 mIoU":<16}: {miou_fg*100:5.1f}%'
                  f'  전체: {miou_all*100:5.1f}%')

            # BEV 시각화
            with torch.no_grad():
                sv = next(iter(va_loader))
                sv_imgs = sv[0].to(device)
                sv_Ks   = sv[1].float().to(device)
                sv_s2e  = sv[2].float().to(device)
                sv_gt   = sv[3]
                pred_v  = model(sv_imgs, sv_Ks, sv_s2e)
            p_np = pred_v[0].argmax(0).cpu().numpy()
            g_np = sv_gt[0].numpy()
            bev_path = os.path.join(CFG['result_dir'],
                                     f'bev_ph2_ep{epoch:03d}.jpg')
            bev_vis(g_np, p_np, f'Ph2-ep{epoch}', bev_path)
            print(f'     📸 BEV: {bev_path}')

            # Best mIoU 갱신
            if miou_fg > best_miou:
                best_miou = miou_fg
                torch.save(model.state_dict(), CFG['ckpt_miou'])
                print(f'  🏆 Best fg-mIoU: {miou_fg*100:.1f}%')

                if miou_fg >= 0.50:
                    git_push(
                        f'🎯 FastOcc v3 mIoU {miou_fg*100:.1f}% 달성! (Ph2 ep{epoch})',
                        repo='..')

            # 10에포크마다 정기 push
            if epoch % 10 == 0:
                git_push(
                    f'FastOcc v3 Ph2 중간 (ep{epoch}, '
                    f'loss={avg_loss:.4f}, fg_mIoU={miou_fg*100:.1f}%)',
                    repo='..')
        else:
            # 평가 안 한 에포크: 10에포크마다 push
            if epoch % 10 == 0:
                git_push(
                    f'FastOcc v3 Ph2 중간 (ep{epoch}, loss={avg_loss:.4f})',
                    repo='..')

        print(f'Ph2 Ep{epoch:03d} | Loss={avg_loss:.4f} | '
              f'LR={cur_lr:.2e} | BestLoss={best_loss:.4f}')

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), CFG['ckpt_best'])
        else:
            no_improve += 1
            if no_improve >= CFG['ph2_patience']:
                print(f'\n  Early Stopping @ Ph2 ep{epoch}')
                break

        with open(log_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, f'{avg_loss:.6f}', f'{cur_lr:.8f}',
                 f'{best_loss:.6f}'] +
                [f'{iou_vals[c]:.4f}' for c in range(NUM_CLASSES)] +
                [f'{iou_vals.mean():.4f}', f'{iou_vals[1:].mean():.4f}'])

        # ★ Fix 2: gc.collect() 에포크 끝에만 (10에포크당 1회)
        if epoch % 10 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    print(f'\n✅ Phase 2 완료 ({elapsed:.1f}분)')
    print(f'   Best Loss : {best_loss:.4f}')
    print(f'   Best fg mIoU: {best_miou*100:.1f}%')

    return model, loss_hist, miou_hist, best_loss, best_miou


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  장치: {device}')
    if device.type == 'cuda':
        print(f'   GPU : {torch.cuda.get_device_name(0)}')
        print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    os.makedirs(CFG['result_dir'], exist_ok=True)

    # ── 데이터셋 (V5: npy 캐싱 + box_dilate=0) ──────
    print('\n[데이터 로드] V5 (npy GT 캐싱 + box_dilate=0)')
    tr_ds = NuScenesV5Dataset(
        CFG['data_root'], CFG['version'],
        is_train=True,
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'],
        img_h=CFG['img_h'], img_w=CFG['img_w'],
        box_dilate=CFG['box_dilate'])
    va_ds = NuScenesV5Dataset(
        CFG['data_root'], CFG['version'],
        is_train=False,
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'],
        img_h=CFG['img_h'], img_w=CFG['img_w'],
        box_dilate=CFG['box_dilate'])

    # ★ Fix 1: 학습 전 GT 전체 사전 캐싱 (I/O 병목 제거)
    print('\n[GT 사전 캐싱] 학습/검증 데이터 전체 처리...')
    tr_ds.pre_cache_all()
    va_ds.pre_cache_all()

    # ★ Fix 2: num_workers=4 (병렬 데이터 로딩)
    tr_loader = DataLoader(
        tr_ds, batch_size=CFG['batch_size'],
        shuffle=True, num_workers=CFG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(CFG['num_workers'] > 0))
    va_loader = DataLoader(
        va_ds, batch_size=CFG['batch_size'],
        shuffle=False, num_workers=CFG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(CFG['num_workers'] > 0))

    # ── GT 분포 출력 ─────────────────────────────────
    print('\n[GT 분포 확인] (box_dilate=0)')
    total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for i in range(min(10, len(tr_ds))):
        _, _, _, gt = tr_ds[i]
        for c in range(NUM_CLASSES):
            total[c] += (gt == c).sum()
    tot = total.sum().item()
    for c, nm in enumerate(CLASS_NAMES):
        n = total[c].item()
        print(f'  {nm:<16}: {n:>10,}  ({n/tot*100:.3f}%)')

    # ════════════════════════════════════════════════
    # Phase 1: Binary Pre-training
    # ════════════════════════════════════════════════
    print('\n[Phase 1 모델] FastOcc — num_classes=2 (Binary)')
    model_bin = FastOcc(
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'],
        num_classes=2,                  # Binary: Free vs Occupied
        fpn_ch=CFG['fpn_ch'], c2h_ch=CFG['c2h_ch'],
        img_h=CFG['img_h'], img_w=CFG['img_w'],
        num_cams=CFG['num_cams'],
    ).to(device)

    model_bin = run_phase1(model_bin, tr_loader, va_loader, device)

    # Phase 1 가중치 저장 (아직 없으면 현재 상태 저장)
    if not os.path.exists(CFG['ph1_ckpt']):
        torch.save(model_bin.state_dict(), CFG['ph1_ckpt'])
        print(f'  Phase 1 가중치 저장: {CFG["ph1_ckpt"]}')

    # ════════════════════════════════════════════════
    # Phase 2: 5-class Fine-tuning
    # ════════════════════════════════════════════════
    print('\n[Phase 2 모델] FastOcc — num_classes=5 (5-class)')
    model_5cls = FastOcc(
        xbound=CFG['xbound'], ybound=CFG['ybound'], zbound=CFG['zbound'],
        num_classes=5,                  # 5-class: Full semantic
        fpn_ch=CFG['fpn_ch'], c2h_ch=CFG['c2h_ch'],
        img_h=CFG['img_h'], img_w=CFG['img_w'],
        num_cams=CFG['num_cams'],
    ).to(device)

    # Phase 1 가중치 로드 (backbone + sampler + c2h만 — 분류기 제외)
    sd_bin = torch.load(CFG['ph1_ckpt'], map_location=device, weights_only=True)
    sd_5   = model_5cls.state_dict()
    compat = {k: v for k, v in sd_bin.items()
              if not k.startswith('classifier.')  # 2-class head 제외
              and k in sd_5
              and v.shape == sd_5[k].shape}
    loaded = model_5cls.load_state_dict(compat, strict=False)
    print(f'  ✅ Phase1 가중치 로드: {len(compat)}/{len(sd_bin)}개 키'
          f' (분류기는 새 초기화)')
    print(f'     Missing: {[k for k in loaded.missing_keys[:5]]}...')

    log_csv = os.path.join(CFG['result_dir'], 'train_log_v3_ph2.csv')
    model_5cls, loss_hist, miou_hist, best_loss, best_miou = \
        run_phase2(model_5cls, tr_loader, va_loader, device, log_csv)

    # ════════════════════════════════════════════════
    # 최종 평가
    # ════════════════════════════════════════════════
    ckpt = CFG['ckpt_miou'] if os.path.exists(CFG['ckpt_miou']) \
           else CFG['ckpt_best']
    model_5cls.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    final_iou = calc_miou_5cls(model_5cls, va_loader, device)
    fg = final_iou[1:].mean().item()

    print('\n' + '═'*46)
    print('  최종 클래스별 IoU (FastOcc v3 — 2단계 학습)')
    for c, nm in enumerate(CLASS_NAMES):
        mark = ' ✅' if final_iou[c] >= 0.5 else ''
        print(f'  {nm:<16}: {final_iou[c]*100:5.1f}%{mark}')
    print(f'  {"전경 mIoU":<16}: {fg*100:5.1f}%')
    print('═'*46)

    # ── 학습 곡선 저장 ────────────────────────────
    if loss_hist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(loss_hist, lw=1.5, color='royalblue')
        ax1.set(xlabel='Ph2 Epoch', ylabel='Loss',
                title=f'FastOcc v3 Phase2 Loss (best={best_loss:.4f})')
        ax1.grid(alpha=.3)

        if miou_hist:
            ep, mi = zip(*miou_hist)
            ax2.plot(ep, mi, lw=1.5, color='tomato', marker='o', ms=4)
            ax2.axhline(50, ls='--', color='green', alpha=.6, label='목표 50%')
            ax2.set(xlabel='Ph2 Epoch', ylabel='Foreground mIoU (%)',
                    title='FastOcc v3 Foreground mIoU')
            ax2.legend()
            ax2.grid(alpha=.3)

        plt.tight_layout()
        plt.savefig(os.path.join(CFG['result_dir'], 'loss_curve_v3.png'),
                    dpi=130)
        plt.close()

    # ── 최종 BEV ─────────────────────────────────
    with torch.no_grad():
        fv = next(iter(va_loader))
        fv_imgs = fv[0].to(device)
        fv_Ks   = fv[1].float().to(device)
        fv_s2e  = fv[2].float().to(device)
        fv_gt   = fv[3]
        pred_v  = model_5cls(fv_imgs, fv_Ks, fv_s2e)
    bev_vis(fv_gt[0].numpy(), pred_v[0].argmax(0).cpu().numpy(),
            'Final', os.path.join(CFG['result_dir'], 'bev_final_v3.jpg'))

    # ── JSON 결과 ─────────────────────────────────
    info = dict(
        model='FastOcc v3 (Binary PreTrain → 5-class FineTune)',
        fixes_applied=[
            'Fix2: num_workers=4, gc.collect() 주기 최적화',
            'Fix3: grid_sample FP32 캐스팅 (model_fastocc.py)',
            'Fix4: gradient accum 마지막 배치 처리',
            'Fix5: box_dilate=0 (Lovász 호환)',
            'Fix6: Binary Pre-training → 5-class Fine-tuning',
        ],
        ph1_epochs=CFG['ph1_epochs'],
        ph2_epochs_trained=len(loss_hist),
        best_loss=round(best_loss, 6),
        best_fg_miou=round(best_miou, 4),
        final_iou={nm: round(final_iou[c].item(), 4)
                   for c, nm in enumerate(CLASS_NAMES)},
        final_fg_miou=round(fg, 4),
    )
    with open(os.path.join(CFG['result_dir'], 'train_info_v3.json'),
              'w', encoding='utf-8') as fj:
        json.dump(info, fj, indent=2, ensure_ascii=False)

    git_push(f'FastOcc v3 완료: fg_mIoU={fg*100:.1f}% (2단계 커리큘럼)',
             repo='..')


if __name__ == '__main__':
    main()
