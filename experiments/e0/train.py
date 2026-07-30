#!/usr/bin/env python3
"""
E0 학습 루프 — C0 MONO / C6 SPLIT-ORACLE

C8 ORACLE-IK 은 학습 파라미터가 0개라 여기 오지 않는다. 바로 evaluate.py 로 간다.

두 arm 은 **완전히 동일한 학습 레시피**를 쓴다 — 같은 옵티마이저, 같은 스텝
수, 같은 스케줄, 같은 seed. 하나라도 다르면 C6-C0 격차를 "튜닝 차이" 로
설명할 수 있게 되고 E0 는 무효가 된다. arm 인자가 바꾸는 것은 오직
`use_goal` 플래그 하나뿐이다.

사용:
    # 배선 검증 (LIBERO/torch 없이도 동작)
    python train.py --smoke

    # 본 실험 (3090)
    python train.py --arm c6 --seed 0 --data $LIBERO_DATA --steps 60000 --batch 64

    # 1660 Super 6GB (배치를 줄이고 grad accumulation 으로 보상)
    python train.py --arm c6 --seed 0 --data $LIBERO_DATA --steps 60000 \
                    --batch 16 --accum 4 --amp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import ACTION_DIM, ARM_DESC, CHUNK_K, build_dataset, E0Dataset
from models import (PolicyConfig, NumpyLinearPolicy, build_policy, count_params,
                    has_torch, require_torch)


# ═════════════════════════════════════════════════════════════════════════════
# 공통
# ═════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int) -> None:
    """
    seed 고정. E0 는 seed 3개를 pool 해서 n=1500 을 만들기 때문에 seed 간
    독립성이 통계 판정의 전제다. 여기서 새는 게 있으면 CI 가 거짓으로 좁아진다.
    """
    np.random.seed(seed)
    if has_torch():
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: Optional[str] = None, smoke: bool = False) -> str:
    """
    ⚠️ smoke 에서는 MPS 를 쓰지 않는다. macOS MPS 백엔드가 이 vision stem 의
       conv 조합에서 'Weights tensor and ndArray input channel mismatch' 로
       죽는 것을 M4 에서 확인했다(재현됨). smoke 는 배선 검증이 목적이므로
       CPU 로 충분하고, 실제 학습은 어차피 CUDA 에서 돈다.
       MPS 에서 굳이 돌려보려면 --device mps 를 명시할 것.
    """
    torch = require_torch()
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if (not smoke and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    """warmup + cosine decay. ACT 계열 표준 레시피."""
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


# ═════════════════════════════════════════════════════════════════════════════
# torch 경로
# ═════════════════════════════════════════════════════════════════════════════


def rgb_to_float(t):
    """
    uint8 (B, n_cam, H, W, 3) → float (B, n_cam, 3, H, W), [-1, 1] 정규화.

    **학습과 rollout 이 반드시 같은 함수를 통과해야 한다.** 전처리가 갈리면
    성공률이 조용히 0 이 되고 원인 찾기가 매우 어렵다 (canonical_rgb 와 같은 이유).

    ImageNet 통계를 쓰지 않는 것은 백본을 scratch 학습하기 때문이다 —
    대칭 정규화가 BatchNorm 과 더 잘 맞는다.
    GPU 로 옮긴 뒤 변환하는 이유: uint8 이 float32 보다 전송량이 4배 작다.
    """
    return t.permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)


def _make_torch_loader(ds: E0Dataset, batch: int, seed: int, workers: int):
    """
    E0Dataset(numpy) → torch DataLoader.

    ⚠️ E0Dataset 을 **그대로** 넘긴다. 지역 클래스로 감싸면 DataLoader worker
       가 pickle 하지 못해 `Can't get local object` 로 죽는다 (macOS/Windows
       기본 start method 가 spawn 이라서. Linux fork 에서는 우연히 통과한다).
       DataLoader 는 __len__/__getitem__ 만 있으면 되고 Dataset 상속은 불필요하다.

    ⚠️ worker 는 fork 에서만 쓴다. spawn 이면 에피소드 전체(500 데모 ≈ 6.4GB)
       가 worker 마다 복사돼 메모리가 터진다. fork 는 copy-on-write 라 안전하다.
    """
    torch = require_torch()
    from torch.utils.data import DataLoader
    import multiprocessing as mp

    ctx = None
    if workers > 0:
        if "fork" in mp.get_all_start_methods():
            ctx = mp.get_context("fork")
        else:
            print(f"[train] ⚠️ 이 플랫폼은 fork 를 지원하지 않습니다 → "
                  f"num_workers {workers} → 0 으로 강제합니다 (메모리 폭발 방지).")
            workers = 0

    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                      pin_memory=(workers > 0), drop_last=True, generator=g,
                      persistent_workers=(workers > 0), multiprocessing_context=ctx)


def train_torch(args) -> dict:
    torch = require_torch()
    import torch.nn.functional as F

    device = pick_device(args.device, smoke=args.smoke)
    set_seed(args.seed)

    cfg = PolicyConfig.smoke() if args.smoke else PolicyConfig()
    ds = build_dataset(smoke=args.smoke, data_root=args.data, split="train",
                       train_demos=args.train_demos, seed=args.seed)
    loader = _make_torch_loader(ds, args.batch, args.seed, args.workers)

    model = build_policy(args.arm, cfg).to(device)
    n_par = count_params(model)
    print(f"[train] arm={args.arm.upper()} ({ARM_DESC[args.arm]})")
    print(f"[train] device={device} params={n_par:,} transitions={len(ds):,} "
          f"batch={args.batch}×accum{args.accum} steps={args.steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = args.amp and device == "cuda"
    # bf16 은 Ampere(3090) 이상만. Turing(1660 Super) 은 fp16 으로 떨어진다.
    amp_dtype = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and amp_dtype == torch.float16))
    if amp:
        print(f"[train] AMP 활성 dtype={amp_dtype}")

    model.train()
    it = iter(loader)
    t0, hist = time.time(), []
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(step, args.steps, args.lr, args.warmup)

        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(args.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp):
                pred = model(rgb_to_float(b["rgb"]), b["proprio"], b["goal"])
                # L1. ACT 가 L2 대신 L1 을 쓰는 이유는 chunk 안의 소수 이상치
                # 스텝(접촉 순간의 큰 보정)이 L2 에서 나머지를 다 눌러버리기 때문.
                l1 = F.l1_loss(pred, b["action"], reduction="none").mean(-1)
                loss = (l1 * b["pad_mask"]).sum() / b["pad_mask"].sum().clamp(min=1)
                loss = loss / args.accum
            scaler.scale(loss).backward()
            acc_loss += float(loss.detach()) * args.accum

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        hist.append(acc_loss / args.accum)

        if step % max(args.log_every, 1) == 0 or step == args.steps - 1:
            el = time.time() - t0
            eta = el / (step + 1) * (args.steps - step - 1)
            print(f"  step {step:6d}/{args.steps}  loss {np.mean(hist[-50:]):.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{(step+1)/el:.2f} it/s  ETA {eta/60:.1f}분")

    ck = args.out or default_ckpt(args.arm, args.seed, args.smoke)
    os.makedirs(os.path.dirname(os.path.abspath(ck)), exist_ok=True)
    torch.save({"backend": "torch", "arm": args.arm, "seed": args.seed,
                "cfg": cfg.to_dict(), "state_dict": model.state_dict(),
                "norm": ds.norm.to_dict(), "steps": args.steps,
                "params": n_par, "final_loss": float(np.mean(hist[-50:])),
                "smoke": bool(args.smoke)}, ck)
    print(f"[train] 체크포인트 저장 → {ck}")
    return {"ckpt": ck, "final_loss": float(np.mean(hist[-50:])),
            "params": n_par, "minutes": (time.time() - t0) / 60}


# ═════════════════════════════════════════════════════════════════════════════
# numpy 폴백 — --smoke 에서 torch 가 없을 때만
# ═════════════════════════════════════════════════════════════════════════════


def train_numpy(args) -> dict:
    """⚠️ 배선 검사 전용. 성능 수치에 의미 없음."""
    assert args.smoke, "numpy 폴백은 --smoke 에서만 허용된다 (본 실험은 torch 필수)"
    print(f"[train] arm={args.arm.upper()} — numpy 폴백 (torch 없음). "
          f"⚠️ 배선 검증 전용이며 성능 수치는 무의미합니다.")
    set_seed(args.seed)
    ds = build_dataset(smoke=True, split="train", seed=args.seed)
    pol = NumpyLinearPolicy(args.arm, seed=args.seed)
    pol.norm = ds.norm

    rng = np.random.default_rng(args.seed)
    t0, hist = time.time(), []
    for step, b in enumerate(ds.batches(args.batch, rng, steps=args.steps)):
        hist.append(pol.step(b, lr=cosine_lr(step, args.steps, 3e-3, args.warmup)))
        if step % max(args.log_every, 1) == 0 or step == args.steps - 1:
            print(f"  step {step:6d}/{args.steps}  loss {np.mean(hist[-50:]):.4f}")

    ck = args.out or default_ckpt(args.arm, args.seed, True)
    pol.save(ck, meta={"steps": args.steps, "seed": args.seed, "smoke": True})
    print(f"[train] 체크포인트 저장 → {ck}")
    return {"ckpt": ck, "final_loss": float(np.mean(hist[-50:])), "params": 0,
            "minutes": (time.time() - t0) / 60}


# ═════════════════════════════════════════════════════════════════════════════


def default_ckpt(arm: str, seed: int, smoke: bool) -> str:
    sub = "smoke" if smoke else "runs"
    ext = "pt" if has_torch() else "pkl"
    return os.path.join(_HERE, "checkpoints", sub, f"{arm}_seed{seed}.{ext}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E0 학습 (C0 / C6)")
    ap.add_argument("--arm", default="c6", choices=["c0", "c6"],
                    help="c8 은 학습 파라미터 0개라 여기 오지 않습니다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.environ.get("LIBERO_DATA"),
                    help="LIBERO 데이터 루트 (env LIBERO_DATA)")
    ap.add_argument("--train-demos", type=int, default=25,
                    help="태스크당 학습 데모 수. 나머지는 eval 로 held out")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1, help="grad accumulation (6GB 카드용)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    ap.add_argument("--out", default=None, help="체크포인트 경로")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true",
                    help="합성 데이터 + 초소형 모델 + 소수 스텝으로 배선만 검증")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.smoke:
        # smoke 기본값을 덮어쓴다 — CPU 에서 수십 초 안에 끝나야 한다
        args.steps = min(args.steps, 300)
        args.batch = min(args.batch, 16)
        args.accum = 1
        args.log_every = min(args.log_every, 10)
        args.warmup = 5
    else:
        if not args.data:
            raise SystemExit("[train] --data <LIBERO 루트> 가 필요합니다 "
                             "(또는 --smoke 로 배선만 검증).")
        # 본 실험에서 numpy 폴백으로 조용히 새는 것을 막는다.
        # 폴백은 합성 데이터로 학습하므로 알아채지 못하면 결과가 통째로 거짓이 된다.
        require_torch()

    r = train_torch(args) if has_torch() else train_numpy(args)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
