#!/usr/bin/env python3
"""
E0 정책 모델 — ACT 계열 transformer + action chunking (K=16), ~25M 파라미터

왜 C0 와 C6 가 **구조까지 같아야** 하는가
──────────────────────────────────────────
E0 가 재려는 것은 오직 "정답 목표 EE 포즈라는 입력의 가치" 하나다.
두 arm 의 파라미터 수나 토큰 수가 다르면 C6-C0 격차가 "용량 차이" 로 오염돼
실험 자체가 무의미해진다. 그래서:

  - C0 와 C6 는 **완전히 동일한 모듈**을 쓴다. goal_proj 도 C0 에 있다.
  - C0 는 goal 입력 자리에 0 벡터를 넣는다 → goal_proj 는 bias 만 내보내는
    **learned null token** 이 된다. 시퀀스 길이도 파라미터 수도 정확히 같다.
  - `assert_matched()` 가 두 arm 의 파라미터 수가 비트 단위로 같은지 검사한다.

C8 은 여기 없다. 학습 파라미터가 0개이므로 모델이 아니다 —
dataset.oracle_to_osc_action() 이 전부다. 그게 C8 의 요점이다.

torch 는 지연 import 한다. torch 가 없는 머신에서도 --smoke 가 돌도록
numpy 전용 폴백 정책(NumpyLinearPolicy)을 함께 제공한다.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import (ACTION_DIM, CHUNK_K, GOAL_DIM, N_CAMERAS, PROPRIO_DIM,
                     ARM_USES_GOAL, NormStats)

_TORCH_HELP = """
PyTorch 를 찾을 수 없습니다.

  학습/평가에는 torch 가 필요합니다:
    # RTX 3090 / GTX 1660 Super (CUDA 12.x)
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    # Mac (배선 확인용, MPS)
    pip install torch

  torch 없이 파이프라인 배선만 확인하려면 --smoke 를 쓰십시오
  (numpy 전용 폴백 정책으로 전체 흐름이 돕니다).
""".strip()


def require_torch():
    """torch 지연 import. 실패 시 설치법을 알려주고 종료."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise SystemExit(f"{_TORCH_HELP}\n\n  (원인: {e})")


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 설정
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class PolicyConfig:
    """
    ~25M 파라미터를 겨냥한 ACT 계열 설정.

    왜 25M 인가: LIBERO 에서 실질적으로 saturate 하는 규모이면서 3090 1장에
    2 arm × 3 seed = 6 run 을 현실적 시간에 돌릴 수 있는 상한이다. 더 키우면
    E0 가 "값싼 반증" 이 아니게 된다.
    """
    d_model: int = 384
    nhead: int = 6
    dim_ff: int = 1536
    enc_layers: int = 6
    dec_layers: int = 6
    dropout: float = 0.1
    chunk_k: int = CHUNK_K
    img_size: int = 128
    n_cameras: int = N_CAMERAS
    proprio_dim: int = PROPRIO_DIM
    goal_dim: int = GOAL_DIM
    action_dim: int = ACTION_DIM

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def smoke(cls) -> "PolicyConfig":
        """CPU 몇 초용 초소형. 배선만 본다."""
        return cls(d_model=64, nhead=4, dim_ff=128, enc_layers=1, dec_layers=1,
                   img_size=32, dropout=0.0)


# ═════════════════════════════════════════════════════════════════════════════
# torch 모듈 — torch 가 있을 때만 정의된다
# ═════════════════════════════════════════════════════════════════════════════


def build_torch_modules():
    """
    torch nn.Module 들을 **함수 안에서** 정의한다.

    모듈 레벨에서 nn.Module 을 상속하면 torch 가 없는 머신에서 이 파일을
    import 하는 순간 죽는다. E0 는 torch 없는 Mac 에서도 --smoke 가 돌아야
    하므로 클래스 정의 자체를 지연시킨다.
    """
    torch = require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        """ResNet 기본 블록. 백본을 밖에서 가져오지 않는 이유: torchvision
        의존을 없애서 3090/1660 어느 쪽이든 pip 한 줄로 끝나게 하려는 것."""

        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(cout)
            self.down = None
            if stride != 1 or cin != cout:
                self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))

        def forward(self, x):
            idt = x if self.down is None else self.down(x)
            y = F.relu(self.b1(self.c1(x)), inplace=True)
            y = self.b2(self.c2(y))
            return F.relu(y + idt, inplace=True)

    class VisionEncoder(nn.Module):
        """
        카메라 공유 백본. 2대(agentview / eye-in-hand)가 **같은 가중치**를
        쓰고 카메라 구분은 뒤에서 더하는 camera embedding 이 담당한다.
        분리 백본은 파라미터를 2배로 먹으면서 LIBERO 규모에서 이득이 없다.

        출력은 pooling 하지 않고 공간 토큰으로 편다 — ACT 의 핵심은 디코더가
        어느 픽셀을 볼지 스스로 고르는 것이고, GAP 하면 그 능력이 사라진다.

        **마지막 stage 만 stride 1** 이다. ResNet18 그대로면 총 stride 32 라
        128×128 입력이 4×4=16 토큰이 되는데, 파지 정렬처럼 수 mm 가 갈리는
        작업에는 한 토큰이 32px(≈수 cm)를 덮어 너무 거칠다. layer3 의 stride
        만 1 로 낮추면 8×8=64 토큰(카메라 2대 128 토큰)이 되고, 추가 연산은
        가장 작은 해상도 단계에만 붙어 카메라당 ~114 MMACs 로 끝난다.
        (파라미터 수는 stride 와 무관하므로 C0/C6 용량 일치도 그대로 유지된다.)
        """

        def __init__(self, d_model: int):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32),
                nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
            self.layer1 = BasicBlock(32, 64, 2)
            self.layer2 = BasicBlock(64, 128, 2)
            self.layer3 = BasicBlock(128, 256, 1)      # ← stride 1
            self.proj = nn.Conv2d(256, d_model, 1)

        def forward(self, x):                       # x: (B*C, 3, H, W)
            x = self.stem(x)
            x = self.layer3(self.layer2(self.layer1(x)))
            return self.proj(x)                     # (B*C, d, h, w)

    def sinusoid_2d(d: int, h: int, w: int, device, dtype):
        """2D sine 위치 인코딩. 학습 파라미터 0 — 해상도가 바뀌어도 안전하다."""
        assert d % 4 == 0
        yv, xv = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                torch.arange(w, device=device, dtype=dtype),
                                indexing="ij")
        omega = torch.exp(torch.arange(0, d // 4, device=device, dtype=dtype)
                          * (-math.log(10000.0) / max(d // 4 - 1, 1)))
        out = []
        for v in (yv, xv):
            a = v.reshape(-1, 1) * omega.reshape(1, -1)
            out += [torch.sin(a), torch.cos(a)]
        return torch.cat(out, dim=1)                # (h*w, d)

    class E0Policy(nn.Module):
        """
        C0 / C6 공용 정책.

        토큰 시퀀스 = [vision(2×h×w) | proprio(1) | goal(1)]
        디코더는 K=16 개의 학습된 query 로 cross-attend 해서 chunk 를 한 번에
        낸다. autoregressive 가 아닌 이유: 20Hz 제어에서 K번 순차 디코딩은
        latency 예산을 그냥 태운다. ACT/Diffusion Policy 가 공통으로 내린 결론.

        `use_goal=False` (C0) 여도 goal_proj 는 존재하며 0 을 먹는다.
        → 파라미터 수가 C6 와 **정확히** 같다. 파일 상단 설명 참고.
        """

        def __init__(self, cfg: PolicyConfig, use_goal: bool):
            super().__init__()
            self.cfg, self.use_goal = cfg, use_goal
            d = cfg.d_model

            self.vision = VisionEncoder(d)
            self.cam_embed = nn.Parameter(torch.zeros(cfg.n_cameras, 1, d))
            self.proprio_proj = nn.Linear(cfg.proprio_dim, d)
            self.goal_proj = nn.Linear(cfg.goal_dim, d)
            self.state_embed = nn.Parameter(torch.zeros(2, 1, d))   # proprio/goal 토큰 구분

            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, cfg.nhead, cfg.dim_ff, cfg.dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True),
                cfg.enc_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d, cfg.nhead, cfg.dim_ff, cfg.dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True),
                cfg.dec_layers, norm=nn.LayerNorm(d))
            self.query = nn.Parameter(torch.zeros(cfg.chunk_k, d))
            self.head = nn.Linear(d, cfg.action_dim)

            nn.init.normal_(self.query, std=0.02)
            nn.init.normal_(self.cam_embed, std=0.02)
            nn.init.normal_(self.state_embed, std=0.02)

        def forward(self, rgb, proprio, goal):
            """
            Args:
                rgb:     (B, n_cam, 3, H, W) float, 이미 정규화됨
                proprio: (B, proprio_dim)
                goal:    (B, goal_dim) — C0 에서는 호출부가 0 을 넣는다
            Returns:
                (B, K, action_dim) 정규화된 action chunk
            """
            B, C = rgb.shape[0], rgb.shape[1]
            f = self.vision(rgb.flatten(0, 1))                  # (B*C, d, h, w)
            _, d, h, w = f.shape
            f = f.flatten(2).transpose(1, 2)                    # (B*C, h*w, d)
            f = f + sinusoid_2d(d, h, w, f.device, f.dtype).unsqueeze(0)
            f = f.reshape(B, C, h * w, d) + self.cam_embed.unsqueeze(0)
            f = f.reshape(B, C * h * w, d)

            if not self.use_goal:
                goal = torch.zeros_like(goal)                   # learned null token
            p = self.proprio_proj(proprio).unsqueeze(1) + self.state_embed[0]
            g = self.goal_proj(goal).unsqueeze(1) + self.state_embed[1]

            mem = self.encoder(torch.cat([f, p, g], dim=1))
            q = self.query.unsqueeze(0).expand(B, -1, -1)
            return self.head(self.decoder(q, mem))

    return dict(E0Policy=E0Policy, VisionEncoder=VisionEncoder, BasicBlock=BasicBlock)


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_policy(arm: str, cfg: Optional[PolicyConfig] = None):
    """arm 이름 → torch 정책. C8 은 모델이 아니므로 거부한다."""
    if arm == "c8":
        raise ValueError("C8 ORACLE-IK 은 학습 파라미터가 0개입니다. "
                         "dataset.oracle_to_osc_action() 을 직접 쓰십시오.")
    if arm not in ARM_USES_GOAL:
        raise ValueError(f"알 수 없는 arm: {arm}")
    cfg = cfg or PolicyConfig()
    M = build_torch_modules()
    return M["E0Policy"](cfg, use_goal=ARM_USES_GOAL[arm])


def assert_matched(cfg: Optional[PolicyConfig] = None, verbose: bool = True) -> int:
    """
    C0 와 C6 의 파라미터 수가 정확히 같은지 확인한다.

    이 assert 가 깨지면 E0 의 결론은 **전부 무효**다. C6-C0 격차를
    "용량이 더 커서" 로 설명할 수 있게 되기 때문이다.
    """
    cfg = cfg or PolicyConfig()
    n0 = count_params(build_policy("c0", cfg))
    n6 = count_params(build_policy("c6", cfg))
    if verbose:
        print(f"[models] C0 {n0:,} / C6 {n6:,} 파라미터  (C8: 0)")
    assert n0 == n6, f"용량 불일치 C0={n0} C6={n6} — 비교가 오염된다"
    return n0


# ═════════════════════════════════════════════════════════════════════════════
# numpy 전용 폴백 — --smoke 에서 torch 가 없을 때만
#
# ⚠️ 이것은 **모델이 아니라 배선 검사기**다. 이 정책의 성능 수치는 어떤
#    과학적 의미도 없다. 존재 이유는 하나: torch 없는 Mac 에서
#    데이터→학습→평가→판정 전체 경로가 끊기지 않는지 몇 초 만에 확인.
# ═════════════════════════════════════════════════════════════════════════════


class NumpyLinearPolicy:
    """(다운샘플 이미지 + proprio + goal) → action chunk 선형 회귀. Adam."""

    def __init__(self, arm: str, chunk_k: int = CHUNK_K, feat_hw: int = 6, seed: int = 0):
        self.arm = arm
        self.use_goal = ARM_USES_GOAL[arm]
        self.chunk_k = chunk_k
        self.feat_hw = feat_hw
        self.in_dim = N_CAMERAS * feat_hw * feat_hw + PROPRIO_DIM + GOAL_DIM
        self.out_dim = chunk_k * ACTION_DIM
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((self.in_dim, self.out_dim)) * 0.01).astype(np.float32)
        self.b = np.zeros(self.out_dim, np.float32)
        self._m = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self._v = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self._t = 0
        self.norm: Optional[NormStats] = None

    def features(self, rgb: np.ndarray, proprio: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """이미지를 회색조 → feat_hw×feat_hw 평균 풀링. 그 이상은 불필요."""
        if rgb.ndim == 4:                       # 단일 샘플 (n_cam,H,W,3)
            rgb, proprio, goal = rgb[None], proprio[None], goal[None]
        B, C, H, W = rgb.shape[:4]
        g = rgb.astype(np.float32).mean(-1) / 255.0                 # (B,C,H,W)
        ph, pw = H // self.feat_hw, W // self.feat_hw
        g = g[:, :, : ph * self.feat_hw, : pw * self.feat_hw]
        g = g.reshape(B, C, self.feat_hw, ph, self.feat_hw, pw).mean((3, 5))
        g = g.reshape(B, -1)
        gl = goal if self.use_goal else np.zeros_like(goal)
        return np.concatenate([g, proprio, gl], 1).astype(np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def step(self, batch: dict, lr: float = 3e-3) -> float:
        """Adam 1 스텝. pad_mask 로 에피소드 끝 padding 을 손실에서 제외."""
        X = self.features(batch["rgb"], batch["proprio"], batch["goal"])
        Y = batch["action"].reshape(len(X), -1)
        m = np.repeat(batch["pad_mask"], ACTION_DIM, axis=1)
        pred = self.forward(X)
        err = (pred - Y) * m
        loss = float((err ** 2).sum() / max(m.sum(), 1))
        gW = X.T @ err * (2.0 / max(m.sum(), 1))
        gb = err.sum(0) * (2.0 / max(m.sum(), 1))

        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(((self.W, gW), (self.b, gb))):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)
        return loss

    def predict(self, rgb, proprio, goal) -> np.ndarray:
        """정규화된 (K, ACTION_DIM) chunk. 역정규화는 호출부(evaluate.py)가 한다."""
        X = self.features(rgb, proprio, goal)
        return self.forward(X).reshape(-1, self.chunk_k, ACTION_DIM)[0]

    # ── 체크포인트 ────────────────────────────────────────────────────────────
    def save(self, path: str, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"arm": self.arm, "W": self.W, "b": self.b,
                         "chunk_k": self.chunk_k, "feat_hw": self.feat_hw,
                         "norm": self.norm.to_dict() if self.norm else None,
                         "backend": "numpy", "meta": meta or {}}, f)

    @classmethod
    def load(cls, path: str) -> "NumpyLinearPolicy":
        with open(path, "rb") as f:
            d = pickle.load(f)
        p = cls(d["arm"], d["chunk_k"], d["feat_hw"])
        p.W, p.b = d["W"], d["b"]
        p.norm = NormStats.from_dict(d["norm"]) if d["norm"] else None
        return p


# ═════════════════════════════════════════════════════════════════════════════


def _self_test(force_numpy: bool = False) -> int:
    print("── models self-test ──")
    if has_torch() and not force_numpy:
        import torch
        cfg = PolicyConfig.smoke()
        n = assert_matched(cfg)
        print(f"  smoke cfg 파라미터 : {n:,}")
        pol = build_policy("c6", cfg).eval()
        rgb = torch.randn(2, N_CAMERAS, 3, cfg.img_size, cfg.img_size)
        with torch.no_grad():
            y = pol(rgb, torch.randn(2, PROPRIO_DIM), torch.randn(2, GOAL_DIM))
        assert y.shape == (2, cfg.chunk_k, ACTION_DIM), y.shape
        print(f"  forward 출력       : {tuple(y.shape)}")

        full = PolicyConfig()
        nfull = assert_matched(full, verbose=False)
        print(f"  본 실험 파라미터   : {nfull:,}  (목표 ~25M)")
        assert 18e6 < nfull < 32e6, f"파라미터 {nfull} 이 ~25M 범위 밖"
        # C0 가 goal 을 정말 안 쓰는지 — 입력을 바꿔도 출력이 같아야 한다
        c0 = build_policy("c0", cfg).eval()
        with torch.no_grad():
            a = c0(rgb, torch.zeros(2, PROPRIO_DIM), torch.zeros(2, GOAL_DIM))
            b = c0(rgb, torch.zeros(2, PROPRIO_DIM), torch.randn(2, GOAL_DIM) * 5)
        assert torch.allclose(a, b, atol=1e-6), "C0 가 goal 을 보고 있다 — 대조군 오염"
        print("  C0 goal 차단 확인  : OK")
    else:
        print("  torch 없음 → numpy 폴백만 검사")

    from dataset import build_dataset
    ds = build_dataset(smoke=True, split="train")
    rng = np.random.default_rng(0)
    for arm in ("c0", "c6"):
        p = NumpyLinearPolicy(arm, seed=0)
        losses = [p.step(b) for b in ds.batches(8, rng, steps=30)]
        print(f"  numpy {arm.upper()} loss  : {losses[0]:.4f} → {losses[-1]:.4f}")
        assert losses[-1] < losses[0], f"{arm}: 손실이 안 줄었다"
        y = p.predict(ds[0]["rgb"], ds[0]["proprio"], ds[0]["goal"])
        assert y.shape == (CHUNK_K, ACTION_DIM), y.shape
    print("── models self-test 통과 ──")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="E0 정책 모델")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--numpy", action="store_true", help="torch 가 있어도 폴백만 검사")
    ap.add_argument("--params", action="store_true", help="본 실험 파라미터 수 출력")
    a = ap.parse_args()
    if a.params:
        print(json.dumps({"config": PolicyConfig().to_dict(),
                          "params": assert_matched()}, indent=2, ensure_ascii=False))
        return 0
    return _self_test(force_numpy=a.numpy)


if __name__ == "__main__":
    raise SystemExit(main())
