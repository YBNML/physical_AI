#!/usr/bin/env python3
"""
E0 rollout 평가 + **통계 판정** — GATE-2

이 파일이 E0 의 결론을 내린다. 성공률 세 개(C0/C6/C8)를 받아 미리 못박아 둔
규칙으로 프로젝트의 진행 여부를 자동 출력한다. 규칙을 실행 **전에** 고정하는
이유는 명백하다 — 숫자를 보고 나서 임계를 정하면 그건 실험이 아니다.

판정 규칙 (하드코딩, 변경 금지)
────────────────────────────────
  C6 - C0 <= +5  또는 95% CI 가 0 포함  →  H1 사망. 가설 자체가 죽는다.
        완벽한 정답 인터페이스로도 monolith 를 못 이기면, 실제로 학습될
        Model 1 은 정답보다 나쁠 수밖에 없으므로 배포판 분리는 확실히 진다.

  C8 >= C6 - 5                          →  학습 Model 2 사망.
        목표 포즈만 있으면 closed-form OSC 로 충분하다는 뜻. 소뇌를 학습시킬
        이유가 없다. Model 1 + IK 로 간다.

  C6-C0 >= +10 AND C6-C8 >= +5          →  진행.

  +5 < C6-C0 < +10                      →  LIBERO-Long / CALVIN ABC→D 재실행.
        신호는 있으나 n=1500 의 MDE(5.1pt)와 구분이 어렵다. 더 긴 horizon 의
        벤치마크에서 분리의 이득이 커지는지 확인해야 한다.

검정력
──────
  two-proportion 양측 검정, α=0.05, power=0.80, p≈0.5(최대분산=보수적) 기준
      n=500  (seed 1개)  → MDE ≈ 8.9pt
      n=1500 (seed 3개)  → MDE ≈ 5.1pt
  즉 **seed 1개로는 ±5pt 규칙을 판정할 수 없다.** 3 seed pooling 이 필수다.

사용:
    python evaluate.py --smoke                                  # 배선 검증
    python evaluate.py --arm c6 --ckpt ... --data $LIBERO_DATA  # 단일 arm rollout
    python evaluate.py --report results/                        # 판정만 재출력
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import (ACTION_DIM, ARM_DESC, ARMS, ARM_USES_GOAL, CHUNK_K, GOAL_DIM,
                     OSC_POS_MAX, OSC_ROT_MAX, PROPRIO_DIM, NormStats, Episode,
                     SYNTH_GAIN, SYNTH_NOISE, SYNTH_TOL,
                     build_dataset, canonical_rgb, oracle_to_osc_action,
                     quat_to_R, rotvec_to_R, R_to_6d, _require_libero)

# ═════════════════════════════════════════════════════════════════════════════
# 판정 상수 — **실행 전에 고정한다. 결과를 보고 바꾸지 말 것.**
# ═════════════════════════════════════════════════════════════════════════════

T_H1_DEATH = 5.0        # [pt] C6-C0 가 이 이하 → H1 사망
T_M2_DEATH = 5.0        # [pt] C6-C8 가 이 이하 → 학습 Model 2 사망
T_PROCEED = 10.0        # [pt] C6-C0 가 이 이상이어야 진행
ALPHA = 0.05
POWER = 0.80

# 문서에 인용된 기준 검정력. verdict 리포트가 자동 출력한다.
MDE_REFERENCE = {500: 8.9, 1500: 5.1}


# ═════════════════════════════════════════════════════════════════════════════
# 통계 — scipy 없이 stdlib 만으로. 회사 머신에 scipy 가 없어도 판정은 나와야 한다
# ═════════════════════════════════════════════════════════════════════════════


def norm_cdf(x: float) -> float:
    """표준정규 CDF. math.erf 로 충분히 정확하다."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    표준정규 분위수 (Acklam 유리근사, |오차| < 1.15e-9).
    scipy.stats.norm.ppf 대체. α 를 바꿔도 검정력 계산이 계속 맞아야 한다.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 는 (0,1) 이어야 합니다: {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def wilson_ci(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """
    단일 비율의 Wilson score CI.

    Wald CI 를 쓰지 않는 이유: 성공률이 0% 나 100% 근처로 가면 Wald 는 폭이
    0 이 되거나 [0,1] 을 벗어난다. E0 의 C0 는 실제로 한 자릿수 성공률이 나올
    수 있으므로 이 경우가 예외가 아니라 기본값이다.
    """
    if n == 0:
        return (0.0, 1.0)
    z = norm_ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int,
                     alpha: float = ALPHA) -> tuple[float, float]:
    """
    두 비율 차의 Newcombe hybrid-score CI (method 10).

    Wald 차 CI 보다 극단 비율에서 coverage 가 훨씬 정직하다. E0 의 판정이
    "CI 가 0 을 포함하는가" 에 직접 걸려 있으므로 여기서 낙관하면 안 된다.
    """
    l1, u1 = wilson_ci(k1, n1, alpha)
    l2, u2 = wilson_ci(k2, n2, alpha)
    p1, p2 = (k1 / n1 if n1 else 0.0), (k2 / n2 if n2 else 0.0)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """pooled-variance 양측 z 검정 + Newcombe CI. 단위는 **pt(퍼센트포인트)**."""
    p1, p2 = (k1 / n1 if n1 else 0.0), (k2 / n2 if n2 else 0.0)
    diff = p1 - p2
    pp = (k1 + k2) / (n1 + n2) if (n1 + n2) else 0.0
    se = math.sqrt(pp * (1 - pp) * (1 / max(n1, 1) + 1 / max(n2, 1)))
    z = diff / se if se > 0 else 0.0
    lo, hi = newcombe_diff_ci(k1, n1, k2, n2)
    return {"diff_pt": diff * 100, "ci_lo_pt": lo * 100, "ci_hi_pt": hi * 100,
            "z": z, "p_value": 2 * (1 - norm_cdf(abs(z))),
            "ci_excludes_zero": (lo > 0) or (hi < 0)}


def mde(n_per_arm: int, alpha: float = ALPHA, power: float = POWER,
        p: float = 0.5) -> float:
    """
    최소탐지효과 [pt].

        MDE = (z_{1-α/2} + z_power) · sqrt(2·p(1-p)/n)

    p=0.5 는 최대분산 → 보수적(가장 큰 MDE). 실제 성공률이 낮으면 MDE 는
    더 작아지므로 여기 나온 값은 상한이다.

    검증: n=500 → 8.86pt, n=1500 → 5.11pt (문서 인용값 8.9 / 5.1 과 일치)
    """
    if n_per_arm <= 0:
        return float("nan")
    return ((norm_ppf(1 - alpha / 2) + norm_ppf(power))
            * math.sqrt(2 * p * (1 - p) / n_per_arm)) * 100


# ═════════════════════════════════════════════════════════════════════════════
# 판정
# ═════════════════════════════════════════════════════════════════════════════


def verdict(res: dict[str, tuple[int, int]]) -> dict:
    """
    Args:
        res: {"c0": (성공수, 시도수), "c6": (...), "c8": (...)}
    Returns:
        규칙별 발화 여부 + 최종 판정 + 근거 수치
    """
    for a in ARMS:
        if a not in res:
            raise ValueError(f"arm {a} 결과가 없습니다. 3개 arm 이 모두 필요합니다.")
    k, n = {a: res[a][0] for a in ARMS}, {a: res[a][1] for a in ARMS}
    rate = {a: (k[a] / n[a] * 100 if n[a] else float("nan")) for a in ARMS}

    t60 = two_proportion_test(k["c6"], n["c6"], k["c0"], n["c0"])   # C6 - C0
    t68 = two_proportion_test(k["c6"], n["c6"], k["c8"], n["c8"])   # C6 - C8

    d60, d68 = t60["diff_pt"], t68["diff_pt"]
    rules = {
        "H1_사망": (d60 <= T_H1_DEATH) or (not t60["ci_excludes_zero"]),
        "학습_Model2_사망": d68 <= T_M2_DEATH,          # == C8 >= C6 - 5
        "진행": (d60 >= T_PROCEED) and (d68 >= T_M2_DEATH),
        "재실행_LIBERO_Long_CALVIN": T_H1_DEATH < d60 < T_PROCEED,
    }
    # 우선순위: 치명적인 것부터. H1 이 죽으면 나머지는 의미가 없다.
    for name in ("H1_사망", "학습_Model2_사망", "진행", "재실행_LIBERO_Long_CALVIN"):
        if rules[name]:
            primary = name
            break
    else:
        primary = "판정불가"

    action = {
        "H1_사망": "가설 자체가 죽었다. 정답 인터페이스로도 monolith 를 못 이긴다 "
                  "→ 2모델 분리를 접고 단일 모델 + 벤더 WBC 로 간다.",
        "학습_Model2_사망": "목표 포즈만 있으면 closed-form OSC 로 충분하다 "
                          "→ 소뇌를 학습시키지 말고 Model 1 + closed-form IK 로 간다.",
        "진행": "분리가 이기고, 그 이득이 학습된 소뇌에서 온다 → E1 로 진행한다.",
        "재실행_LIBERO_Long_CALVIN":
            f"신호는 있으나 n={n['c6']} 의 MDE({mde(n['c6']):.1f}pt)와 구분이 어렵다 "
            "→ LIBERO-Long / CALVIN ABC→D 에서 재실행한다.",
        "판정불가": "규칙이 하나도 발화하지 않았다. 입력 수치를 확인하라.",
    }[primary]

    return {"rate_pct": rate, "successes": k, "trials": n,
            "c6_minus_c0": t60, "c6_minus_c8": t68,
            "rules": rules, "verdict": primary, "action": action,
            "mde_pt": {str(a): mde(n[a]) for a in ARMS},
            "mde_reference": MDE_REFERENCE}


def _dw(s: str) -> int:
    """표시 폭. 한글/전각은 2칸을 먹으므로 str.ljust 로는 표가 어긋난다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int) -> str:
    """표시 폭 기준 좌측 정렬 + 초과분 절단."""
    out = ""
    for c in s:
        if _dw(out + c) > n:
            break
        out += c
    return out + " " * (n - _dw(out))


def print_verdict(v: dict, note: str = "") -> None:
    W = 78
    print("\n" + "═" * W)
    print(" E0 판정 리포트 — GATE-2")
    if note:
        print(f" {note}")
    print("═" * W)
    print(f" {'arm':4s} {_pad('설명', 34)} {'성공/시도':>11s} {'성공률':>7s}  95% CI")
    for a in ARMS:
        k, n = v["successes"][a], v["trials"][a]
        lo, hi = wilson_ci(k, n)
        print(f" {a.upper():4s} {_pad(ARM_DESC[a], 34)} {k:5d}/{n:<5d} "
              f"{v['rate_pct'][a]:6.1f}%  [{lo*100:5.1f}, {hi*100:5.1f}]")

    print("─" * W)
    print(f" {'대비':14s} {'Δ(pt)':>8s}  {'95% CI (Newcombe)':>22s} {'z':>7s} {'p':>10s}")
    for label, key in (("C6 − C0", "c6_minus_c0"), ("C6 − C8", "c6_minus_c8")):
        t = v[key]
        print(f" {label:14s} {t['diff_pt']:+8.1f}  "
              f"[{t['ci_lo_pt']:+8.1f}, {t['ci_hi_pt']:+8.1f}] "
              f"{t['z']:7.2f} {t['p_value']:10.2e}")

    print("─" * W)
    print(" 검정력 (α=0.05 양측, power=0.80, p=0.5 최대분산 기준)")
    for n_ref, m_ref in sorted(MDE_REFERENCE.items()):
        seeds = n_ref // 500
        print(f"   n={n_ref:<5d} (seed {seeds}개 pool) → MDE {mde(n_ref):.1f}pt "
              f"[문서 인용 {m_ref}pt]")
    n_act = v["trials"]["c6"]
    print(f"   실제 n={n_act:<5d}              → MDE {mde(n_act):.1f}pt")
    if mde(n_act) > T_H1_DEATH:
        print(f"   ⚠️  MDE 가 ±{T_H1_DEATH:.0f}pt 임계보다 큽니다 — "
              f"±5pt 규칙은 이 n 으로 신뢰할 수 없습니다. seed 를 더 pool 하십시오.")

    print("─" * W)
    print(" 판정 규칙 (실행 전 고정)")
    desc = {
        "H1_사망": f"C6−C0 ≤ +{T_H1_DEATH:.0f} 또는 CI 가 0 포함",
        "학습_Model2_사망": f"C8 ≥ C6 − {T_M2_DEATH:.0f}",
        "진행": f"C6−C0 ≥ +{T_PROCEED:.0f} AND C6−C8 ≥ +{T_M2_DEATH:.0f}",
        "재실행_LIBERO_Long_CALVIN": f"+{T_H1_DEATH:.0f} < C6−C0 < +{T_PROCEED:.0f}",
    }
    for name, fired in v["rules"].items():
        print(f"   [{'x' if fired else ' '}] {_pad(name, 26)} {_pad(desc[name], 34)} "
              f"→ {'예' if fired else '아니오'}")
    print("─" * W)
    print(f" ▶ 판정: {v['verdict']}")
    print(f"   {v['action']}")
    print("═" * W + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 정책 러너 — torch 체크포인트 / numpy 폴백을 같은 인터페이스로 감싼다
# ═════════════════════════════════════════════════════════════════════════════


class PolicyRunner:
    """predict(rgb, proprio, goal) → (K, ACTION_DIM) **역정규화된** action chunk."""

    def __init__(self, impl, norm: NormStats, use_goal: bool, chunk_k: int = CHUNK_K):
        self.impl, self.norm, self.use_goal, self.chunk_k = impl, norm, use_goal, chunk_k

    def predict(self, rgb: np.ndarray, proprio: np.ndarray, goal: np.ndarray) -> np.ndarray:
        pn = (proprio - self.norm.proprio_mean) / self.norm.proprio_std
        gn = (goal - self.norm.goal_mean) / self.norm.goal_std
        if not self.use_goal:
            gn = np.zeros_like(gn)
        out = self.impl(rgb, pn.astype(np.float32), gn.astype(np.float32))
        return out * self.norm.action_std + self.norm.action_mean


def load_policy(ckpt: str, device: Optional[str] = None) -> PolicyRunner:
    """확장자로 백엔드를 고른다: .pt → torch, .pkl → numpy 폴백."""
    if ckpt.endswith(".pkl"):
        from models import NumpyLinearPolicy
        p = NumpyLinearPolicy.load(ckpt)
        return PolicyRunner(lambda r, pr, g: p.predict(r, pr, g), p.norm,
                            p.use_goal, p.chunk_k)

    from models import PolicyConfig, build_policy, require_torch
    torch = require_torch()
    from train import pick_device
    dev = pick_device(device)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = PolicyConfig(**ck["cfg"])
    model = build_policy(ck["arm"], cfg).to(dev).eval()
    model.load_state_dict(ck["state_dict"])

    from train import rgb_to_float

    def _run(rgb, proprio, goal):
        with torch.no_grad():
            # 학습 로더와 **같은** 전처리 함수를 쓴다 — 갈리면 성공률이 0 이 된다
            r = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(dev)
            y = model(rgb_to_float(r),
                      torch.from_numpy(proprio).unsqueeze(0).to(dev),
                      torch.from_numpy(goal).unsqueeze(0).to(dev))
        return y[0].float().cpu().numpy()

    return PolicyRunner(_run, NormStats.from_dict(ck["norm"]),
                        ARM_USES_GOAL[ck["arm"]], cfg.chunk_k)


# ═════════════════════════════════════════════════════════════════════════════
# rollout 환경
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Obs:
    """arm 3종이 공통으로 소비하는 최소 관측."""
    rgb: np.ndarray          # (n_cam, H, W, 3) uint8, canonical_rgb 적용됨
    proprio: np.ndarray      # (PROPRIO_DIM,)
    ee_pos: np.ndarray       # (3,)
    ee_rot: np.ndarray       # (3,3)


class LiberoRolloutEnv:
    """
    LIBERO OffScreenRenderEnv 래퍼.

    ⚠️ 확인 필요 — 아래 import 경로/메서드는 공개 문서 기준으로 작성했고
       실물로 검증하지 못했다. 회사 첫 실행 시 **이 클래스만** 고치면 된다
       (tools/measure_loop_rate.py 의 G1Adapter 와 같은 패턴).

       확인할 것 4가지:
         1) OffScreenRenderEnv 의 import 경로와 생성 인자
         2) obs 키: agentview_image / robot0_eye_in_hand_image /
                    robot0_joint_pos / robot0_gripper_qpos /
                    robot0_eef_pos / robot0_eef_quat (xyzw?)
         3) step() 이 성공 시 done=True 를 주는지, 아니면 check_success() 인지
         4) task_suite 의 task index 와 hdf5 파일 정렬 순서가 일치하는지
            (일치를 가정하지 않고 **이름으로 매칭**하도록 짜 뒀다)
    """

    def __init__(self, suite: str, task_key: str, img_size: int = 128, seed: int = 0):
        _require_libero()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.task_suite = benchmark.get_benchmark_dict()[suite]()
        self.task_idx = self._resolve_task_index(self.task_suite, task_key)
        task = self.task_suite.get_task(self.task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl,
                                      camera_heights=img_size, camera_widths=img_size)
        self.env.seed(seed)
        self.init_states = self.task_suite.get_task_init_states(self.task_idx)

    @staticmethod
    def _resolve_task_index(task_suite, task_key: str) -> int:
        """
        hdf5 파일 이름으로 benchmark task index 를 찾는다.
        인덱스 순서를 가정하면 조용히 **다른 태스크의 환경**에서 평가하게 되고,
        그 결과는 전부 쓰레기가 되면서 원인 파악이 거의 불가능하다.
        """
        want = task_key.replace("_demo", "").strip().lower()
        n = task_suite.n_tasks
        for i in range(n):
            bd = task_suite.get_task(i).bddl_file
            if os.path.splitext(os.path.basename(bd))[0].strip().lower() == want:
                return i
        avail = [os.path.splitext(os.path.basename(task_suite.get_task(i).bddl_file))[0]
                 for i in range(n)]
        raise SystemExit(f"[eval] task '{want}' 를 benchmark 에서 못 찾음.\n"
                         f"  후보: {avail}")

    def _wrap(self, obs) -> Obs:
        rgb = canonical_rgb(np.stack([obs["agentview_image"],
                                      obs["robot0_eye_in_hand_image"]], 0))
        pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        rot = quat_to_R(obs["robot0_eef_quat"]).astype(np.float32)
        jq = np.asarray(obs["robot0_joint_pos"], np.float32)
        gq = np.asarray(obs["robot0_gripper_qpos"], np.float32)[:2]
        proprio = np.concatenate([jq, gq, pos, R_to_6d(rot)]).astype(np.float32)
        return Obs(np.ascontiguousarray(rgb), proprio, pos, rot)

    def reset(self, ep: Episode) -> Obs:
        self.env.reset()
        # 데모와 같은 초기 상태에서 시작해야 hindsight oracle 이 유효하다.
        obs = self.env.set_init_state(self.init_states[ep.demo_id])
        return self._wrap(obs)

    def step(self, action: np.ndarray) -> tuple[Obs, bool]:
        obs, _r, done, _info = self.env.step(np.asarray(action, np.float64))
        return self._wrap(obs), bool(done)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


class SyntheticRolloutEnv:
    """
    --smoke 전용 합성 rollout.

    OSC_POSE 명령을 그대로 적분하는 1차 플랜트다. 물리도 접촉도 없다.
    ⚠️ 여기서 나오는 성공률은 **어떤 과학적 의미도 없다.** 존재 이유는
       "데이터→학습→rollout→통계→판정" 배선이 안 끊기는지 확인하는 것뿐이다.
    """

    # dataset.py 의 합성 데모 생성과 **같은 상수**를 쓴다. 여기서 갈리면
    # 데모를 완벽히 모방해도 과제가 안 풀린다.
    SUCCESS_TOL = SYNTH_TOL
    GAIN = SYNTH_GAIN
    NOISE = SYNTH_NOISE

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.ep: Optional[Episode] = None
        self.t = 0

    def reset(self, ep: Episode) -> Obs:
        self.ep, self.t = ep, 0
        self.p = ep.ee_pos[0].astype(np.float64).copy()
        self.R = ep.ee_rot[0].astype(np.float64).copy()
        return self._obs()

    def _obs(self) -> Obs:
        ep = self.ep
        i = min(self.t, len(ep.rgb) - 1)
        jq = ep.proprio[i, :7]
        gq = ep.proprio[i, 7:9]
        pos = self.p.astype(np.float32)
        rot = self.R.astype(np.float32)
        proprio = np.concatenate([jq, gq, pos, R_to_6d(rot)]).astype(np.float32)
        return Obs(ep.rgb[i], proprio, pos, rot)

    def step(self, action: np.ndarray) -> tuple[Obs, bool]:
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self.p += a[:3] * OSC_POS_MAX * self.GAIN + self.rng.normal(0, self.NOISE, 3)
        self.R = rotvec_to_R(a[3:6] * OSC_ROT_MAX * self.GAIN) @ self.R
        self.t += 1
        goal = self.ep.ee_pos[-1]
        done = bool(np.linalg.norm(self.p - goal) < self.SUCCESS_TOL and self.t > 5)
        return self._obs(), done

    def close(self) -> None:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# rollout 루프
# ═════════════════════════════════════════════════════════════════════════════


def rollout_episode(arm: str, env, ep: Episode, runner: Optional[PolicyRunner],
                    max_steps: int, exec_horizon: int, c8_kp: float) -> bool:
    """
    한 에피소드 rollout. 성공하면 True.

    C0/C6 는 chunk 를 exec_horizon 스텝마다 다시 뽑는다(open-loop 실행 구간).
    C8 은 closed-form 이라 매 스텝 재계산해도 비용이 0 이므로 그렇게 한다 —
    C8 을 최대한 강하게 두는 것은 "학습 Model 2 사망" 규칙 쪽으로 보수적이다.
    (우리 가설에 불리한 방향으로 기울이는 것이 옳다.)
    """
    obs = env.reset(ep)
    chunk, chunk_t0 = None, 0
    for t in range(max_steps):
        if arm == "c8":
            goal = ep.oracle_at(t, obs.ee_pos, obs.ee_rot)
            a = oracle_to_osc_action(goal, obs.ee_pos, obs.ee_rot,
                                     obs.ee_pos, obs.ee_rot, kp=c8_kp)
        else:
            if chunk is None or (t - chunk_t0) >= exec_horizon or (t - chunk_t0) >= len(chunk):
                goal = (ep.oracle_at(t, obs.ee_pos, obs.ee_rot)
                        if ARM_USES_GOAL[arm] else np.zeros(GOAL_DIM, np.float32))
                chunk = runner.predict(obs.rgb, obs.proprio, goal)
                chunk_t0 = t
            a = np.clip(chunk[t - chunk_t0], -1.0, 1.0)
        obs, done = env.step(a)
        if done:
            return True
    return False


def evaluate_arm(arm: str, episodes: list[Episode], runner: Optional[PolicyRunner],
                 smoke: bool, seed: int, max_steps: int, exec_horizon: int,
                 c8_kp: float, img_size: int, verbose: bool = True) -> dict:
    """arm 하나를 모든 eval 에피소드에 대해 rollout. 태스크별 성공도 함께 기록."""
    per_task: dict[str, list[int]] = {}
    t0, env, key = time.time(), None, None
    n_ok = 0

    for i, ep in enumerate(episodes):
        k = f"{ep.suite}/{ep.task_key}" if ep.suite else f"task{ep.task_id}"
        if smoke:
            env = env or SyntheticRolloutEnv(seed=seed)
        elif k != key:                       # 태스크가 바뀔 때만 env 재생성 (비싸다)
            if env is not None:
                env.close()
            env = LiberoRolloutEnv(ep.suite, ep.task_key, img_size, seed)
            key = k
        ok = rollout_episode(arm, env, ep, runner, max_steps, exec_horizon, c8_kp)
        per_task.setdefault(k, []).append(int(ok))
        n_ok += int(ok)
        if verbose and (i + 1) % max(len(episodes) // 10, 1) == 0:
            print(f"  [{arm.upper()}] {i+1}/{len(episodes)}  누적 {n_ok}  "
                  f"({n_ok/(i+1)*100:.1f}%)  {(time.time()-t0)/60:.1f}분")
    if env is not None:
        env.close()

    n = len(episodes)
    lo, hi = wilson_ci(n_ok, n)
    return {"arm": arm, "seed": seed, "successes": n_ok, "trials": n,
            "success_rate_pct": n_ok / n * 100 if n else 0.0,
            "ci95_pct": [lo * 100, hi * 100],
            "per_task": {k: [int(sum(v)), len(v)] for k, v in per_task.items()},
            "minutes": (time.time() - t0) / 60, "smoke": bool(smoke)}


# ═════════════════════════════════════════════════════════════════════════════
# 리포트 집계
# ═════════════════════════════════════════════════════════════════════════════


def aggregate(paths: list[str]) -> tuple[dict, dict]:
    """
    결과 JSON 여러 개 → arm 별 (성공수, 시도수) pool. seed 3개를 합치는 곳.

    ⚠️ 통계적 주의 — 이 pooling 은 **anti-conservative** 하다.
       seed 3개는 같은 500개 held-out 에피소드를 다시 푼 것이므로 1500 trial
       이 독립이 아니다(에피소드 난이도가 공유된다). 두 비율 검정은 독립을
       가정하므로 진짜 CI 는 여기 출력보다 다소 넓다.
       → 그래서 seed 별 성공률도 함께 출력한다. seed 간 산포가 pooled CI 폭보다
         크면 pooled p-value 를 믿지 말고 seed 를 더 돌려야 한다.
    """
    agg: dict[str, list[int]] = {a: [0, 0] for a in ARMS}
    per_seed: dict[str, list[tuple[int, int, int]]] = {a: [] for a in ARMS}
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        a = r["arm"]
        if a not in agg:
            continue
        agg[a][0] += r["successes"]
        agg[a][1] += r["trials"]
        per_seed[a].append((r["seed"], r["successes"], r["trials"]))
    for a in ARMS:
        per_seed[a].sort()
    return ({a: (agg[a][0], agg[a][1]) for a in ARMS}, per_seed)


def print_per_seed(per_seed: dict) -> None:
    """seed 별 성공률. pooled CI 를 믿어도 되는지 눈으로 판단하는 근거."""
    print("─" * 78)
    print(" seed 별 성공률 (pooling 이 정당한지 확인용 — aggregate() 주의사항 참고)")
    for a in ARMS:
        rows = per_seed[a]
        if not rows:
            continue
        rates = [s / t * 100 for _, s, t in rows if t]
        spread = (max(rates) - min(rates)) if rates else 0.0
        cells = "  ".join(f"s{sd}:{s/t*100:5.1f}%" for sd, s, t in rows if t)
        flag = "  ⚠️ seed 산포가 큼" if spread > 5.0 else ""
        print(f"   {a.upper():3s} {cells}   (산포 {spread:.1f}pt){flag}")
        if len(rows) > 1 and len({s for _, s, _ in rows}) == 1:
            print(f"       ⚠️ {a.upper()} 는 seed 간 결과가 완전히 동일합니다 "
                  f"(결정론적). 실효 n 은 {rows[0][2]} 이며 pooled n 이 아닙니다.")


def report(results_dir: str) -> int:
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    paths = [p for p in paths if not p.endswith("verdict.json")]
    if not paths:
        raise SystemExit(f"[eval] 결과 JSON 이 없습니다: {results_dir}")
    res, per_seed = aggregate(paths)
    missing = [a for a in ARMS if res[a][1] == 0]
    if missing:
        raise SystemExit(f"[eval] arm {missing} 결과 없음 — 3개 arm 이 모두 필요합니다.")

    smoke = any(json.load(open(p)).get("smoke") for p in paths)
    v = verdict(res)
    v["per_seed"] = {a: [{"seed": s, "successes": k, "trials": n}
                         for s, k, n in per_seed[a]] for a in ARMS}
    v["source_files"] = paths
    note = "seed pool: " + ", ".join(
        f"{a.upper()}={[s for s, _, _ in per_seed[a]]}" for a in ARMS)
    if smoke:
        note = "⚠️ SMOKE (합성 데이터) — 아래 수치는 과학적 의미가 없습니다 | " + note
    print_verdict(v, note)
    print_per_seed(per_seed)
    print(" ⚠️ pooled CI 는 anti-conservative 합니다 (같은 에피소드를 seed 마다 재사용).")
    print("═" * 78 + "\n")

    out = os.path.join(results_dir, "verdict.json")
    with open(out, "w") as f:
        json.dump(v, f, indent=2, ensure_ascii=False)
    print(f"[eval] 판정 저장 → {out}")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# 통계 self-test — 문서 인용 수치와 코드가 일치하는지 검사
# ═════════════════════════════════════════════════════════════════════════════


def _stats_self_test() -> None:
    print("── 통계 self-test ──")
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-5, norm_ppf(0.975)
    assert abs(norm_ppf(0.80) - 0.8416212) < 1e-5, norm_ppf(0.80)
    for n, want in MDE_REFERENCE.items():
        got = mde(n)
        print(f"  MDE(n={n:5d}) = {got:.2f}pt   [문서 {want}pt]")
        assert abs(got - want) < 0.1, f"MDE(n={n}) {got:.2f} != 문서값 {want}"

    # 알려진 값 대조: 50/500 vs 100/500
    t = two_proportion_test(100, 500, 50, 500)
    print(f"  100/500 vs 50/500 : Δ{t['diff_pt']:+.1f}pt "
          f"CI[{t['ci_lo_pt']:+.1f},{t['ci_hi_pt']:+.1f}] z={t['z']:.3f} p={t['p_value']:.2e}")
    # 손으로 검산: pooled p=150/1000=0.15,
    #   se = sqrt(0.15·0.85·(1/500+1/500)) = 0.0225832,  z = 0.10/0.0225832 = 4.428
    assert abs(t["diff_pt"] - 10.0) < 1e-9
    assert abs(t["z"] - 4.428) < 0.01, t["z"]
    assert t["ci_excludes_zero"]

    # Wilson: 0/100 은 Wald 와 달리 폭이 0 이 아니어야 한다
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and 0.02 < hi < 0.05, (lo, hi)
    print(f"  Wilson(0/100)     : [{lo*100:.2f}, {hi*100:.2f}]%  (Wald 였다면 폭 0)")

    # 판정 규칙 4갈래가 전부 발화하는지
    cases = {
        "H1_사망": {"c0": (300, 1500), "c6": (310, 1500), "c8": (100, 1500)},
        "학습_Model2_사망": {"c0": (300, 1500), "c6": (600, 1500), "c8": (590, 1500)},
        "진행": {"c0": (300, 1500), "c6": (600, 1500), "c8": (400, 1500)},
        "재실행_LIBERO_Long_CALVIN": {"c0": (300, 1500), "c6": (420, 1500),
                                     "c8": (200, 1500)},
    }
    for want, r in cases.items():
        got = verdict(r)["verdict"]
        print(f"  규칙 {want:26s} → {got}")
        assert got == want, f"판정 규칙 오류: {want} 를 기대했으나 {got}"
    print("── 통계 self-test 통과 ──")


def _synth_self_test() -> None:
    """
    합성 플랜트 정합성.

    **데모 action 을 그대로 재생하면 과제가 풀려야 한다.** 이게 깨지면
    "완벽히 모방해도 실패하는" 환경이 되고, smoke 의 C0/C6 수치는 배선이
    맞는지조차 말해주지 못한다. (초안에서 실제로 이 함정에 빠졌다.)
    """
    print("── 합성 플랜트 self-test ──")
    from dataset import make_synthetic_episodes
    eps = make_synthetic_episodes(n_episodes=8, seed=11)

    env = SyntheticRolloutEnv(seed=0)
    dists = []
    for ep in eps:
        obs = env.reset(ep)
        for t in range(len(ep)):
            obs, _ = env.step(ep.actions[t])
        dists.append(float(np.linalg.norm(obs.ee_pos - ep.ee_pos[-1])))
    print(f"  데모 replay 최종오차 : 평균 {np.mean(dists)*100:.2f}cm "
          f"최대 {np.max(dists)*100:.2f}cm  (허용 {SYNTH_TOL*100:.0f}cm)")
    assert np.max(dists) < SYNTH_TOL, "데모를 그대로 재생해도 도달 못함 — 플랜트 불일치"

    # oracle-IK(C8)도 당연히 풀려야 한다
    ok = sum(rollout_episode("c8", env, ep, None, 60, 8, 1.0) for ep in eps)
    print(f"  C8 oracle-IK 성공    : {ok}/{len(eps)}")
    assert ok == len(eps), "C8 이 합성 플랜트조차 못 푼다 — oracle 변환 확인 필요"
    print("── 합성 플랜트 self-test 통과 ──")


# ═════════════════════════════════════════════════════════════════════════════


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E0 rollout 평가 + 통계 판정")
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    ap.add_argument("--ckpt", default=None, help="C0/C6 체크포인트 (C8 은 불필요)")
    ap.add_argument("--data", default=os.environ.get("LIBERO_DATA"))
    ap.add_argument("--train-demos", type=int, default=25,
                    help="이 개수 이후가 eval held-out. train.py 와 반드시 같게")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400,
                    help="에피소드 최대 스텝 (LIBERO Spatial/Object 관례 220~500)")
    ap.add_argument("--exec-horizon", type=int, default=8,
                    help="C0/C6 가 chunk 를 재계획하는 주기. K=16 의 절반이 표준")
    ap.add_argument("--c8-kp", type=float, default=1.0, help="C8 pure-pursuit 이득")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="결과 JSON 경로")
    ap.add_argument("--results", default=None, help="--report 용 결과 디렉토리")
    ap.add_argument("--report", action="store_true", help="판정만 재출력")
    ap.add_argument("--stats-test", action="store_true", help="통계 함수 self-test")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    if a.stats_test:
        _stats_self_test()
        _synth_self_test()
        return 0
    if a.report:
        return report(a.results or os.path.join(_HERE, "results"))
    if a.arm is None:
        raise SystemExit("--arm {c0,c6,c8} 또는 --report 또는 --smoke 가 필요합니다.")

    if a.smoke:
        a.max_steps = min(a.max_steps, 60)
        a.img_size = 32
    elif not a.data:
        raise SystemExit("[eval] --data <LIBERO 루트> 가 필요합니다.")

    ds = build_dataset(smoke=a.smoke, data_root=a.data, split="eval",
                       train_demos=a.train_demos, seed=a.seed)
    runner = None
    if a.arm != "c8":
        if not a.ckpt:
            raise SystemExit(f"[eval] arm {a.arm} 에는 --ckpt 가 필요합니다.")
        runner = load_policy(a.ckpt, a.device)
        # eval 은 반드시 train 의 정규화 통계를 쓴다 (체크포인트에 들어 있다).
        # 여기서 eval 셋 통계를 다시 fit 하면 입력 분포가 미묘하게 달라진다.

    print(f"[eval] arm={a.arm.upper()} seed={a.seed} 에피소드={len(ds.episodes)} "
          f"max_steps={a.max_steps} exec_horizon={a.exec_horizon}")
    r = evaluate_arm(a.arm, ds.episodes, runner, a.smoke, a.seed,
                     a.max_steps, a.exec_horizon, a.c8_kp, a.img_size)
    print(f"[eval] {a.arm.upper()}: {r['successes']}/{r['trials']} = "
          f"{r['success_rate_pct']:.1f}%  CI[{r['ci95_pct'][0]:.1f},{r['ci95_pct'][1]:.1f}]")

    out = a.out or os.path.join(_HERE, "results", "smoke" if a.smoke else "runs",
                                f"{a.arm}_seed{a.seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"[eval] 결과 저장 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
