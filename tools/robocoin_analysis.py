#!/usr/bin/env python3
"""
RoboCOIN 오프라인 분석 3종 — 로봇 없이 이 프로젝트의 핵심 쟁점 3개를 닫는다.

RoboCOIN: BAAI 공개 실기체 Galbot G1 양팔 데이터셋
          공개(non-gated) 5종 = 2,974 ep / 2,020,721 frames / 18.7h / ~30.8 GB
          (2026-07-31 HF API 로 확인. robot_type='Galbot_G1')
          gated 13종이 추가로 있음 — 전체 18종 164.4 GB, 대부분 비디오

왜 이게 가치 있는가
──────────────────
아래 세 질문은 지금까지 전부 "미확인"으로 남아 있었고, 각각이 설계 결정을
바꿉니다. 그리고 **로봇도 GPU도 시뮬도 없이** 이 데이터만으로 답이 나옵니다.

  A1  목 관절 운동 통계   → 헤드 프레임 인터페이스가 성립하는가
  A2  psi(arm angle) 분포 → 인터페이스에 psi 스칼라가 필수인가
  A3  양손 가시율         → head->base 자가 캘리브레이션 채널이 살아있는가

⚠️ RoboCOIN 에는 F/T 채널이 없다. 힘 관련 질문은 여기서 답할 수 없고,
   컨버터 포크 후 자체 수집으로만 얻는다.

사용
────
    # 데이터 스키마부터 확인 (첫 실행 시 반드시)
    python tools/robocoin_analysis.py --data <path> --inspect

    # 분석 실행
    python tools/robocoin_analysis.py --data <path> --all
    python tools/robocoin_analysis.py --data <path> --neck --psi

스키마는 tools/robocoin_schema.py 에 실측값으로 고정돼 있다.
⚠️ RoboCOIN(21-D) 과 GalbotSDK(23-D) 는 레이아웃이 다르다. 섞으면 조용히 오염된다.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
from g1_kinematics import G1Arm, G1Head, se3_inv  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 스키마
#
# RoboCOIN 스키마는 tools/robocoin_schema.py 가 단일 진실 원천이다.
# (실제 meta/info.json 에서 확인, 2026-07-31)
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ 2026-07-31 정정 — 실제 meta/info.json 확인 결과 RoboCOIN 은 SDK 와 다르다.
#
#     SDK (MCAP, 23-D)  : leg(5)   head(2) L팔(7) L그리퍼(1) R팔(7) R그리퍼(1)
#     RoboCOIN (21-D)   : torso(3) head(2) L팔(7) L그리퍼(1) R팔(7) R그리퍼(1)
#                         ^^^^^^^^ 다리 2개 없음
#
# 이전 코드는 SDK 레이아웃을 썼고, 그대로 돌렸으면 head 를 [5:7] 에서 읽어
# **왼팔 j1,j2 를 목 관절로 착각**했을 것이다. 조용한 오염이고 결과가 전부 무의미해진다.
# 단일 진실 원천은 tools/robocoin_schema.py.
from robocoin_schema import (            # noqa: E402
    STATE_IDX as IDX,
    ACTION_IDX,
    STATE_DIM,
    ACTION_DIM,
    FPS as SCHEMA_FPS,
    PUBLIC_DATASETS,
)

# ⚠️ 확인 필요 — parquet 컬럼명. --inspect 로 확인 후 수정.
COLUMNS = {
    "state": "observation.state",
    "action": "action",
    "episode": "episode_index",
    "frame": "frame_index",
    "timestamp": "timestamp",
}

FPS_DEFAULT = 30.0


def load_episodes(data_dir: str, limit: Optional[int] = None) -> list[np.ndarray]:
    """parquet 들을 읽어 에피소드별 (T, 21) 상태 배열 리스트로 (RoboCOIN 스키마)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow 필요:  mamba install -n physical_ai pyarrow")

    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"parquet 을 찾지 못했습니다: {data_dir}")
    if limit:
        files = files[:limit]

    eps: list[np.ndarray] = []
    for f in files:
        try:
            tbl = pq.read_table(f)
            col = COLUMNS["state"]
            if col not in tbl.column_names:
                continue
            arr = np.array([np.asarray(x, dtype=float) for x in tbl[col].to_pylist()])
            if arr.ndim == 2 and arr.shape[1] == STATE_DIM:
                # 한 파일에 여러 에피소드가 있으면 분리
                ecol = COLUMNS["episode"]
                if ecol in tbl.column_names:
                    e = np.asarray(tbl[ecol].to_pylist())
                    for eid in np.unique(e):
                        eps.append(arr[e == eid])
                else:
                    eps.append(arr)
        except Exception as ex:
            print(f"  [skip] {os.path.basename(f)}: {ex}")
    return eps


def inspect(data_dir: str) -> None:
    """스키마 확인 — 첫 실행 시 반드시."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow 필요:  mamba install -n physical_ai pyarrow")

    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    print(f"parquet 파일 {len(files)}개 발견\n")
    if not files:
        return
    tbl = pq.read_table(files[0])
    print(f"첫 파일: {files[0]}")
    print(f"행 수  : {tbl.num_rows}\n")
    print("컬럼:")
    for name, typ in zip(tbl.column_names, tbl.schema.types):
        sample = tbl[name][0].as_py()
        if isinstance(sample, list):
            info = f"list[{len(sample)}]"
            if len(sample) <= 25:
                info += f" = {np.round(np.asarray(sample, dtype=float), 4).tolist()}"
        else:
            info = repr(sample)[:70]
        print(f"  {name:34s} {str(typ)[:24]:26s} {info}")

    print("\n⚠️ 위 컬럼명을 COLUMNS 딕셔너리와 대조하십시오.")
    print(f"   상태 벡터는 {STATE_DIM}차원이어야 합니다 (RoboCOIN 스키마):")
    print("   [0:3] torso  [3:5] head  [5:12] L팔  [12] L그리퍼  [13:20] R팔  [20] R그리퍼")
    print("   ⚠️ SDK(MCAP) 는 23차원이고 레이아웃이 다릅니다 — 섞으면 조용히 오염됩니다.")



def circ_range(rad) -> float:
    """
    원형 범위 = 모든 각도를 포함하는 최소 호 [rad].

    ⚠️ psi 는 (-pi, pi] 로 감기므로 선형 max-min 을 쓰면 안 된다.
       실측에서 이 버그로 산포가 22.3° 대신 359.7° 로 나왔고,
       판정이 "psi 생략 가능" 에서 "psi 필수" 로 뒤집혔다.
       interface.py 가 Euler/quaternion 을 금지한 것과 같은 이유다.
    """
    a = np.sort(np.mod(np.asarray(rad, dtype=float), 2 * np.pi))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return float(2 * np.pi - np.max(gaps))


# ─────────────────────────────────────────────────────────────────────────────
# A1 — 목 관절 운동 통계
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NeckResult:
    n_episodes: int
    n_frames: int
    fps: float
    pan_range_deg: float
    tilt_range_deg: float
    speed_p50_dps: float
    speed_p95_dps: float
    speed_max_dps: float
    # 청크 길이 동안 누적 각도 (이게 판정 기준)
    chunk_0p5s_p50_deg: float
    chunk_0p5s_p95_deg: float
    chunk_2p0s_p50_deg: float
    chunk_2p0s_p95_deg: float
    # 0.6 m 도달거리 환산 목표 오차
    err_0p5s_p95_mm: float
    err_2p0s_p95_mm: float
    verdict: str


MM_PER_DEG_AT_0P6M = 10.5   # test_kinematics T5 에서 측정


def analyze_neck(eps: list[np.ndarray], fps: float) -> NeckResult:
    """
    목이 실제로 얼마나 움직이는가.

    헤드 프레임 인터페이스의 성패가 여기 달려 있다. Model 1 이 헤드 프레임으로
    목표를 내고 Model 2 가 그것을 소비하는 사이에 목이 움직이면, 그만큼 목표가
    조용히 미끄러진다. 0.6 m 도달거리에서 **목 1도당 약 10.5 mm** (측정값).
    """
    pans, tilts, speeds = [], [], []
    chunk_05, chunk_20 = [], []
    n05, n20 = max(1, int(0.5 * fps)), max(1, int(2.0 * fps))

    for e in eps:
        h = e[:, IDX["head"]]                       # (T, 2)
        if len(h) < 2:
            continue
        pans.append(h[:, 0]); tilts.append(h[:, 1])
        d = np.linalg.norm(np.diff(h, axis=0), axis=1)      # rad/frame
        speeds.append(np.degrees(d) * fps)

        # 슬라이딩 윈도우 내 총 이동량 (누적 절대 변화가 아니라 순 변위)
        for n, acc in ((n05, chunk_05), (n20, chunk_20)):
            if len(h) > n:
                disp = np.linalg.norm(h[n:] - h[:-n], axis=1)
                acc.append(np.degrees(disp))

    if not pans:
        raise ValueError("목 데이터 없음")

    P = np.concatenate(pans); T = np.concatenate(tilts)
    S = np.concatenate(speeds)
    C05 = np.concatenate(chunk_05) if chunk_05 else np.array([0.0])
    C20 = np.concatenate(chunk_20) if chunk_20 else np.array([0.0])

    c05_95, c20_95 = float(np.percentile(C05, 95)), float(np.percentile(C20, 95))
    e05, e20 = c05_95 * MM_PER_DEG_AT_0P6M, c20_95 * MM_PER_DEG_AT_0P6M

    if c20_95 <= 2.0:
        v = ("PASS — 목이 거의 안 움직인다. 헤드 프레임 유지 가능. "
             "촬영시각 변환만 넣으면 충분.")
    elif c20_95 <= 6.0:
        v = ("MARGINAL — 촬영시각 head_joint 로의 변환이 필수. "
             "인터페이스에서 torso_base_link 로 변환할 것.")
    else:
        v = ("FAIL — 헤드 프레임 폐기. Model 1 이 torso_base_link 로 직접 예측해야 함. "
             "이 크기의 오차는 조용히 누적되고 시뮬에서 절대 안 보인다.")

    return NeckResult(
        n_episodes=len(eps), n_frames=int(sum(len(e) for e in eps)), fps=fps,
        pan_range_deg=float(np.degrees(P.max() - P.min())),
        tilt_range_deg=float(np.degrees(T.max() - T.min())),
        speed_p50_dps=float(np.percentile(S, 50)),
        speed_p95_dps=float(np.percentile(S, 95)),
        speed_max_dps=float(S.max()),
        chunk_0p5s_p50_deg=float(np.percentile(C05, 50)),
        chunk_0p5s_p95_deg=c05_95,
        chunk_2p0s_p50_deg=float(np.percentile(C20, 50)),
        chunk_2p0s_p95_deg=c20_95,
        err_0p5s_p95_mm=e05, err_2p0s_p95_mm=e20,
        verdict=v,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2 — psi (arm angle) 분포
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PsiResult:
    n_samples: int
    psi_p05_deg: float
    psi_p50_deg: float
    psi_p95_deg: float
    psi_range_deg: float
    # 같은 (목표포즈, 현재q) bin 안에서의 산포 — 이게 판정 기준
    within_bin_spread_p50_deg: float
    within_bin_spread_p95_deg: float
    n_bins: int
    est_l2_collapse_mm: float
    verdict: str


# test_kinematics T4 에서 G1 로 직접 측정한 붕괴 곡선
_G1_COLLAPSE = [(10, 1.4), (30, 5.7), (45, 11.3), (60, 38.7), (90, 55.3)]


def _collapse_mm(spread_deg: float) -> float:
    xs = [d for d, _ in _G1_COLLAPSE]
    ys = [m for _, m in _G1_COLLAPSE]
    return float(np.interp(spread_deg, xs, ys))


def analyze_psi(eps: list[np.ndarray], side: str = "left",
                pos_bin_m: float = 0.05) -> PsiResult:
    """
    데모 데이터의 psi 산포를 잰다.

    핵심은 전체 범위가 아니라 **같은 (목표포즈, 현재 자세) bin 안에서의 산포**다.
    Model 2 는 현재 q 에 조건화되므로 branch 는 연속성으로 고정되고, L2 회귀가
    실제로 마주치는 다중성은 bin 내부 산포뿐이다.

    산포 <20°  → L2 붕괴 비용 작음, psi 생략 가능
    산포 >60°  → 인터페이스에 psi 스칼라 필수
    """
    arm = G1Arm(side)
    key = f"{side}_arm"

    psis, bins = [], {}
    for e in eps:
        q = e[:, IDX[key]]                       # (T, 7)
        for t in range(len(q)):
            psi = arm.arm_angle(q[t])
            if math.isnan(psi):
                continue
            psis.append(psi)
            # EE 위치로 binning
            p = arm.fk(q[t])[:3, 3]
            b = tuple(np.round(p / pos_bin_m).astype(int))
            bins.setdefault(b, []).append(psi)

    if not psis:
        raise ValueError("psi 샘플 없음")

    A = np.degrees(np.asarray(psis))
    spreads = [math.degrees(circ_range(v)) for v in bins.values() if len(v) >= 5]
    if not spreads:
        spreads = [0.0]

    s50, s95 = float(np.percentile(spreads, 50)), float(np.percentile(spreads, 95))
    collapse = _collapse_mm(s95)

    if s95 < 20:
        v = (f"PASS — bin 내 산포 p95={s95:.1f}°. L2 붕괴 ~{collapse:.1f} mm 로 생존 가능. "
             "psi 를 인터페이스에서 생략해도 됨 (IK 의 자세 태스크로 처리).")
    elif s95 < 60:
        v = (f"MARGINAL — 산포 p95={s95:.1f}°, 붕괴 ~{collapse:.1f} mm. "
             "정밀 작업이면 psi 추가 권장.")
    else:
        v = (f"FAIL — 산포 p95={s95:.1f}°, 붕괴 ~{collapse:.1f} mm. "
             "psi 스칼라가 인터페이스에 필수. 팔당 숫자 1개, 라벨은 FK 로 공짜.")

    return PsiResult(
        n_samples=len(psis),
        psi_p05_deg=float(np.percentile(A, 5)),
        psi_p50_deg=float(np.percentile(A, 50)),
        psi_p95_deg=float(np.percentile(A, 95)),
        psi_range_deg=math.degrees(circ_range(psis)),
        within_bin_spread_p50_deg=s50,
        within_bin_spread_p95_deg=s95,
        n_bins=len(spreads),
        est_l2_collapse_mm=collapse,
        verdict=v,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — 헤드 카메라 내 양손 가시율
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VisibilityResult:
    n_frames: int
    fov_deg: float
    left_visible_frac: float
    right_visible_frac: float
    both_visible_frac: float
    neither_visible_frac: float
    verdict: str


def analyze_visibility(eps: list[np.ndarray], fov_deg: float = 70.0,
                       max_range_m: float = 1.5) -> VisibilityResult:
    """
    로봇 자신의 손이 헤드 카메라에 얼마나 자주 보이는가.

    왜 중요한가: Model 2 가 헤드 프레임 목표를 받고 base 프레임 관절을 내려면
    head->base 변환이 필요한데, 그 변환을 이미지에서 스스로 복구하는 유일한
    경로가 `B_T_H = FK(q)·(H_T_hand)^-1` — 즉 **자기 손을 보는 것**이다.

    손이 자주 안 보이면 그 채널은 죽은 것이고, 인터페이스에서 해석적으로
    변환하는 것이 선택이 아니라 필수가 된다.

    ⚠️ 헤드 카메라 extrinsic 이 URDF 에 없다. 여기서는 head_end_effector_mount
       프레임을 카메라 위치로 근사한다. 실제 extrinsic 을 얻으면 재실행할 것.
    """
    L, R, H = G1Arm("left"), G1Arm("right"), G1Head()
    half = math.radians(fov_deg / 2.0)

    nL = nR = nBoth = nNone = n = 0
    for e in eps:
        for t in range(len(e)):
            qh = e[t, IDX["head"]]
            T_th = H.T_torso_head(qh)
            T_ht = se3_inv(T_th)

            vis = []
            for arm, key in ((L, "left_arm"), (R, "right_arm")):
                p_torso = arm.fk(e[t, IDX[key]])[:3, 3]
                p_head = (T_ht @ np.append(p_torso, 1.0))[:3]
                # 카메라 광축을 head 프레임 +Z 로 가정 (⚠️ extrinsic 미공개)
                z = p_head[2]
                if z <= 0.05 or z > max_range_m:
                    vis.append(False); continue
                ang = math.atan2(math.hypot(p_head[0], p_head[1]), z)
                vis.append(ang < half)

            n += 1
            nL += vis[0]; nR += vis[1]
            nBoth += (vis[0] and vis[1]); nNone += (not vis[0] and not vis[1])

    if n == 0:
        raise ValueError("프레임 없음")

    both = nBoth / n
    if both > 0.7:
        v = (f"자가 캘리브레이션 채널 사용 가능 (양손 동시 가시 {both*100:.0f}%). "
             "그래도 인터페이스 변환이 더 싸고 확실하다.")
    elif both > 0.3:
        v = (f"채널이 약함 (양손 동시 {both*100:.0f}%). "
             "해석적 변환을 기본 경로로 삼을 것.")
    else:
        v = (f"FAIL — 자가 캘리브레이션 채널 사실상 사망 (양손 동시 {both*100:.0f}%). "
             "인터페이스에서 촬영시각 head_joint 로 변환하는 것이 필수.")

    return VisibilityResult(
        n_frames=n, fov_deg=fov_deg,
        left_visible_frac=nL / n, right_visible_frac=nR / n,
        both_visible_frac=both, neither_visible_frac=nNone / n,
        verdict=v,
    )


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="RoboCOIN 오프라인 분석 3종")
    ap.add_argument("--data", required=True, help="RoboCOIN 루트 경로")
    ap.add_argument("--inspect", action="store_true", help="스키마만 확인")
    ap.add_argument("--neck", action="store_true")
    ap.add_argument("--psi", action="store_true")
    ap.add_argument("--visibility", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="파일 수 제한 (빠른 확인용)")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--fov", type=float, default=70.0, help="헤드 카메라 FOV (도)")
    ap.add_argument("--out", default="robocoin_analysis.json")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.data)
        return 0

    if not (args.neck or args.psi or args.visibility or args.all):
        ap.error("--neck / --psi / --visibility / --all 중 하나를 지정하십시오")

    print("=" * 72)
    print("RoboCOIN 오프라인 분석")
    print("=" * 72)
    print(f"  경로: {args.data}")
    print("  로딩 중...")
    eps = load_episodes(args.data, args.limit)
    if not eps:
        sys.exit("에피소드를 읽지 못했습니다. --inspect 로 스키마를 확인하십시오.")
    print(f"  에피소드 {len(eps)}개, 프레임 {sum(len(e) for e in eps):,}개\n")

    out = {"meta": {"data": args.data, "n_episodes": len(eps),
                    "n_frames": int(sum(len(e) for e in eps)), "fps": args.fps}}

    if args.neck or args.all:
        print("── A1  목 관절 운동 " + "─" * 50)
        r = analyze_neck(eps, args.fps)
        out["neck"] = asdict(r)
        print(f"  pan 범위        {r.pan_range_deg:7.1f}°     tilt 범위 {r.tilt_range_deg:6.1f}°")
        print(f"  각속도          p50 {r.speed_p50_dps:6.2f}°/s  p95 {r.speed_p95_dps:6.2f}°/s  "
              f"max {r.speed_max_dps:6.1f}°/s")
        print(f"  0.5s 청크 누적  p50 {r.chunk_0p5s_p50_deg:6.2f}°   p95 {r.chunk_0p5s_p95_deg:6.2f}°"
              f"   → 목표 오차 {r.err_0p5s_p95_mm:6.1f} mm")
        print(f"  2.0s 청크 누적  p50 {r.chunk_2p0s_p50_deg:6.2f}°   p95 {r.chunk_2p0s_p95_deg:6.2f}°"
              f"   → 목표 오차 {r.err_2p0s_p95_mm:6.1f} mm")
        print(f"\n  → {r.verdict}\n")

    if args.psi or args.all:
        print("── A2  psi (arm angle) 분포 " + "─" * 43)
        r = analyze_psi(eps)
        out["psi"] = asdict(r)
        print(f"  샘플 {r.n_samples:,}개   전체 범위 {r.psi_range_deg:.1f}°")
        print(f"  psi 분포        p05 {r.psi_p05_deg:+7.1f}°  p50 {r.psi_p50_deg:+7.1f}°  "
              f"p95 {r.psi_p95_deg:+7.1f}°")
        print(f"  bin 내 산포     p50 {r.within_bin_spread_p50_deg:6.1f}°  "
              f"p95 {r.within_bin_spread_p95_deg:6.1f}°   (bin {r.n_bins}개)")
        print(f"  예상 L2 붕괴    {r.est_l2_collapse_mm:.1f} mm  (G1 실측 곡선 보간)")
        print(f"\n  → {r.verdict}\n")

    if args.visibility or args.all:
        print("── A3  헤드 카메라 내 양손 가시율 " + "─" * 37)
        r = analyze_visibility(eps, args.fov)
        out["visibility"] = asdict(r)
        print(f"  프레임 {r.n_frames:,}개   가정 FOV {r.fov_deg:.0f}°")
        print(f"  왼손 가시   {r.left_visible_frac*100:5.1f}%")
        print(f"  오른손 가시 {r.right_visible_frac*100:5.1f}%")
        print(f"  양손 동시   {r.both_visible_frac*100:5.1f}%")
        print(f"  둘 다 안보임 {r.neither_visible_frac*100:5.1f}%")
        print(f"\n  ⚠️ 헤드 카메라 extrinsic 이 URDF 에 없어 마운트 프레임으로 근사했습니다.")
        print(f"     실제 extrinsic 을 얻으면 재실행하십시오.")
        print(f"\n  → {r.verdict}\n")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: {args.out}")
    print("\n다음: 결과를 docs/PLAN.md §4 판정표에 대조하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
