#!/usr/bin/env python3
"""
SDK FK vs 우리 FK 대조 — URDF 자기일관성을 벗어나는 **첫 외부 검증**

왜 이게 필요한가
────────────────
`robot/test_kinematics.py` 의 T1~T7 은 전부 URDF **자기일관성** 검정이다.
URDF 자체가 실기체와 다르면 7/7 통과해도 아무것도 보증하지 않는다.
그리고 이 FK 는 `tools/mcap2lerobot_fork.py` 가 psi 와 T_rel 을 굽는 데 쓰이므로,
틀리면 그 오차가 데이터셋에 **영구히 구워진다** (F/T 와 똑같이 소급 불가).

2026-07-31 에 실물 시그니처가 확인되면서 이 대조가 가능해졌다:

    forward_kinematics(target_frame, reference_frame='base_link',
                       joint_state: Mapping[str, Sequence[float]] = {}, params)
        -> (MotionStatus, [float])

핵심은 `joint_state` 가 **인자**라는 것이다. 로봇을 그 자세로 움직일 필요가 없다.
그리고 `reference_frame` 을 `torso_base_link` 로 주면 우리 FK 와 같은 기준이 된다.

어떻게 판정하는가 — 두 종류의 상대변환
──────────────────────────────────────
두 FK 가 다를 수 있는 방식은 두 가지이고 **곱해지는 쪽이 다르다**:

    기준(root) 프레임이 다름 :  T' = R · T     (왼쪽 곱)
    tip 프레임이 다름        :  T' = T · X     (오른쪽 곱)

각각을 지우는 불변량이 따로 있다:

    body-relative   A⁻¹B  :  root 차이에 불변, tip 차이에는 불변 아님
    spatial-relative BA⁻¹ :  tip 차이에 불변,  root 차이에는 불변 아님

그래서 둘 다 재면 분리 진단된다:

    body ✅ · spat ✅ → 프레임까지 동일 (또는 사소한 상수 오프셋)
    body ✅ · spat ❌ → root 프레임만 다름. 학습에 치명적이지 않음
    body ❌ · spat ✅ → tip 프레임만 다름. **상수 오차라 학습이 흡수해 조용히 틀림**
    body ❌ · spat ❌ → URDF 자체가 실기체와 다름. 링크 길이·관절 축 문제

⚠️ 상대변환을 **하나만** 재면 tip 문제를 "URDF 가 틀렸다"로 오진한다.
   이 스크립트의 자체 테스트가 실제로 그 오진을 잡아냈다.

출력
────
`robot/assets/fk_crosscheck_<host>.json` 를 남긴다. 이 파일이 있고 임계를
통과했을 때만 대량 변환을 돌리는 것이 안전하다.

사용
────
    # 맥에서: 비교 로직만 검증 (SDK 불필요)
    python tools/fk_crosscheck.py --self-test

    # 회사에서: 실제 대조
    source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh
    python tools/fk_crosscheck.py --n 200

⚠️ 이 스크립트는 로봇을 **움직이지 않는다.** forward_kinematics 는 순수 계산이고,
   set_/move_/execute_ 계열은 호출하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
import sdk_entry  # noqa: E402
from g1_kinematics import (            # noqa: E402
    G1Arm, se3_inv, pose_error, identify_tip_frame, TIP_CANDIDATES,
)

# 판정 임계. 상대변환이 이보다 크게 어긋나면 URDF 가 실기체와 다르다는 뜻이다.
REL_POS_TOL_MM = 2.0
REL_ROT_TOL_DEG = 0.5


def pose_vec_to_T(v) -> np.ndarray:
    """SDK 가 돌려주는 pose 벡터를 4x4 로. 길이로 표현을 추정한다."""
    a = np.asarray(list(v), dtype=float).ravel()
    T = np.eye(4)
    if a.size == 7:                     # [x y z qx qy qz qw] 또는 [x y z qw qx qy qz]
        T[:3, 3] = a[:3]
        q = a[3:]
        # w 가 앞인지 뒤인지 모른다 → 둘 다 만들어 호출부에서 고른다
        T[:3, :3] = _quat_R(q[0], q[1], q[2], q[3])
    elif a.size == 6:                   # [x y z rx ry rz] (axis-angle 또는 euler)
        T[:3, 3] = a[:3]
        T[:3, :3] = _axang_R(a[3:])
    elif a.size == 16:
        T = a.reshape(4, 4)
    elif a.size == 12:
        T[:3, :4] = a.reshape(3, 4)
    else:
        raise ValueError(f"해석할 수 없는 pose 길이 {a.size}: {a[:8]}")
    return T


def _quat_R(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axang_R(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def quat_variants(v) -> list[np.ndarray]:
    """쿼터니언 순서(wxyz vs xyzw)가 미확인이라 두 해석을 다 만든다."""
    a = np.asarray(list(v), dtype=float).ravel()
    if a.size != 7:
        return [pose_vec_to_T(a)]
    out = []
    for order in ("xyzw", "wxyz"):
        T = np.eye(4)
        T[:3, 3] = a[:3]
        q = a[3:]
        T[:3, :3] = (_quat_R(q[0], q[1], q[2], q[3]) if order == "xyzw"
                     else _quat_R(q[1], q[2], q[3], q[0]))
        out.append(T)
    return out


# ─────────────────────────────────────────────────────────────────────────────


def compare(ours: list[np.ndarray], theirs: list[np.ndarray]) -> dict:
    """절대 대조 + **두 종류의 상대변환** 대조.

    왜 상대변환이 두 개인가 — 이게 이 도구의 핵심이다
    ────────────────────────────────────────────────
    두 FK 가 다를 수 있는 방식은 두 가지이고, 수학적으로 **곱해지는 쪽이 다르다**:

        기준(root) 프레임이 다름 :  T' = R · T      (왼쪽 곱, R 은 상수)
        tip 프레임이 다름        :  T' = T · X      (오른쪽 곱, X 는 상수)

    그래서 각각을 지우는 불변량이 따로 있다:

        body-relative  A⁻¹B :  root 차이에 불변  (R 이 소거됨)
                               tip 차이에는 **불변이 아님** (X 로 켤레변환됨)
        spatial-relative BA⁻¹:  tip 차이에 불변   (X 가 소거됨)
                               root 차이에는 불변이 아님

    둘 다 재면 어느 쪽 문제인지 **분리 진단**된다. 하나만 재면 tip 문제를
    "URDF 가 틀렸다"로 오진한다 (실제로 자체 테스트에서 그렇게 나왔다).
    """
    abs_p, abs_r = [], []
    for A, B in zip(ours, theirs):
        dp, dr = pose_error(A, B)
        abs_p.append(dp * 1e3)
        abs_r.append(math.degrees(dr))

    body_p, body_r, spat_p, spat_r = [], [], [], []
    for i in range(len(ours) - 1):
        dp, dr = pose_error(se3_inv(ours[i]) @ ours[i + 1],
                            se3_inv(theirs[i]) @ theirs[i + 1])
        body_p.append(dp * 1e3)
        body_r.append(math.degrees(dr))
        dp, dr = pose_error(ours[i + 1] @ se3_inv(ours[i]),
                            theirs[i + 1] @ se3_inv(theirs[i]))
        spat_p.append(dp * 1e3)
        spat_r.append(math.degrees(dr))

    def stat(v):
        return {"max": float(np.max(v)), "median": float(np.median(v)),
                "mean": float(np.mean(v))} if len(v) else {}

    return {"abs_pos_mm": stat(abs_p), "abs_rot_deg": stat(abs_r),
            "body_rel_pos_mm": stat(body_p), "body_rel_rot_deg": stat(body_r),
            "spat_rel_pos_mm": stat(spat_p), "spat_rel_rot_deg": stat(spat_r),
            "n": len(ours)}


def interpret(c: dict) -> tuple[bool, str, list[str]]:
    """대조 결과를 판정으로 바꾼다.

    진단표
    ──────
      body ✅ · spat ✅ → 프레임까지 동일하거나 차이가 사소함
      body ✅ · spat ❌ → **root(기준) 프레임만 다름.** 학습에 치명적이지 않음
      body ❌ · spat ✅ → **tip 프레임만 다름.** 상수 오차 → 학습이 흡수해 조용히 틀림
      body ❌ · spat ❌ → **URDF 자체가 실기체와 다름.** 링크 길이·관절 축 문제
    """
    notes: list[str] = []
    body_ok = (c["body_rel_pos_mm"]["max"] <= REL_POS_TOL_MM
               and c["body_rel_rot_deg"]["max"] <= REL_ROT_TOL_DEG)
    spat_ok = (c["spat_rel_pos_mm"]["max"] <= REL_POS_TOL_MM
               and c["spat_rel_rot_deg"]["max"] <= REL_ROT_TOL_DEG)
    ap = c["abs_pos_mm"]["median"]
    ar = c["abs_rot_deg"]["median"]
    abs_ok = ap <= REL_POS_TOL_MM and ar <= REL_ROT_TOL_DEG

    if not body_ok and not spat_ok:
        notes.append(
            f"🔴 두 상대변환이 모두 어긋납니다 "
            f"(body {c['body_rel_pos_mm']['max']:.2f}mm / "
            f"spatial {c['spat_rel_pos_mm']['max']:.2f}mm). "
            "프레임 규약 차이로는 설명되지 않습니다 — **URDF 자체가 실기체와 다릅니다** "
            "(링크 길이 또는 관절 축). psi/T_rel 을 데이터셋에 굽지 마십시오.")
        return False, "FAIL — URDF 가 실기체와 불일치", notes

    if abs_ok:
        notes.append("✅ 절대 포즈까지 일치 → tip·기준 프레임 모두 동일합니다.")
        return True, "PASS — 완전 일치", notes

    if spat_ok and not body_ok:
        tip = identify_tip_frame(ap, ar)
        notes.append(
            f"🟡 spatial-relative 는 맞고 body-relative 만 어긋납니다 → "
            f"**tip 프레임 차이**입니다 (상수 오차 {ap:.2f}mm / {ar:.2f}°).")
        if tip:
            notes.append(
                f"   잔차가 `{tip}` 의 서명과 일치합니다. SDK 는 그 링크를 "
                f"EndEffector 로 씁니다 → `G1Arm(tip=...)` 을 맞추면 해소됩니다.")
            return True, f"PASS — tip 프레임만 다름 ({tip})", notes
        notes.append("   다만 잔차가 tip 후보 어느 것과도 안 맞습니다. "
                     "SDK 가 별도의 도구/TCP 오프셋을 붙였을 수 있습니다.")
        return True, "PASS (조건부) — tip 차이, 후보 미상", notes

    if body_ok and not spat_ok:
        notes.append(
            f"🟡 body-relative 는 맞고 spatial-relative 만 어긋납니다 → "
            f"**기준(root) 프레임 차이**입니다. SDK 가 base_link 기준이면 "
            f"다리 5-DoF 만큼 어긋납니다. `--ref-frame` 을 바꿔 재시도하십시오. "
            f"링크 길이·관절 축은 맞으므로 학습에 치명적이지는 않습니다.")
        return True, "PASS (조건부) — 기준 프레임 불일치", notes

    notes.append(
        f"🟡 두 상대변환 모두 임계 이내인데 절대 포즈만 {ap:.2f}mm / {ar:.2f}° "
        "차이납니다. 작은 상수 오프셋으로 보입니다.")
    return True, "PASS (조건부) — 작은 상수 오프셋", notes


# ─────────────────────────────────────────────────────────────────────────────


def self_test() -> int:
    """SDK 없이 비교/판정 로직을 검증한다. 우리 FK 로 가짜 SDK 응답을 만든다."""
    print("=" * 74)
    print("fk_crosscheck 자체 검증 — SDK 불필요")
    print("=" * 74)
    arm = G1Arm("left")
    rng = np.random.default_rng(0)
    lo, hi = arm.limits
    qs = [rng.uniform(lo, hi) for _ in range(30)]
    ours = [arm.fk(q) for q in qs]
    fails = 0

    def chk(name, cond, extra=""):
        nonlocal fails
        print(f"  {'✅' if cond else '❌'} {name} {extra}")
        if not cond:
            fails += 1

    # 1) 자기 자신과 대조하면 완전 일치
    ok, verdict, _ = interpret(compare(ours, ours))
    chk("동일 입력 → PASS 완전 일치", ok and "완전 일치" in verdict, verdict)

    # 2) tip 만 다른 경우 → tip 프레임으로 진단되어야 함
    arm_mount = G1Arm("left", tip="left_arm_end_effector_mount_link")
    theirs = [arm_mount.fk(q) for q in qs]
    c = compare(ours, theirs)
    ok, verdict, notes = interpret(c)
    chk("tip 만 다름 → tip 프레임으로 진단",
        ok and "tip 프레임만 다름" in verdict, verdict)
    chk("  그때 spatial-relative 는 불변",
        c["spat_rel_pos_mm"]["max"] <= REL_POS_TOL_MM,
        f"max {c['spat_rel_pos_mm']['max']:.2e}mm")
    chk("  그때 body-relative 는 어긋남 (켤레변환되므로)",
        c["body_rel_pos_mm"]["max"] > REL_POS_TOL_MM,
        f"max {c['body_rel_pos_mm']['max']:.1f}mm")

    # 3) 기준 프레임만 다른 경우(상수 SE(3) 를 앞에 곱함) → 상대는 일치해야 함
    Toff = np.eye(4)
    Toff[:3, 3] = [0.13, -0.27, 0.41]
    Toff[:3, :3] = _axang_R([0.3, -0.2, 0.1])
    shifted = [Toff @ T for T in ours]
    c3 = compare(ours, shifted)
    chk("root 만 다름 → body-relative 는 불변",
        c3["body_rel_pos_mm"]["max"] <= 1e-6,
        f"max {c3['body_rel_pos_mm']['max']:.2e}mm")
    chk("  그때 spatial-relative 는 어긋남 (켤레변환되므로)",
        c3["spat_rel_pos_mm"]["max"] > REL_POS_TOL_MM,
        f"max {c3['spat_rel_pos_mm']['max']:.1f}mm")
    ok3, v3, _ = interpret(c3)
    chk("  판정은 '기준 프레임 불일치'", ok3 and "기준 프레임" in v3, v3)

    # 4) 링크 길이가 틀린 경우 → 반드시 FAIL 이어야 함 (이게 핵심 안전장치)
    bad = []
    for q in qs:
        T = arm.fk(q).copy()
        T[:3, 3] *= 1.02                      # 2% 스케일 오차
        bad.append(T)
    c4 = compare(ours, bad)
    ok4, v4, _ = interpret(c4)
    chk("링크 길이 2% 오차 → FAIL", (not ok4) and "FAIL" in v4, v4)

    # 5) 쿼터니언 순서 두 해석이 실제로 다른 회전을 만드는가
    v = [0.1, 0.2, 0.3, 0.0, 0.0, 0.7071, 0.7071]
    va = quat_variants(v)
    chk("쿼터니언 순서 2가지 생성", len(va) == 2 and
        not np.allclose(va[0][:3, :3], va[1][:3, :3]))

    print("\n" + "=" * 74)
    if fails:
        print(f"❌ {fails}건 실패")
        return 1
    print("✅ 전부 통과 — 비교/판정 로직이 정상입니다.")
    print("   회사에서:  python tools/fk_crosscheck.py --n 200")
    return 0


# ─────────────────────────────────────────────────────────────────────────────


def run_live(n: int, side: str, ref_frame: str, out: str,
             no_init: bool = False) -> int:
    try:
        import galbot_sdk as sdk
    except ImportError as e:
        print(f"galbot_sdk import 실패: {e}\n"
              "  source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh "
              "(bash 필요)", file=sys.stderr)
        return 2

    # ⚠️ GalbotMotion() 직접 생성은 "No constructor defined!" 로 실패한다 (실측).
    #    pybind11 이 py::init<>() 없이 바인딩했으므로 어딘가에서 받아와야 한다.
    try:
        motion, how = sdk_entry.acquire(sdk, "GalbotMotion")
    except sdk_entry.EntryNotFound as e:
        print(str(e), file=sys.stderr)
        print("\n→ python tools/probe_sdk.py --entry 출력을 공유해주십시오.",
              file=sys.stderr)
        return 2
    print(f"GalbotMotion 획득: {how}", flush=True)
    if no_init:
        print("--no-init: init() 을 건너뜁니다 (forward_kinematics 는 순수 계산)")
        inited = True
    else:
        print("GalbotMotion.init() 호출 중...", flush=True)
        inited = motion.init()
        print(f"GalbotMotion.init() → {inited}", flush=True)
    if not inited:
        print("\n" + "=" * 74)
        print("init() 이 False 를 반환했습니다. 무엇이 되는지 확인해봅니다.")
        print("=" * 74)
        # ⚠️ init 실패 상태에서 **데이터 호출을 더 하면 segfault 로 죽는다** (실측).
        #    그래서 init 전에도 안전했던 소수만 부르고 즉시 멈춘다.
        for m in ("get_supported_chains", "get_supported_frames"):
            if not hasattr(motion, m):
                continue
            try:
                print(f"  motion.{m}() → {str(getattr(motion, m)())[:200]}",
                      flush=True)
            except Exception as e:
                print(f"  motion.{m}() → {type(e).__name__}: {str(e)[:160]}",
                      flush=True)
        try:
            robot, rhow = sdk_entry.acquire(sdk, "GalbotRobot")
            print(f"\n  GalbotRobot 획득: {rhow}", flush=True)
            info = robot.get_device_information()
            print(f"  robot.get_device_information() → {info}", flush=True)
            empty = all(not str(v).strip() for k, v in (info or {}).items()
                        if k != "manufacturer")
            if empty:
                print("\n  🔴 model/serial/firmware 가 전부 비어 있습니다 —")
                print("     **로봇이 실제로 연결돼 있지 않습니다.**")
        except Exception as e:
            print(f"  GalbotRobot 조회 실패: {e}")

        print("\n확인할 것:")
        print("  1. 로봇 전원이 켜져 있고 부팅이 끝났습니까?")
        print("  2. 이 PC 가 로봇 LAN 에 연결돼 있습니까? (ping 으로 확인)")
        print("  3. 다른 프로세스가 이미 SDK 를 점유하고 있지 않습니까?")
        print("     (SDK 는 싱글톤 구조라 중복 연결이 막힐 수 있습니다)")
        print("\n⚠️ 여기서 멈춥니다. init 실패 상태에서 get_* 를 더 부르면")
        print("   segfault 로 프로세스가 죽고 지금까지의 출력도 날아갑니다.")
        return 2

    print("=" * 74)
    print("SDK FK vs 우리 FK 대조")
    print("=" * 74)
    for fn in ("get_supported_chains", "get_supported_ee_frames",
               "get_supported_frames"):
        try:
            print(f"  {fn}: {sorted(getattr(motion, fn)())}")
        except Exception as e:
            print(f"  {fn}: 실패 {type(e).__name__}: {e}")
    try:
        print(f"  end-effector 링크: {motion.get_link_names(only_end_effector=True)}")
    except Exception:
        pass
    try:
        print(f"  현재 chain joint state 키: "
              f"{sorted(motion.get_chain_joint_state().keys())}")
    except Exception as e:
        print(f"  get_chain_joint_state 실패: {e}")

    # 2026-07-31 실측:
    #   get_supported_chains() → {'torso','leg','head','left_arm','right_arm'}
    #   get_supported_frames() → {'map','world','base_link'}   ← torso_base_link 없음
    chains = sorted(motion.get_supported_chains())
    chain = next((c for c in chains if side in c.lower()), chains[0] if chains else side)
    print(f"\n  사용할 chain: '{chain}'   reference_frame: '{ref_frame}'")
    if ref_frame != "torso_base_link":
        print("\n  ℹ️ 우리 FK 는 torso_base_link 기준인데 SDK 는 그 프레임을 지원하지")
        print("     않습니다 (지원: map/world/base_link). 그래서 **절대 포즈는 다리")
        print("     5-DoF 만큼 어긋나는 것이 정상**입니다. 판정은 body-relative")
        print("     (root 차이에 불변) 로 합니다 — 그게 맞으면 URDF 는 옳습니다.")

    arm = G1Arm(side)
    rng = np.random.default_rng(0)
    q_lo, q_hi = arm.limits()

    results: dict[str, dict] = {}
    for tip in TIP_CANDIDATES:
        frame = f"{side}_{tip}"
        ours, theirs = [], []
        bad = 0
        for _ in range(n):
            q = rng.uniform(q_lo, q_hi)
            try:
                st, pose = motion.forward_kinematics(
                    frame, reference_frame=ref_frame, joint_state={chain: list(q)})
            except Exception as e:
                bad += 1
                if bad == 1:
                    print(f"  {frame}: 호출 실패 {type(e).__name__}: {e}")
                break
            if getattr(st, "name", str(st)) != "SUCCESS":
                bad += 1
                if bad == 1:
                    print(f"  {frame}: MotionStatus={getattr(st,'name',st)}")
                break
            cands = quat_variants(pose)
            mine = arm.fk(q)
            # 쿼터니언 순서는 첫 표본에서 오차가 작은 쪽으로 고정
            if not theirs:
                errs = [pose_error(mine, T)[1] for T in cands]
                _quat_pick = int(np.argmin(errs))
                results.setdefault("_meta", {})["quat_order"] = \
                    ("xyzw", "wxyz")[_quat_pick] if len(cands) == 2 else "n/a"
            pick = 0 if len(cands) == 1 else _quat_pick
            ours.append(mine)
            theirs.append(cands[pick])
        if not ours:
            results[frame] = {"error": "호출 실패"}
            continue
        c = compare(ours, theirs)
        ok, verdict, notes = interpret(c)
        results[frame] = {**c, "ok": ok, "verdict": verdict, "notes": notes}
        print(f"\n── {frame}")
        print(f"   절대  pos median {c['abs_pos_mm']['median']:8.3f} mm   "
              f"rot median {c['abs_rot_deg']['median']:7.4f}°")
        print(f"   body-rel  pos max {c['body_rel_pos_mm']['max']:8.3f} mm   "
              f"rot max {c['body_rel_rot_deg']['max']:7.4f}°   (root 차이에 불변)")
        print(f"   spat-rel  pos max {c['spat_rel_pos_mm']['max']:8.3f} mm   "
              f"rot max {c['spat_rel_rot_deg']['max']:7.4f}°   (tip 차이에 불변)")
        print(f"   → {verdict}")
        for nt in notes:
            print(f"      {nt}")

    best = min((k for k in results if not k.startswith("_")
                and "abs_pos_mm" in results[k]),
               key=lambda k: results[k]["abs_pos_mm"]["median"], default=None)

    payload = {
        "host": socket.gethostname(),
        "side": side,
        "chain": chain,
        "reference_frame": ref_frame,
        "motion_acquired_via": how,
        "n_samples": n,
        "tol": {"rel_pos_mm": REL_POS_TOL_MM, "rel_rot_deg": REL_ROT_TOL_DEG},
        "best_match_frame": best,
        "passed": bool(best and results[best]["ok"]),
        "per_frame": results,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print(f"가장 잘 맞는 프레임: {best}")
    print(f"판정: {'✅ 통과' if payload['passed'] else '❌ 불통과'}")
    print(f"저장: {os.path.abspath(out)}")
    print("\n이 파일이 있고 passed=true 여야 대량 변환(mcap2lerobot_fork)이 안전합니다.")
    print("psi 와 T_rel 은 데이터셋에 영구히 구워지고 소급 수정이 불가능합니다.")
    return 0 if payload["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="SDK FK 와 우리 FK 대조")
    ap.add_argument("--self-test", action="store_true", help="SDK 없이 로직 검증")
    ap.add_argument("--n", type=int, default=200, help="표본 수")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--ref-frame", default="base_link",
                    help="SDK FK 의 reference_frame. 2026-07-31 실측 결과 SDK 가 "
                         "지원하는 프레임은 {map, world, base_link} 뿐이고 "
                         "torso_base_link 는 **없다**. 그래서 base_link 로 받고 "
                         "body-relative 대조(root 차이에 불변)로 판정한다")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-init", action="store_true",
                    help="init() 없이 시도 — forward_kinematics 는 순수 계산이라 "
                         "로봇 연결 없이도 될 수 있다")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "robot", "assets",
        f"fk_crosscheck_{socket.gethostname()}.json")
    return run_live(args.n, args.side, args.ref_frame, out, args.no_init)


if __name__ == "__main__":
    raise SystemExit(main())
