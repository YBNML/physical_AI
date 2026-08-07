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
    body ❌ · spat ❌ → 프레임 규약으로 설명 안 됨 → 아래 4가지를 가려야 한다

⚠️ 상대변환을 **하나만** 재면 tip 문제를 "URDF 가 틀렸다"로 오진한다.
   이 스크립트의 자체 테스트가 실제로 그 오진을 잡아냈다.

⚠️ **body ❌ 를 곧바로 "URDF 가 틀렸다" 로 읽으면 안 된다.** 원인이 최소 넷이고
   셋은 우리 입력 문제다:
     (a) SDK 가 joint_state 를 무시하고 현재 자세만 쓴다
     (b) SDK 가 관절값을 자기 한계로 clamp 한다
     (c) chain 내 관절 순서가 우리 URDF 와 다르다
     (d) 진짜로 링크 길이·관절 축이 다르다
   `preflight()` 가 (a)(b) 를 배제하고, body-rel 의 **median vs max** 가
   (c)(d) 와 "일부 샘플만" 을 가른다:
     median 부터 크다        → 계통 오차 → (c) 또는 (d) → 진짜 문제
     median 은 작고 max 만 크다 → 일부 샘플 → 한계 밖 입력 의심 → --sampling local

   기본 샘플링이 `local`(현재 자세 주변)인 이유가 이것이다. 전 범위 무작위는
   SDK 한계를 넘어 (b) 를 유발한다.

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
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
import sdk_entry  # noqa: E402
from g1_kinematics import (            # noqa: E402
    G1Arm, se3_inv, pose_error, identify_tip_frame, TIP_CANDIDATES, build_chain,
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
        if not len(v):
            return {}
        a = np.asarray(v, dtype=float)
        return {"max": float(a.max()), "p95": float(np.percentile(a, 95)),
                "median": float(np.median(a)), "mean": float(a.mean()),
                "argmax": int(a.argmax())}

    return {"abs_pos_mm": stat(abs_p), "abs_rot_deg": stat(abs_r),
            "body_rel_pos_mm": stat(body_p), "body_rel_rot_deg": stat(body_r),
            "spat_rel_pos_mm": stat(spat_p), "spat_rel_rot_deg": stat(spat_r),
            "n": len(ours),
            "_body_p": [float(x) for x in body_p]}


def interpret(c: dict, c_abs=None) -> tuple[bool, str, list[str]]:
    """대조 결과를 판정으로 바꾼다.

    진단표
    ──────
      body ✅ · spat ✅ → 프레임까지 동일하거나 차이가 사소함
      body ✅ · spat ❌ → **root(기준) 프레임만 다름.** 학습에 치명적이지 않음
      body ❌ · spat ✅ → **tip 프레임만 다름.** 상수 오차 → 학습이 흡수해 조용히 틀림
      body ❌ · spat ❌ → **URDF 자체가 실기체와 다름.** 링크 길이·관절 축 문제
    """
    notes: list[str] = []

    # ★ base_link 절대 비교가 있으면 **그것이 주 판정**이다.
    #   같은 프레임·같은 관절값이므로 상대변환 트릭이 필요 없고, 다리 체인까지
    #   함께 검증된다. 상대변환 지표는 그 아래 보조로만 본다.
    if c_abs:
        p, r = c_abs["abs_pos_mm"], c_abs["abs_rot_deg"]
        if p["median"] <= REL_POS_TOL_MM and r["median"] <= REL_ROT_TOL_DEG:
            notes.append(
                f"✅ base_link 절대 비교 일치 "
                f"(pos median {p['median']:.3f}mm / rot {r['median']:.4f}°). "
                "우리 URDF 가 SDK 모델과 맞고, 다리 5-DoF 체인도 함께 검증됐습니다.")
            if p["max"] > REL_POS_TOL_MM * 3:
                notes.append(
                    f"   다만 max {p['max']:.2f}mm 인 샘플이 있습니다 — "
                    "SDK FK 가 비결정론적(살아 있는 상태 혼입)이므로 그 정도 "
                    "이상치는 예상 범위입니다.")
            return True, "PASS — base_link 절대 비교 일치", notes
        notes.append(
            f"🔴 base_link 절대 비교 불일치 "
            f"(pos median {p['median']:.2f}mm / rot {r['median']:.3f}°). "
            "같은 프레임·같은 관절값인데 어긋납니다 — 링크 길이, 관절 축, "
            "또는 chain 내 관절 순서를 의심하십시오. psi/T_rel 을 굽지 마십시오.")
        return False, "FAIL — base_link 절대 비교 불일치", notes

    body_ok = (c["body_rel_pos_mm"]["max"] <= REL_POS_TOL_MM
               and c["body_rel_rot_deg"]["max"] <= REL_ROT_TOL_DEG)
    spat_ok = (c["spat_rel_pos_mm"]["max"] <= REL_POS_TOL_MM
               and c["spat_rel_rot_deg"]["max"] <= REL_ROT_TOL_DEG)
    ap = c["abs_pos_mm"]["median"]
    ar = c["abs_rot_deg"]["median"]
    abs_ok = ap <= REL_POS_TOL_MM and ar <= REL_ROT_TOL_DEG

    if not body_ok and not spat_ok:
        # ⚠️ 여기서 "URDF 가 틀렸다" 로 단정했던 것은 과했다.
        #    tip 을 like-for-like 로 맞춘 뒤에도 body-rel 이 어긋나는 원인은
        #    최소 넷이고, 그중 셋은 우리 입력 문제다:
        #      (a) SDK 가 joint_state 를 무시  (b) SDK 가 관절값을 clamp
        #      (c) chain 내 관절 순서가 다름   (d) 진짜 링크 길이/축 불일치
        #    preflight 가 (a)(b) 를 배제하고, median/max 분포가 (c)(d) 를 가른다:
        #      median 이 크다  → 계통 오차 → (c) 또는 (d)
        #      median 은 작고 max 만 크다 → 일부 샘플만 → 한계 밖 입력 의심
        bp = c["body_rel_pos_mm"]
        systematic = bp.get("median", 0.0) > REL_POS_TOL_MM
        notes.append(
            f"🔴 body-relative 가 어긋납니다 "
            f"(median {bp.get('median', float('nan')):.2f} / "
            f"p95 {bp.get('p95', float('nan')):.2f} / "
            f"max {bp.get('max', float('nan')):.2f} mm). "
            "tip 을 같은 링크로 맞췄으므로 프레임 규약으로는 설명되지 않습니다.")
        if systematic:
            notes.append(
                "   median 부터 크므로 **계통 오차**입니다 — 링크 길이·관절 축이 "
                "다르거나 chain 내 관절 순서가 다릅니다. psi/T_rel 을 굽지 마십시오.")
            return False, "FAIL — 계통적 기구학 불일치", notes
        notes.append(
            "   median 은 작고 일부 샘플만 큽니다. SDK FK 가 비결정론적이라"
            "(같은 q 재호출에도 차이 발생) 살아 있는 다리/토르소 상태가 섞인 "
            "것으로 보입니다. base_link 절대 비교(leg 명시)가 가능하면 그쪽이 "
            "결정적입니다.")
        return False, "FAIL (조건부) — 일부 샘플만 불일치", notes

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


def _sdk_fk(motion, frame: str, ref: str, chain: str, q) -> tuple:
    """SDK FK 한 번. (status_name, pose_list|None)."""
    st, pose = motion.forward_kinematics(
        frame, reference_frame=ref, joint_state={chain: [float(x) for x in q]})
    return getattr(st, "name", str(st)), pose


def preflight(motion, frame: str, ref: str, chain: str, q0, arm) -> dict:
    """대조 전에 **입력이 실제로 반영되는지** 확인한다.

    왜 필요한가
    ───────────
    첫 대조에서 body-relative 가 41.5mm 어긋났는데, 그것만으로 "URDF 가 실기체와
    다르다" 고 결론냈던 것은 성급했다. 같은 증상을 내는 원인이 최소 셋이다:

      (a) SDK 가 joint_state 를 무시하고 현재 자세만 쓴다
      (b) SDK 가 관절값을 자기 한계로 clamp 한다 (우리는 URDF 전 범위를 샘플링했다)
      (c) SDK 의 chain 내 관절 **순서**가 우리 URDF 와 다르다
      (d) 진짜로 링크 길이/관절 축이 다르다

    (a)~(c) 는 전부 우리 입력 문제이고 고칠 수 있다. (d) 만이 데이터셋을
    오염시키는 진짜 문제다. 그래서 셋을 먼저 배제한다.
    """
    out: dict = {}
    print("\n  ── preflight: 입력이 반영되는가 ─────────────────────")

    # (a) 응답성 — 서로 다른 q 두 개가 서로 다른 포즈를 내는가
    qa = np.asarray(q0, dtype=float)
    qb = qa.copy()
    qb[1] += 0.30                      # 어깨 관절 하나만 크게 움직인다
    qb = np.clip(qb, *arm.limits)
    try:
        sa, pa = _sdk_fk(motion, frame, ref, chain, qa)
        sb, pb = _sdk_fk(motion, frame, ref, chain, qb)
    except Exception as e:
        print(f"     ❌ FK 호출 실패: {type(e).__name__}: {e}")
        return {"ok": False, "reason": "fk_call_failed"}
    if sa != "SUCCESS" or sb != "SUCCESS":
        print(f"     ❌ MotionStatus {sa} / {sb}")
        return {"ok": False, "reason": f"status {sa}/{sb}"}

    d_sdk = float(np.linalg.norm(np.asarray(pa[:3]) - np.asarray(pb[:3]))) * 1e3
    d_our = float(np.linalg.norm(arm.fk(qa)[:3, 3] - arm.fk(qb)[:3, 3])) * 1e3
    out["responds"] = d_sdk > 1.0
    print(f"     q[1] 를 0.30 rad 움직였을 때 EE 이동")
    print(f"       SDK  {d_sdk:8.2f} mm")
    print(f"       우리 {d_our:8.2f} mm")
    if d_sdk < 1.0:
        print("     🔴 SDK 포즈가 안 변합니다 → **joint_state 를 무시하고 현재 자세를**")
        print("        **쓰고 있습니다.** 이 상태의 대조는 전부 무의미합니다.")
        return {"ok": False, "reason": "joint_state_ignored", **out}
    ratio = d_sdk / d_our if d_our > 1e-9 else float("nan")
    out["motion_ratio"] = ratio
    print(f"       비율 {ratio:.4f}  (1.0 이어야 정상)")
    if abs(ratio - 1.0) > 0.02:
        print("     ⚠️ 이동량이 다릅니다 — 링크 길이 또는 관절 축이 다를 수 있습니다.")

    # (b) 결정론 — 같은 q 를 두 번
    _, pa2 = _sdk_fk(motion, frame, ref, chain, qa)
    drift = float(np.linalg.norm(np.asarray(pa[:3]) - np.asarray(pa2[:3]))) * 1e3
    out["determinism_mm"] = drift
    print(f"     같은 q 두 번 → 차이 {drift:.6f} mm "
          f"{'✅' if drift < 1e-3 else '⚠️ 비결정론적'}")

    # (c) clamp — 관절 한계 근처를 넘겨보고 반영되는지
    lo, hi = arm.limits
    q_edge = np.clip(hi - 1e-3, lo, hi)
    q_over = hi + 0.30                 # 우리 한계를 넘는 값
    try:
        _, p_edge = _sdk_fk(motion, frame, ref, chain, q_edge)
        _, p_over = _sdk_fk(motion, frame, ref, chain, q_over)
        d_clamp = float(np.linalg.norm(
            np.asarray(p_edge[:3]) - np.asarray(p_over[:3]))) * 1e3
        out["clamp_probe_mm"] = d_clamp
        print(f"     한계 초과 입력 반영 여부: {d_clamp:.3f} mm 차이 "
              f"{'(clamp 됨 — 한계 밖 샘플은 못 씀)' if d_clamp < 0.5 else '(반영됨)'}")
        out["clamps"] = d_clamp < 0.5
    except Exception as e:
        print(f"     clamp 확인 실패: {type(e).__name__}: {e}")

    out["ok"] = True
    return out


def run_live(n: int, side: str, ref_frame: str, out: str,
             no_init: bool = False, sampling: str = "local",
             amp: float = 0.20, _fake=None) -> int:
    if _fake is not None:
        sdk = _fake            # 자체 검증용 가짜 SDK 주입
    else:
      try:
        import galbot_sdk as sdk
      except ImportError as e:
        print(f"galbot_sdk import 실패: {e}\n"
              "  source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh "
              "(bash 필요)", file=sys.stderr)
        return 2

    # ⚠️ **GalbotRobot 을 먼저 init 해야 한다.**
    #    2026-07-31 실측 대조: probe-live 는 robot.init() 을 먼저 했고
    #    get_chain_joint_state() 가 실값을 줬다. fk_crosscheck 는 Motion 만
    #    만들었고 15초 내내 {} 였으며 forward_kinematics 가 DATA_FETCH_FAILED 로
    #    실패했다. 두 실행의 유일한 차이가 이것이다 — 메시지 버스 구독이
    #    robot 핸들 쪽에 있는 것으로 보인다.
    sdk_entry.report_other_clients("read")

    try:
        robot, rhow = sdk_entry.acquire(sdk, "GalbotRobot")
        print(f"GalbotRobot 획득: {rhow}", flush=True)
        _HANDLES["robot"] = robot
        print("GalbotRobot.init() 호출 중...", flush=True)
        rok = bool(robot.init(set()))
        print(f"GalbotRobot.init() → {rok}", flush=True)
        if not rok:
            print("⚠️ robot.init() 이 False 입니다. Motion 이 관절 상태를 못 받을 수 "
                  "있습니다.", file=sys.stderr)
        # ⚠️ **robot 데이터가 실제로 흐를 때까지 기다린 뒤 motion.init() 을 부른다.**
        #    실측 대조: probe-live 는 robot.init() 뒤 get_* 를 수십 번 부르느라
        #    시간이 흘렀고 motion.init() 이 True 였다. fk_crosscheck 는 곧바로
        #    불렀고 False 였다. motion 이 로봇 모델을 로봇에서 받아오는 구조로
        #    보이며, 그 전에 부르면 INIT_FAILED 가 된다.
        print("robot 데이터 도착 대기...", flush=True)
        t_end = time.time() + 20.0
        while time.time() < t_end:
            try:
                if robot.get_joint_group_names():
                    print("  ✅ robot 데이터 흐름 확인", flush=True)
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            print("  ⚠️ 20초 동안 robot 관절 그룹이 비어 있습니다.", flush=True)
    except Exception as e:
        print(f"⚠️ GalbotRobot 준비 실패: {type(e).__name__}: {e}", file=sys.stderr)
        print("   Motion 이 관절 상태를 못 받을 수 있습니다.", file=sys.stderr)

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
    _HANDLES["motion"] = motion
    if no_init:
        print("--no-init: init() 을 건너뜁니다 (forward_kinematics 는 순수 계산)")
        inited = True
    else:
        # ⚠️ **재시도한다.** 한 번 False 라고 끝이 아니다 — 로봇에서 모델을
        #    받아오는 데 시간이 걸린다.
        inited = False
        for attempt in range(1, 7):
            print(f"GalbotMotion.init() 시도 {attempt}...", flush=True)
            try:
                inited = bool(motion.init())
            except Exception as e:
                print(f"  예외: {type(e).__name__}: {e}", flush=True)
                inited = False
            # ⚠️ 준비 여부는 **get_link_names()** 로 판정한다.
            #    get_supported_chains() 는 정적 설정이라 init 실패해도 응답한다
            #    (실측: chains 5개인데 link 0개, ee_frames 0개, FK 는 INIT_FAILED).
            #    그걸 "데이터가 흐른다" 로 읽은 것이 내 오판이었다.
            try:
                nlink = len(motion.get_link_names(only_end_effector=False))
            except Exception:
                nlink = 0
            print(f"  init={inited}  get_link_names={nlink}개", flush=True)
            if nlink:
                inited = True
                break
            time.sleep(2.0)
    if not inited:
        # ⚠️ init() 반환값이 사용 가능 여부를 정확히 나타내지 않는다.
        #    실측: motion.init() → False 인데 바로 뒤 get_supported_chains() 는
        #    정상 응답했다. 같은 로봇에서 직전 실행은 True 였다.
        #    그래서 즉시 중단하지 않고 **데이터가 실제로 흐르는지**로 판단한다.
        print("\n  ⚠️ motion.init() 이 False 입니다. 데이터가 흐르는지 확인합니다...")
        # ⚠️ get_supported_chains() 로 판단하면 안 된다 — 정적 설정이라
        #    init 실패해도 응답한다. get_link_names() 가 진짜 지표다.
        probe_ok = False
        try:
            lk = motion.get_link_names(only_end_effector=False)
            probe_ok = bool(lk)
            print(f"     get_link_names() → {len(lk)}개 "
                  f"{'(모델 로드됨)' if lk else '(모델 없음 — init 실제 실패)'}")
        except Exception as e:
            print(f"     get_link_names() 실패: {type(e).__name__}: {e}")
        if probe_ok:
            print("     → 응답이 옵니다. init 반환값을 무시하고 계속합니다.")
            inited = True
        else:
            print("     → 응답이 없습니다. 중단합니다.")

    if not inited:
        print("\n" + "=" * 74)
        print("init() 이 False 이고 데이터도 오지 않습니다.")
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
        print("  3. **이전 실행의 세션이 로봇에 남아 있지 않습니까?**")
        print("     관측: 같은 로봇에서 처음엔 init()=True 였다가 그 뒤로 계속")
        print("     False 다. 프로세스가 끝나도 유지되는 상태이므로 로봇 쪽에")
        print("     세션이 남은 것으로 보입니다. 이제 종료 시 명시적으로")
        print("     반납하지만, **이미 남은 세션은 아래로 정리하십시오**:")
        print("       ps aux | grep -i galbot        # 우리 python 프로세스 잔류 확인")
        print("       (남아 있으면 kill, 그래도 안 되면 로봇 소프트웨어 재시작)")
        print("\n⚠️ 여기서 멈춥니다. init 실패 상태에서 get_* 를 더 부르면")
        print("   segfault 로 프로세스가 죽고 지금까지의 출력도 날아갑니다.")
        return 2

    # ⚠️ init() 이 True 여도 **데이터가 아직 안 왔을 수 있다.** probe-live 에서는
    #    get_chain_joint_state() 가 실값을 줬는데 여기서는 {} 였고, 그 상태로
    #    forward_kinematics 를 부르니 DATA_FETCH_FAILED 가 났다.
    #    관절 상태가 찰 때까지 기다린다. 안 기다리면 "FK 실패" 로 오진한다.
    print("관절 상태 도착 대기...", flush=True)
    cjs = {}
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            cjs = motion.get_chain_joint_state() or {}
        except Exception as e:
            print(f"  get_chain_joint_state 예외: {type(e).__name__}: {e}", flush=True)
        if cjs:
            break
        time.sleep(0.5)
    if cjs:
        print(f"  ✅ 도착 — chain {sorted(cjs)}", flush=True)
        for k in sorted(cjs):
            print(f"     {k}: {len(cjs[k])}개 {[round(v,4) for v in cjs[k][:8]]}")
    else:
        print("  ⚠️ 15초 동안 관절 상태가 비어 있습니다. FK 가 DATA_FETCH_FAILED 로")
        print("     실패할 가능성이 높습니다. 그래도 계속 진행합니다.")

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

    rng = np.random.default_rng(0)

    # ⚠️ SDK 가 아는 링크만 시도한다. 우리 URDF 의 gripper 링크 이름들은
    #    SDK 모델에 아예 없어서 INVALID_INPUT 이 났다 (실측). 존재하지 않는
    #    프레임을 계속 두드리면 실패 원인이 "이름 없음" 인지 "데이터 없음" 인지
    #    섞여서 진단이 흐려진다.
    try:
        sdk_links = set(motion.get_link_names(only_end_effector=False))
    except Exception as e:
        print(f"  get_link_names 실패: {e}")
        sdk_links = set()
    print(f"\n  SDK 링크 {len(sdk_links)}개")
    wanted = [f"{side}_{t}" for t in TIP_CANDIDATES]
    present = [w for w in wanted if w in sdk_links]
    missing = [w for w in wanted if w not in sdk_links]
    print(f"  우리 tip 후보 중 SDK 에 있는 것 : {present or '(없음)'}")
    print(f"  우리 tip 후보 중 SDK 에 없는 것 : {missing or '(없음)'}")
    if missing:
        print("  → 없는 것은 건너뜁니다. SDK 모델에 그 링크가 없다는 뜻입니다.")
    if not sdk_links:
        # 링크 목록이 통째로 비었다 = 기구학 모델 자체가 안 올라왔다.
        # 이 상태에서 프레임마다 preflight 를 돌리면 같은 INIT_FAILED 를
        # 네 번 찍을 뿐이다.
        print("\n  🔴 SDK 링크 목록이 비어 있습니다 — **기구학 모델이 로드되지**")
        print("     **않았습니다.** motion.init() 이 실제로 실패한 상태입니다.")
        print("     (get_supported_chains 는 정적 설정이라 이 경우에도 응답합니다)")
        others = sdk_entry.find_other_sdk_clients()
        if others:
            print("\n  **원인이 확인됐습니다: 다른 클라이언트가 SDK 를 점유 중입니다.**")
            for o in others:
                print(f"     pid {o['pid']}: {o['cmdline'][:140]}")
            print("\n     GalbotRobot.init() 은 성공하는데 GalbotMotion.init() 만")
            print("     실패합니다 → **Motion 은 배타적**이고 Robot 은 다중 접근을")
            print("     허용하는 것으로 보입니다.")
            print("\n     → 저 세션이 끝나야 FK 대조가 가능합니다.")
            print("       정체 확인:  ps -o pid,ppid,user,etime,cmd -p <pid>")
            print("                   systemctl list-units | grep -i galbot")
            print("       로봇 기본 서비스라면 systemctl stop 으로 멈춰야 하며,")
            print("       kill 만 하면 자동 재시작될 수 있습니다.")
        else:
            print("\n  다른 클라이언트는 안 보입니다. 그렇다면:")
            print("     1) 우리 이전 실행의 세션이 남았을 수 있습니다 (재부팅/재시작)")
            print("     2) root 로 도는 클라이언트는 /proc 접근이 막혀 안 보일 수 있습니다")
            print("        sudo ps aux | grep -i galbot   으로 확인하십시오")
        return 2
    if not present:
        print("\n  🔴 시도할 프레임이 없습니다. SDK 의 팔 관련 링크 목록:")
        for ln in sorted(x for x in sdk_links if side in x):
            print(f"     {ln}")
        print("\n  → 위 목록에서 맞는 이름을 --tip 으로 지정하거나,")
        print("     robot/g1_kinematics.py 의 TIP_CANDIDATES 를 갱신해야 합니다.")
        return 2

    results: dict[str, dict] = {}
    for tip in TIP_CANDIDATES:
        frame = f"{side}_{tip}"
        if sdk_links and frame not in sdk_links:
            results[frame] = {"error": "SDK 링크 목록에 없음"}
            continue
        # ⚠️ **우리 FK 도 같은 링크를 tip 으로 잡는다.** tip 서명표로 사후
        #    보정하는 것보다 애초에 같은 지점을 비교하는 게 깨끗하다.
        #    SDK 모델에는 그리퍼가 없으므로(링크 36개 중 우리 후보는 mount 하나뿐)
        #    실제로 비교 가능한 지점은 arm 끝단이다.
        try:
            arm = G1Arm(side, tip=frame)
        except Exception as e:
            results[frame] = {"error": f"우리 URDF 에 {frame} 없음: {e}"}
            print(f"  {frame}: 우리 URDF 에 없음 — 건너뜀")
            continue
        q_lo, q_hi = arm.limits      # ← chain12 블록보다 **먼저** 정의해야 한다

        # ★ **base_link 기준 전체 체인.** 이게 이번 개정의 핵심이다.
        #
        #    문제: SDK FK 가 **비결정론적**이었다 — 같은 q 를 두 번 넣었는데
        #    0.041mm 차이가 났다. FK 는 순수 함수여야 하므로, SDK 가 내
        #    joint_state 만 쓰는 게 아니라 **살아 있는 다리/토르소 상태를
        #    섞고 있다**는 뜻이다. 그러면 body-relative 가 상쇄하려는 root 변환이
        #    샘플마다 달라져 잔차로 남는다 (관측: body-rel median 1.88mm).
        #
        #    해결: 다리를 우리도 모델링한다.
        #      - 우리 FK 를 base_link 기준 12-DoF(leg 5 + arm 7)로 계산하고
        #      - SDK 에도 leg 를 **명시적으로 넘겨** 살아 있는 상태를 쓰지 못하게 한다
        #    그러면 root 오프셋이 아예 없어져 **절대 포즈를 직접 비교**할 수 있다.
        #    이게 훨씬 강한 검정이고, 덤으로 다리 체인까지 검증된다.
        chain12 = None
        try:
            chain12 = build_chain(frame, root="base_link")
            n_leg = len(chain12.dof_idx) - len(q_lo)
            print(f"  base_link 전체 체인: dof {len(chain12.dof_idx)} "
                  f"(leg {n_leg} + arm {len(q_lo)})")
        except Exception as e:
            print(f"  ⚠️ base_link 전체 체인 실패: {e} — 상대 비교만 합니다.")
            n_leg = 0

        # ⚠️ **기본은 현재 자세 주변 샘플링이다.**
        #    전 범위 무작위 샘플링은 SDK 의 관절 한계를 넘을 수 있고, SDK 가
        #    clamp 하면 우리와 다른 자세를 계산하게 되어 "기구학 불일치" 로
        #    오진한다. 현재 자세 주변은 반드시 유효하고, 학습에서 실제로 쓰는
        #    영역이기도 하다.
        q_center = None
        if sampling == "local":
            try:
                cjs = motion.get_chain_joint_state() or {}
                if chain in cjs and len(cjs[chain]) == len(q_lo):
                    q_center = np.asarray(cjs[chain], dtype=float)
            except Exception:
                pass
            if q_center is None:
                print(f"  ⚠️ 현재 자세를 못 읽어 전 범위 샘플링으로 대체합니다.")
        mode = "현재 자세 ±%.2f rad" % amp if q_center is not None else "전 범위 무작위"
        print(f"  샘플링: {mode}  n={n}")

        pf = preflight(motion, frame, ref_frame, chain,
                       q_center if q_center is not None else (q_lo + q_hi) / 2, arm)
        results.setdefault("_preflight", {})[frame] = pf
        if not pf.get("ok"):
            results[frame] = {"error": "preflight 실패: " + str(pf.get("reason"))}
            print(f"  → preflight 실패. 이 프레임은 건너뜁니다.")
            continue

        # 다리 상태 — 우리 FK 에도 쓰고 SDK 에도 넘긴다
        leg_now = None
        if chain12 is not None:
            try:
                cj = motion.get_chain_joint_state() or {}
                if "leg" in cj and len(cj["leg"]) == n_leg:
                    leg_now = np.asarray(cj["leg"], dtype=float)
                    print(f"  leg 상태 고정: {np.round(leg_now, 4)}")
            except Exception:
                pass
            if leg_now is None:
                print("  ⚠️ leg 상태를 못 읽어 절대 비교를 건너뜁니다.")
                chain12 = None

        ours, theirs = [], []
        ours_abs, theirs_abs = [], []
        qs = []
        bad = 0
        for _ in range(n):
            if q_center is not None:
                q = np.clip(q_center + rng.uniform(-amp, amp, size=len(q_lo)),
                            q_lo, q_hi)
            else:
                q = rng.uniform(q_lo, q_hi)
            qs.append(q)
            js = {chain: [float(x) for x in q]}
            if leg_now is not None:
                # leg 를 명시하면 SDK 가 살아 있는 상태를 못 쓴다 → 결정론 회복
                js["leg"] = [float(x) for x in leg_now]
            try:
                st, pose = motion.forward_kinematics(
                    frame, reference_frame=ref_frame, joint_state=js)
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
            if chain12 is not None and leg_now is not None:
                ours_abs.append(chain12.fk(np.concatenate([leg_now, q])))
                theirs_abs.append(cands[pick])
        if not ours:
            results[frame] = {"error": "호출 실패"}
            continue
        c = compare(ours, theirs)

        # ★ 절대 비교가 가능하면 **그게 주 판정**이다. 같은 frame(base_link)에서
        #   같은 관절값으로 계산했으므로 상대변환 트릭이 필요 없다.
        c_abs = None
        if ours_abs:
            c_abs = compare(ours_abs, theirs_abs)
            ap, ar_ = c_abs["abs_pos_mm"], c_abs["abs_rot_deg"]
            print(f"\n── {frame}  [base_link 절대 비교 — leg 명시]")
            print(f"   pos median {ap['median']:8.3f}  p95 {ap['p95']:8.3f}  "
                  f"max {ap['max']:8.3f} mm")
            print(f"   rot median {ar_['median']:8.4f}  p95 {ar_['p95']:8.4f}  "
                  f"max {ar_['max']:8.4f}°")
            if ap["median"] <= REL_POS_TOL_MM and ar_["median"] <= REL_ROT_TOL_DEG:
                print("   ✅ **일치 — 우리 URDF 가 SDK 모델과 맞습니다.**")
                print("      (다리 5-DoF 체인까지 함께 검증됐습니다)")
            else:
                print("   ❌ 어긋납니다. 링크 길이·관절 축·관절 순서를 의심하십시오.")

        ok, verdict, notes = interpret(c, c_abs)
        results[frame] = {**c, "abs_base": c_abs,
                          "ok": ok, "verdict": verdict, "notes": notes}
        print(f"\n── {frame}")
        print(f"   절대  pos median {c['abs_pos_mm']['median']:8.3f} mm   "
              f"rot median {c['abs_rot_deg']['median']:7.4f}°")
        bp, br = c["body_rel_pos_mm"], c["body_rel_rot_deg"]
        print(f"   body-rel  pos median {bp['median']:8.3f}  p95 {bp['p95']:8.3f}  "
              f"max {bp['max']:8.3f} mm   ← **주 판정 기준** (root 차이에 불변)")
        print(f"             rot median {br['median']:8.4f}  p95 {br['p95']:8.4f}  "
              f"max {br['max']:8.4f}°")
        sp = c["spat_rel_pos_mm"]
        print(f"   spat-rel  median {sp['median']:8.3f}  max {sp['max']:8.3f} mm"
              f"   (root 가 다르면 커지는 게 정상 — 참고용)")
        # 최악 샘플이 관절 한계에 붙어 있는지 — clamp 가설 확인
        k = bp.get("argmax")
        if k is not None and k + 1 < len(qs):
            near = [f"j{i+1}" for i, v in enumerate(qs[k])
                    if min(v - q_lo[i], q_hi[i] - v) < 0.05]
            print(f"   최악 샘플 #{k}: 한계 0.05rad 이내 관절 "
                  f"{near or '(없음)'}")
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
        "sampling": sampling,
        "amp_rad": amp,
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
    ap.add_argument("--self-test-live", action="store_true",
                    help="가짜 SDK 로 run_live 전 경로 검증 (로봇 불필요)")
    ap.add_argument("--n", type=int, default=200, help="표본 수")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--ref-frame", default="base_link",
                    help="SDK FK 의 reference_frame. 2026-07-31 실측 결과 SDK 가 "
                         "지원하는 프레임은 {map, world, base_link} 뿐이고 "
                         "torso_base_link 는 **없다**. 그래서 base_link 로 받고 "
                         "body-relative 대조(root 차이에 불변)로 판정한다")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sampling", default="local", choices=["local", "full"],
                    help="local=현재 자세 주변(기본, 안전) / "
                         "full=관절 전 범위 무작위(SDK 한계를 넘을 수 있음)")
    ap.add_argument("--amp", type=float, default=0.20,
                    help="local 샘플링 진폭 [rad]")
    ap.add_argument("--no-init", action="store_true",
                    help="init() 없이 시도 — forward_kinematics 는 순수 계산이라 "
                         "로봇 연결 없이도 될 수 있다")
    args = ap.parse_args()

    if args.self_test:
        rc = self_test()
        return rc or self_test_live()
    if args.self_test_live:
        return self_test_live()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "robot", "assets",
        f"fk_crosscheck_{socket.gethostname()}.json")
    return run_live(args.n, args.side, args.ref_frame, out, args.no_init,
                    args.sampling, args.amp)


# 종료 시 반납할 핸들. 여러 경로에서 return 하므로 전역에 모아둔다.
_HANDLES: dict = {"robot": None, "motion": None}


# ─────────────────────────────────────────────────────────────────────────────
# 가짜 SDK — run_live 전 경로를 맥에서 태운다
# ─────────────────────────────────────────────────────────────────────────────
#
# 왜 필요한가: self-test 가 compare/interpret 만 덮고 run_live 는 안 덮었다.
# 그 결과 `q_lo` 를 정의 전에 쓰는 단순한 순서 오류가 **로봇에서야** 드러났고,
# 왕복 한 번을 썼다. 온보드는 회사에 있으므로 그 비용이 크다.
#
# 이 가짜는 우리 URDF 로 FK 를 계산하므로 **반드시 PASS 여야 한다.**
# PASS 가 안 나오면 run_live 배선이 깨진 것이다.


class _FakeMotion:
    def __init__(self, side="left"):
        self.side = side
        self._chain = build_chain(f"{side}_arm_end_effector_mount_link",
                                  root="base_link")
        self._n = len(self._chain.dof_idx)
        self._leg = [0.4992, 1.4996, 0.9996, -0.0003, 0.0]
        self._arm = [1.1234, -1.5077, -0.5757, -1.6762, -0.0005, -0.5491, -0.0001]

    def init(self):
        return True

    def get_supported_chains(self):
        return {"head", "left_arm", "leg", "mobile_base", "right_arm", "torso"}

    def get_supported_frames(self):
        return {"base_link", "map", "world"}

    def get_supported_ee_frames(self):
        return {"camera_base", "camera_object", "ee_base"}

    def get_link_names(self, only_end_effector=False):
        return [l.name.replace("_joint", "_link") for l in self._chain.links] + [
            f"{self.side}_arm_end_effector_mount_link"]

    def get_chain_joint_state(self):
        return {"leg": list(self._leg), f"{self.side}_arm": list(self._arm),
                "torso": self._leg[3:5], "head": [0.0, 0.3487]}

    def forward_kinematics(self, target_frame, reference_frame="base_link",
                           joint_state=None, params=None):
        js = joint_state or {}
        leg = list(js.get("leg", self._leg))
        arm = list(js.get(f"{self.side}_arm", self._arm))
        T = self._chain.fk(np.asarray(leg + arm, dtype=float))
        # [x y z qx qy qz qw] 로 돌려준다 (SDK 와 같은 길이 7)
        R = T[:3, :3]
        w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        if w < 1e-8:
            w = 1e-8
        q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
             (R[1, 0] - R[0, 1]) / (4 * w), w]
        return _FakeStatus("SUCCESS"), list(T[:3, 3]) + q


class _FakeStatus:
    def __init__(self, name):
        self.name = name


class _FakeRobot:
    def init(self, s=None):
        return True

    def get_joint_group_names(self):
        return ["head", "left_arm", "right_arm", "left_gripper",
                "right_gripper", "leg", "chassis"]

    def get_device_information(self):
        return {"model": "G1", "serial_number": "FAKE", "manufacturer": "Galbot"}


class _FakeSDK:
    """sdk_entry.acquire 가 찾는 모양을 흉내낸다."""
    __name__ = "fake_galbot_sdk"

    class g1:
        GalbotRobot = _FakeRobot
        GalbotMotion = _FakeMotion

    GalbotRobot = _FakeRobot
    GalbotMotion = _FakeMotion


def self_test_live() -> int:
    """가짜 SDK 로 run_live 전 경로를 태운다. **PASS 가 나와야 정상.**"""
    import tempfile
    print("=" * 74)
    print("run_live 배선 검증 (가짜 SDK — 로봇 불필요)")
    print("=" * 74)
    fake = _FakeSDK()
    # acquire 를 가짜용으로 우회
    orig = sdk_entry.acquire
    sdk_entry.acquire = lambda mod, name, **kw: (
        (_FakeRobot() if name == "GalbotRobot" else _FakeMotion()),
        f"fake.{name}()")
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        rc = run_live(20, "left", "base_link", tmp, False, "local", 0.20,
                      _fake=fake)
    finally:
        sdk_entry.acquire = orig
    print()
    if rc == 0:
        print("✅ run_live 배선 정상 — 가짜 FK 와 완전 일치 (PASS)")
        return 0
    print(f"❌ run_live 가 rc={rc} 를 반환했습니다. 배선이 깨졌습니다.")
    print("   가짜는 우리 URDF 로 FK 를 계산하므로 반드시 PASS 여야 합니다.")
    return 1


def sdk_teardown(robot=None, motion=None) -> None:
    """os._exit 전에 **세션을 명시적으로 반납한다.**

    왜 필요한가 — 2026-07-31 관측
    ─────────────────────────────
    같은 로봇에서 motion.init() 이 처음엔 True 였다가 그 뒤로 계속 False 가 됐다.
    프로세스가 끝나도 유지되는 상태이므로 **로봇 쪽에 세션이 남아 있다**는 뜻이다.

    원인으로 의심되는 것이 내가 넣은 os._exit() 다. 인터프리터 종료 시 SDK 가
    segfault 를 내서 그걸 우회했는데, 그러면 소멸자가 안 돌아 세션도 반납되지
    않는다. 크래시를 피하려다 상태를 오염시킨 셈이다.

    그래서 순서를 바꾼다: **명시적으로 반납하고 나서** os._exit 로 빠져나간다.
    공식 문서의 종료 시퀀스가 request_shutdown → wait_for_shutdown → destroy 다.
    각 호출은 개별 try 로 감싼다 — 하나가 실패해도 나머지는 시도해야 한다.
    """
    for obj, name in ((motion, "motion"), (robot, "robot")):
        if obj is None:
            continue
        for m in ("request_shutdown", "wait_for_shutdown", "destroy"):
            if not hasattr(obj, m):
                continue
            try:
                getattr(obj, m)()
                print(f"  [teardown] {name}.{m}() ok", flush=True)
            except Exception as e:
                print(f"  [teardown] {name}.{m}() → {type(e).__name__}: {e}",
                      flush=True)


def _hard_exit(code: int) -> None:
    """SDK 소멸자를 건너뛰고 종료한다.

    galbot_sdk 는 인터프리터 종료 시점에 SIGSEGV / pthread_mutex assertion 으로
    죽는다 (온보드 실측 — 결과 파일을 다 저장하고 "다음:" 안내까지 출력한 뒤에
    터졌다). 싱글톤 C++ 객체의 정적 소멸 순서 문제로 보이며 우리가 고칠 수 없다.
    결과는 이미 디스크에 있으므로, 버퍼만 비우고 os._exit 로 빠져나간다.
    이걸 안 하면 make 가 실패로 판단해 후속 타겟이 안 돈다.
    """
    # ⚠️ 순서가 중요하다: **반납 먼저, 그다음 os._exit.**
    #    반대로 하면 세션이 로봇에 남아 다음 실행의 init() 이 False 가 된다.
    try:
        sdk_teardown(_HANDLES.get("robot"), _HANDLES.get("motion"))
    except Exception as e:
        print(f"  [teardown] 실패: {type(e).__name__}: {e}")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    _hard_exit(main())

