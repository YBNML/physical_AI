"""
G1 기구학 검증 — 문서에 인용된 수치를 직접 재현한다.

실행:
    python robot/test_kinematics.py

검증 항목
─────────
  T1  S–R–S 구조     : 어깨 j1,2,3 / 손목 j5,6,7 축이 각각 한 점에 집중하는가
  T2  IK 왕복        : q → FK → IK(pose, psi) → FK 가 원래 포즈와 일치하는가
  T3  psi 스윕       : EE 포즈를 고정한 채 psi 를 돌리면 팔꿈치만 움직이는가
                       → 이것이 "6-DoF 포즈로는 7-DoF 형상이 결정되지 않는다"의 기하학
  T4  L2 평균 붕괴   : Δpsi 만큼 다른 두 유효 해를 관절공간에서 평균하면
                       EE 가 얼마나 벗어나는가 (REV.1 §3.5 의 표)
  T5  head→arm_base : 다리/허리/베이스가 정말 소거되는가
  T6  속도           : IK 1회 비용 (학습 모델과의 비교 기준)
  T7  tip 프레임     : tip 후보 4개의 상수 오프셋과 판별표가 URDF 와 맞는가

⚠️ **T1~T7 은 전부 URDF 자기일관성 검정이다.**
   URDF 자체가 실기체와 다르면 7/7 통과해도 아무것도 보증하지 않는다.
   외부 ground truth(SDK FK) 대조는 아직 없다 — pybind11 시그니처가 확정되어야
   호출문을 쓸 수 있기 때문이다 (`docs/RUNBOOK.md` §2-1 의 `make probe` 선행).
   T7 은 그 대조를 **가능하게 만드는 사전 작업**이지 대조 자체가 아니다.
"""

from __future__ import annotations

import math
import time

import numpy as np

from g1_kinematics import (
    G1Arm, G1Head, T_from, se3_inv, pose_error, build_chain, _load_joints,
    identify_tip_frame, TIP_SIGNATURE_FROM_TCP, TIP_CANDIDATES,
)

RNG = np.random.default_rng(0)
OK, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def rand_q(arm: G1Arm, margin: float = 0.15) -> np.ndarray:
    """관절 한계 안쪽에서 무작위 자세 (특이점 회피용 여유 포함)."""
    lo, hi = arm.limits
    span = hi - lo
    return RNG.uniform(lo + margin * span, hi - margin * span)


# ─────────────────────────────────────────────────────────────────────────────


def t1_srs_structure(arm: G1Arm) -> bool:
    """어깨(j1,2,3)와 손목(j5,6,7) 축이 각각 한 점에서 만나는지 최소자승으로 검정."""
    print("\n── T1  S–R–S 구조 " + "─" * 52)
    joints = _load_joints()
    q0 = np.zeros(arm.chain.n_dof)
    Ts = arm.chain.fk_all(q0)
    idx = {arm.chain.links[i].name: i for i in range(len(arm.chain.links))}

    def axis_line(jname):
        T = Ts[idx[jname]]
        p = T[:3, 3]
        a = T[:3, :3] @ np.asarray(joints[jname]["axis"], dtype=float)
        return p, a / np.linalg.norm(a)

    def concurrency(names):
        """여러 직선에 대한 최소자승 교점과 잔차."""
        A = np.zeros((3, 3)); b = np.zeros(3)
        for n in names:
            p, a = axis_line(n)
            P = np.eye(3) - np.outer(a, a)      # 축에 직교 투영
            A += P; b += P @ p
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        resid = max(np.linalg.norm((np.eye(3) - np.outer(a, a)) @ (x - p))
                    for p, a in (axis_line(n) for n in names))
        return x, resid

    s = arm.side
    ok = True
    for label, names in [("어깨 j1,j2,j3", [f"{s}_arm_joint1", f"{s}_arm_joint2", f"{s}_arm_joint3"]),
                         ("손목 j5,j6,j7", [f"{s}_arm_joint5", f"{s}_arm_joint6", f"{s}_arm_joint7"])]:
        pt, res = concurrency(names)
        good = res < 1e-6
        ok &= good
        print(f"  {label:16s} 교점 {np.round(pt, 5)}  잔차 {res*1000:.6f} mm  "
              f"{OK if good else FAIL}")
    print(f"  → canonical S–R–S 이므로 (pose, psi) 에 closed-form 해가 존재")
    return ok


def t2_ik_roundtrip(arm: G1Arm, n: int = 40) -> bool:
    """q → FK → IK(pose, psi) → FK 가 원래 포즈로 돌아오는가."""
    print("\n── T2  IK 왕복 " + "─" * 55)
    errs_p, errs_r, errs_psi, fails = [], [], [], 0
    for _ in range(n):
        q = rand_q(arm)
        T = arm.fk(q)
        psi = arm.arm_angle(q)
        if math.isnan(psi):
            continue
        # 원래 해와 다른 곳에서 시작 (seed 를 흔든다)
        seed = np.clip(q + RNG.normal(0, 0.25, 7), *arm.limits)
        sol = arm.ik(T, psi, q_seed=seed)
        if sol is None:
            fails += 1
            continue
        dp, dr = pose_error(T, arm.fk(sol))
        errs_p.append(dp); errs_r.append(dr)
        errs_psi.append(abs((arm.arm_angle(sol) - psi + math.pi) % (2*math.pi) - math.pi))

    ok = bool(errs_p) and max(errs_p) < 1e-4 and max(errs_r) < 1e-4
    print(f"  시행 {n}  수렴 {len(errs_p)}  실패 {fails}")
    if errs_p:
        print(f"  병진 오차   max {max(errs_p)*1e6:8.3f} µm   median {np.median(errs_p)*1e6:8.3f} µm")
        print(f"  회전 오차   max {math.degrees(max(errs_r))*1e3:8.3f} m°  "
              f"median {math.degrees(np.median(errs_r))*1e3:8.3f} m°")
        print(f"  psi 오차    max {math.degrees(max(errs_psi)):8.5f}°")
    print(f"  {OK if ok else FAIL}")
    return ok


def t3_psi_sweep(arm: G1Arm) -> bool:
    """EE 포즈를 고정한 채 psi 를 돌리면 EE 는 그대로이고 팔꿈치만 움직여야 한다."""
    print("\n── T3  psi 스윕 (self-motion manifold) " + "─" * 31)
    q = rand_q(arm, margin=0.25)
    T = arm.fk(q)
    psi0 = arm.arm_angle(q)

    elbows, ee_err, achieved = [], [], []
    q_seed = q.copy()
    for d in np.linspace(-0.6, 0.6, 13):
        sol = arm.ik(T, psi0 + d, q_seed=q_seed)
        if sol is None:
            continue
        q_seed = sol
        elbows.append(arm.joint_centers(sol)["elbow"])
        ee_err.append(pose_error(T, arm.fk(sol))[0])
        achieved.append(arm.arm_angle(sol) - psi0)

    if len(elbows) < 5:
        print(f"  수렴 샘플 부족 ({len(elbows)})  {FAIL}")
        return False

    E = np.array(elbows)
    span = E.max(axis=0) - E.min(axis=0)
    ok = max(ee_err) < 1e-4 and np.linalg.norm(span) > 0.02
    print(f"  psi 범위    ±{math.degrees(0.6):.0f}°  수렴 {len(elbows)}/13")
    print(f"  EE 이동     max {max(ee_err)*1e6:.3f} µm      ← 0 이어야 함")
    print(f"  팔꿈치 스윕  x {span[0]*1000:6.1f} mm  y {span[1]*1000:6.1f} mm  z {span[2]*1000:6.1f} mm")
    print(f"  → EE 를 고정해도 팔꿈치가 {np.linalg.norm(span)*1000:.0f} mm 움직인다.")
    print(f"    이것이 6-DoF 포즈가 7-DoF 팔에 대해 underdetermined 라는 것의 기하학.")
    print(f"  {OK if ok else FAIL}")
    return ok


def t4_l2_collapse(arm: G1Arm) -> bool:
    """
    Δpsi 만큼 다른 두 유효 IK 해를 관절공간에서 평균하면 EE 가 얼마나 벗어나는가.

    L2 손실로 학습한 회귀 모델이 self-motion manifold 위의 다중 해에
    노출될 때 일어나는 일이다. 평균은 manifold 위에 있지 않다.
    """
    print("\n── T4  L2 평균 붕괴 " + "─" * 50)
    print(f"  {'Δpsi':>8}  {'EE 오차':>12}  {'회전 오차':>12}  {'n':>4}")
    rows = []
    for dpsi_deg in (10, 30, 45, 60, 90):
        d = math.radians(dpsi_deg)
        errs_p, errs_r = [], []
        for _ in range(25):
            q = rand_q(arm, margin=0.28)
            T = arm.fk(q)
            psi = arm.arm_angle(q)
            if math.isnan(psi):
                continue
            a = arm.ik(T, psi - d/2, q_seed=q)
            b = arm.ik(T, psi + d/2, q_seed=q)
            if a is None or b is None:
                continue
            mid = 0.5 * (a + b)                 # ← L2 회귀가 수렴하는 지점
            dp, dr = pose_error(T, arm.fk(mid))
            errs_p.append(dp); errs_r.append(dr)
        if errs_p:
            mp = float(np.median(errs_p))
            mr = math.degrees(float(np.median(errs_r)))
            rows.append((dpsi_deg, mp, mr, len(errs_p)))
            print(f"  {dpsi_deg:6d}°  {mp*1000:9.1f} mm  {mr:9.2f}°  {len(errs_p):4d}")

    ok = len(rows) >= 4 and rows[-1][1] > rows[0][1]
    if ok:
        print(f"\n  → 데모 데이터의 psi 산포가 크면 L2 회귀는 이 크기로 붕괴한다.")
        print(f"    판정 기준: 산포 <20° 면 생존 가능, >60° 면 psi 스칼라가 인터페이스에 필수.")
        print(f"    실제 산포는 RoboCOIN 에 FK 를 돌려 측정한다 (tools/robocoin_psi_audit.py).")
    print(f"  {OK if ok else FAIL}")
    return ok


def t5_head_transform() -> bool:
    """head→arm_base 가 다리/허리/베이스에 불변인지 확인."""
    print("\n── T5  head→arm_base 불변성 " + "─" * 42)
    joints = _load_joints()
    head_chain = build_chain("head_end_effector_mount_link", joints=joints)
    # 어깨 마운트까지 (torso 에서 fixed 로 갈라지는 지점)
    arm_chain = build_chain("torso_left_arm_mount_link", joints=joints)

    # torso_base_link 기준이므로 다리/허리는 애초에 체인에 없다 — 그것이 요점
    head_names = [head_chain.links[i].name for i in head_chain.dof_idx]
    arm_names = [arm_chain.links[i].name for i in arm_chain.dof_idx]
    leg_in_head = [n for n in head_names if n.startswith("leg_")]
    leg_in_arm = [n for n in arm_names if n.startswith("leg_")]

    ok = not leg_in_head and not leg_in_arm and len(head_names) == 2
    print(f"  torso→head  가변 관절: {head_names}")
    print(f"  torso→어깨  가변 관절: {arm_names or '(없음 — 전부 fixed)'}")
    print(f"  head 체인의 다리/허리 관절: {leg_in_head or '없음'}")
    print(f"  → head→arm_base 는 목 관절 {len(head_names)}개에만 의존.")
    print(f"    Model 2 입력(팔 관절 14 + 이미지 + F/T)에는 그 2개가 없다.")

    # 목 오차 → 어깨 프레임 이동 민감도
    H = G1Head()
    print(f"\n  목 오차 → 헤드 프레임 이동 (어깨까지 lever arm 반영):")
    T0 = H.T_torso_head(np.zeros(2))
    for deg in (1, 2, 5, 10):
        T1 = H.T_torso_head(np.array([math.radians(deg), 0.0]))
        dp, dr = pose_error(T0, T1)
        # 0.6 m 도달거리에서의 목표 오차
        reach_err = dp + 0.6 * dr
        print(f"    {deg:2d}°  헤드 이동 {dp*1000:6.2f} mm / {math.degrees(dr):5.2f}°"
              f"   → 0.6 m 도달거리 목표 오차 ≈ {reach_err*1000:6.1f} mm")
    print(f"  {OK if ok else FAIL}")
    return ok


def t6_speed(arm: G1Arm) -> bool:
    """IK 1회 비용. 학습 모델과의 비교 기준."""
    print("\n── T6  IK 속도 " + "─" * 55)
    qs = [rand_q(arm) for _ in range(60)]
    targets = [(arm.fk(q), arm.arm_angle(q), q) for q in qs]
    targets = [t for t in targets if not math.isnan(t[1])]

    # warm seed (연속 제어에서의 실제 조건)
    t0 = time.perf_counter()
    n_ok = 0
    for T, psi, q in targets:
        seed = np.clip(q + RNG.normal(0, 0.03, 7), *arm.limits)
        if arm.ik(T, psi, q_seed=seed) is not None:
            n_ok += 1
    dt_warm = (time.perf_counter() - t0) / len(targets)

    # cold seed
    t0 = time.perf_counter()
    for T, psi, q in targets:
        arm.ik(T, psi, q_seed=None)
    dt_cold = (time.perf_counter() - t0) / len(targets)

    print(f"  warm seed (연속 제어)  {dt_warm*1e6:8.1f} µs/해  →  {1/dt_warm:8.0f} Hz")
    print(f"  cold seed (임의 시작)  {dt_cold*1e6:8.1f} µs/해  →  {1/dt_cold:8.0f} Hz")
    print(f"  수렴률 {n_ok}/{len(targets)}")
    print(f"  → 학습된 pose→joint 모델의 현실적 25–60 Hz 와 비교하십시오.")
    print(f"    (SDK inverse_kinematics_by_state 와는 미비교 — sampling+seed 기반이라")
    print(f"     psi 미지정·비결정론적이다. µs 만 나란히 적으면 오독한다.)")
    print(f"    그리고 IK 는 실패 시 None 을 반환한다 (타입 있는 infeasibility 신호).")
    return dt_warm < 5e-3


# ─────────────────────────────────────────────────────────────────────────────


def t7_tip_frames(side: str = "left") -> bool:
    """tip 후보 4개의 상수 오프셋이 판별표와 맞고, 판별이 유일한가.

    왜 이게 필요한가: 우리 FK 의 tip 은 `*_gripper_tcp_link` 로 **골라 잡은 것**이고,
    SDK FK 가 어느 프레임을 쓰는지는 미확인이다. 후보 4개가 전부 fixed joint 로
    연결돼 있어 오차가 **관절각과 무관한 상수**로 나오는데, 상수 오차는 학습이
    그대로 흡수해 조용히 틀린 채 수렴한다.

    그 상수성이 그대로 진단 도구가 된다 — SDK FK 표본 하나의 잔차만으로
    프레임이 판별되는지 여기서 확인해둔다.
    """
    print("\n── T7 tip 프레임 판별표 " + "─" * 44)
    ok = True

    # 1) 판별표가 URDF 에서 실제로 유도되는가 (하드코딩 표류 방지)
    ref = build_chain(f"{side}_gripper_tcp_link", root=f"{side}_arm_link7")
    T_tcp = ref.fk(np.zeros(len(ref.dof_idx)))
    print(f"  {'후보':34s} {'|Δt| (mm)':>11} {'Δθ (°)':>9}   판별")
    for cand in TIP_CANDIDATES:
        ch = build_chain(f"{side}_{cand}", root=f"{side}_arm_link7")
        T_c = ch.fk(np.zeros(len(ch.dof_idx)))
        dp, dr = pose_error(T_tcp, T_c)
        dmm, ddeg = dp * 1e3, math.degrees(dr)
        got = identify_tip_frame(dmm, ddeg)
        good = (got == cand)
        ok &= good
        print(f"  {cand:34s} {dmm:11.2f} {ddeg:9.2f}   "
              f"{str(got):30s} {OK if good else FAIL}")

    # 2) 잡음이 있어도 유일하게 갈리는가
    noisy = all(identify_tip_frame(d + dd, a + da) == name
                for (d, a), name in TIP_SIGNATURE_FROM_TCP.items()
                for dd, da in ((1.5, 3.0), (-1.5, -3.0)))
    print(f"  잡음 ±1.5mm/±3° 하에서도 유일 판별      {OK if noisy else FAIL}")
    ok &= noisy

    # 3) 표에 없는 잔차는 반드시 None (= root 프레임 문제 신호)
    none_ok = all(identify_tip_frame(d, a) is None
                  for d, a in ((588.0, 12.0), (50.0, 45.0), (143.2, 90.0)))
    print(f"  표 밖·모호 잔차는 None 반환             {OK if none_ok else FAIL}")
    ok &= none_ok

    # 4) torso 아래에 남아 있는 자유도 — root 프레임 오판 시 오차 규모
    leg = build_chain("torso_base_link", root="base_link")
    n_leg = len(leg.dof_idx)
    print(f"  base_link → torso_base_link 사이 DOF   {n_leg}"
          f"  ({', '.join(leg.links[i].name for i in leg.dof_idx)})")
    print(f"    → SDK FK 가 base 기준이면 이 {n_leg}개만큼 어긋난다."
          f" 잔차가 표에 없으면 tip 이 아니라 여기를 의심할 것.")

    return ok


def main() -> int:
    print("=" * 72)
    print("Galbot G1 기구학 검증")
    print("=" * 72)
    arm = G1Arm("left")
    print(f"  상완 {arm.d_se:.4f} m + 전완 {arm.d_ew:.4f} m = {arm.d_se+arm.d_ew:.4f} m")

    results = {
        "T1 S–R–S 구조": t1_srs_structure(arm),
        "T2 IK 왕복": t2_ik_roundtrip(arm),
        "T3 psi 스윕": t3_psi_sweep(arm),
        "T4 L2 평균 붕괴": t4_l2_collapse(arm),
        "T5 head 변환": t5_head_transform(),
        "T6 IK 속도": t6_speed(arm),
        "T7 tip 프레임": t7_tip_frames(),
    }

    print("\n" + "=" * 72)
    for k, v in results.items():
        print(f"  {k:22s} {OK if v else FAIL}")
    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} 통과")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
