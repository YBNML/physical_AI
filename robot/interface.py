"""
Model 1 → Model 2 인터페이스 스키마 v0.1

이 파일이 이 프로젝트의 **제품 자산**이다. 이유:

  - 인터페이스가 동결·버전 관리되면 상위 계층만 재학습해서 출하할 수 있다
    (물체에 닿는 계층을 재검증하지 않고). 이것이 2모델 분리에 대한
    가장 강한 상업적 논거다.
  - 반대로 인터페이스가 바뀌면 양쪽 다 재검증해야 하고 그 이점이 증발한다.
    → `SCHEMA_VERSION` 을 올리는 것은 가벼운 결정이 아니다.
  - 나중에 학습된 Model 2 를 넣든 해석적 WBC 를 쓰든, **이 계약이 같으면
    스왑이 config 변경**이 된다.

설계 근거는 docs/INTERFACE.md 에, 각 필드가 왜 필요한지는 표에 정리돼 있다.
요약하면 — 원안의 "6-DoF 포즈" 는 다음이 전부 빠져 있었다:

    그리퍼(없으면 파지 성공률 구조적으로 0%)
    dt(없으면 접근 속도 = GPU 스케줄링의 함수)
    psi(7-DoF 팔에 6-DoF 포즈는 underdetermined)
    손간 상대 포즈(뺄셈으로 복구 불가 — L2 손실이 공통모드와 차동을 동일하게 벌함)
    타임스탬프/헤드 상태(프레임 변환에 필수)
    stiffness/mode(없으면 F/T 센서에 구동 경로가 없음)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Optional

import numpy as np

SCHEMA_VERSION = (0, 1, 0)      # semver — 올리면 양쪽 계층 재검증 필요
CHUNK_H = 8                     # waypoint 수. Diffusion Policy 최적 Ta=8 과 일치


class Frame(IntEnum):
    """
    좌표계.

    ⚠️ Model 2 는 TORSO_BASE 만 소비해야 한다. Model 1 은 HEAD 로 예측해도
       되지만, 변환은 **인터페이스에서 촬영시각 head_joint 로** 이뤄진다.
       measure: 목 1도당 0.6m 도달거리에서 ~10.5mm.
    """
    TORSO_BASE = 0      # torso_base_link — Model 2 의 입력 프레임
    HEAD = 1            # head_link2 — Model 1 의 내부 표현으로만 허용
    WORLD = 2           # ⚠️ **선언만 있고 미구현.** validate() 가 TORSO_BASE 외
                        #    전부 거부하고, base/world 변환 코드도 아직 없다.
                        #    구현하려면 base_link→torso_base_link 의 다리 5-DoF 를
                        #    태워야 한다 (robot/g1_kinematics.py 의 T7 참조).


class Phase(IntEnum):
    """작업 단계. contact gating 과 stiffness 스케줄에 쓴다."""
    APPROACH = 0
    CONTACT = 1
    TRANSPORT = 2
    MANIPULATE = 3
    RELEASE = 4


class CoordMode(IntEnum):
    """
    양팔 협응 모드.

    TIGHTLY_COUPLED 에서는 두 손이 하나의 강체를 잡으므로
    T_left^-1 · T_right 가 파지 기하로 고정된다 (6개 홀로노믹 구속).
    독립적인 두 절대 포즈는 12 DoF 를 명령하는데 그중 6개는 물체를 움직이지
    못하고 내부 렌치만 만든다. 이 모드에서는 (물체 포즈 + 고정 파지 변환)
    으로 재파라미터화해서 초과 6 DoF 를 **구조적으로 제거**한다.
    """
    INDEPENDENT = 0
    GOAL_COORDINATED = 1
    LOOSELY_COUPLED = 2
    TIGHTLY_COUPLED = 3


class Supervisor(IntEnum):
    """
    상위의 감독 신호. **MTBI 필드 — 없이 출시 금지.**

    상업 지표는 성공률이 아니라 MTBI(개입 간 평균 시간)다.
    0.95^10 ≈ 0.60 이므로 10스텝 작업에서 99% 를 찍으려면 스텝당 99.9% 가
    필요한데, 파라미터 재배분은 그 지수를 바꾸지 않는다. 오류 감지와 복구만
    바꾼다. "나 막혔다" 를 말할 수 없는 상위는 상업적으로 무가치하다.
    """
    CONTINUE = 0
    RETRY_SUBTASK = 1
    ABORT_TO_SAFE = 2
    REQUEST_TELEOP = 3


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Waypoint:
    """
    한 팔의 한 시점 목표.

    포즈는 **chunk 시작 시점의 측정된 EE 포즈 기준 델타**다.
    절대가 아니다 — 실측 EE 절대 69.0% vs EE 델타 89.6%. 델타는 hand-eye
    캘리브레이션 편향(~2.1mm/3.2°)을 양 끝점이 공유하므로 상쇄한다.
    """
    dp: np.ndarray                  # (3,) 병진 델타 [m]
    dR6: np.ndarray                 # (6,) 6D 연속 회전 표현
                                    #      ⚠️ Euler/quaternion 금지 — wrap-around 불연속
    gripper_width: float            # [m] 연속. binary 금지 — 암묵적 파지력 채널이기도 함
    gripper_effort: float           # 정규화 [0,1]. G1 은 그리퍼만 effort 가 유효
    dt: float                       # [s] 이 waypoint 까지의 시간
                                    #      없으면 접근 속도 = Δpose ÷ 추론 jitter
                                    #      → 동일 명령에 접촉력 ~35% 변동
    psi: float                      # [rad] arm angle. 7-DoF 를 완전 결정시킴.
                                    #   psi=0 규약: 어깨→손목 축과 torso_base_link
                                    #   의 +Z 를 포함하는 평면.
                                    #   ⚠️ **이 프로젝트 고유 규약이다.** 대응하는
                                    #   SDK 규약이 없다 — SDK IK 는 SeedType 기반
                                    #   샘플링이고 redundancy 파라미터를 표면에
                                    #   노출하지 않는다. 수신 측이 이 한 줄 없이는
                                    #   규약을 재구성할 수 없으므로 여기 박아둔다.
                                    #   정의 원본: robot/g1_kinematics.py arm_angle()
    stiffness_trans: float = 500.0  # [N/m]
    stiffness_rot: float = 20.0     # [Nm/rad]
    squeeze_force: float = 0.0      # [N] 파지축 방향, 부호 있음

    DIM = 3 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1     # = 16

    def to_vec(self) -> np.ndarray:
        return np.concatenate([
            self.dp, self.dR6,
            [self.gripper_width, self.gripper_effort, self.dt, self.psi,
             self.stiffness_trans, self.stiffness_rot, self.squeeze_force],
        ])

    @classmethod
    def from_vec(cls, v: np.ndarray) -> "Waypoint":
        v = np.asarray(v, dtype=float)
        return cls(dp=v[0:3], dR6=v[3:9], gripper_width=float(v[9]),
                   gripper_effort=float(v[10]), dt=float(v[11]), psi=float(v[12]),
                   stiffness_trans=float(v[13]), stiffness_rot=float(v[14]),
                   squeeze_force=float(v[15]))


@dataclass
class InterArm:
    """
    팔 사이 상대 항. **헤드가 직접 예측해야 하고, 두 절대 포즈의 뺄셈으로
    유도하면 안 된다.**

    결정적 논거는 산술이 아니라 손실이다: 두 절대 포즈에 대한 L2 손실은
    무해한 공통모드 5mm 오차와 물체를 부수는 차동 5mm 오차를 **동일하게**
    벌한다. 네트워크는 상대 구속을 보존하라는 gradient 압력을 한 번도
    받지 않는다.  (UMI 실측: 상대 포즈 제공 시 천 접기 70% vs 30%)
    """
    dp_rel: np.ndarray              # (3,) 목표 T_left→right 병진
    dR6_rel: np.ndarray             # (6,)
    internal_wrench: float = 0.0    # [N] 결합 모드에서 파지축 스퀴즈
    sync_weight: float = 1.0        # [0,1] 이 스텝의 양팔 동시성 엄격도

    DIM = 3 + 6 + 1 + 1             # = 11

    def to_vec(self) -> np.ndarray:
        return np.concatenate([self.dp_rel, self.dR6_rel,
                               [self.internal_wrench, self.sync_weight]])


@dataclass
class Header:
    """
    타입·타임스탬프·버전. 없으면 staleness 가 조용한 지배적 오차항이 된다.
    """
    schema_version: tuple = SCHEMA_VERSION
    seq: int = 0
    t_capture_ns: int = 0           # ⚠️ 카메라 노출 중점. 호스트 수신 시각 아님
    frame: Frame = Frame.TORSO_BASE
    validity_horizon_s: float = 0.75
    latency_offset_s: float = 0.0   # 측정된 Model1→Model2 staleness
                                    # 학습 시에도 같은 분포로 주입할 것

    # 촬영시각의 상체 상태 — 프레임 변환과 도달가능성 맥락에 필요
    q_head: np.ndarray = field(default_factory=lambda: np.zeros(2))
    q_leg: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # 배포 아티팩트 고정: 셋이 섞이면 안 된다
    model1_version: str = ""
    model2_version: str = ""
    calib_version: str = ""


@dataclass
class InterfaceMessage:
    """Model 1 → Model 2 한 메시지."""
    header: Header
    coord_mode: CoordMode = CoordMode.INDEPENDENT
    mode_confidence: float = 1.0
    phase: Phase = Phase.APPROACH
    phase_progress: float = 0.0
    supervisor: Supervisor = Supervisor.CONTINUE

    left: list[Waypoint] = field(default_factory=list)
    right: list[Waypoint] = field(default_factory=list)
    inter: list[InterArm] = field(default_factory=list)

    # TIGHTLY_COUPLED 에서 접촉 진입 시 latch 후 고정
    T_grasp_frozen: Optional[np.ndarray] = None     # (9,) 상대 파지 변환

    # 팔별 병진/회전 불확실성 — RT 계층이 이걸로 속도 제한을 스케일링
    uncertainty: np.ndarray = field(default_factory=lambda: np.zeros(6))

    latent: Optional[np.ndarray] = None             # 선택: 잠재 사이드 채널

    # ── 검증 ────────────────────────────────────────────────────────────────
    def validate(self) -> list[str]:
        """반환된 리스트가 비어 있지 않으면 이 메시지를 실행하면 안 된다."""
        errs: list[str] = []

        if self.header.schema_version != SCHEMA_VERSION:
            errs.append(f"schema 불일치: {self.header.schema_version} != {SCHEMA_VERSION}")

        if self.header.frame != Frame.TORSO_BASE:
            errs.append(
                f"Model 2 는 TORSO_BASE 만 소비해야 함 (받은 값: {self.header.frame.name}). "
                "헤드 프레임 변환은 인터페이스에서 촬영시각 q_head 로 수행할 것.")

        if len(self.left) != len(self.right):
            errs.append(f"양팔 waypoint 수 불일치: {len(self.left)} vs {len(self.right)}")

        if not self.left:
            errs.append("waypoint 없음 — 스텝 단위 인터페이스는 금지 (ACT: k=1 → 1%)")

        for side, wps in (("left", self.left), ("right", self.right)):
            for i, w in enumerate(wps):
                if not np.all(np.isfinite(w.to_vec())):
                    errs.append(f"{side}[{i}] 비유한값")
                if w.dt <= 0:
                    errs.append(f"{side}[{i}] dt<=0 ({w.dt}) — 시간 없는 waypoint 는 궤적이 아님")
                if not (0.0 <= w.gripper_width <= 0.15):
                    errs.append(f"{side}[{i}] gripper_width 범위 밖: {w.gripper_width}")

        if self.coord_mode == CoordMode.TIGHTLY_COUPLED:
            if self.T_grasp_frozen is None:
                errs.append("TIGHTLY_COUPLED 인데 T_grasp_frozen 이 없음 — "
                            "초과 6 DoF 가 내부 렌치가 됨")
            if not self.inter:
                errs.append("TIGHTLY_COUPLED 인데 팔간 상대 항이 없음")

        age = (time.monotonic_ns() - self.header.t_capture_ns) / 1e9
        if self.header.t_capture_ns > 0 and age > self.header.validity_horizon_s:
            errs.append(f"staleness {age*1000:.0f}ms > 지평 "
                        f"{self.header.validity_horizon_s*1000:.0f}ms — "
                        "유지 후 0 으로 감쇠할 것. 절대 외삽 금지")
        return errs

    # ── 직렬화 ──────────────────────────────────────────────────────────────
    def to_vec(self) -> np.ndarray:
        """학습·전송용 평탄화."""
        parts = [
            np.array([self.coord_mode, self.mode_confidence,
                      self.phase, self.phase_progress, self.supervisor],
                     dtype=float),
            self.uncertainty,
        ]
        for w in self.left:
            parts.append(w.to_vec())
        for w in self.right:
            parts.append(w.to_vec())
        for x in self.inter:
            parts.append(x.to_vec())
        if self.T_grasp_frozen is not None:
            parts.append(self.T_grasp_frozen)
        if self.latent is not None:
            parts.append(self.latent)
        return np.concatenate(parts)

    @staticmethod
    def payload_dim(H: int = CHUNK_H, latent_dim: int = 0) -> int:
        return (5 + 6
                + 2 * H * Waypoint.DIM
                + H * InterArm.DIM
                + 9
                + latent_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 6D 회전 표현 (Zhou et al. 2019)
# ─────────────────────────────────────────────────────────────────────────────


def R_to_6d(R: np.ndarray) -> np.ndarray:
    """회전행렬 → 6D. 앞 두 열을 그대로 쓴다."""
    return np.concatenate([R[:, 0], R[:, 1]])


def sixd_to_R(v: np.ndarray) -> np.ndarray:
    """6D → 회전행렬. Gram-Schmidt 로 직교화."""
    a1, a2 = np.asarray(v[:3], dtype=float), np.asarray(v[3:6], dtype=float)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


if __name__ == "__main__":
    H = CHUNK_H
    print(f"인터페이스 스키마 v{'.'.join(map(str, SCHEMA_VERSION))}")
    print(f"  chunk H          : {H}")
    print(f"  waypoint 차원     : {Waypoint.DIM}  (팔당 {H*Waypoint.DIM})")
    print(f"  팔간 항 차원      : {InterArm.DIM}  (총 {H*InterArm.DIM})")
    print(f"  payload (latent 0): {InterfaceMessage.payload_dim(H)} float")
    print(f"  payload (latent 64): {InterfaceMessage.payload_dim(H, 64)} float")
    print(f"  fp16 바이트       : {InterfaceMessage.payload_dim(H, 64)*2} B")
    print(f"  5Hz 대역폭        : {InterfaceMessage.payload_dim(H, 64)*2*5/1024:.1f} KB/s")
    print("\n  → 60.6ms 비전 인코더 대비 전송 비용은 무시 가능.")
    print("    현재의 최소 인터페이스를 유지할 성능상의 논거는 없다.")

    # 왕복 검증
    R = sixd_to_R(np.array([1., 0, 0, 0, 1, 0]))
    assert np.allclose(R, np.eye(3)), "6D 왕복 실패"
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation
    for _ in range(100):
        R0 = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        assert np.allclose(sixd_to_R(R_to_6d(R0)), R0, atol=1e-10)
    print("\n  ✅ 6D 회전 표현 왕복 검증 100/100")
