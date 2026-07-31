#!/usr/bin/env python3
"""
galbot-mcap2lerobot 포크 — 손목 F/T wrench 와 관절별 effort 를 데이터셋에 넣는다

이 파일이 존재하는 이유 (단 하나)
────────────────────────────────
공식 컨버터의 `STATE_TOPICS` 는 `['singorix/wbcs/sensor']` **하나뿐**이고,
출력 parquet 스키마에 **F/T 채널이 0개**다. 소스 메시지에 있는 effort 조차
버린다. 그런데:

  1. F/T 는 이 프로젝트의 유일하게 방어 가능한 승부처다. 벤더 GraspVLA 는
     힘을 쓰지 않고, 자체 실패 분석이 힘 무지를 직접 지목했다
     ("21% of failures involve objects with smooth surfaces slipping
     during grasping"). 동일한 `(목표포즈, q)` 가 자유공간·가벼운 접촉·
     끼인 접촉 전부일 수 있고, 그 셋을 구분하는 유일한 입력이 wrench 다.
  2. **기록된 에피소드에 소급 추가가 불가능하다.** 변환 시점에 없는 채널은
     영원히 없다. 따라서 이 코드는 **에피소드 1개를 수집하기 전에** 끝나야
     하는 blocking 항목이다 (docs/PLAN.md F1).
  3. 공개 G1 데이터(RoboCOIN, ~2,974 에피소드)에도 F/T 채널이 없다.
     이 신호는 여기서만 얻는다.

설계 원칙 — RAW 만 기록한다
──────────────────────────
재기록이 불가능하므로 **어떤 보정도 채널을 덮어쓰지 않는다.** tare(영점),
gravity/tool-load 보상, 좌표 변환은 전부 meta JSON 에 파라미터로만 적고
parquet 에는 센서 원값을 쓴다. 보정은 나중에 다시 할 수 있지만, 보정된
값에서 원값을 되돌리는 것은 못 한다.

공식과 호환을 유지하는 것
────────────────────────
  - 23차원 관절 벡터 레이아웃 (leg5 / head2 / L팔7 / L그리퍼1 / R팔7 / R그리퍼1)
  - `action[t] = state[t+1]` (23차원 전체)
  - 그리퍼 원값 ÷ 1000 → 미터 단위 연속 폭
  - LeRobot v2.1 컬럼 (`observation.state`, `action`, `timestamp`,
    `frame_index`, `episode_index`, `index`, `task_index`)

추가하는 것
──────────
  observation.wrench.left / right      (6,)  [Fx,Fy,Fz,Tx,Ty,Tz] N, N·m
  observation.wrench.dt.left / right   ()    앵커 시각과의 시간차 [s] (staleness)
  observation.effort                   (23,) 관절별 effort [N·m]
  observation.current                  (23,) 관절별 전류 [A]
  observation.psi.left / right         ()    arm angle [rad] — FK 로 계산, 라벨링 0
  observation.T_rel                    (9,)  손간 상대 변환 dp(3)+6D회전(6)

⚠️ 확인 필요 — psi 와 T_rel 은 **URDF 로 유도한 우리 FK** 로만 계산된다
   (robot/assets/g1_joints_raw.json → G1Arm.fk / arm_angle). URDF 가 실기체와
   다르면 그 오차가 데이터셋에 **영구히 구워지고, F/T 와 똑같이 소급 수정이
   불가능하다.** 현재 우리 FK 는 자기일관성만 검증됐고 SDK FK 와 대조한 적이
   없다 (test_kinematics.py T1~T7 전부 URDF 자기일관성).
   → 대량 변환 전에 `make probe-live` 로 SDK FK 와 1회 대조할 것.
     tip 프레임 판별은 g1_kinematics.identify_tip_frame() 이 해준다.

psi 와 T_rel 을 여기서 굽는 이유: 둘 다 기록된 관절각에서 **FK 한 번**으로
공짜로 나오고 (샘플당 수십 us), 학습 루프에서 매번 계산하면 그 비용이
epoch 수만큼 곱해진다. 특히 T_rel 은 **두 절대 포즈의 뺄셈이 아니라 FK 로
직접** 계산한다 — L2 손실은 무해한 공통모드 오차와 물체를 부수는 차동 오차를
동일하게 벌하므로, 상대 항은 first-class 채널이어야 한다.

사용
────
    # 0. 먼저 MCAP 안에 뭐가 있는지 본다 (실제 토픽명을 모르므로 필수)
    python tools/mcap2lerobot_fork.py --list-topics --mcap ep0.mcap --peek

    # 1. 파일럿 1 에피소드 변환 + 강한 검증 (F/T 가 죽어 있으면 종료코드 1)
    python tools/mcap2lerobot_fork.py --mcap ep0.mcap --out ds/ --strict

    # 2. 토픽 자동 탐색이 실패하면 명시적으로 지정
    python tools/mcap2lerobot_fork.py --mcap ep0.mcap --out ds/ \\
        --wrench-topic-left  singorix/sensor/left_ft \\
        --wrench-topic-right singorix/sensor/right_ft

    # 3. MCAP 없이 스키마·동기화·검증 로직만 검증 (Mac 에서 가능)
    python tools/mcap2lerobot_fork.py --dry-run

⚠️ 확인 필요 (회사 Linux 박스에서 실제 MCAP 으로 닫아야 하는 것)
   - 손목 F/T 토픽의 실제 이름과 메시지 스키마
   - wrench 가 어느 프레임(센서/툴/베이스)에서, 어느 부호 규약으로 오는지
   - effort/current 가 wbcs/sensor 안에 있는지 별도 토픽인지
   - 그리퍼 원값 ÷1000 이 F/T 포함 스트림에서도 유효한지
   - 텔레옵 리그가 bilateral 인지 (unilateral 이면 wrench 에 학습 가능한
     구조가 없다 — 이 파일 전체의 전제조건. docs/PLAN.md C6)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "robot"))
from g1_kinematics import G1Arm, se3_inv            # noqa: E402
from interface import R_to_6d, sixd_to_R            # noqa: E402


FORK_VERSION = "0.1.0"
UPSTREAM = "GalaxyGeneralRobotics/galbot-mcap2lerobot (MIT)"


# ─────────────────────────────────────────────────────────────────────────────
# 관절 레이아웃 — 공식 joint_constants.py 와 동일해야 한다.
# 여기가 어긋나면 psi 도 T_rel 도 전부 조용히 틀린다.
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAMES: list[str] = (
    [f"leg_joint{i}" for i in range(1, 6)]          # [0:5]  다리 2 + 허리 3
    + [f"head_joint{i}" for i in range(1, 3)]       # [5:7]  pan, tilt
    + [f"left_arm_joint{i}" for i in range(1, 8)]   # [7:14]
    + ["left_gripper_joint1"]                       # [14]
    + [f"right_arm_joint{i}" for i in range(1, 8)]  # [15:22]
    + ["right_gripper_joint1"]                      # [22]
)
JOINT_DIM = len(JOINT_NAMES)
assert JOINT_DIM == 23

IDX = {
    "leg": slice(0, 5),
    "head": slice(5, 7),
    "left_arm": slice(7, 14),
    "left_gripper": 14,
    "right_arm": slice(15, 22),
    "right_gripper": 22,
}

# 공식 컨버터가 그리퍼 원값을 1000 으로 나눠 미터 단위 연속 폭을 만든다.
# ⚠️ 확인 필요 — F/T 를 포함한 스트림에서도 같은 스케일인지.
GRIPPER_RAW_SCALE = 1e-3
GRIPPER_RANGE_M = (0.0, 0.15)     # interface.Waypoint.gripper_width 검증 범위와 동일

WRENCH_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
WRENCH_DIM = 6

FPS_DEFAULT = 30.0                 # 공식 파이프라인의 30Hz 위치 타깃


# ─────────────────────────────────────────────────────────────────────────────
# 출력 컬럼명
#
# 공식 4개는 그대로 두고 뒤에 덧붙인다. 기존 LeRobot 로더가 모르는 컬럼을
# 무시하도록 되어 있으면 이 데이터셋은 공식 파이프라인에서도 그냥 읽힌다.
# ─────────────────────────────────────────────────────────────────────────────

OUT = {
    "state":       "observation.state",
    "action":      "action",
    "wrench_l":    "observation.wrench.left",
    "wrench_r":    "observation.wrench.right",
    "wrench_dt_l": "observation.wrench.dt.left",
    "wrench_dt_r": "observation.wrench.dt.right",
    "effort":      "observation.effort",
    "current":     "observation.current",
    "psi_l":       "observation.psi.left",
    "psi_r":       "observation.psi.right",
    "T_rel":       "observation.T_rel",
    "timestamp":   "timestamp",
    "frame_index": "frame_index",
    "episode_index": "episode_index",
    "index":       "index",
    "task_index":  "task_index",
}


# ─────────────────────────────────────────────────────────────────────────────
# 기대 wrench 크기 — 검증 리포트의 판정 기준
#
# 손목 6축 F/T 의 정상 동작 범위. 이 숫자를 벗어나면 센서가 아니라 배선이나
# 단위를 의심해야 한다.
#   정지 상태 residual   ~0.5 N   (센서 노이즈 + 툴 중량의 정적 성분)
#   운동 중 residual     ~2 N     (관성 + 중력 성분의 회전)
#   실제 접촉        수 N ~ 수십 N
# ─────────────────────────────────────────────────────────────────────────────

WRENCH_STATIC_N = 0.5
WRENCH_MOTION_N = 2.0
WRENCH_CONTACT_N = 3.0       # 이 이상이면 접촉으로 센다
WRENCH_UNIT_SUSPECT_N = 500.0    # 이 이상은 N 이 아니라 raw count / mN 의심


# ─────────────────────────────────────────────────────────────────────────────
# 토픽 탐색
#
# ⚠️ 확인 필요 — 아래 후보는 전부 추정이다. 공식 컨버터에서 확인된 것은
#    상태 토픽 'singorix/wbcs/sensor' 하나뿐이고, F/T 관련 grep 은 0건이었다.
#    그래서 "정답 하나"가 아니라 "후보 다수 + 점수 + 실패 시 명확한 안내" 로
#    설계했다. --list-topics 로 실제 이름을 확인하면 --wrench-topic-* 로
#    덮어쓰거나 아래 리스트에 추가하면 된다.
# ─────────────────────────────────────────────────────────────────────────────

STATE_TOPICS_OFFICIAL = ["singorix/wbcs/sensor"]       # 공식 (검증됨)

STATE_TOPIC_CANDIDATES = STATE_TOPICS_OFFICIAL + [
    "singorix/wbcs/state",
    "singorix/joint/sensor",
    "/joint_states",
    "joint_states",
]

WRENCH_TOPIC_CANDIDATES = {
    "left": [
        "singorix/sensor/left_ft",
        "singorix/wbcs/left_wrench",
        "singorix/left_arm/wrench",
        "/left_arm/wrench",
        "/ft_sensor/left",
        "/left_wrist_ft",
    ],
    "right": [
        "singorix/sensor/right_ft",
        "singorix/wbcs/right_wrench",
        "singorix/right_arm/wrench",
        "/right_arm/wrench",
        "/ft_sensor/right",
        "/right_wrist_ft",
    ],
}

EFFORT_TOPIC_CANDIDATES = [
    # 기본 가정: effort/current 는 상태 토픽 메시지 안에 이미 있다
    # (JointState 7필드: position/velocity/acceleration/effort/current + ns 타임스탬프).
    # 별도 토픽이면 아래에서 찾는다.
    "singorix/wbcs/effort",
    "singorix/joint/effort",
]

_RE_CAMERA = re.compile(r"(image|camera|compressed|depth|infra|rgb|color|/cam|point_?cloud)", re.I)
_RE_WRENCH = re.compile(r"(wrench|force[_/-]?torque|forcetorque|ft[_/-]?sensor|(^|[_/])ft($|[_/])|f[_/-]t)", re.I)
_RE_FORCE_WEAK = re.compile(r"(force|torque|load[_/-]?cell)", re.I)
_RE_JOINT = re.compile(r"(joint|wbcs/sensor|servo|motor|actuator)", re.I)
_RE_LEFT = re.compile(r"(^|[_/])(left|l)($|[_/])", re.I)
_RE_RIGHT = re.compile(r"(^|[_/])(right|r)($|[_/])", re.I)
_RE_SCHEMA_WRENCH = re.compile(r"(wrench|force_?torque|ftsensor)", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# 자료구조
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TopicInfo:
    """MCAP 안의 토픽 하나에 대한 요약. --list-topics 의 출력 단위."""
    name: str
    schema: str = ""
    encoding: str = ""
    count: int = 0
    t_first_ns: int = 0
    t_last_ns: int = 0
    fields: Optional[list[str]] = None     # --peek 시 첫 메시지의 필드명

    @property
    def hz(self) -> float:
        dt = (self.t_last_ns - self.t_first_ns) / 1e9
        return (self.count - 1) / dt if (dt > 0 and self.count > 1) else 0.0


@dataclass
class Stream:
    """
    한 토픽에서 뽑은 (t_ns, 값) 시계열.

    `clock` 을 필드로 들고 다니는 이유: 모든 센서가 **하나의 monotonic 시계**
    위에 있어야 한다. 관절은 페이로드의 device timestamp 를 쓰고 F/T 는 MCAP
    log_time(호스트 수신 시각)을 쓰면, 둘 사이에 transport latency 만큼의
    **체계적** 오프셋이 생긴다. 손이 1 m/s 로 움직일 때 5ms 오프셋은 5mm 다 —
    접촉 시작 프레임이 통째로 밀린다. 그래서 스트림마다 시계 출처를 기록하고
    섞이면 검증 리포트가 경고한다.
    """
    name: str
    t_ns: np.ndarray            # (N,) int64, 단조 증가로 정렬
    data: np.ndarray            # (N, D) float64
    source_topic: str = ""
    clock: str = "unknown"      # 'payload' | 'log_time' | 'synthetic'

    def __post_init__(self):
        self.t_ns = np.asarray(self.t_ns, dtype=np.int64).reshape(-1)
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        if len(self.t_ns) != len(self.data):
            raise ValueError(f"{self.name}: 타임스탬프 {len(self.t_ns)} != 샘플 {len(self.data)}")
        order = np.argsort(self.t_ns, kind="stable")
        if not np.all(order == np.arange(len(order))):
            self.t_ns = self.t_ns[order]
            self.data = self.data[order]

    @property
    def hz(self) -> float:
        dt = (self.t_ns[-1] - self.t_ns[0]) / 1e9 if len(self.t_ns) > 1 else 0.0
        return (len(self.t_ns) - 1) / dt if dt > 0 else 0.0


@dataclass
class RawStreams:
    """MCAP(또는 합성) 에서 뽑아낸 원시 스트림 묶음. 이 아래는 경로가 하나다."""
    joint_pos: Stream
    joint_effort: Optional[Stream] = None
    joint_current: Optional[Stream] = None
    wrench_left: Optional[Stream] = None
    wrench_right: Optional[Stream] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Frames:
    """앵커 시계에 정렬된 최종 프레임 배열. 이것이 parquet 한 에피소드가 된다."""
    t_ns: np.ndarray                    # (T,)
    state: np.ndarray                   # (T,23)
    action: np.ndarray                  # (T,23)
    wrench_l: Optional[np.ndarray] = None       # (T,6)
    wrench_r: Optional[np.ndarray] = None
    wrench_dt_l: Optional[np.ndarray] = None    # (T,)  [s]
    wrench_dt_r: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None         # (T,23)
    current: Optional[np.ndarray] = None
    psi_l: Optional[np.ndarray] = None          # (T,)
    psi_r: Optional[np.ndarray] = None
    T_rel: Optional[np.ndarray] = None          # (T,9)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t_ns)


# ─────────────────────────────────────────────────────────────────────────────
# 메시지 필드 추출 (duck typing)
#
# ⚠️ 확인 필요 — 실제 메시지 스키마를 모른다. 기록은 "MCAP protobuf, ROS 2
#    메시지 형태 호환" 이라는 것까지만 확인됐다. 그래서 하나의 스키마를
#    가정하지 않고 흔한 형태를 전부 시도한다. 실제 스키마를 확인하면
#    이 절만 좁히면 된다.
# ─────────────────────────────────────────────────────────────────────────────


def _get(obj: Any, *names: str) -> Any:
    """dict / ROS2 객체 / protobuf 메시지에서 첫 번째로 존재하는 필드를 꺼낸다."""
    for n in names:
        try:
            if isinstance(obj, dict):
                if n in obj and obj[n] is not None:
                    return obj[n]
                continue
            if hasattr(obj, "HasField"):
                # protobuf: 미설정 서브메시지도 getattr 은 기본값을 돌려주므로
                # HasField 로 걸러야 한다. repeated/scalar 는 ValueError → 통과.
                try:
                    if not obj.HasField(n):
                        continue
                except (ValueError, TypeError):
                    pass
            v = getattr(obj, n, None)
            if v is not None:
                return v
        except Exception:
            continue
    return None


def _field_names(obj: Any) -> list[str]:
    """--peek 용. 메시지의 최상위 필드명을 뽑는다."""
    if isinstance(obj, dict):
        return sorted(obj.keys())
    if hasattr(obj, "DESCRIPTOR"):
        try:
            return [f.name for f in obj.DESCRIPTOR.fields]
        except Exception:
            pass
    if hasattr(obj, "__slots__") and obj.__slots__:
        return [s.lstrip("_") for s in obj.__slots__]
    return sorted(a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a, None)))


def _xyz(o: Any) -> Optional[np.ndarray]:
    """{x,y,z} 객체 또는 길이 3 배열 → (3,)."""
    if o is None:
        return None
    x, y, z = _get(o, "x"), _get(o, "y"), _get(o, "z")
    if x is not None and y is not None and z is not None:
        return np.array([float(x), float(y), float(z)])
    try:
        a = np.asarray(o, dtype=float).ravel()
        if a.size >= 3:
            return a[:3]
    except Exception:
        pass
    return None


def extract_stamp_ns(msg: Any) -> Optional[int]:
    """
    페이로드에 실린 device 타임스탬프를 ns 로. 없으면 None (호출자가 log_time 사용).

    페이로드 시각을 log_time 보다 우선하는 이유: log_time 은 호스트가 메시지를
    **받은** 시각이라 transport jitter 가 그대로 섞인다. 센서가 자기 시각을
    실어 보내면 그게 진실에 가깝다. 단 모든 센서가 같은 시계를 봐야 한다.
    """
    v = _get(msg, "timestamp_ns", "stamp_ns", "t_ns")
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    hdr = _get(msg, "header")
    stamp = _get(hdr, "stamp", "timestamp") if hdr is not None else _get(msg, "stamp")
    if stamp is not None:
        sec = _get(stamp, "sec", "secs", "seconds")
        nsec = _get(stamp, "nanosec", "nsec", "nanos", "nsecs")
        if sec is not None:
            return int(sec) * 1_000_000_000 + int(nsec or 0)
        try:
            return int(float(stamp) * 1e9)
        except Exception:
            pass
    return None


def extract_wrench(msg: Any) -> Optional[np.ndarray]:
    """
    임의 형태의 wrench 메시지 → (6,) [Fx,Fy,Fz,Tx,Ty,Tz].

    지원 형태:
      geometry_msgs/WrenchStamped   msg.wrench.force.{x,y,z} / .torque.{x,y,z}
      geometry_msgs/Wrench          msg.force / msg.torque
      평탄 배열                      msg.data / msg.values (길이 >= 6)
      스칼라 6필드                   msg.fx .. msg.tz
    """
    node = _get(msg, "wrench", "ft", "force_torque")
    for cand in (node, msg):
        if cand is None:
            continue
        f = _xyz(_get(cand, "force", "f", "linear"))
        t = _xyz(_get(cand, "torque", "moment", "t", "angular"))
        if f is not None and t is not None:
            return np.concatenate([f, t])

    for key in ("data", "values", "wrench_data", "value", "raw"):
        v = _get(msg, key)
        if v is not None:
            try:
                a = np.asarray(v, dtype=float).ravel()
                if a.size >= WRENCH_DIM:
                    return a[:WRENCH_DIM]
            except Exception:
                pass

    comps = [
        _get(msg, "fx", "force_x", "Fx"),
        _get(msg, "fy", "force_y", "Fy"),
        _get(msg, "fz", "force_z", "Fz"),
        _get(msg, "tx", "torque_x", "mx", "Tx"),
        _get(msg, "ty", "torque_y", "my", "Ty"),
        _get(msg, "tz", "torque_z", "mz", "Tz"),
    ]
    if all(c is not None for c in comps):
        return np.asarray([float(c) for c in comps])
    return None


def extract_side_wrenches(msg: Any) -> dict[str, np.ndarray]:
    """한 메시지가 양쪽 F/T 를 동시에 실은 경우 (combined 토픽) 분해."""
    out: dict[str, np.ndarray] = {}
    for side, keys in (("left", ("left", "l", "left_wrench", "left_ft")),
                       ("right", ("right", "r", "right_wrench", "right_ft"))):
        sub = _get(msg, *keys)
        if sub is not None:
            w = extract_wrench(sub)
            if w is not None:
                out[side] = w
    return out


def _to_23(values: Any, names: Optional[list[str]]) -> Optional[np.ndarray]:
    """
    임의 길이 배열 + (있으면) 관절명 리스트 → 23차원 벡터.

    이름이 있으면 **반드시 이름으로 매핑한다.** 순서 가정은 조용히 틀리는
    종류의 버그고, 그 결과 psi 와 T_rel 이 전부 오염된다 (팔 관절 7개가
    한 칸 밀리면 FK 는 여전히 그럴듯한 숫자를 낸다).
    """
    if values is None:
        return None
    try:
        a = np.asarray(values, dtype=float).ravel()
    except Exception:
        return None
    if a.size == 0:
        return None

    if names:
        out = np.full(JOINT_DIM, np.nan)
        pos = {n: i for i, n in enumerate(JOINT_NAMES)}
        hit = 0
        for n, v in zip(names, a):
            i = pos.get(str(n))
            if i is not None:
                out[i] = v
                hit += 1
        # **하나라도 맞으면 이름을 믿는다.** 부분 매칭에서 순서 가정으로
        # 후퇴하면 최악이다 — 이름이 맞은 관절까지 엉뚱한 인덱스로 가고,
        # 결과는 완벽히 그럴듯한 숫자다. 이름이 0개 맞을 때만 (= 발행 쪽이
        # 아예 다른 명명 규칙) 순서 가정으로 간다. 채워지지 않은 칸은 NaN 으로
        # 남고 검증 리포트가 그 비율을 보고한다.
        if hit > 0:
            return out
    if a.size >= JOINT_DIM:
        return a[:JOINT_DIM]
    out = np.full(JOINT_DIM, np.nan)
    out[: a.size] = a
    return out


def extract_joint_fields(msg: Any) -> dict[str, np.ndarray]:
    """
    상태 메시지 → {'position','effort','current','velocity'} 중 있는 것만 23차원으로.

    공식 컨버터는 여기서 position 만 꺼내고 effort 를 버린다. 그 한 줄이
    이 포크가 존재하는 이유의 절반이다.
    """
    node = _get(msg, "joint_state", "joint_states", "joints", "state") or msg
    names = _get(node, "name", "names", "joint_names")
    if names is not None:
        try:
            names = [str(x) for x in names]
        except Exception:
            names = None

    out: dict[str, np.ndarray] = {}
    for key, alts in (
        ("position", ("position", "positions", "q", "pos")),
        ("velocity", ("velocity", "velocities", "dq", "vel")),
        ("effort",   ("effort", "efforts", "torque", "torques", "tau")),
        ("current",  ("current", "currents", "motor_current", "amps")),
    ):
        v = _to_23(_get(node, *alts), names)
        if v is not None and np.any(np.isfinite(v)):
            out[key] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MCAP 어댑터
#
# ⚠️ 확인 필요 — mcap 라이브러리 API 는 검증했으나 Galbot 기록의 실제
#    message_encoding (protobuf vs cdr) 은 미확인이다. 둘 다 시도한다.
# ─────────────────────────────────────────────────────────────────────────────

_MCAP_HELP = """
[mcap] MCAP 라이브러리를 불러오지 못했습니다.

  Galbot 기록은 MCAP 컨테이너입니다. 읽으려면 회사 Linux 박스에서:

      pip install mcap mcap-ros2-support mcap-protobuf-support

  Mac 에서는 --dry-run 으로 스키마·동기화·검증 로직만 검증할 수 있습니다
  (실제 MCAP 없이 합성 프레임 배열로 같은 파이프라인을 태웁니다):

      python tools/mcap2lerobot_fork.py --dry-run
"""


def _decoder_factories() -> list:
    """설치된 디코더를 전부 등록. protobuf 기록과 ros2 기록 둘 다 대응."""
    facs = []
    try:
        from mcap_ros2.decoder import DecoderFactory as Ros2Factory
        facs.append(Ros2Factory())
    except ImportError:
        pass
    try:
        from mcap_protobuf.decoder import DecoderFactory as PbFactory
        facs.append(PbFactory())
    except ImportError:
        pass
    return facs


class McapSource:
    """MCAP 리더 래퍼. 실물 스키마에 맞춰 이 클래스만 고치면 나머지는 그대로다."""

    def __init__(self, path: str):
        self.path = path
        # 존재 확인만 한다 (실제 import 는 _reader 에서). Mac 에서 --dry-run 을
        # 쓸 때 이 모듈 자체는 import 되어야 하므로 최상단 import 는 금지다.
        import importlib.util
        if importlib.util.find_spec("mcap") is None:
            sys.exit(_MCAP_HELP)
        if not os.path.exists(path):
            sys.exit(f"[mcap] 파일이 없습니다: {path}")
        self._facs = _decoder_factories()
        if not self._facs:
            print("[mcap] ⚠️ 디코더 플러그인이 없습니다 "
                  "(mcap-ros2-support / mcap-protobuf-support). "
                  "토픽 목록은 나오지만 디코딩은 실패합니다.")

    def _reader(self, f):
        from mcap.reader import make_reader
        return make_reader(f, decoder_factories=self._facs)

    def list_topics(self, peek: bool = False) -> list[TopicInfo]:
        """토픽 전체 덤프. 실제 이름을 모르므로 이것이 첫 단계다."""
        infos: dict[str, TopicInfo] = {}
        with open(self.path, "rb") as f:
            reader = self._reader(f)
            for schema, channel, message in reader.iter_messages():
                ti = infos.get(channel.topic)
                if ti is None:
                    ti = TopicInfo(
                        name=channel.topic,
                        schema=(schema.name if schema else ""),
                        encoding=getattr(channel, "message_encoding", ""),
                        t_first_ns=message.log_time,
                    )
                    infos[channel.topic] = ti
                ti.count += 1
                ti.t_last_ns = message.log_time

        if peek:
            want = set(infos.keys())
            with open(self.path, "rb") as f:
                reader = self._reader(f)
                for schema, channel, message, decoded in reader.iter_decoded_messages():
                    if channel.topic in want:
                        try:
                            infos[channel.topic].fields = _field_names(decoded)
                        except Exception as e:
                            infos[channel.topic].fields = [f"<디코딩 실패: {e}>"]
                        want.discard(channel.topic)
                    if not want:
                        break
        return sorted(infos.values(), key=lambda t: t.name)

    def read(self, topics: list[str], clock: str = "payload"
             ) -> dict[str, list[tuple[int, Any]]]:
        """지정 토픽들을 한 번의 패스로 읽어 (t_ns, decoded) 리스트로."""
        want = set(t for t in topics if t)
        out: dict[str, list[tuple[int, Any]]] = {t: [] for t in want}
        n_payload = 0
        n_logtime = 0
        with open(self.path, "rb") as f:
            reader = self._reader(f)
            for schema, channel, message, decoded in reader.iter_decoded_messages():
                if channel.topic not in want:
                    continue
                t = None
                if clock == "payload":
                    t = extract_stamp_ns(decoded)
                if t is None:
                    t = message.log_time
                    n_logtime += 1
                else:
                    n_payload += 1
                out[channel.topic].append((int(t), decoded))
        self.clock_stats = {"payload": n_payload, "log_time": n_logtime}
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 토픽 자동 탐색
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TopicSelection:
    state: Optional[str] = None
    wrench_left: Optional[str] = None
    wrench_right: Optional[str] = None
    wrench_combined: Optional[str] = None
    effort: Optional[str] = None
    wrench_ambiguous: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def is_wrench_candidate(ti: TopicInfo) -> bool:
    """
    이 토픽이 wrench 일 수 있는가.

    **스키마 타입이 토픽명보다 강한 증거다.** Galbot 토픽명은 불투명하고
    (확인된 것은 'singorix/wbcs/sensor' 뿐 — 'sensor' 라는 이름에서 관절
    상태라는 걸 알 방법이 없다), F/T 토픽이 'singorix/wbcs/aux3' 같은
    이름일 가능성이 충분하다. 반면 `geometry_msgs/WrenchStamped` 는
    이름이 무엇이든 wrench 다. 그래서 이름과 스키마 중 **하나만** 걸려도
    후보로 올리고, 대신 좌/우 판별은 훨씬 보수적으로 한다 (아래).
    """
    if _RE_CAMERA.search(ti.name):
        return False
    if ti.schema and _RE_SCHEMA_WRENCH.search(ti.schema):
        return True
    if _RE_WRENCH.search(ti.name):
        return True
    if ti.name in WRENCH_TOPIC_CANDIDATES["left"] + WRENCH_TOPIC_CANDIDATES["right"]:
        return True
    # 약한 신호 (force/torque/load_cell). 여기서 false positive 는 싸다 —
    # 6축을 못 뽑으면 build_raw_streams 뒤에서 잡히고 사용자에게 보인다.
    # false negative 는 비싸다 — 사용자가 "F/T 없음"으로 오해하고 포기한다.
    if _RE_FORCE_WEAK.search(ti.name):
        return True
    return False


def _assign_side(cands: list[TopicInfo], side: str) -> Optional[TopicInfo]:
    """
    후보 중 그 쪽 팔의 것을 고른다. **애매하면 고르지 않는다.**

    좌/우를 뒤바꾸는 것은 조용하고 복구 불가능한 오류다 — 변환 결과는
    완벽히 그럴듯하고, 학습은 왼손 힘을 오른손에 귀속시키며, 알아채는
    시점은 정책이 이상하게 행동할 때다. 그래서 이름에 명확한 side 토큰이
    없으면 추측 대신 사용자에게 묻는다.
    """
    exact = [t for t in cands if t.name in WRENCH_TOPIC_CANDIDATES[side]]
    if exact:
        return exact[0]
    pat = _RE_LEFT if side == "left" else _RE_RIGHT
    anti = _RE_RIGHT if side == "left" else _RE_LEFT
    hit = [t for t in cands if pat.search(t.name) and not anti.search(t.name)]
    # 후보가 여러 개면 메시지 수 같은 것으로 고르지 않는다. 'left_ft' 와
    # 'left_gripper_force' 중 무엇이 손목 6축인지는 여기서 알 수 없고,
    # 틀린 선택은 조용하다.
    return hit[0] if len(hit) == 1 else None


def discover_topics(infos: list[TopicInfo], override: TopicSelection) -> TopicSelection:
    """
    후보 + 정규식 + 점수로 토픽을 고른다. 명시 지정이 항상 이긴다.

    자동 탐색을 믿지 않는 이유: F/T 토픽명은 공개 문서 어디에도 없다.
    그래서 골랐다는 사실을 반드시 출력하고, 못 고르면 조용히 넘어가지 않고
    무엇을 봤는지 전부 보여준다.
    """
    sel = TopicSelection(
        state=override.state, wrench_left=override.wrench_left,
        wrench_right=override.wrench_right, effort=override.effort,
    )
    by_name = {t.name: t for t in infos}

    # ── 상태 토픽 ────────────────────────────────────────────────────────────
    if sel.state is None:
        for cand in STATE_TOPIC_CANDIDATES:
            if cand in by_name:
                sel.state = cand
                sel.notes.append(f"상태 토픽: 후보 목록에서 '{cand}' 매칭")
                break
    if sel.state is None:
        best = [t for t in infos if _RE_JOINT.search(t.name) and not _RE_CAMERA.search(t.name)]
        best.sort(key=lambda t: -t.count)
        if best:
            sel.state = best[0].name
            sel.notes.append(f"상태 토픽: 정규식 추정 '{best[0].name}' "
                             f"({best[0].count}개, {best[0].hz:.1f}Hz)")

    # ── wrench 좌/우 ─────────────────────────────────────────────────────────
    cands = [t for t in infos if is_wrench_candidate(t)]
    sel.wrench_ambiguous = [t.name for t in cands]
    for side in ("left", "right"):
        attr = f"wrench_{side}"
        if getattr(sel, attr) is not None:
            continue
        hit = _assign_side(cands, side)
        if hit is not None:
            setattr(sel, attr, hit.name)
            sel.notes.append(f"{side} wrench: '{hit.name}' "
                             f"(스키마 {hit.schema or '?'}, {hit.hz:.1f}Hz)")

    # ── 양쪽이 한 토픽에 실린 경우 ────────────────────────────────────────────
    # 후보가 정확히 1개이고 side 토큰이 없으면 combined 일 수 있다. 2개 이상이면
    # 어느 쪽이 왼손인지 알 방법이 없으므로 추측하지 않고 호출자가 에러를 낸다.
    if sel.wrench_left is None and sel.wrench_right is None and len(cands) == 1:
        sel.wrench_combined = cands[0].name
        sel.notes.append(f"combined wrench 후보: '{cands[0].name}' "
                         "— 한 메시지에 좌/우가 같이 들어있는지 확인하십시오")

    # ── effort ───────────────────────────────────────────────────────────────
    if sel.effort is None:
        for cand in EFFORT_TOPIC_CANDIDATES:
            if cand in by_name:
                sel.effort = cand
                sel.notes.append(f"effort 별도 토픽: '{cand}'")
                break
    if sel.effort is None:
        sel.notes.append("effort: 별도 토픽 없음 → 상태 토픽 메시지 안에서 찾습니다")
    return sel


def _no_wrench_error(infos: list[TopicInfo], ambiguous: list[str]) -> str:
    """
    두 가지 다른 실패를 구분해서 안내한다.

      (A) wrench 스키마 토픽을 찾았는데 **좌/우를 못 가리겠다**
          → 사용자가 한 줄만 추가하면 끝나는 상황. 가장 흔할 것으로 본다
            (Galbot 토픽명이 불투명하므로).
      (B) wrench 로 보이는 토픽이 아예 없다
          → 기록 설정이나 하드웨어 문제. 훨씬 심각하다.
    """
    lines = ["", "=" * 72]
    if ambiguous:
        lines += [
            "❌ wrench 토픽 후보는 찾았으나 **어느 것이 어느 팔인지 확정할 수 없습니다.**",
            "=" * 72,
            "",
            "토픽명에 left/right 토큰이 없거나, 한쪽에 후보가 둘 이상이라",
            "자동 배정을 거부했습니다. 좌우가 뒤바뀐 데이터셋은 조용히 그럴듯하고",
            "복구가 불가능합니다 — 추측하느니 묻는 쪽이 낫습니다.",
            "",
            "후보:",
        ]
        for n in ambiguous:
            ti = next((t for t in infos if t.name == n), None)
            lines.append(f"   {n:46s} {(ti.schema if ti else '')[:28]:30s} "
                         f"{(ti.count if ti else 0):7d}개 {(ti.hz if ti else 0):7.1f}Hz")
        lines += [
            "",
            "어느 쪽이 어느 팔인지 확인한 뒤 (팔 하나를 손으로 눌러보면 즉시 압니다):",
            "  --wrench-topic-left <이름> --wrench-topic-right <이름>",
        ]
    else:
        lines += [
            "❌ 손목 F/T(wrench) 토픽을 찾지 못했습니다.",
            "=" * 72,
            "",
            "이것은 무시하고 넘어갈 수 있는 경고가 아닙니다. F/T 없이 변환하면",
            "그 에피소드에는 힘이 **영원히** 없습니다 (소급 추가 불가).",
            "",
            "다음 중 하나입니다:",
            "  (a) 토픽명도 스키마도 후보와 다르다     → 아래 목록에서 찾아 지정",
            "  (b) 기록 쪽에서 F/T 토픽을 안 담고 있다 → 기록 설정부터 고칠 것",
            "  (c) 센서가 연결/전원 미인가 상태다      → 하드웨어 확인",
            "",
            "MCAP 안의 토픽:",
        ]
        for t in infos:
            mark = "  ← 카메라" if _RE_CAMERA.search(t.name) else ""
            lines.append(f"   {t.name:46s} {t.schema[:28]:30s} "
                         f"{t.count:7d}개 {t.hz:7.1f}Hz{mark}")
        lines += ["", "찾았으면:  --wrench-topic-left <이름> --wrench-topic-right <이름>"]
    lines += ["", "F/T 없이 그래도 변환하려면 --allow-no-wrench (권장하지 않음).", "=" * 72]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MCAP → RawStreams
# ─────────────────────────────────────────────────────────────────────────────


def build_raw_streams(src: McapSource, sel: TopicSelection, clock: str) -> RawStreams:
    """선택된 토픽들을 읽어 스트림 묶음으로. 여기까지가 MCAP 의존 구간이다."""
    topics = [t for t in (sel.state, sel.wrench_left, sel.wrench_right,
                          sel.wrench_combined, sel.effort) if t]
    raw = src.read(topics, clock=clock)
    clock_stats = getattr(src, "clock_stats", {})
    used_clock = "payload" if clock_stats.get("payload", 0) > clock_stats.get("log_time", 0) \
        else "log_time"

    # ── 관절: position / effort / current 를 한 번에 ──────────────────────────
    msgs = raw.get(sel.state or "", [])
    if not msgs:
        sys.exit(f"[state] 상태 토픽 '{sel.state}' 에서 메시지를 하나도 읽지 못했습니다.")
    t_list, pos, eff, cur = [], [], [], []
    for t, m in msgs:
        fields = extract_joint_fields(m)
        p = fields.get("position")
        if p is None:
            continue
        t_list.append(t)
        pos.append(p)
        eff.append(fields.get("effort", np.full(JOINT_DIM, np.nan)))
        cur.append(fields.get("current", np.full(JOINT_DIM, np.nan)))
    if not t_list:
        sys.exit("[state] 상태 메시지에서 position 필드를 찾지 못했습니다. "
                 "--list-topics --peek 로 필드명을 확인한 뒤 extract_joint_fields() 를 "
                 "고치십시오.")

    t_arr = np.asarray(t_list, dtype=np.int64)
    joint_pos = Stream("joint_pos", t_arr, np.asarray(pos), sel.state or "", used_clock)

    eff_arr = np.asarray(eff)
    cur_arr = np.asarray(cur)
    joint_effort = (Stream("joint_effort", t_arr, eff_arr, sel.state or "", used_clock)
                    if np.any(np.isfinite(eff_arr)) else None)
    joint_current = (Stream("joint_current", t_arr, cur_arr, sel.state or "", used_clock)
                     if np.any(np.isfinite(cur_arr)) else None)

    # 별도 effort 토픽이 있으면 그쪽을 우선 (자기 시계를 갖는다)
    if sel.effort and raw.get(sel.effort):
        te, ve = [], []
        for t, m in raw[sel.effort]:
            f = extract_joint_fields(m)
            v = f.get("effort", f.get("position"))
            if v is not None:
                te.append(t)
                ve.append(v)
        if te:
            joint_effort = Stream("joint_effort", np.asarray(te, dtype=np.int64),
                                  np.asarray(ve), sel.effort, used_clock)

    # ── wrench ───────────────────────────────────────────────────────────────
    def _wrench_stream(topic: Optional[str], side: str) -> Optional[Stream]:
        if not topic or not raw.get(topic):
            return None
        tw, vw = [], []
        for t, m in raw[topic]:
            w = extract_wrench(m)
            if w is None:
                sides = extract_side_wrenches(m)
                w = sides.get(side)
            if w is not None:
                tw.append(t)
                vw.append(w)
        if not tw:
            return None
        return Stream(f"wrench_{side}", np.asarray(tw, dtype=np.int64),
                      np.asarray(vw), topic, used_clock)

    wl = _wrench_stream(sel.wrench_left, "left")
    wr = _wrench_stream(sel.wrench_right, "right")
    if sel.wrench_combined:
        wl = wl or _wrench_stream(sel.wrench_combined, "left")
        wr = wr or _wrench_stream(sel.wrench_combined, "right")

    return RawStreams(joint_pos, joint_effort, joint_current, wl, wr,
                      meta={"clock": used_clock, "clock_stats": clock_stats,
                            "topics": asdict(sel)})


# ─────────────────────────────────────────────────────────────────────────────
# 시간 동기화
#
# 왜 이게 별도 절인가
# ──────────────────
# F/T 는 관절 상태와 다른 rate 로 온다 (F/T 는 보통 수백 Hz, 관절 30~100Hz).
# 나이브하게 인덱스로 붙이면 접촉 시작 프레임이 통째로 밀린다. 규칙:
#
#   1. **하나의 monotonic 시계.** 스트림마다 시계 출처(payload / log_time)를
#      기록하고 섞이면 경고한다. 섞이면 오프셋이 랜덤이 아니라 체계적이라
#      학습이 그 편향을 그대로 흡수한다.
#   2. **기본은 nearest, interpolate 아님.** wrench 는 접촉 순간에 계단처럼
#      뛴다. 선형 보간은 그 계단을 프레임 하나에 걸쳐 뭉갠다 — 바로 배우고
#      싶은 신호를 지우는 것이다. --wrench-interp linear 는 제공하되 기본이
#      아니다.
#   3. **시간차를 채널로 남긴다.** observation.wrench.dt 로 프레임마다
#      staleness 를 기록해서, 학습 쪽이 오래된 샘플을 마스킹할 수 있게 한다.
# ─────────────────────────────────────────────────────────────────────────────


def resample_nearest(t_anchor: np.ndarray, src: Stream) -> tuple[np.ndarray, np.ndarray]:
    """최근접 샘플로 정렬. 반환 (값 (T,D), 시간차 (T,) [s])."""
    ts = src.t_ns
    if len(ts) == 1:
        nn = np.zeros(len(t_anchor), dtype=int)
    else:
        hi = np.clip(np.searchsorted(ts, t_anchor), 1, len(ts) - 1)
        lo = hi - 1
        take_lo = np.abs(t_anchor - ts[lo]) <= np.abs(ts[hi] - t_anchor)
        nn = np.where(take_lo, lo, hi)
    dt = np.abs(t_anchor - ts[nn]).astype(np.float64) / 1e9
    return src.data[nn], dt


def resample_linear(t_anchor: np.ndarray, src: Stream) -> tuple[np.ndarray, np.ndarray]:
    """
    선형 보간. 접촉 계단을 뭉개므로 기본이 아니다.

    ns 타임스탬프를 그대로 float64 로 바꾸면 안 된다 — 1e18 ns 는 float64
    가수(약 9e15)를 넘어 ~128ns 단위로 양자화된다. 공통 원점을 빼고 쓴다.
    """
    _, dt = resample_nearest(t_anchor, src)
    t0 = min(int(t_anchor[0]), int(src.t_ns[0]))
    xa = (t_anchor - t0).astype(np.float64)
    xs = (src.t_ns - t0).astype(np.float64)
    out = np.empty((len(t_anchor), src.data.shape[1]))
    for d in range(src.data.shape[1]):
        out[:, d] = np.interp(xa, xs, src.data[:, d])
    return out, dt


def align(t_anchor: np.ndarray, src: Optional[Stream], mode: str = "nearest"
          ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if src is None or len(src.t_ns) == 0:
        return None, None
    if mode == "linear":
        return resample_linear(t_anchor, src)
    return resample_nearest(t_anchor, src)


# ─────────────────────────────────────────────────────────────────────────────
# 파생 채널 — psi 와 T_rel
# ─────────────────────────────────────────────────────────────────────────────


class Derived:
    """
    기록된 관절각에서 FK 로 psi 와 T_rel 을 굽는다. 사람 라벨링 0.

    ⚠️ 확인 필요 — 여기서 쓰는 FK 는 **URDF 자기일관성만 검증**됐다. SDK FK 와
    대조한 적이 없고, URDF 가 실기체와 다르면 그 오차가 데이터셋에 영구히
    구워진다 (F/T 와 동일하게 소급 수정 불가). 대량 변환 전 1회 대조할 것.

    왜 학습 루프가 아니라 변환 시점인가
    ──────────────────────────────────
    비용이 샘플당 수십 us 라 한 번 계산하면 끝이지만, 학습 루프에 두면
    epoch 수만큼 곱해진다. 게다가 psi 는 **인터페이스 필드**라서 (7-DoF 팔에
    6-DoF 포즈는 underdetermined — psi 스윕 시 EE 를 0.16um 안에 고정한 채
    팔꿈치가 263mm 이동) 데이터셋에 없으면 인터페이스 v0.1 을 학습시킬 수
    없다.

    T_rel 은 **두 절대 포즈의 뺄셈이 아니라 FK 로 직접** 만든다. 두 팔 FK 가
    모두 torso_base_link 기준이므로 T_rel = T_left^-1 · T_right 는 정확하고,
    다리/허리/베이스 자유도에 불변이다.
    """

    def __init__(self):
        self.arm_l = G1Arm("left")
        self.arm_r = G1Arm("right")

    def compute(self, state: np.ndarray) -> dict[str, np.ndarray]:
        T = len(state)
        psi_l = np.full(T, np.nan)
        psi_r = np.full(T, np.nan)
        t_rel = np.full((T, 9), np.nan)

        qL_all = state[:, IDX["left_arm"]]
        qR_all = state[:, IDX["right_arm"]]
        for i in range(T):
            qL, qR = qL_all[i], qR_all[i]
            if np.all(np.isfinite(qL)):
                psi_l[i] = self.arm_l.arm_angle(qL)
            if np.all(np.isfinite(qR)):
                psi_r[i] = self.arm_r.arm_angle(qR)
            if np.all(np.isfinite(qL)) and np.all(np.isfinite(qR)):
                Trel = se3_inv(self.arm_l.fk(qL)) @ self.arm_r.fk(qR)
                t_rel[i] = np.concatenate([Trel[:3, 3], R_to_6d(Trel[:3, :3])])
        return {"psi_l": psi_l, "psi_r": psi_r, "T_rel": t_rel}


# ─────────────────────────────────────────────────────────────────────────────
# RawStreams → Frames
# ─────────────────────────────────────────────────────────────────────────────


def build_frames(raw: RawStreams, *, wrench_interp: str = "nearest",
                 apply_gripper_scale: bool = True,
                 derive: bool = True, limit: Optional[int] = None,
                 episode_index: int = 0) -> Frames:
    """
    앵커(관절 상태) 시계에 모든 것을 정렬해 최종 프레임 배열을 만든다.

    앵커를 관절 상태로 잡는 이유: state/action 이 관절 공간이므로 앵커가
    관절이면 그 두 채널에 리샘플링 오차가 0이다. 공식 컨버터는 head-left
    카메라를 마스터 클록으로 쓰는데, 영상까지 합칠 때는 그쪽에 맞춰야 한다.
    ⚠️ 확인 필요 — 영상 합성 단계는 이 포크의 범위 밖이다 (공식 경로 유지).
    """
    t = raw.joint_pos.t_ns.copy()
    state = raw.joint_pos.data.copy()
    if state.shape[1] != JOINT_DIM:
        raise ValueError(f"상태 차원 {state.shape[1]} != {JOINT_DIM}")
    if limit:
        t, state = t[:limit], state[:limit]

    if apply_gripper_scale:
        # 공식 컨버터 호환: 그리퍼 원값 ÷1000 → 미터
        for gi in (IDX["left_gripper"], IDX["right_gripper"]):
            state[:, gi] = state[:, gi] * GRIPPER_RAW_SCALE

    # action[t] = state[t+1] — 공식 동작. 마지막 프레임은 반복한다.
    # ⚠️ 확인 필요 — 공식이 마지막 프레임을 버리는지 반복하는지. 반복은
    #    "정지 명령"이라 학습에 안전한 쪽이다.
    action = np.vstack([state[1:], state[-1:]])

    wl, dtl = align(t, raw.wrench_left, wrench_interp)
    wr, dtr = align(t, raw.wrench_right, wrench_interp)
    eff, _ = align(t, raw.joint_effort, "nearest")
    cur, _ = align(t, raw.joint_current, "nearest")

    fr = Frames(t_ns=t, state=state, action=action,
                wrench_l=wl, wrench_r=wr, wrench_dt_l=dtl, wrench_dt_r=dtr,
                effort=eff, current=cur,
                meta=dict(raw.meta))
    fr.meta["episode_index"] = episode_index
    fr.meta["wrench_interp"] = wrench_interp
    fr.meta["gripper_scale"] = GRIPPER_RAW_SCALE if apply_gripper_scale else 1.0

    if derive:
        t0 = time.perf_counter()
        d = Derived().compute(state)
        fr.psi_l, fr.psi_r, fr.T_rel = d["psi_l"], d["psi_r"], d["T_rel"]
        fr.meta["derive_us_per_frame"] = (time.perf_counter() - t0) / max(1, len(t)) * 1e6
    return fr


# ─────────────────────────────────────────────────────────────────────────────
# 검증 리포트
#
# 변환은 성공했는데 F/T 가 전부 0인 상황이 최악이다 — 파이프라인은 초록불이고
# 데이터는 죽어 있다. 그래서 매 변환마다 자동으로 돌리고, --strict 면 실패를
# 종료코드로 올린다. 200 에피소드를 수집하기 전에 1 에피소드로 이걸 통과시켜야
# 한다.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WrenchCheck:
    side: str
    present: bool
    n: int = 0
    hz: float = 0.0
    all_zero: bool = False
    constant: bool = False
    force_p50: float = 0.0
    force_p95: float = 0.0
    force_max: float = 0.0
    torque_p50: float = 0.0
    torque_p95: float = 0.0
    torque_max: float = 0.0
    contact_frac: float = 0.0
    dt_p50_ms: float = 0.0
    dt_p95_ms: float = 0.0
    dt_max_ms: float = 0.0
    over_tol_frac: float = 0.0
    verdict: str = ""


@dataclass
class ValidationReport:
    n_frames: int
    duration_s: float
    anchor_hz: float
    anchor_jitter_ms: float
    anchor_nonmonotonic: int
    clock: str
    clock_mixed: bool
    wrench: list[WrenchCheck] = field(default_factory=list)
    state_nan_frac: float = 0.0
    has_effort: bool = False
    has_current: bool = False
    effort_nan_frac: float = 1.0
    psi_nan_frac_left: float = 1.0
    psi_nan_frac_right: float = 1.0
    psi_range_left_deg: float = 0.0
    psi_range_right_deg: float = 0.0
    psi_wrap_left: int = 0
    psi_wrap_right: int = 0
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.0
    flags: list[str] = field(default_factory=list)
    ok: bool = True


def _pct(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def _check_wrench(side: str, w: Optional[np.ndarray], dt: Optional[np.ndarray],
                  src: Optional[Stream], tol_ms: float, flags: list[str]) -> WrenchCheck:
    if w is None:
        flags.append(f"wrench_missing_{side}")
        return WrenchCheck(side=side, present=False,
                           verdict="❌ 채널 없음 — 이 에피소드에 힘은 영원히 없습니다")

    c = WrenchCheck(side=side, present=True, n=int(len(w)),
                    hz=(src.hz if src else 0.0))
    F = np.linalg.norm(w[:, 0:3], axis=1)
    T = np.linalg.norm(w[:, 3:6], axis=1)

    c.all_zero = bool(np.all(np.abs(w) < 1e-12))
    c.constant = bool(np.all(np.std(w, axis=0) < 1e-12)) and not c.all_zero
    c.force_p50, c.force_p95, c.force_max = _pct(F, 50), _pct(F, 95), float(np.nanmax(F))
    c.torque_p50, c.torque_p95, c.torque_max = _pct(T, 50), _pct(T, 95), float(np.nanmax(T))
    c.contact_frac = float(np.mean(F > WRENCH_CONTACT_N))

    if dt is not None and len(dt):
        c.dt_p50_ms, c.dt_p95_ms = _pct(dt, 50) * 1e3, _pct(dt, 95) * 1e3
        c.dt_max_ms = float(np.nanmax(dt)) * 1e3
        c.over_tol_frac = float(np.mean(dt * 1e3 > tol_ms))

    # ── 판정 ────────────────────────────────────────────────────────────────
    if c.all_zero:
        flags.append("wrench_all_zero")
        c.verdict = ("❌ 전 프레임 0 — 센서 미연결/전원 미인가, 또는 토픽 오매칭. "
                     "수집을 중단하고 하드웨어부터 확인하십시오")
    elif c.constant:
        flags.append("wrench_constant")
        c.verdict = "❌ 값이 상수 — 센서가 갱신되지 않고 있습니다 (드라이버 정지 의심)"
    elif c.force_p95 > WRENCH_UNIT_SUSPECT_N:
        flags.append("wrench_unit_suspect")
        c.verdict = (f"⚠️ |F| p95 = {c.force_p95:.0f} — 단위가 N 이 아닐 수 있습니다 "
                     "(raw count / mN?). 스케일을 확인하십시오")
    elif c.force_p95 < WRENCH_STATIC_N * 0.5:
        flags.append("wrench_too_quiet")
        c.verdict = (f"⚠️ |F| p95 = {c.force_p95:.3f} N 로 정지 노이즈({WRENCH_STATIC_N} N)"
                     "보다도 작습니다 — 이미 tare 된 상수이거나 접촉이 전혀 없었습니다")
    elif c.contact_frac < 0.01:
        flags.append("wrench_no_contact")
        c.verdict = (f"⚠️ |F| > {WRENCH_CONTACT_N} N 프레임이 {c.contact_frac*100:.1f}% — "
                     "이 에피소드에는 배울 접촉이 사실상 없습니다")
    else:
        c.verdict = (f"✅ 정상 (접촉 프레임 {c.contact_frac*100:.1f}%, "
                     f"|F| p50 {c.force_p50:.2f} N / max {c.force_max:.1f} N)")

    if c.over_tol_frac > 0.05:
        flags.append("sync_poor")
        c.verdict += (f"  |  ⚠️ 동기화: {c.over_tol_frac*100:.0f}% 프레임이 "
                      f"±{tol_ms:.0f}ms 초과 (p95 {c.dt_p95_ms:.1f}ms)")
    return c


def validate(fr: Frames, raw: RawStreams, tol_ms: float = 10.0) -> ValidationReport:
    flags: list[str] = []
    t = fr.t_ns
    dts = np.diff(t) / 1e9 if len(t) > 1 else np.array([0.0])
    dur = float((t[-1] - t[0]) / 1e9) if len(t) > 1 else 0.0
    hz = (len(t) - 1) / dur if dur > 0 else 0.0

    nonmono = int(np.sum(dts <= 0))
    if nonmono:
        flags.append("anchor_nonmonotonic")

    clocks = {s.clock for s in (raw.joint_pos, raw.joint_effort, raw.joint_current,
                                raw.wrench_left, raw.wrench_right) if s is not None}
    mixed = len(clocks) > 1
    if mixed:
        flags.append("clock_mixed")

    rep = ValidationReport(
        n_frames=fr.n, duration_s=dur, anchor_hz=hz,
        anchor_jitter_ms=float(np.std(dts) * 1e3),
        anchor_nonmonotonic=nonmono,
        clock="+".join(sorted(clocks)), clock_mixed=mixed,
    )

    rep.wrench = [
        _check_wrench("left", fr.wrench_l, fr.wrench_dt_l, raw.wrench_left, tol_ms, flags),
        _check_wrench("right", fr.wrench_r, fr.wrench_dt_r, raw.wrench_right, tol_ms, flags),
    ]

    # 23칸 중 안 채워진 칸이 있으면 발행 쪽이 일부 관절만 보내거나 명명 규칙이
    # 다른 것이다. 팔 7관절 중 하나라도 비면 FK 가 죽으므로 조용히 넘기면 안 된다.
    rep.state_nan_frac = float(np.mean(~np.isfinite(fr.state)))
    if rep.state_nan_frac > 0:
        flags.append("state_incomplete")

    rep.has_effort = fr.effort is not None
    rep.has_current = fr.current is not None
    if not rep.has_effort:
        flags.append("no_effort")
    else:
        rep.effort_nan_frac = float(np.mean(~np.isfinite(fr.effort)))
        if np.all(np.abs(np.nan_to_num(fr.effort)) < 1e-12):
            flags.append("effort_all_zero")
    if not rep.has_current:
        flags.append("no_current")

    for side, arr, key in (("left", fr.psi_l, "psi_nan_frac_left"),
                           ("right", fr.psi_r, "psi_nan_frac_right")):
        if arr is None:
            setattr(rep, key, 1.0)
            flags.append(f"psi_missing_{side}")
            continue
        nanf = float(np.mean(~np.isfinite(arr)))
        setattr(rep, key, nanf)
        good = arr[np.isfinite(arr)]
        rng = float(np.ptp(good)) * 180 / math.pi if good.size else 0.0
        setattr(rep, f"psi_range_{side}_deg", rng)
        if nanf > 0.05:
            flags.append(f"psi_nan_high_{side}")

        # psi 는 (-pi, pi] 로 감기는 각도다. 30Hz 에서 인접 프레임이 pi 이상
        # 점프했다면 물리적 운동이 아니라 **wrap 불연속**이다. 회전 표현에
        # Euler/quaternion 을 금지한 것과 같은 이유로 (interface.py Waypoint
        # 주석 참조) L2 회귀가 이 지점에서 부호를 평균내며 깨진다.
        # 학습 쪽에서 (cos psi, sin psi) 로 풀어 쓸지 판단하라는 신호다.
        if good.size > 1:
            wraps = int(np.sum(np.abs(np.diff(good)) > math.pi))
            setattr(rep, f"psi_wrap_{side}", wraps)
            if wraps > 0:
                flags.append(f"psi_wraparound_{side}")

    g = np.concatenate([fr.state[:, IDX["left_gripper"]], fr.state[:, IDX["right_gripper"]]])
    rep.gripper_min_m, rep.gripper_max_m = float(np.nanmin(g)), float(np.nanmax(g))
    if not (GRIPPER_RANGE_M[0] - 1e-6 <= rep.gripper_min_m
            and rep.gripper_max_m <= GRIPPER_RANGE_M[1] + 1e-6):
        flags.append("gripper_range_suspect")

    # 좌/우가 같은 증상이면 플래그가 중복된다 — 순서를 지키며 dedupe.
    # (어느 쪽인지는 rep.wrench[i].verdict 에 이미 있다)
    rep.flags = list(dict.fromkeys(flags))
    fatal = {"wrench_all_zero", "wrench_constant", "wrench_missing_left",
             "wrench_missing_right", "anchor_nonmonotonic"}
    rep.ok = not (fatal & set(rep.flags))
    return rep


def print_report(rep: ValidationReport) -> None:
    print("\n" + "=" * 72)
    print("검증 리포트")
    print("=" * 72)
    print(f"  프레임        {rep.n_frames:,}개 / {rep.duration_s:.2f}s  "
          f"→ {rep.anchor_hz:.2f} Hz (jitter σ {rep.anchor_jitter_ms:.2f} ms)")
    print(f"  시계          {rep.clock}"
          + ("   ⚠️ 스트림마다 시계가 다릅니다 — 체계적 오프셋이 생깁니다"
             if rep.clock_mixed else ""))
    if rep.anchor_nonmonotonic:
        print(f"  ❌ 비단조 타임스탬프 {rep.anchor_nonmonotonic}개")

    print("\n  ── 손목 F/T " + "─" * 57)
    for c in rep.wrench:
        if not c.present:
            print(f"  {c.side:5s}  {c.verdict}")
            continue
        print(f"  {c.side:5s}  소스 {c.hz:6.1f} Hz")
        print(f"         |F|  p50 {c.force_p50:8.3f}  p95 {c.force_p95:8.3f}  "
              f"max {c.force_max:8.2f}  [N]")
        print(f"         |T|  p50 {c.torque_p50:8.3f}  p95 {c.torque_p95:8.3f}  "
              f"max {c.torque_max:8.2f}  [N·m]")
        print(f"         정렬 시간차 p50 {c.dt_p50_ms:6.2f}  p95 {c.dt_p95_ms:6.2f}  "
              f"max {c.dt_max_ms:6.2f}  [ms]")
        print(f"         {c.verdict}")
    print(f"\n         기준: 정지 ~{WRENCH_STATIC_N} N / 운동 중 residual ~{WRENCH_MOTION_N} N "
          f"/ 접촉 판정 >{WRENCH_CONTACT_N} N")

    print("\n  ── 관절 채널 " + "─" * 56)
    print(f"  state    23칸 중 미충전 {rep.state_nan_frac*100:.1f}%"
          + ("   ⚠️ 일부 관절이 발행되지 않거나 명명 규칙이 다릅니다"
             if rep.state_nan_frac > 0 else ""))
    print(f"  effort   {'있음' if rep.has_effort else '❌ 없음'}"
          + (f"  (NaN {rep.effort_nan_frac*100:.1f}%)" if rep.has_effort else ""))
    print(f"  current  {'있음' if rep.has_current else '⚠️ 없음'}")
    print(f"  그리퍼   {rep.gripper_min_m:.4f} ~ {rep.gripper_max_m:.4f} m"
          + ("   ⚠️ 예상 범위 밖 — ÷1000 스케일 가정을 확인하십시오"
             if "gripper_range_suspect" in rep.flags else ""))

    print("\n  ── 파생 (FK) " + "─" * 56)
    print(f"  psi left   NaN {rep.psi_nan_frac_left*100:5.2f}%   "
          f"범위 {rep.psi_range_left_deg:6.1f}°   wrap {rep.psi_wrap_left}회")
    print(f"  psi right  NaN {rep.psi_nan_frac_right*100:5.2f}%   "
          f"범위 {rep.psi_range_right_deg:6.1f}°   wrap {rep.psi_wrap_right}회")
    print("             (NaN = 완전 신전 특이점. 소량은 정상)")
    if rep.psi_wrap_left or rep.psi_wrap_right:
        print("             ⚠️ psi 가 ±pi 에서 감깁니다. L2 회귀는 이 불연속에서 깨집니다"
              " — Euler 를 금지한 것과 같은 이유입니다.")
        print("                학습 입력은 (cos psi, sin psi) 로 풀어 쓰는 것을 검토하십시오.")

    print("\n  플래그: " + (", ".join(rep.flags) if rep.flags else "없음"))
    print(f"  판정  : {'✅ 통과' if rep.ok else '❌ 실패 — 수집을 진행하지 마십시오'}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# 출력 — LeRobot v2.1 parquet + meta
# ─────────────────────────────────────────────────────────────────────────────


def build_table(fr: Frames, task_index: int = 0) -> dict[str, np.ndarray]:
    """
    parquet 컬럼 딕셔너리. pyarrow 없이도 만들 수 있어야 dry-run 이 의미가 있다.

    타임스탬프는 에피소드 시작 기준 초 (LeRobot 관례). 절대 ns 는 meta 에만
    남긴다 — 여러 에피소드를 이어붙일 때 절대 시각이 새면 학습이 그걸 쓴다.
    """
    T = fr.n
    t0 = int(fr.t_ns[0])
    tbl: dict[str, np.ndarray] = {
        OUT["state"]: fr.state.astype(np.float32),
        OUT["action"]: fr.action.astype(np.float32),
        OUT["timestamp"]: ((fr.t_ns - t0) / 1e9).astype(np.float32),
        OUT["frame_index"]: np.arange(T, dtype=np.int64),
        OUT["episode_index"]: np.full(T, fr.meta.get("episode_index", 0), dtype=np.int64),
        OUT["index"]: np.arange(T, dtype=np.int64),
        OUT["task_index"]: np.full(T, task_index, dtype=np.int64),
    }
    if fr.wrench_l is not None:
        tbl[OUT["wrench_l"]] = fr.wrench_l.astype(np.float32)
        tbl[OUT["wrench_dt_l"]] = fr.wrench_dt_l.astype(np.float32)
    if fr.wrench_r is not None:
        tbl[OUT["wrench_r"]] = fr.wrench_r.astype(np.float32)
        tbl[OUT["wrench_dt_r"]] = fr.wrench_dt_r.astype(np.float32)
    if fr.effort is not None:
        tbl[OUT["effort"]] = fr.effort.astype(np.float32)
    if fr.current is not None:
        tbl[OUT["current"]] = fr.current.astype(np.float32)
    if fr.psi_l is not None:
        tbl[OUT["psi_l"]] = fr.psi_l.astype(np.float32)
        tbl[OUT["psi_r"]] = fr.psi_r.astype(np.float32)
        tbl[OUT["T_rel"]] = fr.T_rel.astype(np.float32)
    return tbl


def build_features(tbl: dict[str, np.ndarray]) -> dict:
    """LeRobot meta/info.json 의 features 절. 공식 info.json 에 병합하면 된다."""
    names = {
        OUT["state"]: JOINT_NAMES, OUT["action"]: JOINT_NAMES,
        OUT["effort"]: JOINT_NAMES, OUT["current"]: JOINT_NAMES,
        OUT["wrench_l"]: WRENCH_NAMES, OUT["wrench_r"]: WRENCH_NAMES,
        OUT["T_rel"]: ["dx", "dy", "dz", "r11", "r21", "r31", "r12", "r22", "r32"],
    }
    feats = {}
    for k, v in tbl.items():
        shape = list(v.shape[1:]) or [1]
        feats[k] = {"dtype": str(v.dtype), "shape": shape}
        if k in names:
            feats[k]["names"] = names[k]
    return feats


def write_parquet(tbl: dict[str, np.ndarray], path: str) -> bool:
    """pyarrow 가 없으면 False. dry-run 은 이 단계 없이도 유효하다."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("[write] ⚠️ pyarrow 가 없어 parquet 을 쓰지 못했습니다 "
              "(mamba install -n physical_ai pyarrow)")
        return False
    cols = {}
    for k, v in tbl.items():
        cols[k] = pa.array(v.tolist()) if v.ndim > 1 else pa.array(v)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pq.write_table(pa.table(cols), path)
    return True


def write_meta(out_dir: str, tbl: dict[str, np.ndarray], fr: Frames,
               rep: ValidationReport, sel: Optional[TopicSelection], args) -> str:
    """
    보정 파라미터를 **여기에만** 적는다. parquet 은 raw 다.

    tare(영점)를 parquet 에 적용하지 않는 이유: 재기록이 불가능하므로
    되돌릴 수 없는 변환을 원본에 가하면 안 된다. 게다가 상수 tare 는
    센서 bias 만 지우고 툴 중량의 중력 성분은 손목 자세에 따라 회전하므로
    한 자세에서 잰 영점은 다른 자세에서 틀린다. 제대로 된 보상은 툴 질량과
    CoM 캘리브레이션이 필요하다 (⚠️ 확인 필요).
    """
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    tare = {}
    n = int(args.tare_frames)
    if n > 0:
        for side, w in (("left", fr.wrench_l), ("right", fr.wrench_r)):
            if w is not None and len(w) >= n:
                tare[side] = np.mean(w[:n], axis=0).tolist()

    info = {
        "fork": {
            "name": "mcap2lerobot_fork",
            "version": FORK_VERSION,
            "upstream": UPSTREAM,
            "purpose": "손목 F/T wrench + 관절별 effort 를 LeRobot 스키마에 추가",
        },
        "codebase_version": "v2.1",
        "robot_type": "G1_V2.2B",
        "fps": args.fps,
        "total_frames": fr.n,
        "total_episodes": 1,
        "features": build_features(tbl),
        "wrench": {
            "frame": args.wrench_frame,            # ⚠️ 확인 필요
            "sign_convention": args.wrench_sign,   # ⚠️ 확인 필요
            "units": ["N", "N", "N", "N·m", "N·m", "N·m"],
            "interp": fr.meta.get("wrench_interp"),
            "tare_bias_NOT_APPLIED": tare,
            "note": "parquet 은 센서 raw. tare/중력보상은 적용하지 않았음.",
        },
        "gripper_scale": fr.meta.get("gripper_scale"),
        "clock": fr.meta.get("clock"),
        "topics": asdict(sel) if sel else fr.meta.get("topics"),
        "derive_us_per_frame": fr.meta.get("derive_us_per_frame"),
        "validation": asdict(rep),
    }
    path = os.path.join(meta_dir, "info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# --dry-run — 합성 프레임 배열로 같은 파이프라인을 태운다
#
# 합성 MCAP 을 만들지 않는 이유: MCAP 인코딩은 검증할 게 없다 (라이브러리가
# 한다). 검증해야 하는 것은 **동기화·파생·검증 로직**이고, 그건 RawStreams
# 단계부터 태우면 전부 커버된다. 게다가 합성 MCAP 은 우리가 모르는 실제
# 스키마를 우리가 상상한 스키마로 고정시켜 버려서, 통과해도 아무 의미가 없다.
# ─────────────────────────────────────────────────────────────────────────────


def synth_streams(*, n: int = 300, fps: float = 30.0, wrench_hz: float = 250.0,
                  wrench_mode: str = "normal", wrench_scale: float = 1.0,
                  wrench_offset_ms: float = 0.0, with_effort: bool = True,
                  with_current: bool = True, wrench_clock: str = "synthetic",
                  seed: int = 0) -> RawStreams:
    """실기체를 흉내 낸 스트림. 관절은 한계 안에서 움직이고 F/T 는 접촉 구간을 갖는다."""
    rng = np.random.default_rng(seed)
    t0 = int(time.time_ns())

    # 앵커: fps 에 ±1ms jitter (실제 기록도 완벽히 균일하지 않다)
    dt_ns = int(1e9 / fps)
    t = t0 + np.arange(n, dtype=np.int64) * dt_ns \
        + rng.integers(-1_000_000, 1_000_000, size=n)
    t = np.sort(t)

    armL = G1Arm("left")
    lo, hi = armL.limits
    mid = np.clip((lo + hi) / 2.0, lo, hi)      # 팔꿈치가 굽어 있어 psi 가 정의됨
    amp = 0.25 * (hi - lo) / 2.0
    ts = (t - t[0]) / 1e9

    state = np.zeros((n, JOINT_DIM))
    for k in range(7):
        wave = mid[k] + amp[k] * np.sin(2 * np.pi * 0.3 * ts + 0.7 * k)
        state[:, IDX["left_arm"].start + k] = np.clip(wave, lo[k], hi[k])
        state[:, IDX["right_arm"].start + k] = np.clip(
            mid[k] + amp[k] * np.sin(2 * np.pi * 0.25 * ts + 0.5 * k + 1.1), lo[k], hi[k])
    state[:, IDX["head"]] = 0.2 * np.sin(2 * np.pi * 0.1 * ts)[:, None]
    # 그리퍼는 **원값** (mm 가정) — ÷1000 은 파이프라인이 한다
    state[:, IDX["left_gripper"]] = 40 + 30 * np.sin(2 * np.pi * 0.2 * ts)
    state[:, IDX["right_gripper"]] = 40 + 30 * np.cos(2 * np.pi * 0.2 * ts)

    jp = Stream("joint_pos", t, state, "synthetic/state", "synthetic")

    eff = cur = None
    if with_effort:
        e = 8.0 * np.sin(2 * np.pi * 0.3 * ts)[:, None] * np.ones((1, JOINT_DIM))
        e += rng.normal(0, 0.3, size=(n, JOINT_DIM))
        eff = Stream("joint_effort", t, e, "synthetic/state", "synthetic")
    if with_current:
        c = 1.5 + 0.4 * rng.normal(size=(n, JOINT_DIM))
        cur = Stream("joint_current", t, c, "synthetic/state", "synthetic")

    # F/T: 자체 rate + 오프셋. 접촉 구간에서 힘이 계단처럼 뛴다.
    def _w(side_seed: int) -> Stream:
        m = max(2, int(n / fps * wrench_hz))
        tw = t[0] + int(wrench_offset_ms * 1e6) \
            + (np.arange(m, dtype=np.int64) * int(1e9 / wrench_hz))
        tw = tw + rng.integers(-200_000, 200_000, size=m)
        tw = np.sort(tw)
        r = np.random.default_rng(seed + side_seed)
        w = r.normal(0, WRENCH_STATIC_N / 3.0, size=(m, WRENCH_DIM))
        w[:, 2] += -1.2                                    # 툴 중량의 정적 성분
        rel = (tw - tw[0]) / max(1, (tw[-1] - tw[0]))
        moving = (rel > 0.1) & (rel < 0.9)
        w[moving] += r.normal(0, WRENCH_MOTION_N / 3.0, size=(moving.sum(), WRENCH_DIM))
        contact = (rel > 0.45) & (rel < 0.65)              # 접촉 20%
        w[contact, 0] += 9.0
        w[contact, 2] += -6.0
        w[contact, 4] += 0.8
        if wrench_mode == "zero":
            w[:] = 0.0
        elif wrench_mode == "constant":
            w[:] = w[0]
        return Stream("wrench", tw, w * wrench_scale, "synthetic/ft", wrench_clock)

    return RawStreams(jp, eff, cur, _w(1), _w(2),
                      meta={"clock": "synthetic", "synthetic": True})


def _unit_tests() -> None:
    """동기화·파생의 수치적 정확성. 실패하면 AssertionError 로 즉시 죽는다."""
    # 1) 최근접 매칭이 진짜 최근접인가
    ts = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000     # 0,10,20,30 ms
    src = Stream("u", ts, np.arange(4.0)[:, None])
    ta = np.array([1, 6, 14, 100], dtype=np.int64) * 1_000_000
    v, dt = resample_nearest(ta, src)
    assert v.ravel().tolist() == [0.0, 1.0, 1.0, 3.0], v.ravel().tolist()
    assert np.allclose(dt, [0.001, 0.004, 0.004, 0.070]), dt
    # 2) 정렬되지 않은 입력도 Stream 이 정렬한다
    s2 = Stream("u", ts[::-1], np.arange(4.0)[:, None])
    assert s2.t_ns.tolist() == ts.tolist()
    assert s2.data.ravel().tolist() == [3.0, 2.0, 1.0, 0.0]
    # 3) 선형 보간이 중점에서 평균인가
    v2, _ = resample_linear(np.array([5_000_000], dtype=np.int64), src)
    assert abs(v2[0, 0] - 0.5) < 1e-9, v2
    # 4) T_rel 이 FK 와 정확히 일치하는가 (뺄셈 아님)
    d = Derived()
    q = np.zeros(JOINT_DIM)
    armL = G1Arm("left")
    lo, hi = armL.limits
    q[IDX["left_arm"]] = np.clip((lo + hi) / 2 + 0.1, lo, hi)
    q[IDX["right_arm"]] = np.clip((lo + hi) / 2 - 0.2, lo, hi)
    out = d.compute(q[None, :])
    Trel = se3_inv(d.arm_l.fk(q[IDX["left_arm"]])) @ d.arm_r.fk(q[IDX["right_arm"]])
    assert np.allclose(out["T_rel"][0, :3], Trel[:3, 3], atol=1e-12)
    assert np.allclose(sixd_to_R(out["T_rel"][0, 3:]), Trel[:3, :3], atol=1e-9)
    # 5) 6D 회전이 뺄셈으로 만든 것과 다르다는 것 (상대 항이 first-class 인 이유)
    assert not np.allclose(Trel[:3, 3],
                           d.arm_r.fk(q[IDX["right_arm"]])[:3, 3]
                           - d.arm_l.fk(q[IDX["left_arm"]])[:3, 3])
    print("  ✅ 단위 테스트 5/5 (최근접 매칭 / 정렬 / 보간 / T_rel FK 일치 / 뺄셈≠FK)")


def dry_run(args) -> int:
    print("=" * 72)
    print(f"mcap2lerobot_fork v{FORK_VERSION} — DRY RUN")
    print("=" * 72)
    print("  실제 MCAP 없이 합성 프레임 배열로 동기화·파생·검증 파이프라인을 태웁니다.")
    print("  (MCAP 디코딩만 건너뜁니다 — 그 부분은 실기체 데이터로만 닫힙니다)\n")

    _unit_tests()

    scenarios = [
        # 합성 관절 스윕은 psi 범위가 넓어 ±pi 를 넘는다 → wrap 검출도 여기서 함께 확인
        ("정상 (F/T 250Hz, effort 있음) + psi wrap 검출",
         dict(), {"wrench_all_zero", "wrench_constant", "no_effort", "no_current",
                  "sync_poor", "gripper_range_suspect", "wrench_unit_suspect",
                  "clock_mixed", "psi_nan_high_left", "psi_nan_high_right"},
         {"psi_wraparound_left", "psi_wraparound_right"}, True),
        ("센서 미연결 — wrench 전부 0",
         dict(wrench_mode="zero"), set(), {"wrench_all_zero"}, False),
        ("드라이버 정지 — wrench 상수",
         dict(wrench_mode="constant"), set(), {"wrench_constant"}, False),
        ("동기화 불량 — F/T 4Hz + 60ms offset",
         dict(wrench_hz=4.0, wrench_offset_ms=60.0), set(), {"sync_poor"}, True),
        ("단위 의심 — 1000배 스케일",
         dict(wrench_scale=1000.0), set(), {"wrench_unit_suspect"}, True),
        ("effort/current 없음",
         dict(with_effort=False, with_current=False), set(), {"no_effort", "no_current"}, True),
        ("시계 혼합 — F/T 만 log_time",
         dict(wrench_clock="log_time"), set(), {"clock_mixed"}, True),
    ]

    n_fail = 0
    for name, kw, forbidden, required, expect_ok in scenarios:
        raw = synth_streams(n=args.dry_frames, fps=args.fps, **kw)
        fr = build_frames(raw, wrench_interp=args.wrench_interp)
        rep = validate(fr, raw, tol_ms=args.sync_tol_ms)
        tbl = build_table(fr)
        flags = set(rep.flags)

        problems = []
        missing = required - flags
        if missing:
            problems.append(f"기대 플래그 누락 {sorted(missing)}")
        extra = forbidden & flags
        if extra:
            problems.append(f"뜨면 안 되는 플래그 {sorted(extra)}")
        if rep.ok != expect_ok:
            problems.append(f"판정 {rep.ok} != 기대 {expect_ok}")

        # 스키마 검사 — 모든 시나리오에서 항상
        for key, dim in ((OUT["state"], JOINT_DIM), (OUT["action"], JOINT_DIM)):
            if tbl[key].shape != (fr.n, dim):
                problems.append(f"{key} 모양 {tbl[key].shape}")
        if OUT["wrench_l"] in tbl and tbl[OUT["wrench_l"]].shape != (fr.n, WRENCH_DIM):
            problems.append(f"{OUT['wrench_l']} 모양 {tbl[OUT['wrench_l']].shape}")
        if OUT["T_rel"] in tbl and tbl[OUT["T_rel"]].shape != (fr.n, 9):
            problems.append(f"{OUT['T_rel']} 모양 {tbl[OUT['T_rel']].shape}")

        mark = "✅" if not problems else "❌"
        print(f"  {mark} {name}")
        print(f"       플래그: {', '.join(sorted(flags)) if flags else '없음'}")
        if problems:
            n_fail += 1
            for p in problems:
                print(f"       ↳ {p}")

    # 대표 시나리오 하나는 리포트 전문을 보여준다 (사람이 눈으로 확인할 것)
    raw = synth_streams(n=args.dry_frames, fps=args.fps)
    fr = build_frames(raw, wrench_interp=args.wrench_interp)
    rep = validate(fr, raw, tol_ms=args.sync_tol_ms)
    tbl = build_table(fr)
    print_report(rep)

    print("\n출력 스키마:")
    for k, v in tbl.items():
        shape = f"({v.shape[1]},)" if v.ndim > 1 else "()"
        print(f"  {k:34s} {str(v.dtype):9s} {shape}")
    print(f"\n  파생 비용: {fr.meta.get('derive_us_per_frame', 0):.1f} us/frame "
          "(psi 양팔 + T_rel, FK 포함)")

    if args.out:
        p = os.path.join(args.out, "data", "chunk-000", "episode_000000.parquet")
        if write_parquet(tbl, p):
            print(f"  parquet 저장: {p}")
        print(f"  meta 저장   : {write_meta(args.out, tbl, fr, rep, None, args)}")

    print()
    if n_fail:
        print(f"❌ DRY RUN 실패 — 시나리오 {n_fail}개")
        return 1
    print("✅ DRY RUN 통과 — 스키마/동기화/파생/검증 로직 정상")
    print("   다음: 회사 Linux 박스에서 실제 MCAP 1개로 --list-topics --peek 후 변환")
    return 0


# ─────────────────────────────────────────────────────────────────────────────


def cmd_list_topics(args) -> int:
    src = McapSource(args.mcap)
    infos = src.list_topics(peek=args.peek)
    print("=" * 100)
    print(f"MCAP 토픽 덤프 — {args.mcap}")
    print("=" * 100)
    print(f"  {'토픽':46s} {'스키마':30s} {'개수':>8s} {'Hz':>8s}  추정")
    print("-" * 100)
    for t in infos:
        guess = []
        if _RE_CAMERA.search(t.name):
            guess.append("카메라")
        if is_wrench_candidate(t):
            side = ("left" if _RE_LEFT.search(t.name) else
                    "right" if _RE_RIGHT.search(t.name) else "좌/우 불명")
            guess.append(f"★ F/T 후보 ({side})")
        if t.name in STATE_TOPICS_OFFICIAL:
            guess.append("★ 공식 상태 토픽")
        elif _RE_JOINT.search(t.name):
            guess.append("관절 후보")
        print(f"  {t.name:46s} {t.schema[:30]:30s} {t.count:8d} {t.hz:8.1f}  "
              f"{'/'.join(guess)}")
        if t.fields:
            print(f"      필드: {', '.join(t.fields[:20])}")
    print("-" * 100)
    print("\n다음:")
    print("  ★ F/T 후보를 찾았으면:")
    print("      --wrench-topic-left <이름> --wrench-topic-right <이름>")
    print("  못 찾았으면 기록 쪽 설정에 F/T 토픽이 빠진 것입니다. **수집 전에** 고치십시오")
    print("      — 기록되지 않은 힘은 소급 복원이 불가능합니다.")
    return 0


def cmd_convert(args) -> int:
    print("=" * 72)
    print(f"mcap2lerobot_fork v{FORK_VERSION}  (upstream: {UPSTREAM})")
    print("=" * 72)
    src = McapSource(args.mcap)
    infos = src.list_topics(peek=False)

    override = TopicSelection(state=args.state_topic,
                              wrench_left=args.wrench_topic_left,
                              wrench_right=args.wrench_topic_right,
                              effort=args.effort_topic)
    sel = discover_topics(infos, override)
    print("\n토픽 선택:")
    for n in sel.notes:
        print(f"  · {n}")
    print(f"  state        : {sel.state}")
    print(f"  wrench left  : {sel.wrench_left or sel.wrench_combined or '— 없음'}")
    print(f"  wrench right : {sel.wrench_right or sel.wrench_combined or '— 없음'}")
    print(f"  effort       : {sel.effort or '(상태 메시지 내부)'}")

    if not (sel.wrench_left or sel.wrench_right or sel.wrench_combined):
        print(_no_wrench_error(infos, sel.wrench_ambiguous))
        if not args.allow_no_wrench:
            return 2

    raw = build_raw_streams(src, sel, clock=args.clock)

    # 토픽은 골랐는데 디코딩 결과가 비었다 — combined 로 오판했거나 필드명이
    # extract_wrench() 가 아는 형태가 아니다. 조용히 넘어가면 F/T 0개짜리
    # 데이터셋이 만들어지고, 그건 이 파일이 막으려는 바로 그 결과다.
    if (sel.wrench_left or sel.wrench_right or sel.wrench_combined) \
            and raw.wrench_left is None and raw.wrench_right is None:
        print("\n❌ wrench 토픽은 선택됐으나 메시지에서 6축 값을 뽑지 못했습니다.")
        print("   `--list-topics --peek` 로 필드명을 확인한 뒤 extract_wrench() 를 "
              "고치십시오.")
        if not args.allow_no_wrench:
            return 2
    fr = build_frames(raw, wrench_interp=args.wrench_interp,
                      apply_gripper_scale=not args.no_gripper_scale,
                      derive=not args.no_derive, limit=args.limit_frames,
                      episode_index=args.episode_index)
    rep = validate(fr, raw, tol_ms=args.sync_tol_ms)
    print_report(rep)

    if args.out:
        p = os.path.join(args.out, "data", "chunk-000",
                         f"episode_{args.episode_index:06d}.parquet")
        tbl = build_table(fr, task_index=args.task_index)
        if write_parquet(tbl, p):
            print(f"\n저장: {p}  ({fr.n:,} 프레임, 컬럼 {len(tbl)}개)")
        print(f"메타: {write_meta(args.out, tbl, fr, rep, sel, args)}")

    if args.strict and not rep.ok:
        print("\n❌ --strict: 검증 실패. 수집을 진행하지 마십시오.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="galbot-mcap2lerobot 포크 — 손목 F/T + 관절 effort 를 추가한다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="먼저 `--list-topics --peek` 로 실제 토픽명을 확인하십시오.")
    ap.add_argument("--mcap", help="입력 MCAP 경로")
    ap.add_argument("--out", help="출력 LeRobot 데이터셋 루트")
    ap.add_argument("--list-topics", action="store_true",
                    help="MCAP 안의 토픽을 전부 덤프하고 종료 (F/T 토픽명 확인용)")
    ap.add_argument("--peek", action="store_true",
                    help="--list-topics 시 각 토픽 첫 메시지의 필드명도 출력")
    ap.add_argument("--dry-run", action="store_true",
                    help="MCAP 없이 합성 프레임으로 파이프라인 검증")
    ap.add_argument("--dry-frames", type=int, default=300, help="dry-run 프레임 수")

    ap.add_argument("--state-topic", default=None, help="상태 토픽 명시 지정")
    ap.add_argument("--wrench-topic-left", default=None)
    ap.add_argument("--wrench-topic-right", default=None)
    ap.add_argument("--effort-topic", default=None)
    ap.add_argument("--allow-no-wrench", action="store_true",
                    help="F/T 없이도 변환 (권장하지 않음 — 소급 추가 불가)")

    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--clock", choices=["payload", "log_time"], default="payload",
                    help="타임스탬프 출처. payload=센서 device 시각(기본)")
    ap.add_argument("--sync-tol-ms", type=float, default=10.0,
                    help="F/T 정렬 허용 시간차. 초과 프레임이 5%%를 넘으면 경고")
    ap.add_argument("--wrench-interp", choices=["nearest", "linear"], default="nearest",
                    help="nearest 가 기본 — linear 는 접촉 계단을 뭉갠다")
    ap.add_argument("--no-derive", action="store_true", help="psi/T_rel 계산 생략")
    ap.add_argument("--no-gripper-scale", action="store_true",
                    help="그리퍼 ÷1000 을 적용하지 않는다")
    ap.add_argument("--tare-frames", type=int, default=30,
                    help="앞 N 프레임으로 wrench bias 추정 (meta 에만 기록, 미적용)")
    ap.add_argument("--wrench-frame", default="unknown_TODO",
                    help="⚠️ 확인 필요 — wrench 표현 프레임 (sensor/tool/base)")
    ap.add_argument("--wrench-sign", default="unknown_TODO",
                    help="⚠️ 확인 필요 — 환경→로봇 인지 로봇→환경 인지")

    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="검증 실패 시 종료코드 1 (수집 전 게이트로 쓸 것)")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)
    if not args.mcap:
        ap.error("--mcap 또는 --dry-run 이 필요합니다")
    if args.list_topics:
        return cmd_list_topics(args)
    return cmd_convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
