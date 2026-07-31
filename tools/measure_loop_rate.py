#!/usr/bin/env python3
"""
GATE-1 — Galbot G1 폐루프 명령 대역폭 / latency / jitter 측정

이 프로젝트에서 가장 중요한 단일 측정이다. `set_joint_commands`의 실제
달성 가능한 rate가 모든 공개 출처에서 미문서화이고 (274,000자 API 레퍼런스에
Hz 수치가 한 번도 안 나오며, SDK 자체 example8_real_time_control_loop.cpp는
이름과 달리 sleep_for(1s)가 든 1Hz 블로킹 데모), 하류의 모든 결정이 여기 달려 있다:

  ~30 Hz  → 빠른 Model 2가 Model 1보다 의미 있게 빠를 여지 없음
            → 대뇌/소뇌 분리 전제 붕괴. 단일 모델 + 벤더 WBC로.
  ~125 Hz → admittance가 실제로 작동. 분리 유지 가능.

두 부분:
  PART A — 명령 rate를 올리며 드롭/불연속이 시작되는 천장을 찾는다
  PART B — 스윕 사인을 주입해 위치 추종 루프의 실제 Bode 선도를 얻는다

사용:
    # 드라이런 (로봇 없이 스크립트 검증)
    python measure_loop_rate.py --dry-run

    # 실제 측정 (온보드 HPU 또는 로봇 LAN의 외부 호스트에서)
    python measure_loop_rate.py --host onboard  --out results_onboard.json
    python measure_loop_rate.py --host external --out results_external.json

⚠️  측정 중 해당 머신에서 다른 워크로드를 돌리지 마십시오.
    jitter를 재는 것이므로 p99/p99.9가 오염됩니다.

⚠️  안전: 단일 관절(기본 left_arm_joint4)만 소진폭(±2°)으로 움직입니다.
    실행 전 팔 주변을 비우고 e-stop을 손 닿는 곳에 두십시오.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# SDK 어댑터
#
# 2026-07-31 갱신 — 실물 SDK 표면을 3090에서 확인했다. **클래스/메서드 이름은 확정.**
#
#   설치: /opt/galbot/galbot_sdk/linux-x86_64-gcc940/lib/python/galbot_sdk/
#   ⚠️ import 전에 반드시 (안 하면 ImportError):
#       source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh
#
#   GalbotRobot — 하드웨어 저수준. 확인된 메서드:
#       get_joint_names / get_joint_group_names / get_joint_positions / get_joint_states
#       set_joint_commands / set_joint_commands_batch / set_joint_positions
#       execute_joint_trajectory
#       get_force_sensor_data          ← 손목 F/T
#       get_gripper_state / set_gripper_command / get_sensor_extrinsic
#
#   손목 F/T 센서 id: GalbotOneFoxtrotSensor.LEFT_WRIST_FORCE / RIGHT_WRIST_FORCE
#
#   명령 경로가 **두 개**다 (이 스크립트가 둘 다 잰다):
#       direct     — set_joint_commands 를 고속 반복        → PART A/B
#       trajectory — execute_joint_trajectory + TARGET_TYPE_OVERRIDE → PART C
#
# ⚠️ 아직 모르는 것: **인자 시그니처.** galbot_sdk 는 pybind11 확장이라
#    inspect.signature() 가 전부 실패한다 (`(?)` 만 나옴). 실제 인자는 docstring 에
#    들어 있으므로 먼저 이걸 돌려 확정하십시오:
#
#        python tools/probe_sdk.py --focus
#
#    확정 전까지 아래 코드는 **호출 패턴 사다리**를 순서대로 시도하고, 전부 실패하면
#    해당 메서드의 docstring 을 그대로 출력한다. 추측으로 한 패턴만 박아두면
#    틀렸을 때 원인을 알 수 없기 때문이다.
# ─────────────────────────────────────────────────────────────────────────────

SDK_SETUP = "/opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh"

# 드라이런 전용 — 궤적 덮어쓰기의 가짜 전송 지연. PART C 가 이 값을 되찾아내면
# 측정 코드가 옳다는 뜻이다 (로봇 없이 검증하는 방법).
SIM_TRANSPORT_S = 0.035


def _doc(obj: object) -> str:
    return (getattr(obj, "__doc__", None) or "(docstring 없음)").strip()[:900]


class SDKCallFailed(RuntimeError):
    """호출 패턴을 전부 실패. docstring 을 담아 다음 수정을 안내한다."""

    def __init__(self, name: str, fn: object, tried: list[tuple[str, str]]):
        lines = [
            f"[adapter] `{name}` 호출 패턴을 전부 실패했습니다.",
            "",
            "시도한 것:",
        ]
        lines += [f"  - {pat}\n      → {err}" for pat, err in tried]
        lines += [
            "",
            f"`{name}` 의 실제 시그니처 (pybind11 docstring):",
            "─" * 70,
            _doc(fn),
            "─" * 70,
            "",
            "이 출력을 그대로 공유해주시면 어댑터를 확정합니다.",
            "전체 표면을 한 번에 뜨려면:  python tools/probe_sdk.py --focus",
        ]
        super().__init__("\n".join(lines))


def _try_patterns(name: str, fn, patterns: list[tuple[str, tuple, dict]]):
    """(라벨, args, kwargs) 사다리를 순서대로 시도. 첫 성공을 반환."""
    tried: list[tuple[str, str]] = []
    for label, a, kw in patterns:
        try:
            return fn(*a, **kw), label
        except Exception as e:
            tried.append((label, f"{type(e).__name__}: {e}"[:200]))
    raise SDKCallFailed(name, fn, tried)


class G1Adapter:
    """GalbotSDK 래퍼.

    실물 시그니처가 확정되면 `_resolve_*` 메서드의 패턴 사다리를 확정된 하나로
    줄이면 된다. 나머지 측정 코드는 손댈 필요 없다.
    """

    def __init__(self, dry_run: bool = False, joint_name: str = "left_arm_joint4",
                 tau_s: float = 0.015):
        self.dry_run = dry_run
        self.joint_name = joint_name
        self._robot = None
        self._sdk = None
        self._joint_names: list[str] = []
        self._joint_idx: Optional[int] = None
        self._sim_pos = 0.0
        self._sim_tau = tau_s
        self._sim_t0 = time.perf_counter()

        # 어떤 호출 패턴이 실제로 통했는지 — 결과 JSON 에 기록해 재현 가능하게
        self.resolved: dict[str, str] = {}
        # 상태 타임스탬프가 로봇 것인지 호스트 것인지. 이게 결과 해석을 바꾼다.
        self.ts_source = "sim" if dry_run else "unknown"
        self._ts_attr: Optional[str] = None
        self._read_via = "positions"        # "positions" | "states"
        self.traj_supported: Optional[bool] = None

        if dry_run:
            print(f"[adapter] DRY-RUN — 1차 지연 플랜트 시뮬레이션 (tau={tau_s*1e3:.0f}ms)")
            return

        self._connect_real()

    # ── 연결 ────────────────────────────────────────────────────────────────
    def _connect_real(self) -> None:
        try:
            import galbot_sdk as sdk
        except ImportError as e:
            sys.exit(
                f"[adapter] GalbotSDK import 실패: {e}\n\n"
                f"  SDK 환경을 먼저 로드해야 합니다:\n"
                f"      source {SDK_SETUP}\n\n"
                "  - SDK 는 Linux 전용입니다 (linux-x86_64 / linux-aarch64).\n"
                "    Mac 에서는 --dry-run 으로 스크립트만 검증하십시오.\n"
                "  - 경로가 다르면:  find /opt -maxdepth 6 -type d -name galbot_sdk\n"
            )
        self._sdk = sdk
        print(f"[adapter] galbot_sdk {getattr(sdk, '__file__', '?')}")

        robot, pat = _try_patterns(
            "GalbotRobot()", sdk.GalbotRobot,
            [("GalbotRobot()", (), {})]
            + [(f"GalbotRobot(MachineType.{m})", (getattr(sdk.MachineType, m),), {})
               for m in getattr(getattr(sdk, "MachineType", None), "__members__", {})],
        )
        self._robot = robot
        self.resolved["ctor"] = pat
        print(f"[adapter] 생성 OK — {pat}")

        self._resolve_joint_index()
        self._probe_state_timestamp()

    def _resolve_joint_index(self) -> None:
        """관절 이름 목록을 받아 우리가 흔들 관절의 인덱스를 찾는다.

        이걸 먼저 하는 이유: RoboCOIN(21-D)과 SDK 관절 벡터의 레이아웃이 다르다는
        걸 이미 한 번 데였다. 이름으로 인덱스를 잡으면 그 종류의 조용한 오염이 없다.
        """
        names, pat = _try_patterns(
            "get_joint_names", self._robot.get_joint_names,
            [("get_joint_names()", (), {})]
            + [(f"get_joint_names(G1JointGroup.{m})",
                (getattr(self._sdk.G1JointGroup, m),), {})
               for m in getattr(getattr(self._sdk, "G1JointGroup", None),
                                "__members__", {})],
        )
        self.resolved["get_joint_names"] = pat
        self._joint_names = [str(n) for n in names]
        print(f"[adapter] 관절 {len(self._joint_names)}개 — {pat}")

        if self.joint_name in self._joint_names:
            self._joint_idx = self._joint_names.index(self.joint_name)
        else:
            # 이름 규칙이 다를 수 있다 (left_arm_joint4 vs left_arm_joint_4 등)
            key = self.joint_name.replace("_", "").lower()
            hits = [i for i, n in enumerate(self._joint_names)
                    if n.replace("_", "").lower() == key]
            if not hits:
                sys.exit(
                    f"[adapter] 관절 '{self.joint_name}' 을 찾을 수 없습니다.\n"
                    f"  SDK 가 보고한 관절 이름 {len(self._joint_names)}개:\n"
                    + "\n".join(f"    [{i:2d}] {n}"
                                for i, n in enumerate(self._joint_names))
                    + f"\n\n  --joint <이름> 으로 정확한 이름을 지정하십시오."
                )
            self._joint_idx = hits[0]
            self.joint_name = self._joint_names[hits[0]]
            print(f"[adapter] 이름 정규화 → '{self.joint_name}'")
        print(f"[adapter] 대상 관절 index {self._joint_idx}")

    def _probe_state_timestamp(self) -> None:
        """로봇이 자체 타임스탬프를 주는지 확인.

        중요: 못 주면 호스트 시계로 대체하는데, 그러면 PART A 의 `unique_state_frac`
        과 `state_dt_*` 는 **로봇이 아니라 우리 루프**를 재는 값이 된다. 그 차이를
        모르고 보면 stale 상태를 건강한 것으로 오독한다. 그래서 결과에 명시한다.
        """
        try:
            st, _ = _try_patterns(
                "get_joint_states", self._robot.get_joint_states,
                [("get_joint_states()", (), {})],
            )
            probe = st[0] if isinstance(st, (list, tuple)) and st else st
            for attr in ("timestamp_ns", "timestamp", "stamp", "header"):
                if hasattr(probe, attr) and self._to_ns(getattr(probe, attr)) is not None:
                    self._ts_attr = attr
                    self.ts_source = f"device.{attr}"
                    # 타임스탬프를 실제로 쓰려면 read 경로도 get_joint_states 여야 한다.
                    # get_joint_positions 는 float 만 주므로 타임스탬프가 없다.
                    self._read_via = "states"
                    print(f"[adapter] 상태 타임스탬프: 로봇 제공 ({attr}) — read 경로를 "
                          f"get_joint_states 로 고정")
                    return
        except Exception as e:
            print(f"[adapter] get_joint_states 탐색 실패: {type(e).__name__}: {e}")
        self.ts_source = "host_monotonic"
        self._read_via = "positions"
        print("[adapter] ⚠️ 로봇 타임스탬프 없음 → 호스트 시계 사용.")
        print("          state_dt_* / unique_state_frac 은 '로봇 응답'이 아니라")
        print("          '우리 루프'를 재는 값이 됩니다. 판정에서 천장은")
        print("          '미확인(unconfirmed)'으로 강등됩니다.")

    @staticmethod
    def _to_ns(v: object) -> Optional[int]:
        """SDK 가 어떤 형태로 주든 ns 정수로. 못 바꾸면 None."""
        # ROS 스타일 Header/stamp: sec + nanosec
        for sec_a, nsec_a in (("sec", "nanosec"), ("secs", "nsecs"),
                              ("sec", "nsec")):
            if hasattr(v, sec_a) and hasattr(v, nsec_a):
                try:
                    return int(getattr(v, sec_a)) * 1_000_000_000 + int(getattr(v, nsec_a))
                except Exception:
                    return None
        if hasattr(v, "stamp"):
            return G1Adapter._to_ns(v.stamp)
        if isinstance(v, (int, float)):
            x = float(v)
            if x <= 0:
                return None
            # 단위 추정: 1e18~ = ns, 1e15~ = us, 1e12~ = ms, 그 외 = s
            if x > 1e17:
                return int(x)
            if x > 1e14:
                return int(x * 1e3)
            if x > 1e11:
                return int(x * 1e6)
            return int(x * 1e9)
        return None

    # ── 읽기 ────────────────────────────────────────────────────────────────
    def read_state(self) -> tuple[int, float, Optional[float]]:
        """(timestamp_ns, position_rad, effort_Nm|None).

        타임스탬프는 로봇이 주면 로봇 것을, 아니면 호스트 시계를 쓴다.
        어느 쪽인지는 `self.ts_source` 에 남고 결과 JSON 에 기록된다 — 이 구분이
        `unique_state_frac` / `state_dt_*` 의 의미를 통째로 바꾸기 때문이다.
        """
        if self.dry_run:
            # 궤적 경로가 활성이면 여기서도 플랜트를 전진시킨다.
            # (PART C 는 send 없이 read 만 반복하므로 이게 없으면 응답이 없다)
            if self._sim_traj_at is not None:
                now = time.perf_counter()
                if now >= self._sim_traj_at:
                    dt = now - max(self._sim_t0, self._sim_traj_at)
                    if dt > 0:
                        self._sim_t0 = now
                        alpha = 1.0 - math.exp(-dt / self._sim_tau)
                        self._sim_pos += alpha * (self._sim_target - self._sim_pos)
            return (time.monotonic_ns(), self._sim_pos, 0.0)

        if self._read_via == "states":
            st = self._robot.get_joint_states()
            item = st[self._joint_idx] if isinstance(st, (list, tuple)) else st
            p = float(getattr(item, "position", item))
            ts = self._to_ns(getattr(item, self._ts_attr, None)) if self._ts_attr else None
            return (ts if ts is not None else time.monotonic_ns(), p,
                    getattr(item, "effort", None))

        try:
            pos, pat = _try_patterns(
                "get_joint_positions", self._robot.get_joint_positions,
                [("get_joint_positions()", (), {})],
            )
            self.resolved.setdefault("get_joint_positions", pat)
            p = float(pos[self._joint_idx])
        except Exception:
            st, pat = _try_patterns(
                "get_joint_states", self._robot.get_joint_states,
                [("get_joint_states()", (), {})],
            )
            self.resolved.setdefault("read", pat)
            item = st[self._joint_idx] if isinstance(st, (list, tuple)) else st
            p = float(getattr(item, "position", item))

        return (time.monotonic_ns(), p, None)

    def read_wrench(self) -> Optional[list[float]]:
        """손목 F/T. GATE-1 본체는 아니지만 여기서 한 번에 확인해둔다."""
        if self.dry_run or self._robot is None:
            return None
        E = getattr(self._sdk, "GalbotOneFoxtrotSensor", None)
        if E is None or not hasattr(self._robot, "get_force_sensor_data"):
            return None
        side = "LEFT" if self.joint_name.startswith("left") else "RIGHT"
        mem = f"{side}_WRIST_FORCE"
        if mem not in getattr(E, "__members__", {}):
            return None
        try:
            d = self._robot.get_force_sensor_data(getattr(E, mem))
        except Exception:
            return None
        f = getattr(d, "force", None)
        if f is None:
            return None
        try:
            return [float(x) for x in (list(f) if not hasattr(f, "x")
                                       else [f.x, f.y, f.z])]
        except Exception:
            return None

    # ── 쓰기: 경로 1 — 직접 명령 ────────────────────────────────────────────
    def send_position(self, pos_rad: float) -> None:
        if self.dry_run:
            now = time.perf_counter()
            dt = now - self._sim_t0
            self._sim_t0 = now
            alpha = 1.0 - math.exp(-dt / self._sim_tau)
            self._sim_pos += alpha * (pos_rad - self._sim_pos)
            return

        if "send" in self.resolved:
            self._send_impl(pos_rad)
            return

        # 첫 호출에서만 패턴 사다리를 탄다. 이후엔 확정된 것만 쓴다.
        full = list(self._read_all_positions())
        full[self._joint_idx] = pos_rad
        pats = [
            ("set_joint_positions(names, positions)",
             (self._joint_names, full), {}),
            ("set_joint_positions([name], [pos])",
             ([self.joint_name], [pos_rad]), {}),
            ("set_joint_positions(positions)", (full,), {}),
            ("set_joint_positions({name: pos})", ({self.joint_name: pos_rad},), {}),
        ]
        _, pat = _try_patterns("set_joint_positions",
                               self._robot.set_joint_positions, pats)
        self.resolved["send"] = pat
        print(f"[adapter] 명령 경로 확정 — {pat}")

    def _read_all_positions(self) -> list[float]:
        try:
            pos, _ = _try_patterns("get_joint_positions",
                                   self._robot.get_joint_positions,
                                   [("get_joint_positions()", (), {})])
            return [float(x) for x in pos]
        except Exception:
            return [0.0] * len(self._joint_names)

    def _send_impl(self, pos_rad: float) -> None:
        """확정된 패턴으로만 보낸다 — 측정 루프에서 예외 처리 비용을 없애기 위해."""
        pat = self.resolved["send"]
        if pat.startswith("set_joint_positions(names, positions)"):
            full = self._cached_full
            full[self._joint_idx] = pos_rad
            self._robot.set_joint_positions(self._joint_names, full)
        elif pat.startswith("set_joint_positions([name]"):
            self._robot.set_joint_positions([self.joint_name], [pos_rad])
        elif pat.startswith("set_joint_positions(positions)"):
            full = self._cached_full
            full[self._joint_idx] = pos_rad
            self._robot.set_joint_positions(full)
        else:
            self._robot.set_joint_positions({self.joint_name: pos_rad})

    _cached_full: list[float] = []

    def prime(self) -> None:
        """측정 시작 전 1회 — 패턴 확정과 캐시 워밍을 측정 밖에서 끝낸다."""
        if self.dry_run:
            return
        self._cached_full = self._read_all_positions()
        self.send_position(self._cached_full[self._joint_idx])

    # ── 쓰기: 경로 2 — 궤적 덮어쓰기 ────────────────────────────────────────
    def send_trajectory(self, times_s: list[float], positions_rad: list[float],
                        override: bool = True) -> bool:
        """execute_joint_trajectory + TARGET_TYPE_OVERRIDE.

        chunk 스트리밍의 실제 경로 후보다. 성공 여부를 bool 로 돌려주며,
        시그니처가 안 맞으면 False (측정을 중단시키지 않는다).
        """
        if self.dry_run:
            # 시뮬: 전송 지연 SIM_TRANSPORT 후 마지막 목표로 재조준.
            # PART C 가 이 지연을 되찾아내야 코드가 옳다는 뜻이 된다.
            self._sim_target = positions_rad[-1]
            self._sim_traj_at = time.perf_counter() + SIM_TRANSPORT_S
            return True
        if self._robot is None or not hasattr(self._robot, "execute_joint_trajectory"):
            return False

        sdk = self._sdk
        ttype = getattr(sdk, "TARGET_TYPE_OVERRIDE" if override
                        else "TARGET_TYPE_APPEND", None)
        pats: list[tuple[str, tuple, dict]] = [
            ("execute_joint_trajectory(names, times, positions, TARGET_TYPE_OVERRIDE)",
             ([self.joint_name], times_s, [[p] for p in positions_rad], ttype), {}),
            ("execute_joint_trajectory(names, times, positions)",
             ([self.joint_name], times_s, [[p] for p in positions_rad]), {}),
        ]
        try:
            _, pat = _try_patterns("execute_joint_trajectory",
                                   self._robot.execute_joint_trajectory, pats)
            self.resolved["trajectory"] = pat
            self.traj_supported = True
            return True
        except SDKCallFailed as e:
            if self.traj_supported is None:
                self.traj_supported = False
                print("\n[adapter] 궤적 경로(PART C)를 쓸 수 없습니다 — 시그니처 미확인.")
                print(str(e))
            return False

    _sim_target: float = 0.0
    _sim_traj_at: Optional[float] = None

    def close(self) -> None:
        if self.dry_run or self._robot is None:
            return
        for m in ("disconnect", "shutdown", "close", "stop"):
            if hasattr(self._robot, m):
                try:
                    getattr(self._robot, m)()
                    return
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 통계
# ─────────────────────────────────────────────────────────────────────────────


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


@dataclass
class RateResult:
    target_hz: float
    achieved_hz: float
    n_commands: int
    # 이 램프가 read 를 포함했는가. False 면 achieved_hz 는 순수 송신 천장이다.
    with_read: bool
    # 관측 가능한 상태 갱신률 = unique_state_frac × achieved_hz.
    # 로봇 타임스탬프가 있을 때만 의미가 있다.
    state_hz: float
    # 명령 송신 주기 (우리가 얼마나 규칙적으로 보냈는가)
    send_p50_ms: float
    send_p99_ms: float
    send_p999_ms: float
    send_max_ms: float
    jitter_ratio: float               # p99/p50 — 건강한 스택은 ~1.0
    # 상태 갱신 (로봇이 얼마나 규칙적으로 응답하는가)
    state_dt_p50_ms: float
    state_dt_p99_ms: float
    unique_state_frac: float          # 새 timestamp 비율. 낮으면 상태가 stale
    dropped_est: int


def measure_rate(ad: G1Adapter, target_hz: float, duration_s: float,
                 amplitude_rad: float, center_rad: float,
                 do_read: bool = True) -> RateResult:
    """PART A — 목표 rate로 작은 사인을 스트리밍하며 실제 달성치와 jitter를 측정.

    `do_read=False` 면 송신만 한다. 왜 나눠 재는가:
    한 iteration 에서 send 와 read 를 둘 다 하면 achieved_hz 가 **두 호출의 합산
    주기**를 반영한다. 그러면 천장이 낮게 나왔을 때 그게 명령 경로 탓인지 상태
    조회 탓인지 구분할 수 없다. 두 번 돌려 병기하면 그 오귀속이 사라진다.
    """
    period = 1.0 / target_hz
    send_gaps: list[float] = []
    state_ts: list[int] = []
    state_gaps: list[float] = []

    t_start = time.perf_counter()
    next_t = t_start
    last_send = None
    last_ts = None
    n = 0

    while (time.perf_counter() - t_start) < duration_s:
        now = time.perf_counter()
        if now < next_t:
            # busy-wait: sleep은 이 정밀도에서 신뢰할 수 없다
            continue

        phase = 2.0 * math.pi * 0.5 * (now - t_start)     # 0.5 Hz 소진폭 사인
        ad.send_position(center_rad + amplitude_rad * math.sin(phase))
        n += 1

        if last_send is not None:
            send_gaps.append((now - last_send) * 1e3)
        last_send = now

        if do_read:
            ts, _pos, _eff = ad.read_state()
            state_ts.append(ts)
            if last_ts is not None and ts != last_ts:
                state_gaps.append((ts - last_ts) / 1e6)
            last_ts = ts

        next_t += period
        # 뒤처지면 따라잡기를 포기 (스케줄 붕괴 방지)
        if next_t < now:
            next_t = now + period

    elapsed = time.perf_counter() - t_start
    sg = sorted(send_gaps)
    stg = sorted(state_gaps)
    uniq = (len(set(state_ts)) / len(state_ts)) if state_ts else float("nan")

    p50 = pct(sg, 0.50) if sg else float("nan")
    p99 = pct(sg, 0.99) if sg else float("nan")
    achieved = n / elapsed

    return RateResult(
        target_hz=target_hz,
        achieved_hz=achieved,
        n_commands=n,
        with_read=do_read,
        state_hz=(uniq * achieved) if state_ts else float("nan"),
        send_p50_ms=p50,
        send_p99_ms=p99,
        send_p999_ms=pct(sg, 0.999) if sg else float("nan"),
        send_max_ms=sg[-1] if sg else float("nan"),
        jitter_ratio=(p99 / p50) if (sg and p50 > 0) else float("nan"),
        state_dt_p50_ms=pct(stg, 0.50) if stg else float("nan"),
        state_dt_p99_ms=pct(stg, 0.99) if stg else float("nan"),
        unique_state_frac=uniq,
        dropped_est=max(0, int(target_hz * elapsed) - n),
    )


@dataclass
class BodePoint:
    freq_hz: float
    gain_db: float
    phase_deg: float
    n_samples: int


def measure_bode(ad: G1Adapter, freqs: list[float], amplitude_rad: float,
                 center_rad: float, cycles: int = 12,
                 cmd_hz: float = 200.0) -> list[BodePoint]:
    """PART B — 주파수별 사인 주입 후 상관법으로 크기/위상 추출."""
    out: list[BodePoint] = []
    period_cmd = 1.0 / cmd_hz

    for f in freqs:
        dur = max(1.5, cycles / f)
        t0 = time.perf_counter()
        next_t = t0
        # 직교 상관 누산기
        acc_i = acc_q = 0.0
        n = 0

        while True:
            now = time.perf_counter()
            t = now - t0
            if t >= dur:
                break
            if now < next_t:
                continue

            cmd = center_rad + amplitude_rad * math.sin(2 * math.pi * f * t)
            ad.send_position(cmd)
            _ts, pos, _eff = ad.read_state()

            meas = pos - center_rad
            acc_i += meas * math.sin(2 * math.pi * f * t)
            acc_q += meas * math.cos(2 * math.pi * f * t)
            n += 1
            next_t += period_cmd
            if next_t < now:
                next_t = now + period_cmd

        if n < 8:
            out.append(BodePoint(f, float("nan"), float("nan"), n))
            continue

        # 응답 진폭/위상
        i_c = 2.0 * acc_i / n
        q_c = 2.0 * acc_q / n
        mag = math.hypot(i_c, q_c)
        gain = mag / amplitude_rad if amplitude_rad > 0 else float("nan")
        out.append(BodePoint(
            freq_hz=f,
            gain_db=20.0 * math.log10(gain) if gain > 1e-12 else float("-inf"),
            phase_deg=math.degrees(math.atan2(q_c, i_c)),
            n_samples=n,
        ))
        print(f"  {f:6.2f} Hz   gain {out[-1].gain_db:+6.2f} dB   "
              f"phase {out[-1].phase_deg:+7.1f}°   n={n}")

    return out


@dataclass
class ReplanResult:
    """궤적 덮어쓰기 재계획 지연 — chunk 스트리밍에서 실제로 중요한 양."""
    n_trials: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    detect_threshold_rad: float
    poll_hz: float
    failures: int


def measure_replan(ad: G1Adapter, trials: int, amplitude_rad: float,
                   center_rad: float, hold_s: float = 0.6,
                   horizon_s: float = 0.4) -> Optional[ReplanResult]:
    """PART C — `execute_joint_trajectory` + TARGET_TYPE_OVERRIDE 재계획 지연.

    왜 PART A 와 별개인가
    ─────────────────────
    PART A 는 "Python 이 얼마나 빨리 때릴 수 있나"를 잰다. 그런데 SDK 에는
    궤적 큐를 통째로 덮어쓰는 경로가 따로 있다. 그 경로에서는 상위가 5Hz 로만
    보내도 **온보드가 보간**하므로, 상위 rate 천장이 낮다는 사실 자체는
    분리 전제를 죽이지 않는다.

    대신 그 경로에서 진짜 병목은 **덮어쓰기가 실제 운동에 반영되기까지의 지연**이다.
    이게 크면 접촉 반응이 늦고, 그건 rate 를 아무리 올려도 해결되지 않는다.
    action chunk 를 5Hz 로 갈아끼우는 설계에서 이 값이 200ms 를 넘으면
    "청크 경계마다 이미 지난 상황에 반응"하게 된다.

    측정: 중심에서 정지 → 계단 궤적을 OVERRIDE 로 투입 → 위치가 임계를 넘는
    첫 순간까지의 시간. 폴링으로 검출하므로 폴링 주기가 분해능 하한이다.
    """
    if not ad.send_trajectory([horizon_s], [center_rad]):
        return None

    thresh = 0.10 * abs(amplitude_rad)
    lat: list[float] = []
    fails = 0
    poll_dt = 0.001
    # 검출 구간만 따로 센다. 홀드 구간까지 섞으면 poll_hz 가 실제 분해능이 아니라
    # busy-spin 속도가 되어 의미를 잃는다.
    det_polls = 0
    det_time = 0.0

    for k in range(trials):
        # 홀드: 중심으로 안정화. sleep 을 넣어 SDK 를 불필요하게 두들기지 않는다.
        ad.send_trajectory([horizon_s], [center_rad])
        t_end = time.perf_counter() + hold_s
        while time.perf_counter() < t_end:
            ad.read_state()
            time.sleep(poll_dt)
        _, base, _ = ad.read_state()

        # 계단을 OVERRIDE 로 투입
        target = center_rad + (amplitude_rad if k % 2 == 0 else -amplitude_rad)
        t0 = time.perf_counter()
        if not ad.send_trajectory([horizon_s], [target]):
            fails += 1
            continue

        hit = None
        deadline = t0 + 2.0
        while time.perf_counter() < deadline:
            _, p, _ = ad.read_state()
            det_polls += 1
            if abs(p - base) >= thresh:
                hit = time.perf_counter()
                break
            time.sleep(poll_dt)
        det_time += time.perf_counter() - t0
        if hit is None:
            fails += 1
            continue
        lat.append((hit - t0) * 1e3)
        print(f"  trial {k+1:2d}/{trials}  {lat[-1]:7.2f} ms")

    if not lat:
        return ReplanResult(trials, float("nan"), float("nan"), float("nan"),
                            thresh, 0.0, fails)

    s = sorted(lat)
    return ReplanResult(
        n_trials=len(lat),
        latency_p50_ms=pct(s, 0.50),
        latency_p95_ms=pct(s, 0.95),
        latency_max_ms=s[-1],
        detect_threshold_rad=thresh,
        poll_hz=det_polls / det_time if det_time > 0 else float("nan"),
        failures=fails,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 판정
# ─────────────────────────────────────────────────────────────────────────────


def verdict(rates: list[RateResult], bode: list[BodePoint],
            replan: Optional[ReplanResult] = None,
            ts_source: str = "unknown") -> dict:
    """GATE-1 판정.

    2026-07-31 개정 — SDK 표면 확인 후 **경로가 둘**임이 드러나 판정을 분리했다.

      경로 A (direct)     : 관절 위치 명령을 상위에서 고속 반복
                            (현재 구현은 set_joint_positions 사다리를 탄다.
                             set_joint_commands / _batch 는 아직 미측정이다)
      경로 B (trajectory) : execute_joint_trajectory + TARGET_TYPE_OVERRIDE

    이전 판정은 경로 A 만 가정하고 "천장 <50Hz → 분리 전제 붕괴"라고 썼다.
    경로 B 가 실제로 동작한다면 그 논리는 성립하지 않는다 — 상위가 느려도
    온보드가 보간하기 때문이다. 그래서 **경로 B 가 살아 있으면 replan latency 가
    주 판정 기준**이 되고, 경로 A 천장은 보조 지표로 내려간다.

    ⚠️ 천장의 신뢰성
    ───────────────
    `achieved_hz` 는 **우리가 함수를 호출한 횟수**를 센 값이다. 송신 API 가
    fire-and-forget 비동기 publish 라면 로봇이 실제로 실행했는지와 무관하게
    올라가고, 그러면 천장이 부풀려져 **거짓 PASS** 가 난다.

    이걸 반증할 유일한 증거는 **로봇이 제공한 타임스탬프로 본 상태 갱신률**이다.
    로봇 타임스탬프가 없으면(호스트 시계 대체) 그 증거가 원리적으로 없으므로
    천장을 PASS 로 승인하지 않고 `unconfirmed` 로 강등한다.
    """
    device_ts = ts_source.startswith("device")

    # 경로 A 천장은 send-only 램프가 있으면 그쪽을 쓴다 (read 오염 제거)
    send_only = [r for r in rates if not r.with_read]
    pool = send_only or rates

    ok = [r for r in pool
          if r.achieved_hz >= 0.9 * r.target_hz
          and (math.isnan(r.jitter_ratio) or r.jitter_ratio < 2.0)]
    ceiling = max((r.target_hz for r in ok), default=0.0)

    # 관측 가능한 천장 — 로봇 타임스탬프가 있을 때만 의미가 있다
    obs = [r for r in rates
           if r.with_read and not math.isnan(r.state_hz)
           and r.state_hz >= 0.9 * r.target_hz]
    state_ceiling = max((r.target_hz for r in obs), default=0.0)
    ceiling_confirmed = bool(device_ts and state_ceiling >= ceiling > 0)

    bw = None
    for p in bode:
        if not math.isnan(p.gain_db) and p.gain_db <= -3.0:
            bw = p.freq_hz
            break

    # ── 경로 B 가 살아 있으면 그쪽이 주 기준 ────────────────────────────────
    traj_alive = replan is not None and not math.isnan(replan.latency_p50_ms)
    if traj_alive:
        p95 = replan.latency_p95_ms
        if p95 <= 50:
            v = "PASS (경로 B) — 궤적 덮어쓰기 재계획이 빠름"
            impl = ("5Hz chunk 스트리밍이 성립. 상위 rate 천장이 낮아도 무방하다. "
                    "온보드 보간이 받쳐주므로 분리 유지 가능.")
        elif p95 <= 200:
            v = "MARGINAL (경로 B) — 재계획 지연 경계"
            impl = (f"재계획 p95 {p95:.0f}ms. 청크 주기를 이보다 길게 잡아야 하며, "
                    "그만큼 접촉 반응이 늦는다. 빠른 충돌 반응은 포기하고 "
                    "느린 삽입/닦기로 작업 범위를 한정할 것.")
        else:
            v = "FAIL (경로 B) — 재계획이 너무 느림"
            impl = (f"재계획 p95 {p95:.0f}ms 는 청크 경계마다 이미 지난 상황에 "
                    "반응한다는 뜻이다. 경로 A 천장을 함께 보고 판단할 것.")
    elif ceiling >= 100 and ceiling_confirmed:
        v = "PASS (경로 A) — admittance 작동 가능. 분리 유지 가능"
        impl = ("빠른 Model 2가 의미를 가짐. 인터페이스 수정(그리퍼·dt·psi·상대 포즈) 후 "
                "residual 구조로 진행.")
    elif ceiling >= 100:
        v = "PASS 미확인 (경로 A) — 천장은 높으나 로봇이 실행했다는 증거가 없음"
        impl = (f"호출 rate {ceiling:.0f}Hz 는 나왔지만, 상태 갱신률로 확인되지 않았다"
                f"{'' if device_ts else ' (로봇 타임스탬프 자체가 없음)'}. "
                "송신이 fire-and-forget 비동기 publish 라면 이 값은 로봇의 성능이 "
                "아니라 우리 for 문의 속도다. 이 상태로 PASS 를 인정하면 안 된다. "
                "probe-live 로 상태 타임스탬프 유무를 먼저 확정할 것.")
    elif ceiling >= 50:
        v = "MARGINAL (경로 A) — 경계"
        impl = ("admittance 대역폭이 5-8Hz 수준. 느린 삽입/닦기는 되고 충격 흡수는 안 됨. "
                "접촉 작업 범위를 서면으로 한정할 것.")
    else:
        v = "FAIL — 두 경로 모두 부족"
        impl = ("경로 A 천장이 낮고 경로 B(궤적 덮어쓰기)도 확보되지 않았다. "
                "단일 모델 + 벤더 WBC + closed-form S-R-S IK로 전환. "
                "메인 문서 REV.2 최종 권고 참조. "
                "단, 경로 B 실패가 '미지원'이 아니라 '시그니처 미확인' 때문이면 "
                "probe_sdk.py 로 확정한 뒤 재측정할 것 — 판정이 뒤집힐 수 있다.")

    return {
        "command_ceiling_hz": ceiling,
        "command_ceiling_source": "send-only 램프" if send_only else "send+read 램프",
        "state_ceiling_hz": state_ceiling,
        "ceiling_confirmed": ceiling_confirmed,
        "state_timestamp_source": ts_source,
        "tracking_bandwidth_3db_hz": bw,
        "replan_latency_p50_ms": replan.latency_p50_ms if replan else None,
        "replan_latency_p95_ms": replan.latency_p95_ms if replan else None,
        "trajectory_path_available": bool(traj_alive),
        "primary_criterion": "경로 B (replan latency)" if traj_alive
                             else "경로 A (command ceiling)",
        "verdict": v,
        "implication": impl,
    }


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="GATE-1: G1 폐루프 대역폭 측정")
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 스크립트 검증")
    ap.add_argument("--host", default="unspecified",
                    choices=["onboard", "external", "unspecified"],
                    help="측정 위치 (결과에 기록됨)")
    ap.add_argument("--joint", default="left_arm_joint4")
    ap.add_argument("--amp-deg", type=float, default=2.0, help="진폭 (도)")
    ap.add_argument("--center-deg", type=float, default=0.0)
    ap.add_argument("--dwell", type=float, default=6.0, help="rate당 측정 시간 (초)")
    ap.add_argument("--rates", default="10,20,30,50,75,100,150,200,300,500")
    ap.add_argument("--bode-freqs", default="0.5,1,2,3,5,7,10")
    ap.add_argument("--skip-bode", action="store_true")
    ap.add_argument("--ramp-mode", default="both",
                    choices=["both", "send-only", "send-read"],
                    help="PART A 램프 방식. both 면 두 번 돌려 천장을 병기한다")
    ap.add_argument("--skip-replan", action="store_true",
                    help="PART C(궤적 덮어쓰기 재계획 지연) 건너뜀")
    ap.add_argument("--replan-trials", type=int, default=12)
    ap.add_argument("--out", default="gate1_results.json")
    args = ap.parse_args()

    amp = math.radians(args.amp_deg)
    center = math.radians(args.center_deg)
    rates = [float(x) for x in args.rates.split(",")]
    bfreqs = [float(x) for x in args.bode_freqs.split(",")]

    print("=" * 72)
    print("GATE-1 — Galbot G1 폐루프 명령 대역폭 측정")
    print("=" * 72)
    print(f"  관절     : {args.joint}")
    print(f"  진폭     : ±{args.amp_deg}° (±{amp:.4f} rad)")
    print(f"  측정 위치: {args.host}")
    if args.dry_run:
        print("  모드     : DRY-RUN (tau=15ms 시뮬레이션 플랜트)")
    else:
        print("\n  ⚠️  팔 주변을 비우고 e-stop을 손 닿는 곳에 두십시오.")
        print("  ⚠️  이 머신에서 다른 워크로드를 돌리지 마십시오 (jitter 오염).")
        try:
            input("\n  준비되면 Enter, 중단하려면 Ctrl-C: ")
        except KeyboardInterrupt:
            print("\n중단됨")
            return 1
    print()

    ad = G1Adapter(dry_run=args.dry_run, joint_name=args.joint)
    ad.prime()
    results = {
        "meta": {
            "host": args.host,
            "joint": ad.joint_name,
            "amp_deg": args.amp_deg,
            "dwell_s": args.dwell,
            "dry_run": args.dry_run,
            "unix_time": time.time(),
            # 어떤 호출 패턴이 실제로 통했는지 — 재현과 어댑터 확정에 필요
            "resolved_calls": dict(ad.resolved),
            # ⚠️ 이 값이 host_monotonic 이면 state_dt_*/unique_state_frac 은
            #    로봇이 아니라 우리 루프를 재는 값이다
            "state_timestamp_source": ad.ts_source,
            "sdk_joint_count": len(ad._joint_names) or None,
        }
    }

    try:
        rate_results: list[RateResult] = []
        # send-only 를 먼저 — 순수 명령 천장. 그다음 send+read 로 관측 가능성까지.
        # 나눠 재지 않으면 천장이 낮게 나왔을 때 명령 탓인지 조회 탓인지 모른다.
        modes = ([(False, "send-only  (순수 명령 천장)"),
                  (True, "send+read  (관측 포함)")]
                 if args.ramp_mode == "both"
                 else [(args.ramp_mode == "send-read", args.ramp_mode)])

        for do_read, label in modes:
            print(f"\n── PART A — 명령 rate 램프 · {label} " + "─" * 18)
            print(f"{'target':>8} {'achieved':>9} {'p50':>8} {'p99':>8} {'p99.9':>8} "
                  f"{'max':>8} {'jitter':>7} {'state_uniq':>11} {'state_hz':>9}")
            for hz in rates:
                r = measure_rate(ad, hz, args.dwell, amp, center, do_read=do_read)
                rate_results.append(r)
                uq = "—" if math.isnan(r.unique_state_frac) else f"{r.unique_state_frac:.2f}"
                sh = "—" if math.isnan(r.state_hz) else f"{r.state_hz:.1f}"
                print(f"{r.target_hz:8.0f} {r.achieved_hz:9.1f} {r.send_p50_ms:8.2f} "
                      f"{r.send_p99_ms:8.2f} {r.send_p999_ms:8.2f} {r.send_max_ms:8.2f} "
                      f"{r.jitter_ratio:7.2f} {uq:>11} {sh:>9}")
                # 목표의 절반도 못 내면 천장을 지난 것
                if r.achieved_hz < 0.5 * r.target_hz:
                    print(f"  → 달성률 50% 미만. 천장 통과로 보고 램프 중단.")
                    break
        results["rate_ramp"] = [asdict(r) for r in rate_results]

        bode_results: list[BodePoint] = []
        if not args.skip_bode:
            print("\n── PART B — 스윕 사인 Bode " + "─" * 40)
            bode_results = measure_bode(ad, bfreqs, amp, center)
            results["bode"] = [asdict(p) for p in bode_results]

        replan: Optional[ReplanResult] = None
        if not args.skip_replan:
            print("\n── PART C — 궤적 덮어쓰기 재계획 지연 " + "─" * 26)
            print("   execute_joint_trajectory + TARGET_TYPE_OVERRIDE")
            replan = measure_replan(ad, args.replan_trials, amp, center)
            if replan is None:
                print("   건너뜀 — 궤적 경로를 쓸 수 없습니다.")
                print("   (미지원인지 시그니처 미확인인지는 위 진단을 보십시오.")
                print("    후자면 probe_sdk.py 로 확정 후 재측정하면 판정이 바뀔 수 있습니다.)")
            else:
                results["replan"] = asdict(replan)
                results["meta"]["trajectory_call"] = ad.resolved.get("trajectory")
                print(f"   p50 {replan.latency_p50_ms:.2f} ms · "
                      f"p95 {replan.latency_p95_ms:.2f} ms · "
                      f"max {replan.latency_max_ms:.2f} ms  "
                      f"(폴링 {replan.poll_hz:.0f} Hz, 실패 {replan.failures})")
                if args.dry_run:
                    err = abs(replan.latency_p50_ms - SIM_TRANSPORT_S * 1e3)
                    ok = err < 15.0
                    print(f"   [드라이런 자체검증] 주입한 전송 지연 "
                          f"{SIM_TRANSPORT_S*1e3:.0f} ms 를 "
                          f"{replan.latency_p50_ms:.1f} ms 로 복원 "
                          f"→ {'✅ 통과' if ok else '❌ 실패'}")

        # 손목 F/T 가 실제로 값을 주는지 여기서 한 번 확인해둔다 (회사 방문 1회 절약)
        w = ad.read_wrench()
        if w is not None:
            results["wrench_sample"] = w
            print(f"\n  손목 F/T 표본: {[round(x, 3) for x in w]}")

        v = verdict(rate_results, bode_results, replan, ts_source=ad.ts_source)
        results["verdict"] = v

        print("\n" + "=" * 72)
        print("판정")
        print("=" * 72)
        print(f"  주 기준            : {v['primary_criterion']}")
        print(f"  명령 rate 천장     : {v['command_ceiling_hz']:.0f} Hz   (경로 A, "
              f"{v['command_ceiling_source']})")
        print(f"  관측 가능 천장     : {v['state_ceiling_hz']:.0f} Hz   "
              f"→ 천장 확인 {'✅' if v['ceiling_confirmed'] else '❌ 미확인'}")
        bw = v["tracking_bandwidth_3db_hz"]
        print(f"  추종 대역폭(-3dB)  : {bw if bw else '측정 범위 내 없음'} Hz")
        if v["replan_latency_p95_ms"] is not None:
            print(f"  재계획 지연 p95    : {v['replan_latency_p95_ms']:.1f} ms  (경로 B)")
        else:
            print(f"  재계획 지연        : 측정 못 함 (경로 B 미확보)")
        if ad.ts_source == "host_monotonic":
            print(f"\n  ⚠️ 상태 타임스탬프가 호스트 시계입니다 — state_dt_*/unique_state_frac")
            print(f"     은 로봇 응답이 아니라 우리 루프를 잰 값입니다. 그래서 천장을")
            print(f"     '확인됨'으로 승격할 수 없습니다.")
        print(f"\n  → {v['verdict']}")
        print(f"     {v['implication']}")

    finally:
        ad.close()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {args.out}")
    print("\n다음: 이 결과를 docs/PLAN.md §4 판정표에 대조하십시오.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
