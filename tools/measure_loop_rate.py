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
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdk_entry  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# SDK 어댑터
#
# 2026-07-31 — **실물 시그니처 확정.** `make probe` 로 3090 에서 pybind11 docstring
# 246/296 을 복원했다. 더 이상 추측하지 않는다.
#
#   설치: /opt/galbot/galbot_sdk/linux-x86_64-gcc940/
#   ⚠️ import 전 반드시:  source .../setup.sh   (bash 문법이므로 sh 로는 안 된다)
#
# 확정된 시그니처 중 측정에 직결되는 것들
# ────────────────────────────────────────
#   set_joint_commands(joint_commands: [JointCommand], joint_groups=[],
#                      joint_names=[], time_from_start_s=10.0) -> ControlStatus
#       ↳ 공식 문서 표현: "high-frequency, no interpolation".
#         **이것이 스트리밍 경로다.** 벤더 예제는 time_from_start_s=0.0 을 쓴다.
#   set_joint_commands_batch(trajectory: Trajectory) -> ControlStatus
#       ↳ 공식 문서 표현: "multiple future frames".
#         **이것이 action chunk 경로다.**
#   set_joint_positions(joint_positions, joint_groups=[], joint_names=[],
#                       is_blocking=True, speed_rad_s=0.2, timeout_s=15.0)
#                      -> ControlStatus
#   execute_joint_trajectory(trajectory: Trajectory, is_blocking=True)
#                      -> ControlStatus
#   set_joint_commands_batch(trajectory: Trajectory) -> ControlStatus
#   get_joint_positions(joint_groups, joint_names=[]) -> [float]
#   get_joint_states(joint_group_vec, joint_names_vec=[]) -> [JointState]
#   get_joint_names(only_active_joint=True, joint_groups=[]) -> [str]
#   get_joint_group_names() -> [str]
#   check_trajectory_execution_status(joint_groups=[]) -> [TrajectoryControlStatus]
#   stop_trajectory_execution() -> ControlStatus
#   get_force_sensor_data(sensor_type: GalbotOneFoxtrotSensor) -> dict
#   init(enable_sensor_set: {SensorType} = set()) -> bool
#
# 이 시그니처들이 뒤집은 것 — 셋 다 측정을 무의미하게 만들 수 있었다
# ───────────────────────────────────────────────────────────────────
#  1. `is_blocking` 기본값이 **True** 다. 그냥 부르면 rate 램프가 명령 전송률이
#     아니라 **모션 완료 시간**을 잰다. 측정 경로에서는 전부 False 로 넘긴다.
#  2. `get_joint_positions` 의 `joint_groups` 는 **필수 인자**다 (기본값 없음).
#     무인자 호출은 TypeError 다.
#  3. `set_joint_commands` 의 `time_from_start_s` 기본값이 **10.0 초**다.
#     고속 스트리밍에서 그대로 두면 매 명령이 "10초에 걸쳐 도달" 을 뜻해
#     사실상 움직이지 않는다. 벤더 고주파 예제가 쓰는 **0.0**(보간 없음)이
#     스트리밍의 올바른 값이다.
#
# ⚠️ 공식 문서(1.7.0)와 실물 바이너리의 **인자 이름이 다르다.** 바이너리가 진실이다:
#       문서 joint_pos          → 실물 joint_positions
#       문서 max_speed          → 실물 speed_rad_s
#       문서 time_from_start_sec→ 실물 time_from_start_s
#       문서 is_block           → 실물 is_blocking
#    그래서 이 파일은 문서를 그대로 베끼지 않고 probe 로 뽑은 시그니처를 쓴다.
#
# 관측 가능성 — 여기서 방향이 바뀌었다
# ────────────────────────────────────
#   `JointState` 에는 **timestamp 가 없다** (acceleration/current/effort/
#   position/velocity 뿐). 즉 "로봇 타임스탬프로 상태 갱신률을 본다" 는 원래
#   계획은 이 SDK 에서 **불가능**하다.
#   대신 모든 `set_*` 가 **ControlStatus 를 반환**한다. 이게 대체 증거 채널이다:
#   명령이 SUCCESS 로 수락됐는지 호출별로 알 수 있다. 완전한 "실행됨" 증거는
#   아니지만 (fire-and-forget 이면 SUCCESS 도 즉시 돌아올 수 있다),
#   PUBLISH_FAIL / COMM_DISCONNECTED / TIMEOUT 은 확실히 잡아낸다.
#   궤적 경로에는 `check_trajectory_execution_status` 라는 더 강한 채널이 있다
#   (RUNNING / COMPLETED / STOPPED_UNREACHED / ERROR).
#
# ⚠️ TARGET_TYPE_* 는 `execute_joint_trajectory` 인자가 아니다.
#    실제로는 TargetConfig(target_type/target_data/target_sampling/...) →
#    TargetGroupTrajectory → SingoriXTarget → publish_target 경로 소속이다.
#    값이 **비트 플래그**다: OVERRIDE(10) = CLEAR(2)|APPEND(8),
#    PROVERRIDE(14) = CLEAR(2)|PREPENDNOW(4)|APPEND(8).
#    SingoriXTarget 의 필드 구성이 미확인이라 이 경로는 아직 못 탄다.
#    그래서 PART C 는 `execute_joint_trajectory` 재발행으로 대체 측정한다.
# ─────────────────────────────────────────────────────────────────────────────

SDK_SETUP = "/opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh"

# 드라이런 전용 — 궤적 재발행의 가짜 전송 지연. PART C 가 이 값을 되찾아내면
# 측정 코드가 옳다는 뜻이다 (로봇 없이 검증하는 방법).
SIM_TRANSPORT_S = 0.035

# 팔 관절이 속한 그룹 이름 후보. 실물은 get_joint_group_names() 로 확인한다.
_ARM_GROUP_HINTS = ("left_arm", "right_arm", "arm")


def _doc(obj: object) -> str:
    return (getattr(obj, "__doc__", None) or "(docstring 없음)").strip()[:900]


class G1Adapter:
    """GalbotSDK 래퍼 — 실물 시그니처 기준.

    측정 경로에서는 **절대 blocking 호출을 쓰지 않는다.** is_blocking=True 로
    부르면 명령 전송률이 아니라 모션 완료 시간을 재게 된다.
    """

    def __init__(self, dry_run: bool = False, joint_name: str = "left_arm_joint4",
                 tau_s: float = 0.015, horizon_s: float = 0.0,
                 send_api: str = "commands"):
        self.dry_run = dry_run
        self.joint_name = joint_name
        self.horizon_s = horizon_s          # time_from_start_s
        self.send_api = send_api            # "commands" | "positions"
        # PART C 경로. batch = 공식 문서의 "multiple future frames" = chunk 경로.
        self.traj_api = "trajectory"        # "trajectory" | "batch"
        self._robot = None
        self._motion = None
        self._sdk = None
        self._joint_names: list[str] = []
        self._groups: list[str] = []
        self._arm_group: Optional[str] = None
        self._joint_idx: Optional[int] = None
        self._sim_pos = 0.0
        self._sim_tau = tau_s
        self._sim_t0 = time.perf_counter()

        self.resolved: dict[str, str] = {}
        # JointState 에 timestamp 가 없음이 확인됐으므로 실물은 항상 host 다.
        self.ts_source = "sim" if dry_run else "host_monotonic"
        self.traj_supported: Optional[bool] = None
        # ControlStatus ack 집계 — 타임스탬프 대신 쓰는 관측 채널
        self.ack_ok = 0
        self.ack_total = 0
        self.ack_bad: dict[str, int] = {}

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
                f"  SDK 환경을 먼저 로드해야 합니다 (bash 필요):\n"
                f"      source {SDK_SETUP}\n\n"
                "  - SDK 는 Linux 전용입니다. Mac 에서는 --dry-run 만 됩니다.\n"
                "  - 경로가 다르면:  find /opt -maxdepth 6 -type d -name galbot_sdk\n"
            )
        self._sdk = sdk
        print(f"[adapter] galbot_sdk {getattr(sdk, '__file__', '?')}")

        # ⚠️ GalbotRobot() 직접 생성은 실패할 수 있다 — pybind11 이 py::init<>()
        #    없이 바인딩한 클래스는 "No constructor defined!" 가 난다 (GalbotMotion
        #    에서 실측). 그래서 획득 전략을 sdk_entry 에 위임한다.
        try:
            self._robot, how = sdk_entry.acquire(sdk, "GalbotRobot")
        except sdk_entry.EntryNotFound as e:
            sys.exit(f"[adapter] {e}\n\n"
                     "→ python tools/probe_sdk.py --entry 로 진입점을 찾으십시오.")
        print(f"[adapter] GalbotRobot 획득: {how}")
        # init(enable_sensor_set) — 센서를 안 켜면 카메라 스트림 비용이 안 든다.
        # F/T 는 SensorType 이 아니라 GalbotOneFoxtrotSensor 라 여기 포함되지 않는다.
        if not self._robot.init(set()):
            sys.exit("[adapter] GalbotRobot.init() 이 False 를 반환했습니다. "
                     "로봇 전원/네트워크를 확인하십시오.")
        self.resolved["ctor"] = f"{how}; init(set())"
        print(f"[adapter] init OK · is_running={self._robot.is_running()}")

        self._groups = list(self._robot.get_joint_group_names())
        print(f"[adapter] joint group {len(self._groups)}개: {self._groups}")

        self._joint_names = list(self._robot.get_joint_names(only_active_joint=True))
        print(f"[adapter] active joint {len(self._joint_names)}개")

        self._resolve_joint_index()
        self._resolve_arm_group()
        self._acquire()

    def _resolve_joint_index(self) -> None:
        """관절을 **이름으로** 잡는다. 인덱스 가정은 조용한 오염의 원천이다."""
        names = self._joint_names
        if self.joint_name in names:
            self._joint_idx = names.index(self.joint_name)
        else:
            key = self.joint_name.replace("_", "").lower()
            hits = [i for i, n in enumerate(names)
                    if n.replace("_", "").lower() == key]
            if not hits:
                sys.exit(
                    f"[adapter] 관절 '{self.joint_name}' 을 찾을 수 없습니다.\n"
                    f"  SDK 가 보고한 관절 {len(names)}개:\n"
                    + "\n".join(f"    [{i:2d}] {n}" for i, n in enumerate(names))
                    + "\n\n  --joint <이름> 으로 지정하십시오."
                )
            self._joint_idx = hits[0]
            self.joint_name = names[hits[0]]
            print(f"[adapter] 이름 정규화 → '{self.joint_name}'")
        print(f"[adapter] 대상 관절 '{self.joint_name}' index {self._joint_idx}")

    def _resolve_arm_group(self) -> None:
        """대상 관절이 속한 joint group. get_joint_positions 의 필수 인자다."""
        for g in self._groups:
            try:
                if self.joint_name in self._robot.get_joint_names(
                        only_active_joint=True, joint_groups=[g]):
                    self._arm_group = g
                    break
            except Exception:
                continue
        if self._arm_group is None:
            side = self.joint_name.split("_")[0]
            for g in self._groups:
                if g.startswith(side) and "arm" in g:
                    self._arm_group = g
                    break
        if self._arm_group is None:
            sys.exit(f"[adapter] '{self.joint_name}' 이 속한 joint group 을 못 찾음. "
                     f"그룹 목록: {self._groups}")
        # 그룹 내 인덱스 — get_joint_positions(그룹) 결과에서 우리 관절 위치
        gnames = list(self._robot.get_joint_names(
            only_active_joint=True, joint_groups=[self._arm_group]))
        self._group_names = gnames
        self._group_idx = gnames.index(self.joint_name)
        print(f"[adapter] joint group '{self._arm_group}' "
              f"({len(gnames)}개) 내 index {self._group_idx}")

    def _acquire(self) -> None:
        """컨트롤러 확보. 이게 없으면 명령이 조용히 무시될 수 있다."""
        try:
            st = self._robot.acquire_controller(self._arm_group)
            print(f"[adapter] acquire_controller('{self._arm_group}') → "
                  f"{self._name_of(st)}")
            self.resolved["acquire"] = self._name_of(st)
        except Exception as e:
            print(f"[adapter] ⚠️ acquire_controller 실패: {type(e).__name__}: {e}")
            print("          명령이 무시될 수 있습니다. 활성 컨트롤러:")
            try:
                print(f"          {self._robot.get_active_controller(self._arm_group)}")
            except Exception:
                pass

    def _name_of(self, status: object) -> str:
        return getattr(status, "name", str(status))

    def _track(self, status: object) -> None:
        """ControlStatus ack 집계. timestamp 가 없는 이 SDK 의 관측 채널."""
        self.ack_total += 1
        nm = self._name_of(status)
        if nm == "SUCCESS":
            self.ack_ok += 1
        else:
            self.ack_bad[nm] = self.ack_bad.get(nm, 0) + 1

    def ack_frac(self) -> float:
        return (self.ack_ok / self.ack_total) if self.ack_total else float("nan")

    def reset_ack(self) -> None:
        self.ack_ok = self.ack_total = 0
        self.ack_bad = {}

    # ── 읽기 ────────────────────────────────────────────────────────────────
    def read_state(self) -> tuple[int, float, Optional[float]]:
        """(timestamp_ns, position_rad, effort_Nm|None).

        ⚠️ 타임스탬프는 **항상 호스트 시계**다. `JointState` 에 timestamp 필드가
        없음이 실물 확인됐다 (acceleration/current/effort/position/velocity 뿐).
        따라서 state_dt_* / unique_state_frac 은 로봇 응답이 아니라 우리 루프를
        재는 값이고, 판정에서 천장은 절대 '확인됨'으로 승격되지 않는다.
        관측 증거는 ControlStatus ack 쪽에서 얻는다.
        """
        if self.dry_run:
            if self._sim_traj_at is not None:
                now = time.perf_counter()
                if now >= self._sim_traj_at:
                    dt = now - max(self._sim_t0, self._sim_traj_at)
                    if dt > 0:
                        self._sim_t0 = now
                        alpha = 1.0 - math.exp(-dt / self._sim_tau)
                        self._sim_pos += alpha * (self._sim_target - self._sim_pos)
            return (time.monotonic_ns(), self._sim_pos, 0.0)

        pos = self._robot.get_joint_positions([self._arm_group])
        return (time.monotonic_ns(), float(pos[self._group_idx]), None)

    def read_effort(self) -> Optional[float]:
        """JointState.effort — 타임스탬프는 없지만 effort/current 는 있다."""
        if self.dry_run or self._robot is None:
            return None
        try:
            st = self._robot.get_joint_states([self._arm_group])
            return float(getattr(st[self._group_idx], "effort", float("nan")))
        except Exception:
            return None

    def read_wrench(self) -> Optional[dict]:
        """손목 F/T. get_force_sensor_data 는 dict 를 돌려준다 (ForceData 아님).

        ForceData 에는 timestamp_ns 가 있으므로, 반환 dict 에도 있을 가능성이 크다.
        있으면 **F/T 쪽에서는 로봇 타임스탬프를 쓸 수 있다** — 관절 상태에는 없지만.
        """
        if self.dry_run or self._robot is None:
            return None
        E = getattr(self._sdk, "GalbotOneFoxtrotSensor", None)
        if E is None:
            return None
        side = "LEFT" if self.joint_name.startswith("left") else "RIGHT"
        mem = f"{side}_WRIST_FORCE"
        if mem not in getattr(E, "__members__", {}):
            return None
        try:
            return self._robot.get_force_sensor_data(getattr(E, mem))
        except Exception as e:
            print(f"[adapter] get_force_sensor_data 실패: {type(e).__name__}: {e}")
            return None

    # ── 쓰기: 경로 A — 직접 명령 ────────────────────────────────────────────
    def send_position(self, pos_rad: float) -> None:
        if self.dry_run:
            now = time.perf_counter()
            dt = now - self._sim_t0
            self._sim_t0 = now
            alpha = 1.0 - math.exp(-dt / self._sim_tau)
            self._sim_pos += alpha * (pos_rad - self._sim_pos)
            return

        if self.send_api == "positions":
            # ⚠️ is_blocking=False 필수. 기본값 True 로 부르면 모션 완료를 기다린다.
            st = self._robot.set_joint_positions(
                [pos_rad],
                joint_names=[self.joint_name],
                is_blocking=False,
                timeout_s=self.horizon_s * 3,
            )
        else:
            cmd = self._sdk.JointCommand()
            cmd.position = pos_rad
            # ⚠️ time_from_start_s 기본값 10.0 초를 반드시 덮어쓴다.
            st = self._robot.set_joint_commands(
                [cmd],
                joint_names=[self.joint_name],
                time_from_start_s=self.horizon_s,
            )
        self._track(st)

    def prime(self) -> None:
        """측정 시작 전 1회 — 첫 호출 비용을 측정 밖으로 뺀다."""
        if self.dry_run:
            return
        _, p, _ = self.read_state()
        self.send_position(p)
        print(f"[adapter] prime OK — send_api={self.send_api}, "
              f"time_from_start_s={self.horizon_s}, ack={self._fmt_ack()}")
        self.reset_ack()

    def _fmt_ack(self) -> str:
        if not self.ack_total:
            return "n/a"
        bad = ", ".join(f"{k}×{v}" for k, v in self.ack_bad.items())
        return f"{self.ack_ok}/{self.ack_total}" + (f" ({bad})" if bad else "")

    # ── 쓰기: 경로 B — 궤적 ─────────────────────────────────────────────────
    def _build_trajectory(self, times_s: list[float],
                          positions_rad: list[float]):
        T = self._sdk.Trajectory()
        T.joint_names = [self.joint_name]
        T.joint_groups = []
        pts = []
        for t, p in zip(times_s, positions_rad):
            jc = self._sdk.JointCommand()
            jc.position = p
            tp = self._sdk.TrajectoryPoint()
            tp.joint_command_vec = [jc]
            tp.time_from_start_second = float(t)
            pts.append(tp)
        T.points = pts
        return T

    def send_trajectory(self, times_s: list[float], positions_rad: list[float],
                        stop_first: bool = False,
                        api: Optional[str] = None) -> bool:
        """execute_joint_trajectory 로 궤적을 발행한다 (non-blocking).

        ⚠️ TARGET_TYPE_OVERRIDE 는 여기 인자가 아니다 (SingoriXTarget 경로 소속).
        따라서 "덮어쓰기" 는 **재발행이 기존 궤적을 대체하는가**로 실측한다.
        `stop_first=True` 면 stop_trajectory_execution() 을 먼저 호출한다 —
        두 경우의 지연 차이가 곧 큐 의미론에 대한 증거다.
        """
        if self.dry_run:
            self._sim_target = positions_rad[-1]
            self._sim_traj_at = time.perf_counter() + SIM_TRANSPORT_S
            return True
        if self._robot is None:
            return False
        use = api or self.traj_api
        try:
            if stop_first:
                self._robot.stop_trajectory_execution()
            traj = self._build_trajectory(times_s, positions_rad)
            if use == "batch":
                # 공식 문서: "multiple future frames" — action chunk 경로다.
                st = self._robot.set_joint_commands_batch(traj)
                label = "set_joint_commands_batch(Trajectory)"
            else:
                st = self._robot.execute_joint_trajectory(traj, is_blocking=False)
                label = "execute_joint_trajectory(Trajectory, is_blocking=False)"
            self._track(st)
            self.traj_supported = True
            self.resolved["trajectory"] = (
                label + (" + stop_first" if stop_first else ""))
            return self._name_of(st) in ("SUCCESS", "IN_PROGRESS", "RUNNING")
        except Exception as e:
            if self.traj_supported is None:
                self.traj_supported = False
                print(f"\n[adapter] 궤적 경로(PART C, {use}) 실패: "
                      f"{type(e).__name__}: {e}")
                for m in ("execute_joint_trajectory", "set_joint_commands_batch"):
                    if hasattr(self._robot, m):
                        print(f"  {m} 실제 시그니처:")
                        print("  " + _doc(getattr(self._robot, m)).splitlines()[0])
            return False

    def traj_status(self) -> list[str]:
        """RUNNING / COMPLETED / STOPPED_UNREACHED / ERROR — 궤적 경로의 관측 채널."""
        if self.dry_run or self._robot is None:
            return []
        try:
            return [self._name_of(s) for s in
                    self._robot.check_trajectory_execution_status([self._arm_group])]
        except Exception:
            return []

    _sim_target: float = 0.0
    _sim_traj_at: Optional[float] = None
    _group_names: list[str] = []
    _group_idx: int = 0

    def close(self) -> None:
        if self.dry_run or self._robot is None:
            return
        # 공식 문서의 종료 시퀀스:
        #   request_shutdown() → wait_for_shutdown() → destroy()
        for fn, arg in (("stop_trajectory_execution", None),
                        ("release_controller", self._arm_group),
                        ("request_shutdown", None),
                        ("wait_for_shutdown", None),
                        ("destroy", None)):
            if not hasattr(self._robot, fn):
                continue
            try:
                getattr(self._robot, fn)() if arg is None \
                    else getattr(self._robot, fn)(arg)
            except Exception as e:
                print(f"[adapter] close: {fn} → {type(e).__name__}: {e}")


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
    # ControlStatus ack — JointState 에 timestamp 가 없는 이 SDK 의 관측 채널.
    # SUCCESS 비율이 떨어지기 시작하는 지점이 실제 수락 천장이다.
    ack_frac: float
    ack_bad: str
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

    ad.reset_ack()
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
        ack_frac=ad.ack_frac(),
        ack_bad=", ".join(f"{k}×{v}" for k, v in ad.ack_bad.items()),
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
                   horizon_s: float = 0.4,
                   stop_first: bool = False) -> Optional[ReplanResult]:
    """PART C — 궤적 재발행이 실제 운동에 반영되기까지의 지연.

    왜 PART A 와 별개인가
    ─────────────────────
    PART A 는 "Python 이 얼마나 빨리 때릴 수 있나"를 잰다. 그런데 SDK 에는
    궤적을 통째로 발행하는 경로가 따로 있다. 그 경로에서는 상위가 5Hz 로만
    보내도 **온보드가 보간**하므로, 상위 rate 천장이 낮다는 사실 자체는
    분리 전제를 죽이지 않는다.

    대신 그 경로의 진짜 병목은 **재발행이 실제 운동에 반영되기까지의 지연**이다.
    이건 rate 를 올려도 해결되지 않는다. action chunk 를 5Hz 로 갈아끼우는
    설계에서 이 값이 200ms 를 넘으면 "청크 경계마다 이미 지난 상황에 반응"한다.

    ⚠️ 2026-07-31 정정 — TARGET_TYPE_OVERRIDE 는 `execute_joint_trajectory` 의
    인자가 **아니다.** 실물 시그니처는 `(trajectory, is_blocking=True)` 뿐이고,
    TARGET_TYPE_* 는 TargetConfig → TargetGroupTrajectory → SingoriXTarget →
    publish_target 경로 소속이다 (SingoriXTarget 필드 미확인이라 아직 못 탄다).
    그래서 덮어쓰기 의미론을 **실측으로** 확인한다:

      stop_first=False : 실행 중에 그냥 재발행 → 대체되는가, 큐에 쌓이는가
      stop_first=True  : stop_trajectory_execution() 후 발행 → 명시적 취소 비용

    두 값의 차이가 곧 큐 의미론에 대한 증거다. 둘 다 재는 이유가 이것이다.
    """
    if not ad.send_trajectory([horizon_s], [center_rad], stop_first=stop_first):
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
        ad.send_trajectory([horizon_s], [center_rad], stop_first=stop_first)
        t_end = time.perf_counter() + hold_s
        while time.perf_counter() < t_end:
            ad.read_state()
            time.sleep(poll_dt)
        _, base, _ = ad.read_state()

        # 계단 궤적을 재발행 — 이게 기존 궤적을 대체하는지가 측정 대상이다
        target = center_rad + (amplitude_rad if k % 2 == 0 else -amplitude_rad)
        t0 = time.perf_counter()
        if not ad.send_trajectory([horizon_s], [target], stop_first=stop_first):
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

    원래는 로봇 타임스탬프로 상태 갱신률을 보려 했으나, 2026-07-31 실물 확인
    결과 **`JointState` 에 timestamp 필드가 없다** (acceleration/current/
    effort/position/velocity 뿐). 그 채널은 이 SDK 에서 원리적으로 불가능하다.

    대신 모든 `set_*` 가 **ControlStatus 를 반환**한다. SUCCESS 비율이 떨어지기
    시작하는 지점이 실제 수락 천장이다. 완전한 "실행됨" 증거는 아니지만
    (fire-and-forget 이면 SUCCESS 도 즉시 돌아올 수 있다) PUBLISH_FAIL /
    COMM_DISCONNECTED / TIMEOUT 은 확실히 잡는다. 그래서 천장 승인 조건을
    **ack ≥ 99%** 로 둔다. ack 가 무너지는 rate 는 천장이 아니다.
    """
    device_ts = ts_source.startswith("device")
    ACK_MIN = 0.99

    # 경로 A 천장은 send-only 램프가 있으면 그쪽을 쓴다 (read 오염 제거)
    send_only = [r for r in rates if not r.with_read]
    pool = send_only or rates

    ok = [r for r in pool
          if r.achieved_hz >= 0.9 * r.target_hz
          and (math.isnan(r.jitter_ratio) or r.jitter_ratio < 2.0)]
    ceiling = max((r.target_hz for r in ok), default=0.0)

    # ack 가 살아 있는 천장 — 이 SDK 에서 실제로 쓸 수 있는 관측 증거
    acked = [r for r in pool
             if r.achieved_hz >= 0.9 * r.target_hz
             and (math.isnan(r.jitter_ratio) or r.jitter_ratio < 2.0)
             and (math.isnan(r.ack_frac) or r.ack_frac >= ACK_MIN)]
    ack_ceiling = max((r.target_hz for r in acked), default=0.0)

    # 상태 갱신률 기반 천장 — 로봇 타임스탬프가 있어야만 의미가 있다.
    # 이 SDK 에는 없으므로 실물에서는 항상 0 이다.
    obs = [r for r in rates
           if r.with_read and not math.isnan(r.state_hz)
           and r.state_hz >= 0.9 * r.target_hz]
    state_ceiling = max((r.target_hz for r in obs), default=0.0)

    have_ack = any(not math.isnan(r.ack_frac) for r in pool)
    ceiling_confirmed = bool(
        (device_ts and state_ceiling >= ceiling > 0)
        or (have_ack and ack_ceiling >= ceiling > 0))
    if have_ack and ack_ceiling < ceiling:
        ceiling = ack_ceiling      # ack 가 무너진 rate 는 천장이 아니다

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
        "ack_ceiling_hz": ack_ceiling,
        "ceiling_confirmed": ceiling_confirmed,
        "confirmation_channel": ("ControlStatus ack" if have_ack
                                 else "로봇 타임스탬프" if device_ts else "없음"),
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
    ap.add_argument("--send-api", default="commands",
                    choices=["commands", "positions"],
                    help="경로 A 에 쓸 API. commands=set_joint_commands "
                         "(time_from_start_s 지정 가능), positions=set_joint_positions")
    ap.add_argument("--horizon-s", type=float, default=0.0,
                    help="set_joint_commands 의 time_from_start_s. "
                         "SDK 기본값 10.0 은 고속 스트리밍에 쓸 수 없고, "
                         "벤더 고주파 예제는 0.0(보간 없음)을 쓴다")
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
                    help="PART C(궤적 재발행 지연) 건너뜀")
    ap.add_argument("--skip-stopfirst", action="store_true",
                    help="PART C 의 stop_trajectory_execution 변형만 건너뜀")
    ap.add_argument("--skip-batch", action="store_true",
                    help="PART C 의 set_joint_commands_batch 변형만 건너뜀")
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

    ad = G1Adapter(dry_run=args.dry_run, joint_name=args.joint,
                   horizon_s=args.horizon_s, send_api=args.send_api)
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
            "send_api": args.send_api,
            "time_from_start_s": args.horizon_s,
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
                  f"{'max':>8} {'jitter':>7} {'state_hz':>9} {'ack':>7}")
            for hz in rates:
                r = measure_rate(ad, hz, args.dwell, amp, center, do_read=do_read)
                rate_results.append(r)
                sh = "—" if math.isnan(r.state_hz) else f"{r.state_hz:.1f}"
                ak = "—" if math.isnan(r.ack_frac) else f"{r.ack_frac*100:.1f}%"
                print(f"{r.target_hz:8.0f} {r.achieved_hz:9.1f} {r.send_p50_ms:8.2f} "
                      f"{r.send_p99_ms:8.2f} {r.send_p999_ms:8.2f} {r.send_max_ms:8.2f} "
                      f"{r.jitter_ratio:7.2f} {sh:>9} {ak:>7}"
                      + (f"   ⚠️ {r.ack_bad}" if r.ack_bad else ""))
                # 목표의 절반도 못 내면 천장을 지난 것
                if r.achieved_hz < 0.5 * r.target_hz:
                    print(f"  → 달성률 50% 미만. 천장 통과로 보고 램프 중단.")
                    break
                # ack 가 무너지면 그 위는 재도 의미가 없다
                if not math.isnan(r.ack_frac) and r.ack_frac < 0.5:
                    print(f"  → ack {r.ack_frac*100:.0f}%. 로봇이 명령을 거부하기 "
                          f"시작했으므로 램프 중단.")
                    break
        results["rate_ramp"] = [asdict(r) for r in rate_results]

        bode_results: list[BodePoint] = []
        if not args.skip_bode:
            print("\n── PART B — 스윕 사인 Bode " + "─" * 40)
            bode_results = measure_bode(ad, bfreqs, amp, center)
            results["bode"] = [asdict(p) for p in bode_results]

        replan: Optional[ReplanResult] = None
        if not args.skip_replan:
            print("\n── PART C — 궤적 재발행 지연 " + "─" * 34)
            print("   execute_joint_trajectory(Trajectory, is_blocking=False)")
            print("   ⚠️ TARGET_TYPE_OVERRIDE 는 이 호출의 인자가 아니다 —")
            print("      SingoriXTarget 경로 소속. 덮어쓰기 의미론을 실측한다.")
            variants = [("trajectory", False, "execute_joint_trajectory 재발행")]
            if not args.skip_stopfirst:
                variants.append(("trajectory", True,
                                 "stop_trajectory_execution 후 재발행"))
            if not args.skip_batch:
                # 공식 문서가 "multiple future frames" 라고 부르는 경로.
                # action chunk 스트리밍의 진짜 후보다.
                variants.append(("batch", False,
                                 "set_joint_commands_batch (chunk 경로)"))

            for tapi, sf, label in variants:
                print(f"\n   [{label}]")
                ad.traj_api = tapi
                r = measure_replan(ad, args.replan_trials, amp, center,
                                   stop_first=sf)
                if r is None:
                    print("   건너뜀 — 궤적 경로를 쓸 수 없습니다 (위 진단 참조).")
                    continue
                key = ("replan_batch" if tapi == "batch"
                       else "replan_stopfirst" if sf else "replan")
                results[key] = asdict(r)
                results.setdefault("replan_calls", {})[key] = \
                    ad.resolved.get("trajectory")
                if tapi == "trajectory" and not sf:
                    replan = r
                print(f"   p50 {r.latency_p50_ms:.2f} ms · "
                      f"p95 {r.latency_p95_ms:.2f} ms · "
                      f"max {r.latency_max_ms:.2f} ms  "
                      f"(폴링 {r.poll_hz:.0f} Hz, 실패 {r.failures})")
                st = ad.traj_status()
                if st:
                    print(f"   궤적 상태: {st}")

            results["meta"]["trajectory_call"] = ad.resolved.get("trajectory")
            if "replan_batch" in results and "replan" in results:
                db = (results["replan_batch"]["latency_p50_ms"]
                      - results["replan"]["latency_p50_ms"])
                print(f"\n   batch vs trajectory 차이 {db:+.1f} ms →", end=" ")
                if db < -5:
                    print("batch 가 빠르다. chunk 스트리밍은 batch 로 갈 것.")
                elif db > 5:
                    print("trajectory 가 빠르다.")
                else:
                    print("사실상 동일.")

            if "replan" in results and "replan_stopfirst" in results:
                d = (results["replan_stopfirst"]["latency_p50_ms"]
                     - results["replan"]["latency_p50_ms"])
                print(f"\n   두 방식 차이 {d:+.1f} ms →", end=" ")
                if abs(d) < 5:
                    print("사실상 동일. 재발행이 그 자체로 대체한다고 볼 수 있다.")
                elif d > 0:
                    print("명시적 stop 이 더 느리다. 그냥 재발행하는 편이 낫다.")
                else:
                    print("stop 이 더 빠르다 → 재발행만으로는 큐에 쌓일 가능성.")

            if replan is not None:
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
            results["wrench_sample"] = {k: str(v)[:120] for k, v in w.items()} \
                if isinstance(w, dict) else str(w)[:400]
            print(f"\n  손목 F/T 표본 ({type(w).__name__}):")
            if isinstance(w, dict):
                for k, val in w.items():
                    print(f"    {k:22s} {str(val)[:80]}")
                if "timestamp_ns" in w:
                    print("    → timestamp_ns 있음. 관절 상태에는 없지만 F/T 에는 "
                          "로봇 시계가 있다.")
            else:
                print(f"    {w}")

        v = verdict(rate_results, bode_results, replan, ts_source=ad.ts_source)
        results["verdict"] = v

        print("\n" + "=" * 72)
        print("판정")
        print("=" * 72)
        print(f"  주 기준            : {v['primary_criterion']}")
        print(f"  명령 rate 천장     : {v['command_ceiling_hz']:.0f} Hz   (경로 A, "
              f"{v['command_ceiling_source']})")
        print(f"  ack 유지 천장      : {v['ack_ceiling_hz']:.0f} Hz   "
              f"(확인 채널: {v['confirmation_channel']})")
        print(f"  상태갱신 천장      : {v['state_ceiling_hz']:.0f} Hz   "
              f"→ 천장 확인 {'✅' if v['ceiling_confirmed'] else '❌ 미확인'}")
        bw = v["tracking_bandwidth_3db_hz"]
        print(f"  추종 대역폭(-3dB)  : {bw if bw else '측정 범위 내 없음'} Hz")
        if v["replan_latency_p95_ms"] is not None:
            print(f"  재계획 지연 p95    : {v['replan_latency_p95_ms']:.1f} ms  (경로 B)")
        else:
            print(f"  재계획 지연        : 측정 못 함 (경로 B 미확보)")
        if ad.ts_source == "host_monotonic":
            print(f"\n  ℹ️ 상태 타임스탬프는 호스트 시계입니다 — 이 SDK 의 JointState 에는")
            print(f"     timestamp 필드가 없습니다(실물 확인). state_dt_*/unique_state_frac")
            print(f"     은 로봇 응답이 아니라 우리 루프를 잰 값이니 읽지 마십시오.")
            print(f"     대신 ack 열(ControlStatus SUCCESS 비율)을 보십시오.")
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
