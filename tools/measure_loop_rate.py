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
# ⚠️ 확인 필요 — 아래 import 경로와 메서드 시그니처는 공개 문서에서 확인한
#    이름을 근거로 작성했으나 실물 SDK로 검증하지 못했다. 회사에서 첫 실행 시
#    이 클래스만 고치면 나머지는 그대로 동작한다.
#
# 문서에서 확인된 것:
#   - robot.set_joint_commands(...)  : 표준 관절은 position만 유효
#                                      (velocity/acceleration/effort는 무시됨)
#   - JointState 에 timestamp_ns, position, velocity, effort, current 존재
#   - 그리퍼 관절만 velocity/effort 유효
# ─────────────────────────────────────────────────────────────────────────────


class G1Adapter:
    """GalbotSDK 래퍼. 실물 API에 맞춰 이 클래스만 수정하면 된다."""

    def __init__(self, dry_run: bool = False, joint_name: str = "left_arm_joint4"):
        self.dry_run = dry_run
        self.joint_name = joint_name
        self._robot = None
        self._sim_pos = 0.0
        self._sim_t0 = time.perf_counter()

        if dry_run:
            print("[adapter] DRY-RUN — 1차 지연 플랜트를 시뮬레이션합니다 (로봇 없음)")
            return

        # ── 여기부터 실물 SDK ────────────────────────────────────────────────
        try:
            from galbot_sdk import Robot  # ⚠️ 확인 필요: 실제 모듈/클래스명
        except ImportError as e:
            sys.exit(
                f"[adapter] GalbotSDK import 실패: {e}\n"
                "  - SDK는 Linux 전용입니다 (ubuntu 20-24, x86_64 또는 aarch64)\n"
                "  - Mac에서는 실행할 수 없습니다. --dry-run 으로 스크립트만 검증하십시오.\n"
                "  - import 경로가 다르면 G1Adapter.__init__ 을 수정하십시오."
            )
        self._robot = Robot()          # ⚠️ 확인 필요: 생성자 인자 (IP 등)
        self._robot.connect()          # ⚠️ 확인 필요: 연결 메서드명

    # ── 읽기 ────────────────────────────────────────────────────────────────
    def read_state(self) -> tuple[int, float, Optional[float]]:
        """(timestamp_ns, position_rad, effort_Nm|None) 반환."""
        if self.dry_run:
            # 1차 지연 플랜트: tau=15ms, 100Hz 이상에서 위상 지연이 보이도록
            now = time.perf_counter()
            return (time.monotonic_ns(), self._sim_pos, 0.0)

        st = self._robot.get_joint_state(self.joint_name)   # ⚠️ 확인 필요
        return (st.timestamp_ns, st.position, getattr(st, "effort", None))

    # ── 쓰기 ────────────────────────────────────────────────────────────────
    def send_position(self, pos_rad: float) -> None:
        if self.dry_run:
            # tau=15ms 1차 응답
            now = time.perf_counter()
            dt = now - self._sim_t0
            self._sim_t0 = now
            alpha = 1.0 - math.exp(-dt / 0.015)
            self._sim_pos += alpha * (pos_rad - self._sim_pos)
            return

        self._robot.set_joint_commands(                      # ⚠️ 확인 필요
            {self.joint_name: {"position": pos_rad}}
        )

    def close(self) -> None:
        if not self.dry_run and self._robot is not None:
            try:
                self._robot.disconnect()                      # ⚠️ 확인 필요
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
                 amplitude_rad: float, center_rad: float) -> RateResult:
    """PART A — 목표 rate로 작은 사인을 스트리밍하며 실제 달성치와 jitter를 측정."""
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
    uniq = len(set(state_ts)) / max(1, len(state_ts))

    p50 = pct(sg, 0.50) if sg else float("nan")
    p99 = pct(sg, 0.99) if sg else float("nan")

    return RateResult(
        target_hz=target_hz,
        achieved_hz=n / elapsed,
        n_commands=n,
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


# ─────────────────────────────────────────────────────────────────────────────
# 판정
# ─────────────────────────────────────────────────────────────────────────────


def verdict(rates: list[RateResult], bode: list[BodePoint]) -> dict:
    """GATE-1 판정."""
    # 달성률이 목표의 90% 이상이고 jitter가 2배 미만인 최고 rate
    ok = [r for r in rates
          if r.achieved_hz >= 0.9 * r.target_hz
          and (math.isnan(r.jitter_ratio) or r.jitter_ratio < 2.0)]
    ceiling = max((r.target_hz for r in ok), default=0.0)

    # -3dB 대역폭
    bw = None
    for p in bode:
        if not math.isnan(p.gain_db) and p.gain_db <= -3.0:
            bw = p.freq_hz
            break

    if ceiling >= 100:
        v = "PASS — admittance 작동 가능. 분리 유지 가능"
        impl = ("빠른 Model 2가 의미를 가짐. 인터페이스 수정(그리퍼·dt·psi·상대 포즈) 후 "
                "residual 구조로 진행.")
    elif ceiling >= 50:
        v = "MARGINAL — 경계"
        impl = ("admittance 대역폭이 5-8Hz 수준. 느린 삽입/닦기는 되고 충격 흡수는 안 됨. "
                "접촉 작업 범위를 서면으로 한정할 것.")
    else:
        v = "FAIL — 대뇌/소뇌 분리 전제 붕괴"
        impl = ("빠른 Model 2가 Model 1보다 의미 있게 빠를 여지가 없음. "
                "단일 모델 + 벤더 WBC + closed-form S-R-S IK로 전환. "
                "메인 문서 REV.2 최종 권고 참조.")

    return {
        "command_ceiling_hz": ceiling,
        "tracking_bandwidth_3db_hz": bw,
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
    results = {
        "meta": {
            "host": args.host,
            "joint": args.joint,
            "amp_deg": args.amp_deg,
            "dwell_s": args.dwell,
            "dry_run": args.dry_run,
            "unix_time": time.time(),
        }
    }

    try:
        print("── PART A — 명령 rate 램프 " + "─" * 40)
        print(f"{'target':>8} {'achieved':>9} {'p50':>8} {'p99':>8} {'p99.9':>8} "
              f"{'max':>8} {'jitter':>7} {'state_uniq':>11}")
        rate_results: list[RateResult] = []
        for hz in rates:
            r = measure_rate(ad, hz, args.dwell, amp, center)
            rate_results.append(r)
            print(f"{r.target_hz:8.0f} {r.achieved_hz:9.1f} {r.send_p50_ms:8.2f} "
                  f"{r.send_p99_ms:8.2f} {r.send_p999_ms:8.2f} {r.send_max_ms:8.2f} "
                  f"{r.jitter_ratio:7.2f} {r.unique_state_frac:11.2f}")
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

        v = verdict(rate_results, bode_results)
        results["verdict"] = v

        print("\n" + "=" * 72)
        print("판정")
        print("=" * 72)
        print(f"  명령 rate 천장     : {v['command_ceiling_hz']:.0f} Hz")
        bw = v["tracking_bandwidth_3db_hz"]
        print(f"  추종 대역폭(-3dB)  : {bw if bw else '측정 범위 내 없음'} Hz")
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
