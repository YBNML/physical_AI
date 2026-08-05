#!/usr/bin/env python3
"""
GalbotSDK 표면 정밀 조사 — pybind11 docstring 추출 + 읽기 전용 실물 probe

왜 이 도구가 필요한가
─────────────────────
`galbot_sdk` 는 **pybind11 C++ 확장 모듈**이다. 그래서 `inspect.signature()` 가
모든 클래스/메서드에서 실패하고 `(?)` 만 나온다. 실제 인자 목록은 사라진 게 아니라
**`__doc__` 첫 줄**에 문자열로 박혀 있다. pybind11 이 이렇게 생성한다:

    set_joint_commands(self: galbot_sdk.GalbotRobot, commands: List[...], ...) -> bool

    Send joint commands to the robot.

오버로드가 있으면 이렇게 나온다:

    method(*args, **kwargs)
    Overloaded function.

    1. method(self: X, a: int) -> None
    2. method(self: X, a: str) -> None

따라서 **docstring 을 파싱해야** 실제 시그니처를 안다. 이 스크립트가 그걸 한다.

안전
────
기본 모드는 **정적 조사만** 한다. 로봇에 연결하지 않고, 아무것도 움직이지 않는다.
Mac 에서는 SDK import 자체가 안 되므로 이 스크립트도 Linux 전용이다.

`--live` 를 줘야 실물 호출을 시도한다. 그 경우에도:
  - `get_*` 만 호출한다. `set_*` / `move_*` / `execute_*` 는 **하드 차단**되어 있다
    (호출 직전 이름 검사 + assert). 코드를 고치지 않는 한 로봇은 움직이지 않는다.
  - 각 호출은 개별 try/except 로 감싸서, 실패해도 나머지 조사가 계속된다.

사용
────
    source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh   # ← 반드시 먼저
    conda activate physical_ai

    # 1) 정적 조사만 (로봇 불필요, 안전)
    python tools/probe_sdk.py --out sdk_surface.json --md sdk_surface.md

    # 2) 관심 심볼만 화면에 크게 (가장 먼저 볼 것)
    python tools/probe_sdk.py --focus

    # 3) 읽기 전용 실물 probe (로봇 전원 + 연결 필요, 움직이지 않음)
    python tools/probe_sdk.py --live --out sdk_live.json

결과 파일 2개를 그대로 주시면 어댑터를 정확히 맞출 수 있습니다.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import traceback
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdk_entry  # noqa: E402

SETUP_HINT = (
    "GalbotSDK import 실패.\n"
    "  1) SDK 환경을 먼저 로드했습니까?\n"
    "       source /opt/galbot/galbot_sdk/linux-x86_64-gcc940/setup.sh\n"
    "  2) SDK 는 Linux 전용입니다 (linux-x86_64 / linux-aarch64).\n"
    "     Mac 에서는 실행할 수 없습니다.\n"
    "  3) 경로가 다르면:  find /opt -name 'galbot_sdk' -maxdepth 6 -type d\n"
)

# 실물 호출이 절대 건드리면 안 되는 접두사. --live 에서 하드 차단.
FORBIDDEN_PREFIXES = ("set_", "move_", "execute_", "start_", "stop_", "enable_",
                      "disable_", "reset_", "clear_", "send_", "shutdown", "close")

# 우리 프로젝트가 실제로 쓸 심볼. --focus 에서 이것만 크게 출력한다.
FOCUS = {
    "classes": ["GalbotRobot", "GalbotMotion", "JointState", "JointStates",
                "JointCommand", "GroupCommand", "Trajectory", "TrajectoryPoint",
                "TargetGroupTrajectory", "Wrench", "ForceData", "GripperState",
                "IKSolverConfig", "TargetConfig", "Pose", "FrameTriad"],
    "enums": ["G1JointGroup", "G1ControllerName", "GalbotOneFoxtrotSensor",
              "SensorType", "MachineType", "SeedType", "ControlStatus",
              "MotionStatus", "TrajectoryControlStatus", "RobotStatesType"],
    "functions": ["create_joint_state", "create_pose_state", "create_parameter",
                  "check_motion_status"],
    "const_prefixes": ["TARGET_DATA_", "TARGET_TYPE_"],
}


# ─────────────────────────────────────────────────────────────────────────────
# pybind11 docstring 파싱
# ─────────────────────────────────────────────────────────────────────────────

_SIG_RE = re.compile(r"^\s*(?:\d+\.\s+)?([A-Za-z_]\w*)\((.*)\)\s*->\s*(.+?)\s*$")


def _sig_name(sig: str) -> Optional[str]:
    """시그니처 줄에서 함수명만. 매칭 안 되면 None."""
    m = _SIG_RE.match(sig)
    return m.group(1) if m else None


def parse_pybind_doc(doc: Optional[str]) -> dict:
    """pybind11 이 생성한 docstring 에서 시그니처 줄들을 뽑아낸다.

    반환: {"signatures": [...], "overloaded": bool, "text": "설명부"}
    """
    if not doc:
        return {"signatures": [], "overloaded": False, "text": ""}

    lines = doc.splitlines()
    sigs: list[str] = []
    text_lines: list[str] = []
    overloaded = any("Overloaded function" in ln for ln in lines[:4])

    for ln in lines:
        m = _SIG_RE.match(ln)
        if m and "(" in ln:
            # `method(*args, **kwargs)` 는 오버로드 껍데기이므로 의미 없음
            if m.group(2).strip() in ("*args, **kwargs", "*args", ""):
                if not overloaded:
                    sigs.append(ln.strip())
                continue
            sigs.append(ln.strip())
        elif ln.strip() and "Overloaded function" not in ln:
            text_lines.append(ln.rstrip())

    return {
        "signatures": sigs,
        "overloaded": overloaded,
        "text": "\n".join(text_lines).strip(),
    }


def is_pybind_enum(obj: Any) -> bool:
    return inspect.isclass(obj) and hasattr(obj, "__members__")


def describe_enum(cls: Any) -> dict:
    members = {}
    try:
        for name, val in cls.__members__.items():
            try:
                members[name] = int(val)
            except Exception:
                members[name] = repr(val)
    except Exception as e:                                   # pragma: no cover
        members = {"<error>": str(e)}
    return {"kind": "enum", "doc": (cls.__doc__ or "").strip()[:400],
            "members": members, "n_members": len(members)}


def describe_class(cls: Any, deep: bool = False) -> dict:
    out: dict = {
        "kind": "class",
        "doc": (cls.__doc__ or "").strip()[:600],
        "init": parse_pybind_doc(getattr(cls.__init__, "__doc__", None)),
        "methods": {},
        "properties": {},
    }
    for name in sorted(dir(cls)):
        if name.startswith("_") and not deep:
            continue
        if name.startswith("__"):
            continue
        try:
            attr = inspect.getattr_static(cls, name)
        except Exception:
            try:
                attr = getattr(cls, name)
            except Exception:
                continue

        doc = getattr(attr, "__doc__", None)
        parsed = parse_pybind_doc(doc)

        # `staticmethod` 래퍼 자체의 docstring("staticmethod(function) -> method")이
        # 딸려오는 경우가 있다. 시그니처의 함수명이 속성명과 다르면 버린다.
        parsed["signatures"] = [s for s in parsed["signatures"]
                                if _sig_name(s) in (name, None)]

        if isinstance(attr, property):
            out["properties"][name] = {
                "doc": (doc or "").strip()[:300],
                "readonly": attr.fset is None,
            }
        elif callable(attr) or parsed["signatures"]:
            out["methods"][name] = {
                "signatures": parsed["signatures"],
                "overloaded": parsed["overloaded"],
                "doc": parsed["text"][:400],
                "static": isinstance(attr, staticmethod),
            }
        else:
            # pybind11 은 데이터 멤버를 property 가 아닌 descriptor 로 노출하기도 한다
            out["properties"][name] = {
                "doc": (doc or "").strip()[:300],
                "readonly": None,
                "note": "descriptor (property 아님)",
            }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 정적 조사
# ─────────────────────────────────────────────────────────────────────────────


def probe_static(g: Any, deep: bool = False) -> dict:
    report: dict = {
        "module_file": getattr(g, "__file__", "?"),
        "module_version": getattr(g, "__version__", None),
        "constants": {},
        "enums": {},
        "classes": {},
        "functions": {},
        "other": {},
    }

    for name in sorted(dir(g)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(g, name)
        except Exception as e:
            report["other"][name] = f"<getattr 실패: {e}>"
            continue

        # 모듈 재수출(sys, platform, importlib 등)은 건너뜀
        if inspect.ismodule(obj):
            continue

        if is_pybind_enum(obj):
            report["enums"][name] = describe_enum(obj)
        elif inspect.isclass(obj):
            report["classes"][name] = describe_class(obj, deep=deep)
        elif inspect.isroutine(obj) or inspect.isbuiltin(obj):
            p = parse_pybind_doc(getattr(obj, "__doc__", None))
            report["functions"][name] = {
                "signatures": p["signatures"],
                "overloaded": p["overloaded"],
                "doc": p["text"][:400],
            }
        else:
            entry: dict = {"type": type(obj).__name__, "repr": repr(obj)[:200]}
            try:
                entry["int"] = int(obj)
            except Exception:
                pass
            # enum 인스턴스(모듈 최상위로 노출된 enum 멤버)이면 소속을 기록
            parent = type(obj).__name__
            if parent not in ("int", "float", "str", "bool", "NoneType"):
                entry["enum_class"] = parent
            report["constants"][name] = entry

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 읽기 전용 실물 probe
# ─────────────────────────────────────────────────────────────────────────────


def _assert_readonly(method_name: str) -> None:
    """호출 직전 마지막 방어선. 이 함수를 지우지 마십시오."""
    lo = method_name.lower()
    for pre in FORBIDDEN_PREFIXES:
        if lo.startswith(pre):
            raise RuntimeError(
                f"안전 차단: probe_sdk.py 는 읽기 전용입니다. '{method_name}' 는 "
                f"상태를 바꿀 수 있어 호출하지 않습니다."
            )


def _try(label: str, fn, *args, **kw) -> dict:
    """호출 하나를 안전하게 시도하고 결과/예외를 기록한다."""
    try:
        val = fn(*args, **kw)
        return {"ok": True, "repr": _short(val), "type": type(val).__name__,
                "detail": _expand(val)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=2)[-600:]}


def _short(v: Any, n: int = 400) -> str:
    try:
        s = repr(v)
    except Exception:
        s = f"<repr 실패 {type(v).__name__}>"
    return s if len(s) <= n else s[:n] + f"... (총 {len(s)}자)"


def _expand(v: Any) -> Any:
    """반환 객체가 pybind11 타입이면 공개 속성을 펼쳐 본다."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return {"len": len(v), "head": [_short(x, 120) for x in list(v)[:8]]}
    out = {}
    for a in dir(v):
        if a.startswith("_"):
            continue
        try:
            av = getattr(v, a)
        except Exception:
            continue
        if callable(av):
            continue
        out[a] = _short(av, 160)
    return out or _short(v)


def _construct(sdk: Any, name: str) -> tuple[Optional[Any], dict]:
    """핸들 획득. 여러 전략을 시도하고 실패 시 정확한 진단을 담는다.

    2026-07-31 — `GalbotMotion()` 이 "No constructor defined!" 로 실패했다.
    pybind11 이 py::init<>() 없이 바인딩한 것이므로 직접 생성이 아니라
    어딘가에서 받아오는 구조다. 그래서 sdk_entry 로 탐색을 위임한다.
    """
    cls = getattr(sdk, name, None)
    doc = parse_pybind_doc(getattr(getattr(cls, "__init__", None), "__doc__", None))
    try:
        obj, how = sdk_entry.acquire(sdk, name)
        return obj, {"ok": True, "method": how, "init_doc": doc}
    except sdk_entry.EntryNotFound as e:
        return None, {"ok": False, "diagnosis": str(e), "init_doc": doc}


def probe_live(g: Any) -> dict:
    """로봇에 연결해 **읽기 전용** 호출만 시도한다."""
    live: dict = {"note": "읽기 전용. set_/move_/execute_ 는 하드 차단됨."}

    # ── GalbotRobot ─────────────────────────────────────────────────────────
    if not hasattr(g, "GalbotRobot"):
        live["GalbotRobot"] = {"ok": False, "error": "심볼 없음"}
        return live

    print("  → GalbotRobot 핸들 획득 시도...", flush=True)
    robot, ctor = _construct(g, "GalbotRobot")
    live["GalbotRobot_ctor"] = ctor
    if ctor.get("ok"):
        print(f"  ✅ {ctor['method']}", flush=True)
    if robot is None:
        live["GalbotRobot"] = {"ok": False,
                               "error": ctor.get("diagnosis", "획득 실패")}
        print("\n" + ctor.get("diagnosis", "GalbotRobot 획득 실패"))
        return live

    # ⚠️ **init() 없이 데이터 메서드를 부르면 segfault 다.**
    #    2026-07-31 실측: 획득 직후 get_joint_names() 를 부르자
    #    "Segmentation fault (core dumped)". SDK 가 초기화되지 않은 내부 상태를
    #    역참조하는 것으로 보인다. 예외가 아니라 SIGSEGV 라 못 잡는다.
    #    따라서 init() 이 False 면 **여기서 멈춘다.** 진행하면 프로세스가 죽고
    #    지금까지 모은 정보까지 같이 날아간다.
    print("  → robot.init() 호출 중...", flush=True)
    try:
        inited = bool(robot.init(set()))
    except Exception as e:
        inited = False
        print(f"  init() 예외: {type(e).__name__}: {e}", flush=True)
    print(f"  init() → {inited}", flush=True)
    live["init"] = {"ok": inited}

    if not inited:
        info = {}
        # get_device_information 은 init 전에도 안전했다 (실측)
        try:
            info = robot.get_device_information()
            print(f"  get_device_information() → {info}", flush=True)
            live["device_information"] = info
        except Exception as e:
            print(f"  get_device_information() → {type(e).__name__}: {e}")
        empty = all(not str(v).strip() for k, v in (info or {}).items()
                    if k != "manufacturer")
        print("\n" + "=" * 72)
        print("🔴 init() 이 False 입니다. 여기서 중단합니다.")
        print("=" * 72)
        if empty:
            print("  device 정보(model/serial/firmware)가 전부 비어 있습니다.")
            print("  → **로봇이 실제로 연결돼 있지 않습니다.** SDK 프로세스만 살아 있고")
            print("     반대편에 기체가 없는 상태입니다.")
        print("\n  확인할 것:")
        print("   1. 로봇 전원이 켜져 있고 부팅이 끝났는가")
        print("   2. 이 PC 가 로봇 LAN 에 있는가 (ping 으로 확인)")
        print("   3. 다른 프로세스가 SDK 를 점유하고 있지 않은가 (싱글톤 구조)")
        print("\n  ⚠️ 이 상태에서 get_* 를 부르면 segfault 로 프로세스가 죽습니다.")
        print("     그래서 더 진행하지 않습니다.")
        return live

    READ_CALLS = [
        "get_joint_names",          # ★ 관절 순서/개수 확정 — 21 vs 23 논쟁 종결
        "get_joint_group_names",    # ★ G1JointGroup 실제 값
        "get_joint_positions",
        "get_joint_states",         # ★ effort/current 필드가 실제로 채워지는가
        "get_gripper_state",        # ★ 엔드이펙터 좌우 구성 단서
        "get_dexterous_hand_state",
        "get_suction_cup_state",
        "get_robot_states",
    ]
    for m in READ_CALLS:
        if not hasattr(robot, m):
            live[m] = {"ok": False, "error": "메서드 없음"}
            continue
        _assert_readonly(m)
        # ⚠️ 진행 위치를 호출 **전에** 찍는다. SDK 가 segfault 로 죽으면
        #    예외가 아니라 SIGSEGV 라 아무 정보도 못 남기므로, 마지막으로
        #    출력된 줄이 곧 범인이다. (Makefile 이 python -u 로 실행한다)
        print(f"    → {m}() 호출 중...", flush=True)
        fn = getattr(robot, m)
        r = _try(m, fn)
        if not r["ok"]:
            # 인자가 필요한 경우가 대부분 — docstring 을 같이 실어 준다
            r["signature_doc"] = parse_pybind_doc(getattr(fn, "__doc__", None))
        live[m] = r

    # ── 손목 F/T ★★ 이게 이 probe 의 가장 큰 수확 ────────────────────────────
    if hasattr(robot, "get_force_sensor_data"):
        _assert_readonly("get_force_sensor_data")
        fn = robot.get_force_sensor_data
        ft: dict = {"signature_doc": parse_pybind_doc(getattr(fn, "__doc__", None))}
        sensors = []
        for enum_name in ("GalbotOneFoxtrotSensor", "SensorType"):
            E = getattr(g, enum_name, None)
            if E is None or not hasattr(E, "__members__"):
                continue
            for mem in E.__members__:
                if "FORCE" in mem.upper() or "WRENCH" in mem.upper():
                    sensors.append((f"{enum_name}.{mem}", getattr(E, mem)))
        ft["candidates"] = [s[0] for s in sensors]
        for label, val in sensors:
            ft[label] = _try(label, fn, val)
        live["get_force_sensor_data"] = ft

    # ── 카메라 extrinsic ★ RUNBOOK 벤더 문의 항목을 자체 해결할 수 있는 곳 ────
    if hasattr(robot, "get_sensor_extrinsic"):
        _assert_readonly("get_sensor_extrinsic")
        fn = robot.get_sensor_extrinsic
        ex: dict = {"signature_doc": parse_pybind_doc(getattr(fn, "__doc__", None))}
        cams = []
        for enum_name in ("GalbotOneFoxtrotSensor", "SensorType"):
            E = getattr(g, enum_name, None)
            if E is None or not hasattr(E, "__members__"):
                continue
            for mem in E.__members__:
                u = mem.upper()
                if any(k in u for k in ("CAM", "RGB", "DEPTH", "HEAD", "STEREO")):
                    cams.append((f"{enum_name}.{mem}", getattr(E, mem)))
        ex["candidates"] = [c[0] for c in cams]
        for label, val in cams[:12]:
            ex[label] = _try(label, fn, val)
        live["get_sensor_extrinsic"] = ex

    # ── GalbotMotion FK — 우리 자체 FK 와 대조할 근거 ────────────────────────
    if hasattr(g, "GalbotMotion"):
        print("  → GalbotMotion 핸들 획득 시도...", flush=True)
        motion, mctor = _construct(g, "GalbotMotion")
        live["GalbotMotion_ctor"] = mctor
        if mctor.get("ok"):
            print(f"  ✅ {mctor['method']}", flush=True)
        if motion is not None:
            for m in ("get_robot_states", "get_chain_joint_state"):
                if hasattr(motion, m):
                    _assert_readonly(m)
                    fn = getattr(motion, m)
                    r = _try(m, fn)
                    if not r["ok"]:
                        r["signature_doc"] = parse_pybind_doc(
                            getattr(fn, "__doc__", None))
                    live[f"Motion.{m}"] = r
            # FK 는 인자가 필요하므로 시그니처만 확보한다
            for m in ("forward_kinematics_by_state", "inverse_kinematics_by_state"):
                if hasattr(motion, m):
                    live[f"Motion.{m}.signature"] = parse_pybind_doc(
                        getattr(getattr(motion, m), "__doc__", None))

    return live


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────


def print_focus(rep: dict) -> None:
    """우리 프로젝트가 실제로 쓸 것만 크게 출력."""
    print("=" * 78)
    print("FOCUS — 이 프로젝트가 실제로 쓰는 심볼의 실제 시그니처")
    print("=" * 78)

    for cname in FOCUS["classes"]:
        c = rep["classes"].get(cname)
        if not c:
            continue
        print(f"\n{'─'*78}\n■ {cname}\n{'─'*78}")
        init_sigs = c["init"]["signatures"]
        if init_sigs:
            print("  [생성자]")
            for s in init_sigs:
                print(f"    {s}")
        else:
            print("  [생성자] docstring 에 시그니처 없음")
        if c["methods"]:
            print("  [메서드]")
            for m, info in sorted(c["methods"].items()):
                if info["signatures"]:
                    for s in info["signatures"]:
                        print(f"    {s}")
                else:
                    print(f"    {m}(?)   ← docstring 에 시그니처 없음")
        if c["properties"]:
            props = ", ".join(sorted(c["properties"]))
            print(f"  [속성] {props}")

    print(f"\n{'='*78}\nENUM 실제 멤버\n{'='*78}")
    for ename in FOCUS["enums"]:
        e = rep["enums"].get(ename)
        if not e:
            continue
        print(f"\n■ {ename}  ({e['n_members']}개)")
        for k, v in e["members"].items():
            print(f"    {k:44s} = {v}")

    print(f"\n{'='*78}\n모듈 상수\n{'='*78}")
    for pre in FOCUS["const_prefixes"]:
        hits = {k: v for k, v in rep["constants"].items() if k.startswith(pre)}
        if not hits:
            continue
        print(f"\n■ {pre}*")
        for k, v in sorted(hits.items()):
            iv = v.get("int")
            print(f"    {k:40s} = {iv if iv is not None else v['repr']}"
                  f"   [{v.get('enum_class', v['type'])}]")

    print(f"\n{'='*78}\n자유 함수\n{'='*78}")
    for fname in FOCUS["functions"]:
        f = rep["functions"].get(fname)
        if not f:
            continue
        if f["signatures"]:
            for s in f["signatures"]:
                print(f"  {s}")
        else:
            print(f"  {fname}(?)")


def write_md(rep: dict, live: Optional[dict], path: str) -> None:
    L: list[str] = []
    L.append("# GalbotSDK 표면 조사 결과\n")
    L.append(f"- 모듈: `{rep['module_file']}`")
    L.append(f"- 클래스 {len(rep['classes'])} · enum {len(rep['enums'])} · "
             f"함수 {len(rep['functions'])} · 상수 {len(rep['constants'])}\n")
    L.append("> pybind11 확장이라 `inspect.signature()` 는 실패합니다. "
             "아래 시그니처는 모두 **docstring 에서 파싱**한 것입니다.\n")

    L.append("\n## 1. 핵심 클래스\n")
    for cname in FOCUS["classes"]:
        c = rep["classes"].get(cname)
        if not c:
            continue
        L.append(f"\n### `{cname}`\n")
        if c["init"]["signatures"]:
            L.append("**생성자**\n")
            L.append("```")
            L += c["init"]["signatures"]
            L.append("```")
        if c["methods"]:
            L.append("\n**메서드**\n")
            L.append("```")
            for m, info in sorted(c["methods"].items()):
                if info["signatures"]:
                    L += info["signatures"]
                else:
                    L.append(f"{m}(?)")
            L.append("```")
        if c["properties"]:
            L.append(f"\n**속성**: {', '.join('`'+p+'`' for p in sorted(c['properties']))}\n")

    L.append("\n## 2. enum\n")
    for ename, e in sorted(rep["enums"].items()):
        L.append(f"\n### `{ename}` ({e['n_members']})\n")
        L.append("```")
        for k, v in e["members"].items():
            L.append(f"{k} = {v}")
        L.append("```")

    L.append("\n## 3. 모듈 상수\n")
    L.append("| 이름 | 값 | 타입 |")
    L.append("|---|---|---|")
    for k, v in sorted(rep["constants"].items()):
        iv = v.get("int")
        L.append(f"| `{k}` | {iv if iv is not None else v['repr'][:40]} | "
                 f"{v.get('enum_class', v['type'])} |")

    if live:
        L.append("\n## 4. 실물 probe (읽기 전용)\n")
        L.append("```json")
        L.append(json.dumps(live, indent=2, ensure_ascii=False)[:60000])
        L.append("```")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ─────────────────────────────────────────────────────────────────────────────


def self_test() -> int:
    """SDK 없이(=Mac 에서) 파서가 맞는지 확인. 실제 pybind11 출력 형식을 그대로 씀."""
    print("=" * 78)
    print("probe_sdk 자체 검증 — SDK 불필요")
    print("=" * 78)
    fails = 0

    def chk(name: str, cond: bool, got: Any = None) -> None:
        nonlocal fails
        if cond:
            print(f"  ✅ {name}")
        else:
            fails += 1
            print(f"  ❌ {name}   got={got!r}")

    # 1) 단일 시그니처 — pybind11 표준 형식
    d1 = ("set_joint_commands(self: galbot_sdk.GalbotRobot, "
          "commands: List[galbot_sdk.JointCommand], blocking: bool = False) -> bool\n"
          "\n"
          "Send joint commands to the robot.\n")
    p1 = parse_pybind_doc(d1)
    chk("단일 시그니처 1개 추출", len(p1["signatures"]) == 1, p1["signatures"])
    chk("반환 타입까지 포함", p1["signatures"] and p1["signatures"][0].endswith("-> bool"),
        p1["signatures"])
    chk("설명부 분리", "Send joint commands" in p1["text"], p1["text"])
    chk("오버로드 아님", p1["overloaded"] is False, p1["overloaded"])

    # 2) 오버로드 형식
    d2 = ("get_joint_states(*args, **kwargs)\n"
          "Overloaded function.\n"
          "\n"
          "1. get_joint_states(self: galbot_sdk.GalbotRobot) -> galbot_sdk.JointStates\n"
          "\n"
          "2. get_joint_states(self: galbot_sdk.GalbotRobot, "
          "group: galbot_sdk.G1JointGroup) -> galbot_sdk.JointStates\n")
    p2 = parse_pybind_doc(d2)
    chk("오버로드 감지", p2["overloaded"] is True, p2["overloaded"])
    chk("오버로드 2개 추출", len(p2["signatures"]) == 2, p2["signatures"])
    chk("껍데기 (*args) 제외",
        all("*args" not in s for s in p2["signatures"]), p2["signatures"])
    chk("번호 접두사 유지되어 읽힘",
        p2["signatures"] and p2["signatures"][0].startswith("1."), p2["signatures"])

    # 3) docstring 없음
    p3 = parse_pybind_doc(None)
    chk("None 안전", p3["signatures"] == [] and p3["text"] == "", p3)

    # 4) 시그니처 없는 산문 docstring
    p4 = parse_pybind_doc("Represents a 6-axis wrench.\n\nMembers:\n  force\n  torque\n")
    chk("산문에서 오탐 없음", p4["signatures"] == [], p4["signatures"])

    # 5) enum 판별 + describe_enum
    class FakeEnum:
        __members__ = {"LEFT_WRIST_FORCE": 3, "RIGHT_WRIST_FORCE": 4}
        __doc__ = "sensor ids"
    chk("enum 판별", is_pybind_enum(FakeEnum), False)
    de = describe_enum(FakeEnum)
    chk("enum 멤버 2개", de["n_members"] == 2, de)
    chk("enum 값 보존", de["members"]["RIGHT_WRIST_FORCE"] == 4, de["members"])

    # 6) 안전 차단이 실제로 막는가 — 이게 뚫리면 로봇이 움직인다
    for bad in ("set_joint_commands", "move_whole_body_joint_zero",
                "execute_joint_trajectory", "set_gripper_command"):
        try:
            _assert_readonly(bad)
            chk(f"차단: {bad}", False, "차단 실패!")
        except RuntimeError:
            chk(f"차단: {bad}", True)
    for good in ("get_joint_names", "get_force_sensor_data", "get_sensor_extrinsic"):
        try:
            _assert_readonly(good)
            chk(f"통과: {good}", True)
        except RuntimeError as e:
            chk(f"통과: {good}", False, str(e))

    # 7) describe_class 가 실제 클래스에서 죽지 않는가
    class FakeCls:
        """A fake."""
        def __init__(self): ...
        def get_thing(self): ...
        @property
        def value(self): return 1
    dc = describe_class(FakeCls)
    chk("클래스 메서드 수집", "get_thing" in dc["methods"], list(dc["methods"]))
    chk("property 수집", "value" in dc["properties"], list(dc["properties"]))

    print("\n" + "=" * 78)
    if fails:
        print(f"❌ {fails}건 실패")
        return 1
    print("✅ 전부 통과 — 파서와 안전 차단이 정상입니다.")
    print("   이제 3090 에서 SDK 를 source 한 뒤 --focus 로 실행하십시오.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GalbotSDK 표면 정밀 조사 (pybind11 docstring 추출)")
    ap.add_argument("--out", default="sdk_surface.json", help="JSON 저장 경로")
    ap.add_argument("--md", default=None, help="사람이 읽을 markdown 저장 경로")
    ap.add_argument("--focus", action="store_true",
                    help="우리가 쓸 심볼만 화면에 크게 출력")
    ap.add_argument("--deep", action="store_true", help="_ 로 시작하는 멤버도 포함")
    ap.add_argument("--live", action="store_true",
                    help="로봇에 연결해 읽기 전용 호출 시도 (움직이지 않음)")
    ap.add_argument("--self-test", action="store_true",
                    help="SDK 없이 파서/안전차단 검증 (Mac 가능)")
    ap.add_argument("--entry", action="store_true",
                    help="GalbotRobot/GalbotMotion 진입점 조사 + __init__.py 원문 출력")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    try:
        import galbot_sdk as g
    except ImportError as e:
        print(f"{SETUP_HINT}\n원본 에러: {e}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("GalbotSDK 표면 조사")
    print("=" * 78)
    if args.entry:
        print(sdk_entry.entry_report(g))
        return 0

    print(f"  모듈 : {getattr(g, '__file__', '?')}")
    print(f"  플랫폼: {sys.platform}  python {sys.version.split()[0]}")

    rep = probe_static(g, deep=args.deep)
    print(f"  클래스 {len(rep['classes'])} · enum {len(rep['enums'])} · "
          f"함수 {len(rep['functions'])} · 상수 {len(rep['constants'])}")

    # docstring 에서 시그니처를 실제로 건졌는지 즉시 알려준다
    got = sum(1 for c in rep["classes"].values()
              for m in c["methods"].values() if m["signatures"])
    total = sum(len(c["methods"]) for c in rep["classes"].values())
    print(f"  시그니처 복원: {got}/{total} 메서드")
    if got == 0:
        print("\n  ⚠️ docstring 에서 시그니처를 하나도 못 건졌습니다.")
        print("     이 빌드는 pybind11 docstring 없이 컴파일되었을 수 있습니다")
        print("     (-DPYBIND11_DETAILED_ERROR_MESSAGES 없이 빌드된 경우).")
        print("     그렇다면 SDK 동봉 헤더/예제로만 시그니처를 알 수 있습니다:")
        print("       ls /opt/galbot/galbot_sdk/linux-x86_64-gcc940/")
        print("       find /opt/galbot -name '*.h' -o -name '*.hpp' -o -name 'example*'")

    live = None
    if args.live:
        print("\n" + "=" * 78)
        print("실물 probe — 읽기 전용")
        print("=" * 78)
        print("  set_/move_/execute_ 는 하드 차단됩니다. 로봇은 움직이지 않습니다.")
        live = probe_live(g)
        ok = sum(1 for v in live.values() if isinstance(v, dict) and v.get("ok"))
        print(f"  성공한 호출 {ok}건")
        for k, v in live.items():
            if isinstance(v, dict) and "ok" in v:
                mark = "✅" if v["ok"] else "❌"
                detail = v.get("repr", v.get("error", ""))
                print(f"  {mark} {k:32s} {str(detail)[:90]}")

    if args.focus:
        print()
        print_focus(rep)

    payload = {"static": rep, "live": live}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n저장: {os.path.abspath(args.out)}")

    if args.md:
        write_md(rep, live, args.md)
        print(f"저장: {os.path.abspath(args.md)}")

    print("\n다음: 이 파일을 그대로 공유하시면 G1Adapter 를 실물에 맞춰 확정합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
