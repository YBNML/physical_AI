"""
GalbotSDK 핸들 획득 — `GalbotRobot` / `GalbotMotion` 을 어떻게 얻는가

답 — 2026-07-31 3090 실측으로 확정
─────────────────────────────────

    robot  = galbot_sdk.GalbotRobot.get_instance()
    motion = galbot_sdk.GalbotMotion.get_instance()

**네 개 상위 클래스가 전부 싱글톤이다** — GalbotRobot / GalbotMotion /
GalbotPerception / GalbotNavigation. 넷 다 `py::init<>()` 없이 바인딩돼 있어
직접 생성하면 이렇게 실패한다:

    >>> galbot_sdk.GalbotMotion()
    TypeError: galbot_sdk.GalbotMotion: No constructor defined!

`__init__.py` 는 팩토리를 정의하지 않는다. 그 파일이 하는 일은 python 버전과
아키텍처에 맞는 `.so` 를 골라 동적 로드하고 공개 심볼을 재수출하는 것뿐이다
(그래서 진입점이 거기 있을 거라는 초기 추정은 빗나갔다 — 싱글톤은 C++ 쪽에 있었다).

부수 발견
─────────
  - `galbot_sdk.pyi` **타입 스텁이 동봉돼 있다.** pybind11 docstring 보다
    풍부한 소스다 — `read_stub()` 참조.
  - Python 서브모듈 `g1` / `s1` 이 있다 — `probe_submodules()` 참조.
  - `.so` 가 python 3.8~3.14 용으로 7개 동봉돼 있다.

이 모듈은 확정 경로를 먼저 쓰고, SDK 가 바뀌었을 때를 대비해 탐색 사다리를
fallback 으로 남겨둔다. 사다리는 후보를 **실제로 호출**하므로, 상태를 바꿀 수
있는 이름은 `_is_autocallable()` 로 차단한다.
"""

from __future__ import annotations

import inspect
import os
import pkgutil
from typing import Any, Optional

# 팩토리로 의심할 이름 패턴 (반환 타입으로 최종 확인한다)
_FACTORY_HINTS = ("create", "make", "get", "instance", "build", "new", "init",
                  "connect", "open")

# ⚠️ `acquire()` 는 후보를 **실제로 호출**한다. 그래서 힌트에 걸렸더라도
#    상태를 바꿀 수 있는 이름은 자동 호출하지 않는다. 힌트에 init/connect/open 이
#    들어 있으므로 이 가드가 없으면 원치 않는 부작용이 날 수 있다.
_AUTOCALL_DENY = ("set_", "move_", "execute_", "start_", "stop_", "enable_",
                  "disable_", "reset_", "clear_", "send_", "shutdown",
                  "destroy", "release", "init", "connect", "open")


def _is_autocallable(name: str) -> bool:
    """자동 호출해도 안전한 팩토리 이름인가.

    허용: `*instance*` 또는 create_/make_/new_/get_ 으로 시작하는 것.
    거부: 상태를 바꿀 수 있는 접두사 (위 목록).
    """
    lo = name.lower()
    if any(lo.startswith(d) or lo == d.rstrip("_") for d in _AUTOCALL_DENY):
        return False
    return ("instance" in lo
            or lo.startswith(("create", "make", "new", "get")))


def package_dir(sdk: Any) -> Optional[str]:
    f = getattr(sdk, "__file__", None)
    return os.path.dirname(f) if f else None


def read_init_source(sdk: Any, max_chars: int = 20000) -> str:
    """`__init__.py` 는 Python 소스다. 진입점이 여기 적혀 있을 가능성이 크다."""
    f = getattr(sdk, "__file__", None)
    if not f or not f.endswith(".py") or not os.path.exists(f):
        return "(__init__.py 를 읽을 수 없음)"
    try:
        with open(f, "r", errors="replace") as fh:
            s = fh.read()
        return s if len(s) <= max_chars else s[:max_chars] + f"\n... (총 {len(s)}자)"
    except Exception as e:                                   # pragma: no cover
        return f"(읽기 실패: {e})"


def list_submodules(sdk: Any) -> list[str]:
    d = package_dir(sdk)
    if not d or not os.path.isdir(d):
        return []
    out = []
    try:
        for m in pkgutil.iter_modules([d]):
            out.append(m.name)
    except Exception:
        pass
    return sorted(out)


def list_package_files(sdk: Any) -> list[str]:
    d = package_dir(sdk)
    if not d or not os.path.isdir(d):
        return []
    out = []
    for root, _dirs, files in os.walk(d):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), d)
            out.append(rel)
        if len(out) > 200:
            break
    return sorted(out)[:200]


def find_factories(sdk: Any, cls_name: str) -> list[tuple[str, str]]:
    """모듈 최상위에서 `cls_name` 을 돌려줄 법한 호출 가능 객체를 찾는다.

    반환 타입 annotation 이 pybind11 docstring 에 있으므로 그것으로 판별한다.
    """
    hits: list[tuple[str, str]] = []
    for name in dir(sdk):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(sdk, name)
        except Exception:
            continue
        if not callable(obj) or inspect.isclass(obj):
            continue
        doc = (getattr(obj, "__doc__", None) or "")
        first = doc.strip().splitlines()[0] if doc.strip() else ""
        if f"-> galbot_sdk.{cls_name}" in first or f"-> {cls_name}" in first:
            hits.append((name, first))
        elif any(h in name.lower() for h in _FACTORY_HINTS) and cls_name.lower() in first.lower():
            hits.append((name, first))
    return hits


def _first_doc_line(attr: Any) -> str:
    """staticmethod/classmethod 래퍼는 __doc__ 이 None 일 수 있다 —
    그때는 감싼 함수(`__func__`)의 docstring 을 본다."""
    for src in (attr, getattr(attr, "__func__", None)):
        if src is None:
            continue
        doc = (getattr(src, "__doc__", None) or "").strip()
        if doc:
            return doc.splitlines()[0]
    return ""


def find_class_factories(cls: Any) -> list[tuple[str, str]]:
    """클래스 자신의 static/class 메서드 중 인스턴스를 돌려줄 법한 것.

    ⚠️ **docstring 이 없어도 버리지 않는다.** 초기 구현은 docstring 을 요구했는데,
    그러면 docstring 없이 빌드된 static 메서드가 통째로 누락된다 (자체 테스트에서
    실제로 `get_instance` 를 놓쳤다). staticmethod/classmethod 인 것은 정의상
    인스턴스 없이 호출 가능하므로 doc 유무와 무관하게 후보로 삼는다.
    잘못 골라도 `acquire()` 가 반환 타입을 확인하므로 안전하다.
    """
    hits: list[tuple[str, str]] = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if not any(h in name.lower() for h in _FACTORY_HINTS):
            continue
        try:
            attr = inspect.getattr_static(cls, name)
        except Exception:
            continue
        if not isinstance(attr, (staticmethod, classmethod)) and not callable(attr):
            continue

        first = _first_doc_line(attr)
        is_static = isinstance(attr, (staticmethod, classmethod))
        # doc 이 있으면 첫 인자가 self 인지로 걸러내고, 없으면 static 여부로 판단
        if first:
            takes_self = "self" in first.split("(", 1)[-1].split(",")[0]
            if takes_self and not is_static:
                continue
        elif not is_static:
            continue
        hits.append((name, first))
    return hits


class EntryNotFound(RuntimeError):
    """핸들 획득 실패. 다음에 무엇을 봐야 하는지 담아 던진다."""


def acquire(sdk: Any, cls_name: str, verbose: bool = True) -> tuple[Any, str]:
    """`cls_name` 인스턴스를 얻는다. (객체, 사용한 방법) 반환.

    전략을 순서대로 시도하고, 성공한 방법을 문자열로 돌려준다 —
    그 문자열을 결과 JSON 에 기록해두면 다음 사람이 재현할 수 있다.
    """
    cls = getattr(sdk, cls_name, None)
    if cls is None:
        raise EntryNotFound(f"galbot_sdk 에 `{cls_name}` 심볼이 없습니다.")

    tried: list[str] = []

    # 0) 확정된 경로 — 2026-07-31 3090 실측.
    #    GalbotRobot / GalbotMotion / GalbotPerception / GalbotNavigation 네 개
    #    전부 py::init<>() 없이 바인딩된 **싱글톤**이고 `get_instance()` 로 얻는다.
    #    아래 탐색 사다리는 SDK 가 바뀌었을 때를 위한 fallback 으로 남겨둔다.
    if hasattr(cls, "get_instance"):
        try:
            obj = cls.get_instance()
            if isinstance(obj, cls):
                return obj, f"{cls_name}.get_instance()"
            tried.append(f"{cls_name}.get_instance()  →  "
                         f"{type(obj).__name__} 반환 (기대 {cls_name})")
        except Exception as e:
            tried.append(f"{cls_name}.get_instance()  →  {type(e).__name__}: {e}")

    # 1) 직접 생성
    try:
        return cls(), f"{cls_name}()"
    except Exception as e:
        tried.append(f"{cls_name}()  →  {type(e).__name__}: {e}")

    # 2) MachineType 인자
    MT = getattr(sdk, "MachineType", None)
    for mem in getattr(MT, "__members__", {}) or {}:
        try:
            return cls(getattr(MT, mem)), f"{cls_name}(MachineType.{mem})"
        except Exception as e:
            tried.append(f"{cls_name}(MachineType.{mem})  →  {type(e).__name__}: {e}")

    # 3) 클래스 자신의 팩토리 — 2026-07-31 실측으로 `get_instance` 가 정답이었다.
    #    Galbot 의 4개 상위 클래스(Robot/Motion/Perception/Navigation)가 전부
    #    싱글톤이고 py::init<>() 없이 바인딩돼 있다.
    for name, sig in find_class_factories(cls):
        if not _is_autocallable(name):
            tried.append(f"{cls_name}.{name}()  →  건너뜀 (자동 호출 비허용)")
            continue
        try:
            obj = getattr(cls, name)()
            if isinstance(obj, cls):
                return obj, f"{cls_name}.{name}()"
            tried.append(f"{cls_name}.{name}()  →  {type(obj).__name__} 반환 (기대 {cls_name})")
        except Exception as e:
            tried.append(f"{cls_name}.{name}()  →  {type(e).__name__}: {e}")

    # 4) 모듈 최상위 팩토리
    for name, sig in find_factories(sdk, cls_name):
        if not _is_autocallable(name):
            tried.append(f"{name}()  →  건너뜀 (자동 호출 비허용)")
            continue
        try:
            obj = getattr(sdk, name)()
            if isinstance(obj, cls):
                return obj, f"{name}()"
            tried.append(f"{name}()  →  {type(obj).__name__} 반환")
        except Exception as e:
            tried.append(f"{name}()  →  {type(e).__name__}: {e}")

    # 5) 서브모듈에서 같은 이름 찾기
    for sub in list_submodules(sdk):
        try:
            mod = __import__(f"{sdk.__name__}.{sub}", fromlist=[sub])
        except Exception:
            continue
        alt = getattr(mod, cls_name, None)
        if alt is None or alt is cls:
            continue
        try:
            return alt(), f"{sdk.__name__}.{sub}.{cls_name}()"
        except Exception as e:
            tried.append(f"{sdk.__name__}.{sub}.{cls_name}()  →  {type(e).__name__}: {e}")

    raise EntryNotFound(_report(sdk, cls_name, tried))


def _report(sdk: Any, cls_name: str, tried: list[str]) -> str:
    L = [
        f"`{cls_name}` 인스턴스를 얻지 못했습니다.",
        "",
        "시도한 것:",
    ]
    L += [f"  - {t}" for t in tried]
    L += [
        "",
        "'No constructor defined!' 는 pybind11 이 py::init<>() 없이 바인딩했다는 뜻입니다.",
        "즉 이 클래스는 직접 생성하는 것이 아니라 어딘가에서 받아오는 것입니다.",
        "",
        f"패키지 위치: {package_dir(sdk)}",
        f"서브모듈: {list_submodules(sdk) or '(없음)'}",
        "",
        "다음을 확인하십시오 — __init__.py 가 Python 소스이므로 거기 답이 있을 가능성이 큽니다:",
        f"  python tools/probe_sdk.py --entry",
        "",
        "또는 SDK 동봉 예제:",
        "  find /opt/galbot -name 'example*' -o -name '*.py' | head -50",
    ]
    return "\n".join(L)


def read_stub(sdk: Any, grep: Optional[list[str]] = None,
              max_chars: int = 30000) -> str:
    """`galbot_sdk.pyi` 타입 스텁.

    2026-07-31 발견 — 패키지에 `.pyi` 가 동봉돼 있다. pybind11 docstring 보다
    **더 풍부한 소스**다: docstring 이 비어 있는 심볼(생성자 등)의 시그니처와
    `@staticmethod` 표시, 오버로드까지 들어 있다.
    """
    d = package_dir(sdk)
    if not d:
        return "(패키지 경로 없음)"
    p = os.path.join(d, "galbot_sdk.pyi")
    if not os.path.exists(p):
        return f"(스텁 없음: {p})"
    try:
        with open(p, "r", errors="replace") as fh:
            src = fh.read()
    except Exception as e:                                   # pragma: no cover
        return f"(읽기 실패: {e})"

    if not grep:
        return src if len(src) <= max_chars else src[:max_chars] + f"\n... (총 {len(src)}자)"

    # 관심 심볼 주변만 — 전체를 다 붙이면 너무 길다
    lines = src.splitlines()
    keep: list[str] = []
    for i, ln in enumerate(lines):
        if any(g in ln for g in grep):
            lo, hi = max(0, i - 2), min(len(lines), i + 3)
            keep.append(f"{i+1:5d}| " + "\n     | ".join(lines[lo:hi]))
    return "\n".join(keep) if keep else f"(스텁에 {grep} 없음)"


def probe_submodules(sdk: Any) -> str:
    """`g1` / `s1` 같은 Python 서브모듈에 무엇이 있는가."""
    L: list[str] = []
    for sub in list_submodules(sdk):
        if sub in ("galbot_sdk", "galbot_sdk_logger"):
            continue
        try:
            mod = __import__(f"{sdk.__name__}.{sub}", fromlist=[sub])
        except Exception as e:
            L.append(f"\n── {sub}: import 실패 {type(e).__name__}: {e}")
            continue
        names = [n for n in dir(mod) if not n.startswith("_")]
        L.append(f"\n── {sdk.__name__}.{sub}  ({len(names)}개 심볼)")
        for n in names[:60]:
            try:
                o = getattr(mod, n)
            except Exception:
                continue
            if hasattr(o, "__members__"):
                L.append(f"    {n}  [enum] {list(o.__members__)[:12]}")
            elif inspect.isclass(o):
                L.append(f"    {n}  [class]")
            elif callable(o):
                doc = (getattr(o, '__doc__', '') or '').strip().splitlines()
                L.append(f"    {n}  [fn] {doc[0] if doc else ''}")
            else:
                L.append(f"    {n} = {repr(o)[:60]}")
    return "\n".join(L) if L else "(서브모듈 없음)"


def self_test() -> int:
    """SDK 없이 획득 사다리를 검증한다. 실물과 같은 모양의 가짜 모듈을 만든다."""
    import types
    print("=" * 74)
    print("sdk_entry 자체 검증 — SDK 불필요")
    print("=" * 74)
    fails = 0

    def chk(name: str, cond: bool, extra: str = "") -> None:
        nonlocal fails
        print(f"  {'✅' if cond else '❌'} {name} {extra}")
        if not cond:
            fails += 1

    # 실물과 동일한 모양: py::init<>() 없음 + get_instance 싱글톤
    class FakeRobot:
        _inst = None

        def __init__(self):
            raise TypeError("galbot_sdk.FakeRobot: No constructor defined!")

        @staticmethod
        def get_instance():
            obj = object.__new__(FakeRobot)
            FakeRobot._inst = obj
            return obj
    FakeRobot.get_instance.__doc__ = "get_instance() -> galbot_sdk.FakeRobot\n"

    # 자동 호출되면 안 되는 것들 — 실제로 불리면 흔적을 남긴다
    touched: list[str] = []

    class Danger:
        def __init__(self):
            raise TypeError("No constructor defined!")

        @staticmethod
        def init():
            touched.append("init")
            return Danger.__new__(Danger)

        @staticmethod
        def connect():
            touched.append("connect")
            return Danger.__new__(Danger)
    Danger.init.__doc__ = "init() -> galbot_sdk.Danger\n"
    Danger.connect.__doc__ = "connect() -> galbot_sdk.Danger\n"

    fake = types.ModuleType("fake_sdk")
    fake.__name__ = "fake_sdk"
    fake.FakeRobot = FakeRobot
    fake.Danger = Danger
    fake.MachineType = None

    obj, how = acquire(fake, "FakeRobot")
    chk("get_instance 싱글톤을 찾아낸다", isinstance(obj, FakeRobot), how)
    chk("  사용한 방법을 정확히 보고한다", how == "FakeRobot.get_instance()", how)

    # 안전 가드 — init/connect 는 힌트에 걸리지만 자동 호출되면 안 된다
    try:
        acquire(fake, "Danger")
        chk("init/connect 는 자동 호출 안 함", False, "호출되어 획득 성공해버림")
    except EntryNotFound:
        chk("init/connect 는 자동 호출 안 함", not touched,
            f"touched={touched}" if touched else "")

    for n, want in (("get_instance", True), ("create_robot", True),
                    ("make_thing", True), ("new_handle", True),
                    ("init", False), ("connect", False), ("open", False),
                    ("set_joint_commands", False), ("destroy", False),
                    ("release_controller", False), ("stop_base", False)):
        chk(f"  _is_autocallable({n}) == {want}",
            _is_autocallable(n) == want)

    # 심볼이 없을 때
    try:
        acquire(fake, "NoSuchClass")
        chk("없는 심볼 → EntryNotFound", False)
    except EntryNotFound:
        chk("없는 심볼 → EntryNotFound", True)

    print("\n" + "=" * 74)
    if fails:
        print(f"❌ {fails}건 실패")
        return 1
    print("✅ 전부 통과 — 획득 사다리와 안전 가드가 정상입니다.")
    return 0


def entry_report(sdk: Any) -> str:
    """진입점 조사 보고서 — `probe_sdk.py --entry` 가 출력한다.

    후보를 나열만 하지 않고 **실제로 호출해본다.** 나열만 하면 사람이 다시
    확인해야 하고, 그 왕복이 회사 시간을 쓴다.
    """
    L: list[str] = []
    L.append("=" * 78)
    L.append("진입점 조사 — GalbotRobot / GalbotMotion 을 어떻게 얻는가")
    L.append("=" * 78)
    L.append(f"\n패키지: {package_dir(sdk)}")
    L.append(f"서브모듈: {list_submodules(sdk) or '(없음)'}")

    files = list_package_files(sdk)
    L.append(f"\n패키지 파일 {len(files)}개:")
    L += [f"  {f}" for f in files[:60]]

    L.append("\n" + "=" * 78)
    L.append("핸들 획득 — 후보를 실제로 호출해본다")
    L.append("=" * 78)
    for cname in ("GalbotRobot", "GalbotMotion", "GalbotPerception",
                  "GalbotNavigation"):
        cls = getattr(sdk, cname, None)
        if cls is None:
            continue
        L.append(f"\n── {cname}")
        cf = find_class_factories(cls)
        L.append(f"  클래스 팩토리 후보: {[n for n, _ in cf] or '(없음)'}")
        for n, sig in cf:
            L.append(f"    {n}: {sig or '(docstring 없음)'}")
        try:
            obj, how = acquire(sdk, cname)
            L.append(f"  ✅ 획득 성공 — {how}")
            L.append(f"     type={type(obj).__name__}")
            for m in ("is_running", "get_joint_group_names", "get_frame_names",
                      "get_supported_chains"):
                if hasattr(obj, m):
                    try:
                        L.append(f"     {m}() → {str(getattr(obj, m)())[:160]}")
                    except Exception as e:
                        L.append(f"     {m}() → {type(e).__name__}: {str(e)[:120]}")
        except EntryNotFound as e:
            L.append("  ❌ 획득 실패")
            L += [f"     {ln}" for ln in str(e).splitlines()[:14]]

    L.append("\n" + "=" * 78)
    L.append("서브모듈 내용")
    L.append("=" * 78)
    L.append(probe_submodules(sdk))

    L.append("\n" + "=" * 78)
    L.append("galbot_sdk.pyi 타입 스텁 — 관심 심볼 주변")
    L.append("=" * 78)
    L.append(read_stub(sdk, grep=["get_instance", "class GalbotRobot",
                                  "class GalbotMotion", "def init",
                                  "class SingoriXTarget", "class TargetConfig"]))

    L.append("\n" + "=" * 78)
    L.append("__init__.py 원문")
    L.append("=" * 78)
    L.append(read_init_source(sdk))
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(self_test())
