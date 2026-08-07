"""
GalbotSDK 핸들 획득 — `GalbotRobot` / `GalbotMotion` 을 어떻게 얻는가

답 — 2026-07-31 3090 실측
─────────────────────────

    motion = galbot_sdk.g1.GalbotMotion()      # ← 이것이 실제로 성공한 경로

**로봇별 서브모듈(`g1` / `s1`)에 구상 클래스가 있다.** 최상위의 동명 클래스는
`py::init<>()` 없는 베이스라 직접 생성하면 실패한다:

    >>> galbot_sdk.GalbotMotion()
    TypeError: galbot_sdk.GalbotMotion: No constructor defined!

🔴 **`get_instance()` 를 부르지 마십시오.** 후보로 잡히지만 최상위 베이스
클래스에서 호출하면 **SIGSEGV 로 프로세스가 죽는다** (probe-live 에서 실측:
"Segmentation fault (core dumped)"). try/except 로 못 잡는다.
그래서 `_RISKY_AUTOCALL` 에 넣어 기본 차단하고, `allow_risky=True` 를 명시해야만
시도한다.

정정 이력: 처음엔 `get_instance` 를 "확정 경로" 로 박았는데 틀렸다.
후보 목록에 있다는 것과 실제로 동작한다는 것은 다르고, 그 차이를 확인하지 않고
확정이라고 쓴 것이 실수였다.

`__init__.py` 는 팩토리를 정의하지 않는다. python 버전/아키텍처에 맞는 `.so` 를
골라 동적 로드하고 공개 심볼을 재수출할 뿐이다.

부수 발견
─────────
  - `galbot_sdk.pyi` **타입 스텁이 동봉돼 있다.** pybind11 docstring 보다
    풍부한 소스다 — `read_stub()` 참조.
  - Python 서브모듈 `g1` / `s1` — `probe_submodules()` 참조.
  - `.so` 가 python 3.8~3.14 용으로 7개 동봉돼 있다.
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

# 🔴 **프로세스를 죽이는 것들.** try/except 로 못 잡는다 — segfault 는 예외가
#    아니라 SIGSEGV 라서 프로세스가 통째로 죽고 출력도 날아간다.
#
#    2026-07-31 실측: `make probe-live` 가 여기서 "Segmentation fault (core
#    dumped)" 로 죽었다. 최상위 GalbotRobot/GalbotMotion 은 py::init<>() 없는
#    베이스 클래스이고, 인스턴스가 없는 상태에서 `get_instance()` 를 부르면
#    C++ 쪽에서 null 을 역참조하는 것으로 보인다.
#
#    실제 경로는 로봇별 서브모듈이다: `galbot_sdk.g1.GalbotMotion()`.
#    그래서 아래 이름들은 `allow_risky=True` 를 명시하지 않는 한 호출하지 않는다.
_RISKY_AUTOCALL = ("get_instance",)


def _is_autocallable(name: str, allow_risky: bool = False) -> bool:
    """자동 호출해도 안전한 팩토리 이름인가.

    허용: `*instance*` 또는 create_/make_/new_/get_ 으로 시작하는 것.
    거부: 상태를 바꿀 수 있는 접두사, 그리고 프로세스를 죽이는 것으로 실측된 이름.
    """
    lo = name.lower()
    if not allow_risky and lo in _RISKY_AUTOCALL:
        return False
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


def acquire(sdk: Any, cls_name: str, allow_risky: bool = False) -> tuple[Any, str]:
    """`cls_name` 인스턴스를 얻는다. (객체, 사용한 방법) 반환.

    전략을 순서대로 시도하고, 성공한 방법을 문자열로 돌려준다 —
    그 문자열을 결과 JSON 에 기록해두면 다음 사람이 재현할 수 있다.
    """
    cls = getattr(sdk, cls_name, None)
    if cls is None:
        raise EntryNotFound(f"galbot_sdk 에 `{cls_name}` 심볼이 없습니다.")

    tried: list[str] = []

    # 0) 로봇별 서브모듈 — 2026-07-31 실측으로 **이것이 실제 경로**다.
    #    `galbot_sdk.g1.GalbotMotion()` 이 성공했다. g1/s1 서브모듈에 로봇별
    #    구상 클래스가 있고, 최상위의 동명 클래스는 py::init<>() 없는 베이스다.
    for sub in ("g1", "s1"):
        if sub not in list_submodules(sdk):
            continue
        try:
            mod = __import__(f"{sdk.__name__}.{sub}", fromlist=[sub])
        except Exception as e:
            tried.append(f"import {sdk.__name__}.{sub}  →  {type(e).__name__}: {e}")
            continue
        alt = getattr(mod, cls_name, None)
        if alt is None:
            continue
        try:
            return alt(), f"{sdk.__name__}.{sub}.{cls_name}()"
        except Exception as e:
            tried.append(f"{sdk.__name__}.{sub}.{cls_name}()  →  "
                         f"{type(e).__name__}: {e}")

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
        if not _is_autocallable(name, allow_risky):
            why = ("프로세스를 죽인 이력 — allow_risky 필요"
                   if name.lower() in _RISKY_AUTOCALL else "자동 호출 비허용")
            tried.append(f"{cls_name}.{name}()  →  건너뜀 ({why})")
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
        if not _is_autocallable(name, allow_risky):
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

    chk("get_instance 는 기본적으로 차단 (segfault 이력)",
        not _is_autocallable("get_instance"))
    chk("  allow_risky=True 면 허용",
        _is_autocallable("get_instance", allow_risky=True))
    obj, how = acquire(fake, "FakeRobot", allow_risky=True)
    chk("allow_risky 로 get_instance 싱글톤을 찾아낸다",
        isinstance(obj, FakeRobot), how)
    chk("  사용한 방법을 정확히 보고한다", how == "FakeRobot.get_instance()", how)
    try:
        acquire(fake, "FakeRobot")
        chk("기본값에서는 get_instance 를 안 부른다", False, "불렸다")
    except EntryNotFound:
        chk("기본값에서는 get_instance 를 안 부른다", True)

    # 안전 가드 — init/connect 는 힌트에 걸리지만 자동 호출되면 안 된다
    try:
        acquire(fake, "Danger")
        chk("init/connect 는 자동 호출 안 함", False, "호출되어 획득 성공해버림")
    except EntryNotFound:
        chk("init/connect 는 자동 호출 안 함", not touched,
            f"touched={touched}" if touched else "")

    for n, want in (("create_robot", True),
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


def entry_report(sdk: Any, emit=None) -> str:
    """진입점 조사 보고서 — `probe_sdk.py --entry` 가 출력한다.

    ⚠️ **즉시 출력한다.** 초기 구현은 문자열을 모아 마지막에 한 번 출력했는데,
    SDK 가 중간에 segfault 로 죽으면 아무것도 안 남는다 (실측: sdk_entry_*.txt 가
    두 번 연속 빈 파일이었다). python -u 를 붙여도 **출력 자체가 없으면 소용없다.**
    그래서 emit 으로 줄 단위 즉시 출력한다.

    ⚠️ **초기화되지 않은 핸들에 데이터 메서드를 부르지 않는다.** init() 전에
    get_joint_names() 를 부르면 segfault 다 (실측). 여기서는 획득까지만 한다.
    """
    emit = emit or (lambda s: print(s, flush=True))
    L: list[str] = []

    def out(line: str) -> None:
        L.append(line)
        emit(line)
    out("=" * 78)
    out("진입점 조사 — GalbotRobot / GalbotMotion 을 어떻게 얻는가")
    out("=" * 78)
    out(f"패키지: {package_dir(sdk)}")
    out(f"서브모듈: {list_submodules(sdk) or '(없음)'}")

    files = list_package_files(sdk)
    out(f"\n패키지 파일 {len(files)}개:")
    for f in files[:60]:
        out(f"  {f}")

    out("\n" + "=" * 78)
    out("핸들 획득 (데이터 메서드는 부르지 않는다 — init 전이라 segfault 난다)")
    out("=" * 78)
    for cname in ("GalbotRobot", "GalbotMotion", "GalbotPerception",
                  "GalbotNavigation"):
        cls = getattr(sdk, cname, None)
        if cls is None:
            continue
        out(f"\n── {cname}")
        cf = find_class_factories(cls)
        out(f"  클래스 팩토리 후보: {[n for n, _ in cf] or '(없음)'}")
        for n, sig in cf:
            out(f"    {n}: {sig or '(docstring 없음)'}")
        try:
            obj, how = acquire(sdk, cname)
            out(f"  ✅ 획득 성공 — {how}   type={type(obj).__name__}")
        except EntryNotFound as e:
            out("  ❌ 획득 실패")
            for ln in str(e).splitlines()[:14]:
                out(f"     {ln}")

    out("\n" + "=" * 78)
    out("서브모듈 내용")
    out("=" * 78)
    for ln in probe_submodules(sdk).splitlines():
        out(ln)

    out("\n" + "=" * 78)
    out("galbot_sdk.pyi 타입 스텁 — 관심 심볼 주변")
    out("=" * 78)
    for ln in read_stub(sdk, grep=["get_instance", "class GalbotRobot",
                                   "class GalbotMotion", "def init",
                                   "class SingoriXTarget",
                                   "class TargetConfig"]).splitlines():
        out(ln)
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(self_test())


# ─────────────────────────────────────────────────────────────────────────────
# 다른 SDK 클라이언트 탐지 — 2026-07-31 온보드에서 필요해졌다
# ─────────────────────────────────────────────────────────────────────────────
#
# 관측: motion.init() 이 계속 False 였는데, 원인은 우리 세션 잔류가 아니라
#       **다른 프로세스가 SDK 를 점유**하고 있었기 때문이다.
#
#           galbot_g1_client.py --server ws://... --id galbot_g1_no2
#           LD_LIBRARY_PATH=/data/galbot/lib     ← 우리와 같은 SDK
#           CPU 132%                              ← 활발히 동작 중
#
#       GalbotRobot.init() 은 성공하는데 GalbotMotion.init() 만 실패한다.
#       즉 **Motion 은 배타적**이고 Robot 은 다중 접근을 허용하는 것으로 보인다.
#
# 왜 이게 안전 문제인가
# ─────────────────────
# GATE-1 은 팔을 실제로 움직인다. 다른 클라이언트가 동시에 로봇을 제어 중이면
# **명령원이 둘**이 되어 예측할 수 없는 움직임이 난다. 읽기 전용 도구는
# 경고만 하면 되지만, 움직이는 도구는 **막아야 한다.**

_CLIENT_MARKERS = ("galbot_g1_client", "open_bridge", "galbot_sdk",
                   "/data/galbot/lib")


def find_other_sdk_clients(exclude_self: bool = True) -> list:
    """이 기계에서 SDK 를 쓰는 **다른** 프로세스를 찾는다.

    /proc 를 읽는다 (Linux 전용, 실패해도 조용히 빈 목록). 같은 사용자의
    프로세스만 읽히므로 root 데몬은 안 보일 수 있다 — 그건 한계로 인정한다.
    """
    out = []
    me = os.getpid()
    proc = "/proc"
    if not os.path.isdir(proc):
        return out
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        p = int(pid)
        if exclude_self and p in (me, os.getppid()):
            continue
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except Exception:
            continue
        if not cmd:
            continue
        # 우리 자신의 도구는 제외
        if "tools/probe_sdk.py" in cmd or "tools/fk_crosscheck.py" in cmd \
                or "tools/measure_loop_rate.py" in cmd:
            continue
        hay = cmd
        try:
            with open(f"{proc}/{pid}/environ", "rb") as f:
                hay += " " + f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except Exception:
            pass
        if any(m in hay for m in _CLIENT_MARKERS):
            out.append({"pid": p, "cmdline": cmd[:200]})
    return out


def report_other_clients(purpose: str = "read") -> list:
    """탐지 결과를 출력한다. purpose='move' 면 위험을 강하게 알린다.

    반환: 발견된 클라이언트 목록 (호출자가 차단 여부를 결정한다)
    """
    others = find_other_sdk_clients()
    if not others:
        print("  [sdk] 다른 SDK 클라이언트 없음 ✅", flush=True)
        return others
    print("\n  ⚠️ **다른 프로세스가 SDK 를 쓰고 있습니다:**", flush=True)
    for o in others:
        print(f"     pid {o['pid']}: {o['cmdline'][:150]}", flush=True)
    if purpose == "move":
        print("\n  🔴 이 도구는 **로봇을 실제로 움직입니다.** 명령원이 둘이 되면")
        print("     예측할 수 없는 동작이 발생합니다. 상대 작업이 끝난 뒤 실행하십시오.")
    else:
        print("\n  ℹ️ 읽기 전용이라 위험하지는 않지만, **GalbotMotion 은 배타적**이라")
        print("     motion.init() 이 실패할 수 있습니다 (실측). 그 경우 상대 작업이")
        print("     끝나야 FK 대조가 가능합니다.")
    print("     ⚠️ 이 프로세스를 임의로 kill 하지 마십시오 — 다른 사람의 작업입니다.",
          flush=True)
    return others
