"""
GalbotSDK 핸들 획득 — `GalbotRobot` / `GalbotMotion` 을 어떻게 얻는가

문제
────
2026-07-31 실측:

    >>> galbot_sdk.GalbotMotion()
    TypeError: galbot_sdk.GalbotMotion: No constructor defined!

이건 pybind11 이 `py::init<>()` **없이** 바인딩한 클래스에서 나는 에러다.
즉 이 클래스들은 **직접 생성하는 것이 아니라 어딘가에서 받아오는** 것이다.
`GalbotRobot` 도 마찬가지로 생성에 실패했다 (probe-live 성공 호출 0건).

단서: `galbot_sdk/__init__.py` 는 `.so` 가 아니라 **Python 소스 파일**이다.
    /opt/galbot/galbot_sdk/linux-x86_64-gcc940/lib/python/galbot_sdk/__init__.py
따라서 그 파일이 네이티브 모듈을 재수출하면서 팩토리나 래퍼를 정의하고 있을
가능성이 크다.

이 모듈은 **추측하지 않고 찾는다.** 여러 획득 전략을 순서대로 시도하고,
전부 실패하면 다음에 무엇을 봐야 하는지 정확히 알려준다.
"""

from __future__ import annotations

import inspect
import os
import pkgutil
from typing import Any, Optional

# 팩토리로 의심할 이름 패턴 (반환 타입으로 최종 확인한다)
_FACTORY_HINTS = ("create", "make", "get", "instance", "build", "new", "init",
                  "connect", "open")


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


def find_class_factories(cls: Any) -> list[tuple[str, str]]:
    """클래스 자신의 static/class 메서드 중 인스턴스를 돌려줄 법한 것."""
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
        doc = (getattr(attr, "__doc__", None) or "").strip()
        first = doc.splitlines()[0] if doc else ""
        # self 를 받지 않는 것만 = 인스턴스 없이 부를 수 있는 것
        if first and "self" not in first.split("(", 1)[-1].split(",")[0]:
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

    # 3) 클래스 자신의 팩토리 (instance/create/get_instance …)
    for name, sig in find_class_factories(cls):
        try:
            obj = getattr(cls, name)()
            if isinstance(obj, cls):
                return obj, f"{cls_name}.{name}()"
            tried.append(f"{cls_name}.{name}()  →  {type(obj).__name__} 반환 (기대 {cls_name})")
        except Exception as e:
            tried.append(f"{cls_name}.{name}()  →  {type(e).__name__}: {e}")

    # 4) 모듈 최상위 팩토리
    for name, sig in find_factories(sdk, cls_name):
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


def entry_report(sdk: Any) -> str:
    """진입점 조사 보고서 — `probe_sdk.py --entry` 가 출력한다."""
    L: list[str] = []
    L.append("=" * 78)
    L.append("진입점 조사 — GalbotRobot / GalbotMotion 을 어떻게 얻는가")
    L.append("=" * 78)
    L.append(f"\n패키지: {package_dir(sdk)}")
    L.append(f"서브모듈: {list_submodules(sdk) or '(없음)'}")

    files = list_package_files(sdk)
    L.append(f"\n패키지 파일 {len(files)}개:")
    L += [f"  {f}" for f in files[:60]]

    for cname in ("GalbotRobot", "GalbotMotion", "GalbotPerception", "GalbotNavigation"):
        cls = getattr(sdk, cname, None)
        if cls is None:
            continue
        L.append(f"\n── {cname}")
        try:
            cls()
            L.append("  ✅ 직접 생성 가능")
        except Exception as e:
            L.append(f"  ❌ 직접 생성 불가 — {type(e).__name__}: {e}")
        cf = find_class_factories(cls)
        L.append(f"  클래스 팩토리 후보: {[n for n, _ in cf] or '(없음)'}")
        mf = find_factories(sdk, cname)
        L.append(f"  모듈 팩토리 후보  : {[n for n, _ in mf] or '(없음)'}")

    L.append("\n" + "=" * 78)
    L.append("__init__.py 원문 (Python 소스이므로 진입점이 여기 있을 수 있음)")
    L.append("=" * 78)
    L.append(read_init_source(sdk))
    return "\n".join(L)
