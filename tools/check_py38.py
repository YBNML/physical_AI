#!/usr/bin/env python3
"""
python 3.8 호환성 검사 — 로봇 온보드가 3.8 이기 때문에 필요하다.

기계별 python 버전 (2026-07-31 실측)
    Mac mini / 3090   : 3.11.15
    로봇 온보드        : **3.8.10**

3.8 은 PEP 585(`list[int]`) / PEP 604(`int | None`) 를 **런타임에** 지원하지 않는다.
`from __future__ import annotations` 가 있으면 annotation 은 문자열로 남아 안전하지만,
그 import 를 빠뜨린 파일이 하나라도 있으면 온보드에서만 터진다. 그리고 온보드는
회사에 있어서 왕복 비용이 크다 — 그래서 맥에서 미리 잡는다.

검사 항목
  1. `ast.parse(feature_version=(3,8))` 로 문법 자체가 3.8 에서 파싱되는가
  2. PEP585/604 annotation 을 쓰면서 `from __future__ import annotations` 가 없는가
  3. 3.9+ 에서 추가된 stdlib API 를 쓰는가

사용:  python tools/check_py38.py
"""

from __future__ import annotations

import ast
import os
import re
import sys

# 온보드에서 실제로 돌릴 파일들. numpy 필요 여부와 무관하게 문법은 다 맞아야 한다.
TARGETS = [
    "tools/probe_sdk.py",
    "tools/sdk_entry.py",
    "tools/measure_loop_rate.py",
    "tools/fk_crosscheck.py",
    "tools/check_py38.py",
    "robot/g1_kinematics.py",
    "robot/test_kinematics.py",
    "robot/interface.py",
]

# 3.9+ 에서 추가된 것들. 문자열 검색이라 오탐이 날 수 있으므로 단어 경계를 쓴다.
NEW_API = {
    r"\.removeprefix\(": "str.removeprefix (3.9+)",
    r"\.removesuffix\(": "str.removesuffix (3.9+)",
    r"\bimport graphlib\b": "graphlib (3.9+)",
    r"\bimport zoneinfo\b": "zoneinfo (3.9+)",
    r"\bstdlib_module_names\b": "sys.stdlib_module_names (3.10+)",
    r"\bitertools\.pairwise\b": "itertools.pairwise (3.10+)",
    r"\bfunctools\.cache\b": "functools.cache (3.9+)",
    r"\banext\(": "anext (3.10+)",
    r"\baiter\(": "aiter (3.10+)",
    r"\bmath\.lcm\b": "math.lcm (3.9+)",
    r"\bmath\.nextafter\b": "math.nextafter (3.9+)",
}

# PEP585/604 를 annotation 에 쓰는 흔한 형태
PEP585 = re.compile(
    r":\s*(list|dict|tuple|set|frozenset|type)\[|"
    r"->\s*(list|dict|tuple|set|frozenset|type)\[|"
    r":\s*\w+\s*\|\s*None|->\s*\w+\s*\|\s*None")


# 이 파일 자신은 NEW_API 패턴 문자열을 정의하고 있으므로 API 스캔에서 제외한다.
# (문법 검사와 future import 검사는 그대로 받는다)
SKIP_API_SCAN = {"tools/check_py38.py"}


def check(path: str) -> list[str]:
    problems: list[str] = []
    try:
        src = open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return [f"파일 없음: {path}"]

    # 1) 3.8 문법으로 파싱되는가
    try:
        ast.parse(src, feature_version=(3, 8))
    except SyntaxError as e:
        problems.append(f"3.8 문법 오류: {e.msg} (line {e.lineno})")
    except TypeError:
        # feature_version 미지원 파이썬에서 돌릴 때
        try:
            ast.parse(src)
        except SyntaxError as e:
            problems.append(f"문법 오류: {e.msg} (line {e.lineno})")

    # 2) PEP585/604 를 쓰면서 future import 가 없는가
    has_future = "from __future__ import annotations" in src
    if PEP585.search(src) and not has_future:
        problems.append(
            "PEP585/604 annotation 을 쓰는데 "
            "`from __future__ import annotations` 가 없다 "
            "→ 3.8 에서 import 시점에 TypeError")

    # 3) 3.9+ stdlib API
    if path.replace(os.sep, "/") not in SKIP_API_SCAN:
        for pat, label in NEW_API.items():
            if re.search(pat, src):
                problems.append(f"3.9+ API 사용: {label}")

    return problems


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print("=" * 70)
    print("python 3.8 호환성 검사 (로봇 온보드가 3.8.10)")
    print("=" * 70)
    total = 0
    for t in TARGETS:
        probs = check(t)
        total += len(probs)
        mark = "✅" if not probs else "❌"
        print(f"  {mark} {t}")
        for p in probs:
            print(f"       {p}")

    print()
    if total:
        print(f"❌ {total}건 — 온보드(3.8)에서 터집니다.")
        return 1
    print("✅ 전부 3.8 호환. 온보드에서 conda 없이 시스템 python3 로 돌릴 수 있습니다.")
    print("   (probe_sdk / sdk_entry / measure_loop_rate 는 stdlib 만 씁니다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
