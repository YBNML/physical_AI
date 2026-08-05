#!/usr/bin/env bash
# GalbotSDK 를 이 기계에서 찾는다.
#
# 왜 필요한가: 기계마다 설치 위치와 방식이 다르다.
#   3090(x86_64)  : /opt/galbot/galbot_sdk/linux-x86_64-gcc940/
#   로봇 온보드    : /opt/galbot/galbot_sdk 가 **비어 있었다** — 다른 데 있다.
#
# 로봇 온보드는 자기 자신이 Galbot 소프트웨어를 돌리는 기계이므로,
# SDK 가 이미 환경에 잡혀 있어 source 없이 import 되는 경우도 있다.
# 그래서 "그냥 import 해보기" 부터 시도한다.
#
# 아무것도 설치하거나 변경하지 않는다. 읽기만 한다.

set -uo pipefail
PY="${1:-python3}"

echo "=============================================================="
echo "GalbotSDK 위치 찾기"
echo "=============================================================="
echo "  기계   : $(hostname 2>/dev/null || echo ?)  $(uname -m)"
echo "  python : $PY  ($($PY -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo ?))"
echo

# ── 1. 이미 import 되는가 ────────────────────────────────────────────────────
echo "── 1. source 없이 그냥 import 되는가"
if $PY -c "import galbot_sdk, sys; print('  ✅ 됨:', galbot_sdk.__file__)" 2>/dev/null; then
    echo
    echo "  → SDK 가 이미 환경에 잡혀 있습니다. setup.sh 를 source 할 필요가 없습니다."
    echo "     make SDK_SETUP=/dev/null <타겟>   으로 실행하십시오."
    echo "     (/dev/null 을 source 하면 아무 일도 안 일어나고 통과합니다)"
else
    echo "  ❌ 안 됨"
fi
echo

# ── 2. 현재 환경 변수 ────────────────────────────────────────────────────────
echo "── 2. 관련 환경 변수"
for v in PYTHONPATH LD_LIBRARY_PATH GALBOT_SDK_PATH GALBOT_HOME; do
    val="${!v:-}"
    printf "  %-18s %s\n" "$v" "${val:-(비어 있음)}"
done
echo

# ⚠️ 검색은 반드시 시간 제한을 건다. $HOME 이 크면 find 가 몇 분씩 걸린다
#    (맥에서 실측). 진단 도구가 멈춰 있으면 쓰지 않게 된다.
#    macOS 에는 `timeout` 이 없다 (gtimeout 은 coreutils 설치 시에만). 그래서
#    없으면 백그라운드 실행 + 감시 kill 로 직접 구현한다. 이 스크립트는 로봇
#    온보드(Linux)용이지만, 맥에서 먼저 돌려보고 회사에 가는 흐름이라 양쪽 다 돼야 한다.
_TIMEOUT_S=12

_run_bounded() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$_TIMEOUT_S" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$_TIMEOUT_S" "$@"
    else
        "$@" &
        local pid=$!
        ( sleep "$_TIMEOUT_S"; kill -9 "$pid" 2>/dev/null ) &
        local watcher=$!
        wait "$pid" 2>/dev/null
        local rc=$?
        kill "$watcher" 2>/dev/null
        wait "$watcher" 2>/dev/null
        return $rc
    fi
}

_find() {  # _find <root> <maxdepth> <find-args...>
    local root="$1" depth="$2"; shift 2
    [ -d "$root" ] || return 0
    _run_bounded find "$root" -maxdepth "$depth" "$@" 2>/dev/null | head -20
}

# 홈은 얕게만 본다. 시스템 경로는 조금 더 깊게.
SYS_ROOTS="/opt /usr/local /usr/lib /srv /galbot /data"

# setup.sh 와 galbot_sdk 를 **한 번의 find 로** 같이 찾는다. 루트마다 두 번씩
# 돌면 호출 수가 배로 늘고 그만큼 최악 대기시간도 배가 된다.
echo "── 3. galbot 관련 파일 검색 (루트당 ${_TIMEOUT_S}초 제한)"
found=0
for root in $SYS_ROOTS; do
    while IFS= read -r f; do echo "  $f"; found=1; done < <(
        _find "$root" 7 \( -name 'galbot_sdk' -o -name 'galbot_sdk*.so' \
              -o \( -name 'setup.sh' -path '*galbot*' \) \))
done
while IFS= read -r f; do echo "  $f"; found=1; done < <(
    _find "$HOME" 3 \( -name 'galbot_sdk' -o -name 'galbot_sdk*.so' \
          -o \( -name 'setup.sh' -path '*galbot*' \) \))
[ "$found" -eq 0 ] && echo "  (없음)"
echo

# ── 4. /opt/galbot 실제 내용 ─────────────────────────────────────────────────
echo "── 4. /opt/galbot 트리 (깊이 3)"
if [ -d /opt/galbot ]; then
    find /opt/galbot -maxdepth 3 2>/dev/null | head -40 | sed 's/^/  /'
else
    echo "  (/opt/galbot 자체가 없습니다)"
fi
echo

# ── 6. pip 로 설치돼 있는가 ──────────────────────────────────────────────────
echo "── 5. pip 설치 여부"
$PY -m pip list 2>/dev/null | grep -i galbot | sed 's/^/  /' || echo "  (pip 목록에 galbot 없음)"
echo

# ── 7. 컨테이너 안에 있을 가능성 ─────────────────────────────────────────────
echo "── 6. docker 컨테이너"
if command -v docker >/dev/null 2>&1; then
    docker ps --format '  {{.Names}}\t{{.Image}}' 2>/dev/null | head -10 \
        || echo "  (docker 접근 불가 — sudo 필요할 수 있음)"
    echo "  → SDK 가 컨테이너 안에 있으면 컨테이너 안에서 실행해야 합니다:"
    echo "     docker exec -it <이름> bash"
else
    echo "  (docker 없음)"
fi
echo

# ── 8. 실행 중인 galbot 프로세스 ─────────────────────────────────────────────
echo "── 7. 실행 중인 galbot 관련 프로세스"
ps aux 2>/dev/null | grep -i galbot | grep -v grep | head -8 | sed 's/^/  /' \
    || echo "  (없음)"
echo "  ⚠️ 프로세스가 돌고 있으면 SDK 가 싱글톤이라 중복 연결이 막힐 수 있습니다."
echo

echo "=============================================================="
echo "다음"
echo "=============================================================="
echo "  - 1번이 ✅ 였다면:   make SDK_SETUP=/dev/null probe-entry"
echo "  - setup.sh 를 찾았다면:  make SDK_SETUP=<그 경로> probe-entry"
echo "  - 아무것도 없으면 이 출력을 그대로 공유해주십시오."
