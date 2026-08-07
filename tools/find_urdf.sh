#!/usr/bin/env bash
# 로봇이 실제로 쓰는 기구학 모델(URDF) 을 찾는다.
#
# 왜 필요한가 — 2026-07-31 fk-check 결과
# ──────────────────────────────────────
# base_link 절대 비교가 계통적으로 어긋났다:
#     pos median 4.65 mm   rot median 1.22°
#     관절별 모멘트암 차이 j1 +6.8mm / j2 +13.7mm / j5~j7 -3 ~ -4mm
#     방향차는 전부 0.6° 미만 → **축은 맞고 길이가 다르다**
#
# 그리고 출처를 확인해보니:
#     우리 URDF : galbot_one_golf_description   ← **Golf** 세대
#     실제 로봇 : GalbotOneFoxtrotSensor, hardware_version 2.2  ← **Foxtrot**
#
# 세대가 다르다. 관절 수와 위상은 같지만 링크 길이가 1~3% 다른 것이 정확히
# 이 증상이다. 로봇 온보드에 SDK 가 로드하는 모델이 있을 것이므로 그걸 찾는다.
#
# 찾으면 그것으로 robot/assets/g1_joints_raw.json 을 다시 만들어야 한다.
# 그게 psi / T_rel 을 데이터셋에 굽기 전에 반드시 해결해야 할 일이다.
#
# 아무것도 설치하거나 변경하지 않는다. 읽기만 한다.

set -uo pipefail

_T=12
_run() {
    if command -v timeout >/dev/null 2>&1; then timeout $_T "$@"
    elif command -v gtimeout >/dev/null 2>&1; then gtimeout $_T "$@"
    else "$@"; fi
}

echo "=============================================================="
echo "로봇의 기구학 모델(URDF) 찾기"
echo "=============================================================="
echo "  기계: $(hostname 2>/dev/null || echo ?)  $(uname -m)"
echo

echo "── 1. URDF / xacro / SRDF 파일"
found=0
for root in /data/galbot /userdata /opt/galbot /data /usr/share "$HOME"; do
    [ -d "$root" ] || continue
    while IFS= read -r f; do
        printf "  %-72s %s\n" "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
        found=1
    done < <(_run find "$root" -maxdepth 6 \
             \( -name '*.urdf' -o -name '*.URDF' -o -name '*.xacro' \
                -o -name '*.srdf' \) 2>/dev/null | head -30)
done
[ "$found" -eq 0 ] && echo "  (없음)"
echo

echo "── 2. description / model / kinematics 디렉터리"
found=0
for root in /data/galbot /userdata /opt/galbot /data; do
    [ -d "$root" ] || continue
    while IFS= read -r d; do
        echo "  $d"
        found=1
    done < <(_run find "$root" -maxdepth 5 -type d \
             \( -name '*description*' -o -name '*model*' -o -name '*urdf*' \
                -o -name '*kinemat*' -o -name '*robot*' \) 2>/dev/null | head -25)
done
[ "$found" -eq 0 ] && echo "  (없음)"
echo

echo "── 3. 캘리브레이션 / 설정 파일 (개체별 보정이 여기 있을 수 있음)"
echo "     로봇에 eyehand_calib_publish 가 돌고 있으므로 관련 파일이 있을 것이다."
found=0
for root in /userdata /data/galbot; do
    [ -d "$root" ] || continue
    while IFS= read -r f; do
        printf "  %-72s %s\n" "$f" "$(du -h "$f" 2>/dev/null | cut -f1)"
        found=1
    done < <(_run find "$root" -maxdepth 5 -type f \
             \( -name '*calib*' -o -name '*dh*.yaml' -o -name '*kinemat*' \) \
             2>/dev/null | head -25)
done
[ "$found" -eq 0 ] && echo "  (없음)"
echo

echo "── 4. /userdata/user_config 내용 (카메라 cfg 가 여기 있었다)"
if [ -d /userdata/user_config ]; then
    ls -la /userdata/user_config 2>/dev/null | head -25 | sed 's/^/  /'
else
    echo "  (없음)"
fi
echo

echo "── 5. SDK 가 열어둔 모델 파일 (실행 중 프로세스가 잡고 있는 것)"
echo "     이게 가장 확실하다 — SDK 가 실제로 읽는 파일이다."
found=0
for pid in $(pgrep -f '/data/galbot/bin/' 2>/dev/null | head -20); do
    while IFS= read -r f; do
        echo "  pid $pid: $f"
        found=1
    done < <(ls -l /proc/$pid/fd 2>/dev/null | grep -Ei 'urdf|xacro|srdf|model|calib' \
             | sed 's/.*-> //' | head -5)
done
[ "$found" -eq 0 ] && echo "  (없음 — 이미 읽고 닫았을 수 있습니다)"
echo

echo "=============================================================="
echo "다음"
echo "=============================================================="
echo "  URDF 를 찾으면 그 경로를 알려주십시오."
echo "  robot/assets/g1_joints_raw.json 을 그것으로 다시 만들어야 합니다."
echo "  (현재는 galbot_one_golf_description = Golf 세대에서 뽑은 것이고,"
echo "   이 로봇은 Foxtrot hw 2.2 라 링크 길이가 1~3% 다릅니다)"
