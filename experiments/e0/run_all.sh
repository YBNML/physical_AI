#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  E0 전체 실행 — 3 arm × 3 seed + 판정 리포트
#
#  E0 = GATE-2. 이 프로젝트에서 가장 중요한 단일 실험이고, 가능한 가장 값싼
#  반증이다. oracle 인터페이스로 저수준 정책의 **아키텍처 천장**을 잰다.
#
#  seed 3개를 반드시 다 돌려야 한다. seed 1개는 n=500 → MDE 8.9pt 라서
#  판정 규칙의 ±5pt 임계를 아예 판정할 수 없다. 3개를 pool 해야 n=1500,
#  MDE 5.1pt 가 된다.
#
#  사용:
#    bash run_all.sh --smoke                       # LIBERO/GPU 없이 배선 검증
#    bash run_all.sh --data /path/to/libero        # 본 실험 (3090 기본값)
#    bash run_all.sh --data ... --profile 1660     # 6GB 카드 프로파일
#    bash run_all.sh --report-only                 # 판정만 재출력
#
#  중단 후 재시작하면 이미 만들어진 체크포인트/결과는 건너뛴다.
#  (3090 은 다른 업무와 공유하므로 중단이 정상 상황이다.)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python 선택 순서 — Makefile/setup.sh 와 같은 규칙이다. conda activate 를
# 안 한 셸에서 그냥 `bash run_all.sh` 를 쳤을 때 시스템 python3(torch 없음)로
# 떨어져 학습 직전에 죽는 것을 막는다.
#   1) PYTHON=... 명시   2) conda 환경 physical_ai 의 python   3) python3
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
else
  _BASE="${CONDA_EXE:-}"; _BASE="${_BASE%/bin/conda}"
  [[ -z "$_BASE" ]] && _BASE="$(conda info --base 2>/dev/null || true)"
  _ENVPY="$_BASE/envs/${ENV_NAME:-physical_ai}/bin/python"
  if [[ -n "$_BASE" && -x "$_ENVPY" ]]; then PY="$_ENVPY"; else PY="python3"; fi
fi
echo "[run_all] python = $PY"

SMOKE=0
REPORT_ONLY=0
DATA="${LIBERO_DATA:-}"
SEEDS="0 1 2"
PROFILE="3090"
STEPS=60000
BATCH=64
ACCUM=1
AMP="--amp"
WORKERS=4
TRAIN_DEMOS=25
MAX_STEPS=400
EXTRA=""
BATCH_OVR=""
ACCUM_OVR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)        SMOKE=1; shift ;;
    --report-only)  REPORT_ONLY=1; shift ;;
    --data)         DATA="$2"; shift 2 ;;
    --seeds)        SEEDS="$2"; shift 2 ;;
    --profile)      PROFILE="$2"; shift 2 ;;
    --steps)        STEPS="$2"; shift 2 ;;
    --batch)        BATCH_OVR="$2"; shift 2 ;;
    --accum)        ACCUM_OVR="$2"; shift 2 ;;
    --train-demos)  TRAIN_DEMOS="$2"; shift 2 ;;
    --max-steps)    MAX_STEPS="$2"; shift 2 ;;
    --extra)        EXTRA="$2"; shift 2 ;;
    -h|--help)      sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done

# ── GPU 프로파일 ────────────────────────────────────────────────────────────
# 근거는 README.md 의 "배치 크기 / 예상 시간" 표 참고. effective batch 는
# 두 프로파일 모두 64 로 맞춘다 — 배치가 다르면 C0/C6 비교가 아니라
# 하드웨어 비교가 돼 버린다.
case "$PROFILE" in
  3090) BATCH=64; ACCUM=1; WORKERS=4; AMP="--amp" ;;   # 24GB, bf16 (Ampere)
  1660) BATCH=32; ACCUM=2; WORKERS=2; AMP="--amp" ;;   # 6GB, fp16 (Turing, tensor core 없음)
                                                       #   OOM 이면: --batch 16 --accum 4
  cpu)  BATCH=8;  ACCUM=8; WORKERS=0; AMP="" ;;
  *)    echo "알 수 없는 프로파일: $PROFILE (3090 | 1660 | cpu)"; exit 2 ;;
esac
# 프로파일보다 명시 인자가 우선. effective batch(= BATCH×ACCUM)는 64 로 유지할 것 —
# 바뀌면 C0/C6 비교가 아니라 하드웨어 비교가 된다.
[[ -n "$BATCH_OVR" ]] && BATCH="$BATCH_OVR"
[[ -n "$ACCUM_OVR" ]] && ACCUM="$ACCUM_OVR"
echo "[run_all] effective batch = $((BATCH * ACCUM))"

if [[ $SMOKE -eq 1 ]]; then
  SUB="smoke"; FLAG="--smoke"; STEPS=300; SEEDS="0 1 2"
else
  SUB="runs"; FLAG=""
  # --report-only 는 results/*.json 만 읽으므로 LIBERO 데이터가 필요 없다.
  # (여기서 막으면 실험이 끝난 뒤 판정만 다시 보는 것이 불가능해진다)
  if [[ -z "$DATA" && $REPORT_ONLY -eq 0 ]]; then
    echo "오류: --data <LIBERO 루트> 또는 env LIBERO_DATA 가 필요합니다."
    echo "      배선만 확인하려면: bash run_all.sh --smoke"
    echo "      이미 끝난 실험의 판정만 보려면: bash run_all.sh --report-only"
    exit 2
  fi
fi

CKPT_DIR="$HERE/checkpoints/$SUB"
RES_DIR="$HERE/results/$SUB"
LOG_DIR="$HERE/logs/$SUB"
mkdir -p "$CKPT_DIR" "$RES_DIR" "$LOG_DIR"

report() {
  "$PY" "$HERE/evaluate.py" --report --results "$RES_DIR"
}

if [[ $REPORT_ONLY -eq 1 ]]; then
  report; exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo " E0 — GATE-2   profile=$PROFILE  smoke=$SMOKE"
echo " seeds=$SEEDS  steps=$STEPS  batch=${BATCH}×accum${ACCUM}"
echo " 결과 → $RES_DIR"
echo "═══════════════════════════════════════════════════════════════"

# ── 0) 사전 self-test. 여기서 걸리면 GPU 시간을 태우기 전에 멈춘다 ──────────
"$PY" "$HERE/dataset.py"  --smoke
"$PY" "$HERE/models.py"   --smoke
"$PY" "$HERE/evaluate.py" --stats-test

DATA_ARG=()
[[ -n "$DATA" && $SMOKE -eq 0 ]] && DATA_ARG=(--data "$DATA")

# ── 1) 학습 — C0 / C6 만. C8 은 학습 파라미터가 0개다 ───────────────────────
for SEED in $SEEDS; do
  for ARM in c0 c6; do
    CK=$(ls "$CKPT_DIR/${ARM}_seed${SEED}."{pt,pkl} 2>/dev/null | head -1 || true)
    if [[ -n "$CK" ]]; then
      echo "[skip] 학습 $ARM seed=$SEED — 체크포인트 존재: $CK"
      continue
    fi
    echo "── 학습 $ARM seed=$SEED ──"
    "$PY" "$HERE/train.py" --arm "$ARM" --seed "$SEED" ${DATA_ARG[@]+"${DATA_ARG[@]}"} \
        --steps "$STEPS" --batch "$BATCH" --accum "$ACCUM" --workers "$WORKERS" \
        --train-demos "$TRAIN_DEMOS" $AMP $FLAG $EXTRA \
        2>&1 | tee "$LOG_DIR/train_${ARM}_seed${SEED}.log"
  done
done

# ── 2) 평가 — 3 arm 전부 ────────────────────────────────────────────────────
for SEED in $SEEDS; do
  for ARM in c0 c6 c8; do
    OUT="$RES_DIR/${ARM}_seed${SEED}.json"
    if [[ -f "$OUT" ]]; then
      echo "[skip] 평가 $ARM seed=$SEED — 결과 존재: $OUT"
      continue
    fi
    CK_ARG=()
    if [[ "$ARM" != "c8" ]]; then
      CK=$(ls "$CKPT_DIR/${ARM}_seed${SEED}."{pt,pkl} 2>/dev/null | head -1 || true)
      if [[ -z "$CK" ]]; then
        echo "오류: $ARM seed=$SEED 체크포인트가 없습니다."; exit 1
      fi
      CK_ARG=(--ckpt "$CK")
    fi
    echo "── 평가 $ARM seed=$SEED ──"
    "$PY" "$HERE/evaluate.py" --arm "$ARM" --seed "$SEED" ${DATA_ARG[@]+"${DATA_ARG[@]}"} \
        ${CK_ARG[@]+"${CK_ARG[@]}"} --train-demos "$TRAIN_DEMOS" --max-steps "$MAX_STEPS" \
        --out "$OUT" $FLAG \
        2>&1 | tee "$LOG_DIR/eval_${ARM}_seed${SEED}.log"
  done
done

# ── 3) 판정 ─────────────────────────────────────────────────────────────────
report

if [[ $SMOKE -eq 1 ]]; then
  echo "⚠️  SMOKE 실행이었습니다. 위 성공률과 판정은 합성 데이터에서 나온 것이며"
  echo "    어떤 과학적 의미도 없습니다. 배선이 끊기지 않았다는 것만 확인된 것입니다."
fi
