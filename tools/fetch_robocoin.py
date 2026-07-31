#!/usr/bin/env python3
"""
RoboCOIN Galbot G1 데이터 받기 — **parquet 만.**

왜 parquet 만인가
─────────────────
전체 18종은 164.4 GB 인데 대부분이 비디오(mp4)다. 우리 분석 3종
(목 운동 / psi 분포 / 양손 가시율)은 **관절 상태만** 쓰므로 비디오가 전혀
필요 없다. parquet 만 받으면 공개 5종이 수백 MB 수준으로 끝난다.

비디오가 필요해지는 시점은 실제로 정책을 학습할 때이고, 그건 3090 에서 한다.

공개(non-gated) 5종 — 로그인 없이 바로 받아짐:
    Galbot_G1_fold_clothes            593 ep
    Galbot_G1_fold_clothes_1          581 ep
    Galbot_G1_use_dryer               952 ep
    Galbot_G1_use_dryer_1             620 ep
    Galbot_G1_steamer_storage_baozi   228 ep
                                    ────────
                                    2,974 ep / 2,020,721 frames / 18.7 h

gated 13종은 HF 로그인 + 각 데이터셋 페이지에서 동의가 필요하다.
`--include-gated` 로 시도하되, 401/403 이 나면 안내만 하고 넘어간다.

사용
────
    # 무엇을 받을지 먼저 보기 (다운로드 안 함)
    python tools/fetch_robocoin.py --plan

    # 1개만 받아서 스키마 확인 (권장 — 첫 실행)
    python tools/fetch_robocoin.py --one --out data/robocoin

    # 공개 5종 전부
    python tools/fetch_robocoin.py --all-public --out data/robocoin

    # 받은 뒤 검증
    python tools/fetch_robocoin.py --verify --out data/robocoin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robocoin_schema import (  # noqa: E402
    PUBLIC_DATASETS, GATED_DATASETS, STATE_DIM, ACTION_DIM, STATE_IDX, FPS,
)

HF = "https://huggingface.co"
ORG = "RoboCOIN"
UA = {"User-Agent": "physical_AI/0.1"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=dict(UA))
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_files(ds: str) -> list[dict]:
    """데이터셋의 파일 목록. parquet 만 골라내기 위해."""
    try:
        raw = _get(f"{HF}/api/datasets/{ORG}/{ds}?full=true")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PermissionError(
                f"{ds} 는 gated 입니다. {HF}/datasets/{ORG}/{ds} 에서 동의 후\n"
                f"  export HF_TOKEN=<your_token>  로 토큰을 넣고 다시 실행하십시오."
            ) from e
        raise
    j = json.loads(raw)
    return j.get("siblings") or []


def download(ds: str, rel: str, out_dir: str, retries: int = 3) -> Optional[str]:
    dst = os.path.join(out_dir, ds, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst                      # 멱등 — 이미 있으면 건너뜀
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{HF}/datasets/{ORG}/{ds}/resolve/main/{rel}"
    for k in range(retries):
        try:
            data = _get(url, timeout=180)
            with open(dst, "wb") as f:
                f.write(data)
            return dst
        except Exception as e:
            if k == retries - 1:
                print(f"    ✗ {rel}: {e}")
                return None
            time.sleep(2 ** k)
    return None


def fetch_dataset(ds: str, out_dir: str, max_episodes: Optional[int] = None) -> dict:
    """meta 전부 + parquet (max_episodes 개까지)."""
    print(f"\n── {ds} " + "─" * max(0, 56 - len(ds)))
    try:
        sibs = list_files(ds)
    except PermissionError as e:
        print(f"  ⚠️ {e}")
        return {"dataset": ds, "skipped": "gated"}

    metas = [s["rfilename"] for s in sibs if s["rfilename"].startswith("meta/")]
    parqs = sorted(s["rfilename"] for s in sibs if s["rfilename"].endswith(".parquet"))
    vids = [s for s in sibs if s["rfilename"].endswith(".mp4")]

    print(f"  meta {len(metas)}개 · parquet {len(parqs)}개 · mp4 {len(vids)}개 (mp4 는 건너뜀)")

    for m in metas:
        download(ds, m, out_dir)

    take = parqs if max_episodes is None else parqs[:max_episodes]
    ok = 0
    for i, p in enumerate(take):
        if download(ds, p, out_dir):
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(take)} ...")
    print(f"  ✅ parquet {ok}/{len(take)} 받음")
    return {"dataset": ds, "meta": len(metas), "parquet": ok, "available": len(parqs)}


def verify(out_dir: str) -> int:
    """받은 데이터가 실제로 우리 스키마와 맞는지."""
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow/numpy 필요:  mamba install -n physical_ai pyarrow numpy")

    print("=" * 72)
    print("검증")
    print("=" * 72)
    bad = 0
    for ds in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, ds)
        if not os.path.isdir(d):
            continue
        info_p = os.path.join(d, "meta", "info.json")
        parqs = []
        for root, _, files in os.walk(d):
            parqs += [os.path.join(root, f) for f in files if f.endswith(".parquet")]
        if not parqs:
            continue

        print(f"\n── {ds}")
        if os.path.exists(info_p):
            j = json.load(open(info_p))
            print(f"  robot_type {j.get('robot_type')}  "
                  f"episodes {j.get('total_episodes')}  fps {j.get('fps')}")
            if j.get("robot_type") != "Galbot_G1":
                print(f"  ❌ robot_type 이 Galbot_G1 이 아닙니다 — 다른 로봇입니다!")
                bad += 1

        tbl = pq.read_table(parqs[0])
        st = np.asarray(tbl["observation.state"].to_pylist()[0], dtype=float)
        ac = np.asarray(tbl["action"].to_pylist()[0], dtype=float)
        print(f"  parquet {len(parqs)}개  state {st.shape}  action {ac.shape}")

        if st.shape[0] != STATE_DIM:
            print(f"  ❌ state 차원 {st.shape[0]} != {STATE_DIM} — 스키마 불일치")
            bad += 1
        if ac.shape[0] != ACTION_DIM:
            print(f"  ❌ action 차원 {ac.shape[0]} != {ACTION_DIM}")
            bad += 1

        # 목 관절이 정말 목처럼 보이는가 (범위 sanity)
        head = st[STATE_IDX["head"]]
        print(f"  head[{STATE_IDX['head'].start}:{STATE_IDX['head'].stop}] = "
              f"{np.round(head, 4)}  (pan 한계 ±1.5208, tilt -0.2143..+0.4936)")
        if abs(head[0]) > 1.6 or head[1] < -0.3 or head[1] > 0.6:
            print(f"  ⚠️ 목 관절 값이 URDF 한계를 벗어납니다 — 인덱스가 틀렸을 수 있습니다")
            bad += 1

        if "eef_sim_pose_state" in tbl.column_names:
            print(f"  🎯 eef_sim_pose_state 있음 — FK 대조 검증 가능")

    print("\n" + "=" * 72)
    if bad:
        print(f"❌ 문제 {bad}건. tools/robocoin_schema.py 를 확인하십시오.")
    else:
        print("✅ 스키마 일치. 다음:")
        print(f"   python tools/robocoin_analysis.py --data {out_dir} --inspect")
        print(f"   python tools/robocoin_analysis.py --data {out_dir} --all")
    return 1 if bad else 0


def plan() -> None:
    print("=" * 72)
    print("RoboCOIN Galbot G1 — 받을 것")
    print("=" * 72)
    tot_ep = tot_fr = 0
    print(f"\n공개 (로그인 불필요):")
    for ds, m in PUBLIC_DATASETS.items():
        tot_ep += m["episodes"]; tot_fr += m["frames"]
        print(f"  {ds:34s} {m['episodes']:5d} ep  {m['frames']:9,d} fr  "
              f"(전체 {m['gb']:5.2f} GB 중 parquet 만)")
    print(f"  {'합계':34s} {tot_ep:5d} ep  {tot_fr:9,d} fr  = {tot_fr/FPS/3600:.1f} h")
    print(f"\ngated ({len(GATED_DATASETS)}종, HF 로그인 + 동의 필요):")
    for ds in GATED_DATASETS[:4]:
        print(f"  {ds}")
    print(f"  ... 외 {len(GATED_DATASETS)-4}종")
    print(f"\n⚠️ mp4 는 받지 않습니다. 분석 3종은 관절 상태만 씁니다.")
    print(f"   비디오는 실제 학습 단계(3090)에서 필요해집니다.")
    print(f"\n권장 순서:")
    print(f"  1. --one          1개만 받아 스키마 확인")
    print(f"  2. --verify       스키마 일치 확인")
    print(f"  3. --all-public   나머지")


def main() -> int:
    ap = argparse.ArgumentParser(description="RoboCOIN Galbot G1 parquet 다운로드")
    ap.add_argument("--out", default="data/robocoin")
    ap.add_argument("--plan", action="store_true", help="계획만 출력")
    ap.add_argument("--one", action="store_true", help="가장 작은 것 1종만")
    ap.add_argument("--all-public", action="store_true", help="공개 5종 전부")
    ap.add_argument("--include-gated", action="store_true")
    ap.add_argument("--dataset", help="특정 데이터셋 하나")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="데이터셋당 parquet 개수 제한 (빠른 확인용)")
    ap.add_argument("--verify", action="store_true", help="받은 것 검증만")
    args = ap.parse_args()

    if args.plan:
        plan(); return 0
    if args.verify:
        return verify(args.out)

    targets: list[str] = []
    if args.dataset:
        targets = [args.dataset]
    elif args.one:
        # 가장 작은 것 = steamer_storage_baozi (228 ep)
        targets = ["Galbot_G1_steamer_storage_baozi"]
    elif args.all_public:
        targets = list(PUBLIC_DATASETS)
    else:
        ap.error("--plan / --one / --all-public / --dataset 중 하나를 지정하십시오")

    if args.include_gated:
        targets += GATED_DATASETS

    os.makedirs(args.out, exist_ok=True)
    print(f"출력: {os.path.abspath(args.out)}")
    print(f"대상: {len(targets)}종  (mp4 제외, parquet + meta 만)")
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN 감지됨 — gated 접근 시도 가능")

    results = [fetch_dataset(ds, args.out, args.max_episodes) for ds in targets]

    print("\n" + "=" * 72)
    got = sum(r.get("parquet", 0) for r in results)
    skipped = [r["dataset"] for r in results if r.get("skipped")]
    print(f"parquet {got}개 받음")
    if skipped:
        print(f"gated 로 건너뜀 {len(skipped)}종: {', '.join(skipped[:3])}...")
        print(f"  → 각 페이지에서 동의 후 HF_TOKEN 설정하고 재실행")
    print(f"\n다음: python tools/fetch_robocoin.py --verify --out {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
