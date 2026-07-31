"""
RoboCOIN Galbot G1 데이터셋 스키마 — **실제 meta/info.json 에서 확인한 값**

⚠️ 중요: RoboCOIN 의 레이아웃은 GalbotSDK 의 23차원 관절 벡터와 **다르다.**
   두 개를 섞으면 값이 엉뚱한 관절에 들어가고, 그 오염은 조용하다.

    SDK (MCAP, 23-D):  leg(5) head(2) L팔(7) L그리퍼(1) R팔(7) R그리퍼(1)
    RoboCOIN (21-D) : torso(3) head(2) L팔(7) L그리퍼(1) R팔(7) R그리퍼(1)
                      ^^^^^^^^ 다리 2개가 없고 torso 3개만 기록됨

확인 방법 (2026-07-31, HF API):
    curl -sL https://huggingface.co/datasets/RoboCOIN/Galbot_G1_use_dryer_1/\\
         resolve/main/meta/info.json

공개(non-gated) Galbot G1 데이터셋 5종 — 원자료의 2,974 ep 수치와 일치:
    Galbot_G1_fold_clothes        593 ep /  520,473 frames   8.82 GB
    Galbot_G1_fold_clothes_1      581 ep /  536,843 frames   8.93 GB
    Galbot_G1_use_dryer           952 ep /  450,316 frames   6.83 GB
    Galbot_G1_use_dryer_1         620 ep /  242,533 frames   3.54 GB
    Galbot_G1_steamer_storage_baozi 228 ep / 270,556 frames  2.66 GB
    ────────────────────────────────────────────────────────────────
    합계                        2,974 ep / 2,020,721 frames  ~30.8 GB
                                        = 18.7 시간 @ 30fps

gated(auto) 13종이 추가로 있으나 HF 로그인 + 동의가 필요하다.
전체 18종 합계는 164.4 GB 이고 대부분이 비디오다 — **분석에는 parquet 만 필요.**
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# observation.state — 21차원
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM = 21

STATE_IDX = {
    "torso": slice(0, 3),          # ⚠️ SDK 의 leg(5) 가 아니라 torso(3)
    "head": slice(3, 5),           # ⚠️ SDK 는 [5:7], 여기는 [3:5]
    "left_arm": slice(5, 12),
    "left_gripper": 12,            # left_gripper_open
    "right_arm": slice(13, 20),
    "right_gripper": 20,
}

STATE_NAMES = [
    "torso_joint_1_rad", "torso_joint_2_rad", "torso_joint_3_rad",
    "head_joint_1_rad", "head_joint_2_rad",
    "left_arm_joint_1_rad", "left_arm_joint_2_rad", "left_arm_joint_3_rad",
    "left_arm_joint_4_rad", "left_arm_joint_5_rad", "left_arm_joint_6_rad",
    "left_arm_joint_7_rad",
    "left_gripper_open",
    "right_arm_joint_1_rad", "right_arm_joint_2_rad", "right_arm_joint_3_rad",
    "right_arm_joint_4_rad", "right_arm_joint_5_rad", "right_arm_joint_6_rad",
    "right_arm_joint_7_rad",
    "right_gripper_open",
]

# ─────────────────────────────────────────────────────────────────────────────
# action — 16차원. **state 와 순서가 다르다.**
#
# state 는 팔-그리퍼-팔-그리퍼 순인데 action 은 팔-팔-그리퍼-그리퍼 순이다.
# 이것도 조용한 오염원이다.
# ─────────────────────────────────────────────────────────────────────────────

ACTION_DIM = 16

ACTION_IDX = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_gripper": 14,            # ⚠️ state 는 [12], action 은 [14]
    "right_gripper": 15,
}

# ─────────────────────────────────────────────────────────────────────────────
# 부가 채널 — 여기가 이 데이터셋의 진짜 가치다
# ─────────────────────────────────────────────────────────────────────────────

EXTRA = {
    # ⚠️⚠️ 2026-07-31 확인 — **이것은 우리 FK 와 다른 것이다. 대조 검증에 쓸 수 없다.**
    #
    #   이름의 "sim" 이 핵심이다. RoboCOIN 공식 문서:
    #     "due to inconsistencies in coordinate system definitions across different
    #      robotic SDKs, we employed a simulation-based approach to obtain the
    #      end-effector poses of each robot expressed in a unified coordinate system"
    #
    #   즉 **15개 이종 로봇을 cross-embodiment 학습용으로 정규화한 좌표계**이지
    #   Galbot G1 의 실제 FK 가 아니다.
    #     - 프레임: x-forward / y-left / z-up, 원점은 base 또는 양발 중앙
    #     - 회전: **Euler** (norm 이 3.33~4.38 로 pi 를 넘음 → axis-angle 이 아님)
    #
    #   우리 URDF FK 와 실측 비교 (에피소드 1개, 921 프레임):
    #     거리 오차 median 588 mm
    #     Δ방향 상관 cos = -0.24        (단순 프레임 오프셋이면 +1.0 이어야 함)
    #     Kabsch 잔차 0.52              (순수 회전 관계도 아님)
    #     양손 거리 비율 1.33 vs 경로 길이 비율 0.78  ← 서로 모순. 단일 스케일 아님
    #     양손 거리 상관 0.89           (관련은 있으나 동일하지 않음 = retarget)
    #
    #   ⚠️ 이것을 실제 EE 포즈로 착각하고 학습하면 조용히 틀린다.
    #      실제 EE 포즈가 필요하면 observation.state 의 관절값에 **우리 FK** 를 돌릴 것.
    "eef_sim_pose_state": (12, ["left_pos_xyz(3)", "left_rot_euler(3)",
                                "right_pos_xyz(3)", "right_rot_euler(3)"]),
    "eef_sim_pose_action": (12, ["동일, action 쪽"]),

    # 그리퍼가 두 표현으로 들어 있다:
    #   observation.state[12]/[20]  = *_gripper_open      (원값)
    #   gripper_open_scale_state    = *_gripper_open_scale (정규화?)
    # ⚠️ 어느 쪽이 미터 단위인지 확인 필요. SDK 는 원값/1000 = 미터였다.
    "gripper_open_scale_state": (2, ["left", "right"]),
    "gripper_open_scale_action": (2, ["left", "right"]),

    # 이산 라벨들 — 용도 불명, RoboCOIN 논문 확인 필요
    "eef_direction_state": (2, ["int32"]),
    "eef_velocity_state": (2, ["int32"]),
    "eef_acc_mag_state": (2, ["int32"]),
    "gripper_mode_state": (2, ["int32"]),
    "gripper_activity_state": (2, ["int32"]),
}

# 카메라 3대
CAMERAS = {
    "observation.images.cam_front_head_rgb": (480, 640, 3),
    "observation.images.cam_left_wrist_rgb": (368, 640, 3),
    "observation.images.cam_right_wrist_rgb": (368, 640, 3),
}

FPS = 30

# ─────────────────────────────────────────────────────────────────────────────
# 공개 데이터셋
# ─────────────────────────────────────────────────────────────────────────────

PUBLIC_DATASETS = {
    "Galbot_G1_fold_clothes":          dict(episodes=593, frames=520_473, gb=8.82),
    "Galbot_G1_fold_clothes_1":        dict(episodes=581, frames=536_843, gb=8.93),
    "Galbot_G1_use_dryer":             dict(episodes=952, frames=450_316, gb=6.83),
    "Galbot_G1_use_dryer_1":           dict(episodes=620, frames=242_533, gb=3.54),
    "Galbot_G1_steamer_storage_baozi": dict(episodes=228, frames=270_556, gb=2.66),
}

GATED_DATASETS = [
    "Galbot_g1_fold_clothe_b", "Galbot_g1_fold_clothe_c", "Galbot_g1_fold_clothe_e",
    "Galbot_g1_steamer_storage_baozi_a", "Galbot_g1_steamer_storage_baozi_b",
    "Galbot_g1_steamer_storage_baozi_c", "Galbot_g1_steamer_storage_baozi_d",
    "Galbot_g1_steamer_storage_baozi_e", "Galbot_g1_steamer_storage_baozi_f",
    "Galbot_g1_steamer_storage_baozi_g", "Galbot_g1_steamer_storage_baozi_h",
    "Galbot_g1_steamer_storage_baozi_i", "Galbot_g1_steamer_storage_baozi_j",
]

# ─────────────────────────────────────────────────────────────────────────────
# ⚠️ 이 데이터셋의 한계 — 분석 결과 해석 시 반드시 고려
# ─────────────────────────────────────────────────────────────────────────────

CAVEATS = """
1. 작업이 3종뿐이다 — 빨래 개기 / 찜기에 만두 / 건조기 사용.
   전부 **테이블 위 준정적 양팔 작업**이고, 접촉 집약(삽입 등)이 아니다.
   → 목 운동 통계가 이 3작업에 편향된다. 작업별로 나눠 보고할 것.

2. **F/T 채널이 없다.** wrench 관련 질문은 여기서 답할 수 없다.
   컨버터 포크 후 자체 수집으로만 얻는다.

3. **다리 관절이 없다** (torso 3개만). 토르소 높이 변화가 기록되는지
   불확실하므로, "환경 고정 목표에서 미관측 10 DoF" 문제를 이 데이터로는
   완전히 재현할 수 없다.

4. ⚠️ eef_sim_pose_* 는 **cross-embodiment 정규화 좌표계**이지 G1 의 실제 FK 가
   아니다 (공식 문서 확인). Euler 회전, x-forward/y-left/z-up, 원점은 base.
   우리 FK 와 588mm 차이나고 Δ방향 상관이 -0.24 다. **FK 검증에 쓸 수 없고,
   이것을 실제 EE 포즈로 알고 학습하면 조용히 틀린다.**
   → 실제 EE 포즈는 observation.state 관절값 + 우리 FK 로 계산할 것.

5. `*_gripper_open` 과 `*_gripper_open_scale` 의 단위 관계 불명.
"""


def summary() -> str:
    tot_ep = sum(d["episodes"] for d in PUBLIC_DATASETS.values())
    tot_fr = sum(d["frames"] for d in PUBLIC_DATASETS.values())
    tot_gb = sum(d["gb"] for d in PUBLIC_DATASETS.values())
    return (
        f"RoboCOIN Galbot G1 (robot_type='Galbot_G1' 확인)\n"
        f"  공개 5종  : {tot_ep:,} ep / {tot_fr:,} frames / "
        f"{tot_fr/FPS/3600:.1f} h / {tot_gb:.1f} GB\n"
        f"  gated 13종: 추가 (HF 로그인 + 동의 필요)\n"
        f"  state {STATE_DIM}-D (SDK 23-D 와 다름), action {ACTION_DIM}-D\n"
        f"  카메라 3대, {FPS} fps\n"
        f"  ⚠️ eef_sim_pose_state 는 cross-embodiment 정규화 좌표계 — 실제 FK 아님"
    )


if __name__ == "__main__":
    print(summary())
    print(CAVEATS)
