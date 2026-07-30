I independently verified the load-bearing facts rather than relying on the research summary. Everything marked **[VERIFIED HERE]** I established today by parsing Galbot's official Apache-2.0 `galbot_one_golf_description` URDF and by running the official MJCF in MuJoCo on this M4 Mac mini.

---

# Galbot G1 — Design Questions Resolved Against the Real Robot

## 0. What I verified myself (not taken on trust)

I cloned `github.com/GalaxyGeneralRobotics/galbot_one_golf_description` (official Galbot GitHub org, Apache-2.0) and parsed `urdf/galbot_one_golf.urdf` directly. "Galbot One Golf" is the G1. Local artifacts: `/private/tmp/claude-501/-Users-khj-YBNML-macmini-physical-AI/000857d0-9f55-4208-9ae8-2ba918f3b08a/scratchpad/g1desc/`

This resolved **five items the platform research explicitly listed as unverified**, including the single most important one (the link tree). It also **falsifies one thing the research assumed**.

---

## Q1 — Arm command interface: **POSITION-ONLY. Confirmed.**

The official GalbotSDK C++/Python API reference says verbatim, for both `set_joint_commands` and `execute_joint_trajectory`:

> "For standard joints (head, legs, arms), only `JointCommand::position` is effective in current versions; velocity, acceleration, and effort are currently ignored."

The `JointCommand` struct *carries* `position, velocity, acceleration, effort (N·m), Kp, Kd` — the plumbing for impedance control is visibly there — but for the arms every field except `position` is discarded. There is no impedance mode, no variable-stiffness mode, no Cartesian wrench channel. `set_end_effector_command()` takes poses and frames only; no stiffness argument.

**One exception, and it matters:** "For gripper joints, the position field represents gripper width and both velocity and effort fields are supported and effective." Force-limited grasping *is* available — but only at the gripper.

### Consequence for the design

**Reframe Model 2 honestly. It is not a cerebellum.** It cannot modulate torque, stiffness, or damping. Everything the biological metaphor implies about compliance is unreachable through this interface. Since Model 2's actuation channel is byte-for-byte identical to what an IK solver would use, its value must come entirely from *choosing better joint targets* — not from a different kind of command.

**What a learned low level can genuinely do that analytic IK cannot, on this specific robot:**

1. **Vision-conditioned obstacle avoidance.** This is the strongest one and it is a documented hole. `GalbotMotion` states plainly: *"galbotMotion does not have real-time obstacle perception. When `enable_collision_check=true`, collision checking is evaluated against self-collision and the Motion-side environment objects that the user loads manually via `add_obstacle()`."* And: *"integrating real-time perception into galbotMotion is a planned future feature and has limited internal validation."* A policy that implicitly avoids what it sees beats the shipped planner. This is defensible.
2. **Learned redundancy resolution** — see Q2. The null-space posture in human teleop is task- and clutter-dependent in ways a fixed IK cost function is not.
3. **Bimanual coordination and mutual-arm collision** across the 417 mm shoulder separation **[VERIFIED HERE]**.
4. **Admittance behavior** — F/T in, position-target offset out. This is real but *bandwidth-limited by the loop rate*, which is undocumented (see week-1 measurement).

**What it cannot do: learned compliance.** Drop that claim from the pitch.

**Unverified lead worth one email to Galbot engineering:** the SDK exposes `LEFT_ARM_PVT_BYPASS_CTRL` / `RIGHT_ARM_PVT_BYPASS_CTRL` alongside the normal `*_PVT_CTRL` controllers. "PVT" = Position-Velocity-Torque. Zero documentation exists on what "bypass" bypasses. Likewise the raw `publish_target(SingoriXTarget)` path is described as the lowest-level channel and its fields are not enumerated publicly. If either forwards effort or Kp/Kd, the entire compliance-oriented version of Model 2 becomes viable. **This is the highest-value question to ask the vendor and it is likely answerable only by a sales/FAE contact.**

---

## Q2 — Redundancy: 7-DoF confirmed, and the arm is a textbook S-R-S

**7-DoF per arm is confirmed four independent ways** — the official spec table (手臂 7x2), the official 23-D joint vector (`left_arm_joint1..7`, `right_arm_joint1..7`), academic papers, and now the URDF itself.

**[VERIFIED HERE] — new result the research did not have.** I computed the world-frame position and axis direction of all 7 left-arm joints at zero configuration and tested axis concurrency by least squares. The result is exact (residual 0.000 mm):

| Cluster | Joints | Common point | Structure |
|---|---|---|---|
| Shoulder | j1, j2, j3 | `[0.1069, 0.2084, 0.7257]` | **spherical, 3 axes concurrent** |
| Elbow | j4 | `[0.1069, 0.5584, 0.7257]` | single revolute |
| Wrist | j5, j6, j7 | `[0.1069, 0.9184, 0.7257]` | **spherical, 3 axes concurrent** |

Upper arm = **0.350 m**, forearm = **0.360 m**, total **0.710 m** — matching the official 710 mm shoulder-to-wrist spec exactly.

**The G1 arm is a canonical 7-DoF Spherical–Revolute–Spherical (S–R–S) manipulator.** This is much better news than a generic 7R.

### The exact fix

A 6-DoF pose leaves a **1-dimensional self-motion manifold per arm**: with the wrist center fixed, the elbow sweeps a circle about the shoulder→wrist axis. Parameterize it with the classical **arm angle (swivel angle) ψ**:

> ψ = the signed angle of the elbow point about the shoulder→wrist axis, measured from a reference plane. Use the plane containing the shoulder→wrist line and the `torso_base_link` vertical axis as the ψ = 0 reference — torso-anchored, so it is invariant to neck motion and consistent across torso heights.

**Concrete interface change:**

- **Model 1 output per arm: 8 numbers** — `[x, y, z, qx, qy, qz, qw, ψ]`, i.e. 16 numbers for both hands instead of 14.
- **Labels are free.** Compute ψ from your teleop demonstrations by forward kinematics on the recorded 7 joint angles. No extra annotation, no extra hardware. The TM01 leader arm is *isomorphic* (7-DoF leader → 7-DoF follower), so the recorded ψ is genuinely the human operator's chosen posture, not an IK artifact.
- **Because the arm is S–R–S, given `(pose, ψ)` the 7 joint angles have a closed-form analytic solution** (Shimizu et al. 2008 style), up to finitely many discrete branches. So you get an exact analytic baseline for Model 2 for free — and Model 2 now has to beat a *fully determined* analytic solver, not an underdetermined one. That is a harder but much more honest bar.
- Enforce ψ limits from the joint limits I extracted: `j1 ±3.004, j2 ±1.608, j3 ±2.917, j4 −2.568..+1.870 (left; mirrored right), j5 ±2.917, j6 −0.823..+0.735, j7 ±1.538` rad. Efforts `60/60/30/30/10/10/10` N·m, velocity `1.5` rad/s on all seven. **[VERIFIED HERE]**

⚠️ The uniform 1.5 rad/s velocity limit across all seven joints looks like a placeholder, not a measured hardware limit. Treat it as unverified.

---

## Q4 — The head-frame problem

### The full kinematic chain — **[VERIFIED HERE], no longer an assumption**

The research flagged this as its biggest unproven assumption. I resolved it from the official URDF. The tree is:

```
base_footprint
 └─ base_link                    (fixed)
     └─ omni_chassis_base_link   (fixed)   ← 3 planar DoF live BELOW here (x, y, yaw), not in the URDF
         └─ omni_chassis_leg_mount_link (fixed)
             └─ leg_base_link    (fixed)
                 └─ leg_joint1   REVOLUTE   [0.000 .. 0.937 rad]
                     └─ leg_joint2  REVOLUTE [0.000 .. 2.585]   link 0.45 m
                         └─ leg_joint3 REVOLUTE [0.000 .. 2.326] link 0.39 m
                             └─ leg_joint4 REVOLUTE [±1.591]  ← WAIST YAW (±91°)
                                 └─ leg_joint5 REVOLUTE [±0.165]
                                     └─ torso_base_link          ★ SINGLE RIGID TORSO LINK
                                         ├─ (fixed) → head_base_link → head_joint1 PAN → head_joint2 TILT → head_link2
                                         ├─ (fixed, xyz +0.208 +0.172) → left_arm_base_link
                                         └─ (fixed, xyz −0.208 +0.172) → right_arm_base_link
```

**Three findings that change the analysis:**

1. **The neck and BOTH shoulders descend from one rigid `torso_base_link` via FIXED joints.** All 5 leg/waist joints — including the ±91° waist yaw — sit strictly *below* it. The research's central assumption is **confirmed**, not assumed. This is the good news.
2. **The 5 "leg" joints are REVOLUTE, not prismatic.** It is a 5-link folding column (0.45 m + 0.39 m segments), not a telescoping lift. The research listed this as unverified; it is now settled. `leg_joint4` is the waist yaw and it is large (±91°).
3. **Neck ROM, previously unpublished anywhere:** `head_joint1` pan **±1.5208 rad = ±87.1°**; `head_joint2` tilt **−0.2143 .. +0.4936 rad = −12.3° .. +28.3°**. **The tilt range is only 40.6° total.** This has a consequence nobody has noted: *the G1 physically cannot look far down at a table by tilting its head.* To see a work surface it must fold its legs and lower the torso. **So the torso WILL move during manipulation — constantly.** Hold that thought.

### How many DoF of head→arm-base are NOT recoverable from the 14 arm joints?

**Exactly 2:** `head_joint1` (pan) and `head_joint2` (tilt). Nothing else.

Because the 5 leg/waist joints and the 3 planar base DoF are **common ancestors** of both the neck branch and both arm branches, they cancel algebraically out of the *relative* head→arm-base transform. Verified geometrically: I computed the transform at multiple leg configurations and it is invariant.

But those 2 joints are not a small 2-D corner of SE(3). The head pivot sits **0.205 m above and 0.208 m lateral** of the shoulder, so pan and tilt move the shoulder on a lever arm: **all 3 translation components and 2 of 3 rotation components** of the 6-DoF transform vary. Roughly 5 of 6 pose components move, and **none are recoverable from arm joints alone.**

Model 2's proposed input (Model 1 output + head image + wrist F/T + 14 arm joints) is **structurally missing exactly 2 scalars** required to make the head→joint mapping well-posed.

### Error magnitude at realistic motion — **[COMPUTED HERE from the real URDF]**

Sensitivity of the head→left-shoulder transform to neck motion:

| Neck error | Shoulder translation shift | Shoulder rotation shift |
|---|---|---|
| 1° | 3.6 mm | 1.00° |
| 5° | 18.2 mm | 5.00° |
| 10° | 36.3 mm | 10.00° |

End-to-end target error, for a target expressed in head frame and consumed in shoulder frame with a stale/unknown neck angle:

| Reach distance | 1° neck error | 2° | 5° | 10° |
|---|---|---|---|---|
| 0.4 m | 7.0 mm | 14.0 mm | 34.9 mm | 69.7 mm |
| **0.6 m (typical)** | **10.5 mm** | **20.9 mm** | **52.3 mm** | **104.6 mm** |
| 0.8 m | 14.0 mm | 27.9 mm | 69.8 mm | 139.4 mm |

**Rule of thumb: ≈10 mm of target error per degree of neck error, at 0.6 m reach.**

Translating to realistic motion:

- **Neck pan at 30 °/s, one 30 Hz frame of staleness (33 ms):** 1.0° → **~10 mm**. Marginal but survivable for grasping a mug; fatal for insertion.
- **Neck pan at 60 °/s, 100 ms end-to-end latency** (plausible for an off-board Mac over Ethernet): 6° → **~63 mm**. Complete miss.
- **Neck pan at 30 °/s over a 500 ms action chunk:** 15° → **~157 mm**. Catastrophic.

**This is not hypothetical.** Galbot's own converter sets `action[t] = state[t+1]` over all 23 dims, and `generate_modality_json()` lists all 23 as action names — so **in the vendor's own reference teleop recordings the neck is a commanded action dimension that moves within episodes.** The "head is fixed once positioned" assumption is refuted by vendor code.

### Now the much bigger number

Everything above is for a **body-relative** target. For an **environment-anchored** target — the actual case for any manipulation where the object sits on a table while the robot repositions — the unobservable count **jumps from 2 to 10**: 2 neck + 5 leg/waist + 3 planar base. None appear in the 14 arm joints, and **there is no base odometry channel anywhere in the 23-D state vector or in the official converter's topic list.**

The error here is **unattenuated, 1:1**:

- **Torso lift of 10 cm** during a reach → **100 mm** of target error. And recall finding (3): with only 40.6° of neck tilt, torso motion during manipulation is *mandatory*, not optional. The full travel is 650 mm.
- **Base drift of 2 cm** (omnidirectional wheel creep, or a deliberate reposition) → **20 mm**, invisible to the model.
- **Waist yaw** (±91°) → tens of centimeters.

### The fix, ranked by cost

**(b) Convert head-frame → arm-base frame at the interface, using capture-time state — DO THIS. Cheapest, most correct.**
At the moment Model 1 produces a target, compose it with the live `head_joint1/2` values sampled from the *same* synchronized observation, and hand Model 2 a target in `torso_base_link` (or `base_link`) frame. Cost: two joint reads and one 4×4 multiply, using a transform I have already computed from the public URDF. It kills the moving-frame problem *and* the environment-anchored problem in one move, because the arm-base frame is by construction invariant to neck motion, and re-anchoring each step handles torso/base motion. This is what NVIDIA's official Isaac Lab Galbot task does — every observation there is `*_in_base_frame` rooted at `base_link`. It is also what Galbot's own GraspVLA does: it predicts grasp pose **in the robot base frame**. The SDK's `set_end_effector_command` defaults to `world` frame. **Nothing in the entire ecosystem works in head-image frame. You would be the only one.**

**(a) Add the missing joint states to Model 2's input — do this too, it is nearly free.**
Widen Model 2's state input from 14 → 21: 14 arm + 2 head + 5 leg/waist. All seven extra values are already in the 23-D dataset at zero marginal cost. Even after (b) removes the geometric need for the head joints, the **5 leg/waist joints remain valuable as posture/reachability context** — the natural null-space posture ψ in your demonstrations will correlate strongly with torso height across the 650 mm range, and the policy cannot learn that correlation from variables it cannot see. Do (a) *and* (b); they are complementary, not alternatives.

**(c) Freeze the neck/torso/base in v1 — do NOT rely on this.**
Two problems. First, **I could not verify that the SDK exposes a "freeze head" mode at all** — it is not in the public quick-start manual or SDK overview, and the `HEAD_PVT_CTRL` controller lifecycle functions do not document a hold/lock semantic. You would need to hold position by continuously commanding the current head angle, which is not the same as locking. Second, and worse, **the 40.6° tilt limit means a frozen head cannot see a table without a specific torso pose** — so freezing the neck forces you to also freeze the torso, which forces one fixed working height out of a 650 mm range. You would be discarding most of the robot's workspace to avoid a two-number interface change. Not worth it. *Useful only as a controlled A/B to isolate the frame error during debugging.*

**(d) Predict directly in arm-base frame — this is the right long-term answer and it is what (b) converges to.**
The distinction between (b) and (d) is only *where* the transform happens. (b) keeps Model 1's internal representation head-centric and converts at the boundary; (d) trains Model 1 to emit base-frame poses directly. Start with (b) because it is a wrapper you can add today without retraining. Move to (d) once you have data, because it removes the transform from the latency path entirely and matches every dataset (RoboCOIN's `eef_sim_pose_state`/`eef_sim_pose_action` are in a robot frame, not head frame) and every vendor API you will ever call.

### One more head-frame tax you have not costed

**[VERIFIED HERE] There is no head camera in the official robot description.** I grepped `urdf/`, `xacro/`, `mjcf/`, and `config/`: the only camera links that exist are `left_wrist_camera_link` and `right_wrist_camera_link`. There is a `head_end_effector_mount_link` (an empty frame on `head_link2`) but **no camera link, no optical frame, no intrinsics, and no stereo baseline.** The MJCF contains **0 `<camera>` elements and 0 `<site>` elements**.

So the head-camera-to-`head_link2` extrinsic — the single most load-bearing number in a head-frame architecture — is **not published anywhere**. You must obtain it from Galbot or measure it by hand-eye calibration on real hardware. `get_camera_intrinsic()` returns intrinsics at runtime from the robot, but that is a real-robot call; it does not help you in sim, and I found no documented extrinsic accessor.

**Also: the head has no depth sensor.** The spec says 双目相机x1 (one binocular/stereo camera) and the SDK `SensorType` enum has exactly `HEAD_LEFT_CAMERA` and `HEAD_RIGHT_CAMERA`, both RGB. There is no head depth enum. Metric depth from the head requires running `FOUNDATION_STEREO` or `LIGHT_STEREO` — *learned* stereo models — which is a GPU inference cost competing with your policy for the same Orin. Meanwhile the **wrist** cameras are genuine RGB-D (RealSense D405-class per the xacro, with real depth). Note the source conflict: the MobileH2R paper calls the G1 head "a head depth camera," contradicting the manual. **Verify on your unit.**

Net: head-image frame costs you an unpublished extrinsic, a learned-stereo depth pipeline, a moving reference frame, and divergence from every dataset and API in the ecosystem. **Stop paying the head-frame tax.**

---

## Gripper — absent from the diagram, and that is a task-critical omission

**What the G1's end effector actually is:** In the shipped `G1_V2.2B` configuration, **1 DoF per side with continuous aperture**. Galbot's converter divides the raw value by 1000 to yield **gripper width in metres** — a regression target, *not* a binary open/close flag. **[VERIFIED HERE]** in the URDF: `left_gripper_joint` / `right_gripper_joint`, range `0 .. 1.703 rad`, effort 50 N·m, velocity 0.5 rad/s, driving 5 mimic joints per side (parallel-jaw linkage), TCP at `left_gripper_tcp_link` 0.14 m out. Both arms carry parallel grippers in the default preset, and the MJCF has 23 position actuators including both grippers.

⚠️ **The end-effector configuration is contradictory across sources and you must confirm it in writing before purchase.** `xacro/robot.xacro` exposes `left_ee_type` and `right_ee_type` as **independent** arguments **[VERIFIED HERE]**, and the repo ships a doc image literally named `galbot_one_golf_left_hitbot_gripper_right_suction_cup_urdf.png`. Meanwhile: Chinese product copy says **left suction + right adaptive gripper**; NVIDIA Isaac Lab's `galbot_one_charlie` asset says **left parallel gripper + right suction cup**; the `G1_V2.2B` joint vector names **both sides "gripper."** These cannot all describe the same machine. **If your unit ships with a suction cup on one arm, half your action space is a vacuum toggle rather than a graspable hand, and true bimanual coordination — the entire premise — becomes impossible.** Specify dual grippers explicitly in the purchase order.

### Where the gripper command must enter the pipeline

**Model 2's output, as a continuous width. Expand Model 2 from 14-D to 16-D.**

Reasoning:

- It **cannot** enter at Model 1 as part of a pose. A 6-DoF pose has no aperture dimension. If you want Model 1 to have a say, it should emit a *discrete grasp phase* (approach / close / lift / release), not a width.
- It **must** be Model 2's output because grasp timing is contact-conditioned, and Model 2 is the only model that sees wrist F/T. Closing the gripper at the right instant is exactly the decision that force feedback informs. This is also the one place where your F/T input has a *direct actuation consequence* rather than merely nudging a position target.
- **Training a 14-D policy on data collected through the official pipeline silently discards 9 of 23 action dimensions — including grasp open/close.** Your policy would be structurally unable to pick anything up.

**Two capabilities the diagram is throwing away:**

1. **The gripper is the ONE joint where velocity and effort commands actually work** (SDK, verbatim: *"For gripper joints, the position field represents gripper width and both velocity and effort fields are supported and effective"*). So **force-limited grasping is available on this robot right now**, even though arm-level compliance is not. For deformable or fragile objects this is significant, and it partly rescues the force-aware story.
2. If your unit has a suction cup, the command is `set_suction_cup_command` and the feedback is `SuctionCupState` with **pressure in Pascals** (negative for suction) plus success/fail states — a genuinely useful contact/grasp-confirmation signal, and a completely different action space that your model must be architected for from day one.

**Unit conversion warning:** the URDF drives the gripper in radians (0..1.703) while the SDK and dataset use metres of width. You must own that mapping. It is not documented; measure it.

---

## Wrist F/T — **YES, it is standard hardware. Your key dependency holds.**

Confirmed three independent ways:

1. **Official hardware spec table:** 腕部（左右合计）深度相机x2，**腕部六维力传感器x2** — 2 wrist six-axis force/torque sensors, listed **without an "optional" marker**.
2. **First-class SDK API:** `robot.get_force_sensor_data(GalbotOneFoxtrotSensor.LEFT_WRIST_FORCE / RIGHT_WRIST_FORCE)` returning a `ForceData` struct with `force.x/y/z` in **Newtons**, `torque.x/y/z` in **N·m**, and `timestamp_ns`.
3. **A dedicated example program** ships in the SDK: `get_force_sensor_data_example.cpp` / `.py`.

This was the stated make-or-break unknown and the answer is unambiguously yes.

### But three practical caveats, all of which cost you work

**(a) It is NOT in the default data path.** The official `mcap2lerobot` converter's `STATE_TOPICS` is only `['singorix/wbcs/sensor']` and `IMAGE_TOPICS` is the 4 cameras. Grepping the converter for force/wrench returns nothing. The output parquet schema is 7 columns with **zero F/T channels**. **Fork the converter and add the wrench channel to both recording and conversion BEFORE you collect a single episode.** Retrofitting force onto already-collected demonstrations is impossible. This is not a config flag; budget for it.

**(b) There is no public G1 data with F/T.** BAAI's RoboCOIN release (~2,974 episodes / 2.02 M frames / ~18.7 h of real G1 bimanual data — genuinely useful, download it) has no F/T channel either. So F/T is the one Model 2 input you can neither pretrain on public data nor obtain for free.

**(c) [VERIFIED HERE] It is not in the sim assets.** I grepped `urdf/`, `xacro/`, `mjcf/` for force/torque/ft_sensor. The 27 hits in the MJCF are all `forcerange` **actuator** attributes. There are **0 `<site>` elements and 0 force/torque sensor elements**. You must hand-add MuJoCo `<site>` + `<sensor type="force"/torque">` elements and **guess the mounting frame**, because the F/T sensor's mounting pose is not published.

### The fallback, and how much weaker it is

If F/T turned out absent (it does not — but the question is worth answering because you may need to degrade gracefully): **`JointState` carries per-joint `effort` in N·m AND `current` in Amperes** — you can *observe* torque even though you cannot *command* it. The converter has a `use_effort` flag exposing joint effort.

**How much weaker — three distinct degradations:**

1. **Rank deficiency.** Joint torques give you contact wrench only through Jᵀ. Wrench components lying in the null space of Jᵀ are *structurally invisible* — most obviously, any force along a joint axis. A dedicated 6-axis wrist sensor observes all six components unconditionally. This is not a resolution problem; it is an observability problem, and no filtering fixes it.
2. **Contamination.** The measured joint torque is contact torque *plus* gravity, link inertia, Coriolis, gear friction, and harmonic-drive ripple. On a geared arm, static friction alone is commonly 10–30% of rated torque, and it is hysteretic, so it does not subtract cleanly. You would need a well-identified dynamic model just to get started, and the G1's inertial parameters in the URDF are already suspect — **[VERIFIED HERE] the URDF's total mass sums to 116.18 kg against the official 92.5 kg spec**, mostly from 40 passive omni-wheel roller links, so do not trust it for dynamics without auditing.
3. **Resolution.** Realistically you would resolve contact forces to a few Newtons at best, versus the sub-Newton floor typical of a purpose-built wrist F/T cell. Call it **roughly one to two orders of magnitude worse**, with the caveat below.

⚠️ **Unverified, and it matters a great deal:** I could find **no make, model, measurement range, resolution, noise floor, sampling rate, overload limit, or mounting frame** for the G1's F/T sensors, and **no statement of whether the reading is gravity/payload-compensated or raw.** A raw uncompensated wrench with a 5 kg payload hanging off it is nearly useless until you subtract the tool weight yourself. **Ask the vendor for the F/T datasheet and the compensation semantics.** This is likely a sales-contact item.

⚠️ Also unverified: the SDK enum is named `GalbotOneFoxtrotSensor` — a *variant-specific* name — while the converter reports `robot_type = "G1_V2.2B"` and the open-source description is "Golf." The generation lineage appears to be Charlie → Foxtrot → Golf and **the joint counts differ between them** (Isaac Lab's Charlie has `leg_joint1..4`; Golf has `leg_joint1..5`). **Confirm which generation you are buying and that it carries the F/T sensors.**

### Does F/T justify the learned low level?

**Partially, and conditionally.** Because you cannot command torque, F/T can only enter as an **observation that shifts position targets** — making Model 2 an *admittance* controller. Admittance quality is bounded entirely by loop rate, and the loop rate is **undocumented everywhere**.

- If the achievable closed loop is **~30 Hz** (the dataset fps, the camera rate, the action definition — everything converges here), your admittance bandwidth is maybe 3–5 Hz. Enough for slow insertion-with-search and wiping. Not enough for impact absorption or slip arrest.
- If it is **125 Hz** (the SDK's own VLA example uses `dt = 0.008 s`), admittance becomes genuinely useful.

**At 30 Hz, the two-model split also loses its main rationale** — a separate "fast" Model 2 has little room to be meaningfully faster than Model 1, which undercuts the entire slow-planner/fast-reflex premise. This is why the loop rate is the week-one measurement.

⚠️ **Red flag:** the SDK's own `example8_real_time_control_loop.cpp`, despite its name, is a 1 Hz blocking waypoint demo with `sleep_for(1 second)` between waypoints and `max_speed = 0.1 rad/s`. **Not one example in the entire SDK demonstrates a closed-loop high-rate streaming controller, and no document anywhere states a frequency.** Do not assume a fast loop until you have measured it.

---

## Sim plan — **viable, but only on MuJoCo, and the assets are missing exactly what you need**

### The gate is PASSED — I re-verified it on this machine today

I installed MuJoCo 3.11.0 into a venv on this M4 Mac mini (arm64, 32 GB) and loaded the official `mjcf/galbot_one_golf_fixed_base.xml`. **[VERIFIED HERE]**

```
nq=33  nv=33  nu=23  nbody=65  ngeom=268  ncam=0  nsensor=21  timestep=0.002
3000 steps in 0.067 s  ->  44,735 steps/s  =  89.5x realtime  (single-threaded)
offscreen render 480x640: OK
23 actuators: leg 1-5, head 1-2, left_arm 1-7, left_gripper, right_arm 1-7, right_gripper
```

Zero load errors. Galbot ships a genuinely simulation-ready model: real inertias (91 of 104 links), real joint/effort/velocity limits, mesh collision geometry with convex decomposition, proper `<mimic>` gripper linkages, and tuned position actuators. **You can start building against true G1 kinematics today, on the hardware you already own, with zero vendor contact.** That is a far stronger starting position than most Chinese humanoid platforms offer.

### But the CUDA-only lockout is real and total

| Simulator | G1 asset? | macOS / Apple Silicon? |
|---|---|---|
| **MuJoCo / MJX** | **Official MJCF, Apache-2.0** | **Yes — verified 89.5× realtime here** |
| Isaac Sim / Isaac Lab | **Best G1 support that exists** — first-party `GALBOT_ONE_CHARLIE_CFG`, 5 registered Gym tasks, RMPFlow controllers, and 4 `isaaclab_mimic` envs for demo amplification | **No.** x86_64 + NVIDIA RTX only. Unreachable. |
| ManiSkill / SAPIEN | No G1 asset | Docs: *"no support for MacOS at the moment"* |
| RoboTwin 2.0 | **No** — 5 embodiments, G1 not among them | SAPIEN-based, same restriction |
| Genesis | No G1 asset found | Claims Apple Silicon via MPS; **untested with this MJCF** |

**So: the sim-first plan is viable, but "sim-first" here means "MuJoCo-first," and you are locked out of the richest G1 tooling in existence.** The Isaac Lab Galbot suite — the one thing that would give you photorealistic rendering and automatic demo expansion — needs x86_64 Linux + RTX.

### What the MuJoCo assets are missing, specifically for *your* architecture

**[VERIFIED HERE]** — these are not minor:

1. **`ncam = 0`.** No cameras at all in the MJCF. You cannot render *any* policy observation out of the box.
2. **No head camera link anywhere** in URDF, xacro, or MJCF. Only wrist camera links exist. The head-camera extrinsic is unpublished, so you cannot even add it correctly without hand-eye calibration on real hardware.
3. **0 sites, 0 F/T sensors.** Model 2's defining input cannot be simulated without hand-authoring sites at a mounting frame nobody has published.
4. **No photorealism.** MuJoCo rendering will not survive sim-to-real visual transfer for a VLA. Galbot's own SynGrasp-1B pipeline used MuJoCo *only for physics validation* and re-rendered everything in Isaac Sim with ray tracing for the actual training images.

### Recommended split

- **Mac mini / MuJoCo:** kinematics, the S–R–S IK + arm-angle solver, frame-transform correctness, controller and interface development, self-collision and reachability studies across the 650 mm torso range, ψ-parameterization validation, unit tests. All of this is high-value and none of it needs a GPU.
- **Do NOT plan on MuJoCo→real visual transfer.** It will not work.
- **If you need the NVIDIA half** (Isaac Lab tasks, Mimic demo amplification, photorealistic rendering): rent a cloud L4/A10 Linux instance, or buy one used RTX box. This is a few thousand dollars, not a rearchitecture. **Budget for it explicitly — do not architect as if one Mac is sufficient.**

### And a separate, harder blocker for the Mac

**The GalbotSDK is Linux-only.** README: Ubuntu 20–24, Python 3.8–3.14. The shipped binary directories are literally `linux-x86_64-gcc940` and `linux-aarch64-gcc940`. There is no macOS build, no macOS wheel, no Darwin mention anywhere in a 34,000-file repo. **The M4 Mac mini physically cannot link the SDK, cannot talk to the robot, cannot run teleop collection, and cannot host deployment.** (Technically inferred from exhaustive absence rather than an explicit statement, but treat as certain.)

Your options: (a) run inference on the onboard **AGX Orin 64 GB / 275 TOPS**, which is already on the robot and eliminates a network hop; (b) put a small Linux box on the robot LAN as SDK host and let the Mac serve inference over a socket; (c) keep the Mac for training and MuJoCo only. **Option (a) is right for Model 2.** The Orin is a stronger inference target than the M4 for this workload, and it removes exactly the latency that determines whether your F/T feedback means anything. External-host control *is* a supported mode (`system.cfg` has `device_type: "pc"`, with a documented PC/XCU/HPU IP topology) — but every control cycle then crosses Ethernet, and you have not measured that cost.

⚠️ Two procurement issues to raise with Galbot **now**, because neither is fixable in your software layer: **no ISO 10218 / ISO 13849 PL / ISO/TS 15066 / TÜV / CE evidence exists in any source** for a 92.5 kg bimanual machine intended to work near people; and the SDK ships **hardcoded root credentials over a flat LAN** (`XCU root/<redacted>`, `HPU galbot/<redacted>`), a posture no enterprise customer will accept unchanged.

---

## The wedge question — direct answer

**"Galbot ships their own VLA for this robot" is half true, and the half that is false is the half that matters.**

Here is what actually exists:

- **GraspVLA** (CoRL 2025) is published with code and weights. But: it is **CC BY-NC 4.0 — non-commercial**, ships **inference-only with no finetuning code**, its **dataset (SynGrasp-1B) is unreleased**, and — decisively — **its real-robot experiments ran on a single-arm Franka Panda, not a Galbot G1.** The paper's own Limitations section says so: *"our data generation and evaluation are conducted exclusively on the Franka Panda arm... We leave this engineering effort as future work."* **I found no paper, repo, or demo showing GraspVLA controlling a G1.**
- **AstraBrain (银河星脑)** and **GroceryVLA** have **no paper, no arXiv entry, no weights, no model size, no control rate, no benchmark, no interface spec, and no developer API.** Every claim traces to Chinese press coverage of company statements.

**So "finetune the vendor model" is not actually on the table for a commercial product.** The licence forecloses it and the artifacts do not exist.

### The real choice, and my answer

The question is not "vendor model vs. own model." It is **"finetune an open model vs. train from scratch."** And the answer is unambiguous:

**Finetune an open, permissively-licensed model (π0 / π0.5 / SmolVLA / GR00T-class) in Galbot's own LeRobot format, against Galbot's own shipped WBC.** Training from scratch is indefensible: SynGrasp-1B cost **160 × RTX 4090 for 10 days ≈ 38,400 GPU-hours** — for *single-arm, gripper-only, tabletop grasping with no force*. You cannot reproduce that, and it is not even your task.

Look at the other end of Galbot's own pipeline instead: **few-shot post-training worked with 100, 100, and 10 demonstrations.** That is the actionable number. Size your teleop budget in **hundreds of demos for adaptation**, not tens of thousands for pretraining.

### And now the harder truth about your architecture

**Cut the learned Model 2 from v1.** Not forever — from v1.

The SDK already ships your Model 2 in analytic form, and it is better than you think:

- `set_end_effector_command(poses=[[x,y,z,qx,qy,qz,qw],...], end_effector_frames, reference_frames)` — **exactly Model 1's proposed output signature, natively supported.**
- `get_wbc_end_effector_poses()` returns `lee_pose`, `ree_pose`, `head_pose` — exactly Model 1's proposed input.
- Backed by `GalbotMotion`: `inverse_kinematics`, `forward_kinematics`, `get_jacobian`, single/multi-waypoint planning, RRT/RRT*, self-collision checking, tool attach/detach.
- Streamable at 125 Hz per the vendor's own VLA example, with the docs explicitly steering you to `set_joint_commands` / `set_joint_commands_batch` for *"per-frame model inference output"* and explicitly warning **away** from `set_joint_positions` for that purpose.
- **And now, thanks to the S–R–S structure I verified, you also have a closed-form analytic IK of your own** as a second baseline.

**Galbot's own researchers did not learn a cerebellum.** In GraspVLA's real deployment they used a **hand-written Cartesian impedance controller** with Jacobian transform and singularity handling, a receding-horizon scheme over the action chunk, and a **triple-cascaded first-order Butterworth filter** (chosen over Bessel and Chebyshev-II to avoid overshoot). They had the option to learn it and chose not to.

You would be spending months training, from self-collected teleop, a neural replacement for a component the vendor ships working and the vendor's own researchers deliberately hand-wrote. **That is the weakest part of the plan and it should be the first thing cut.**

### So what IS the wedge? Force. And it is real.

Assemble the facts:

- The G1 has **2× wrist 6-axis F/T sensors in hardware**, with a clean first-party API.
- **GraspVLA uses no force and no tactile at all.** Its inputs are RGB + text + proprioception.
- **GraspVLA's own failure analysis blames force-blindness:** *"21% of failures involve objects with smooth surfaces (e.g., plastic balls) slipping during grasping, which tactile feedback might help resolve."*
- **Galbot's own blessed data converter silently DROPS force/torque** — it does not even export the `effort` field that is present in the source protobuf.
- **All public G1 datasets have zero F/T channels.**
- **The sim assets have zero F/T frames.**
- **I found no Galbot model that consumes force, anywhere.**

The vendor has first-party force hardware that **none of its published models use and its public data pipeline discards.** Meanwhile, the deployment numbers everyone cites (>95% grasp, >99.97% coffee, 370 orders/day) are all **grasp-attempt success or throughput — never task completion — and human intervention rate is never reported in any source, in either language.** A 40 m² pharmacy with a fixed 6,000-slot planogram is close to structured pick-and-place that the SDK's classical planner could largely solve; no source disambiguates whether a VLA is even in that loop.

**Contact-rich, force-aware, BIMANUAL manipulation is a genuine hole in everything Galbot has published.** That is your wedge. Not "a better VLA." Not "a learned IK."

### One design idea worth stealing outright

GraspVLA's **Progressive Action Generation** is the most transferable result in the paper: by forcing the model to emit a **2D bounding box first** — an intermediate target that cheap *internet grounding data* can also supervise — Galbot got open-vocabulary generalization from a policy trained on only 240 object categories. Web-category success jumped from 40.0 (π0) to **93.3**.

**Your Model 1 is structurally the same move**: an intermediate spatial target supervisable from non-robot data. **This is the strongest argument FOR your two-level design** — and it points at a concrete tactic: **co-train Model 1 on internet grounding/pose data** so it generalizes past the objects in your teleop set. Note the cost, though: those CoT tokens are **122 of GraspVLA's 195 ms**. The reasoning is what makes it slow, not the action head. Budget Model 1 at 1–5 Hz.

---

## Revised recommendation

**Build a single vision-language policy that emits both-hands targets in `torso_base_link` frame, and drive the robot with the vendor's shipped whole-body controller plus your own closed-form S–R–S arm-angle IK — no learned low level in v1.** Concretely: fork `galbot-mcap2lerobot` *before collecting anything* and add the wrist wrench channel plus per-joint effort to the schema, since force is your only defensible wedge and it cannot be retrofitted onto recorded episodes; define your interface as **8 numbers per hand** — `[x, y, z, qx, qy, qz, qw, ψ]` where ψ is the arm angle, whose labels you get free by FK on isomorphic TM01 teleop — and convert to arm-base frame at capture time using the live `head_joint1/2` values, killing the moving-frame problem outright; feed the policy the **full 21-D proprioceptive state** (14 arm + 2 head + 5 leg/waist), not 14, because with only 40.6° of neck tilt the torso must move constantly and the policy cannot learn reachability it cannot observe; output **16-D** (14 arm + 2 continuous gripper widths in metres) so the robot can actually grasp; initialize from a **permissively-licensed open VLA** (π0-class) finetuned on RoboCOIN's ~2,974 real G1 episodes plus a few hundred of your own force-annotated demos, never from CC-BY-NC GraspVLA weights; use MuJoCo on the Mac for kinematics, IK, and interface correctness, but run inference on the robot's own **AGX Orin**, not over Ethernet from the Mac — the SDK has no macOS build, so the Mac cannot talk to the robot regardless. Reintroduce a learned Model 2 **only** at the specific point where you can demonstrate the analytic WBC failing — most likely vision-conditioned obstacle avoidance, since `GalbotMotion` has no real-time obstacle perception by the vendor's own admission — and keep Model 1's output contract fixed at that 8-per-hand pose so a learned Model 2 can be swapped in later at the identical interface.

### The single week-one measurement

**Measure the closed-loop command bandwidth and end-to-end latency of `set_joint_commands`, from the on-robot HPU and from an external host, in the same session.**

Method: stream small position deltas to one arm joint (`left_arm_joint4`) while logging `JointState.timestamp_ns`, `position`, and `effort`. Two parts: **(1)** ramp the command rate until commands are dropped or motion becomes visibly discontinuous — that is your ceiling; **(2)** inject a swept sine (±2°, 0.5→10 Hz) and read the magnitude and phase of the measured response to get an actual Bode plot of the position-tracking loop.

**Why this one and nothing else:** the loop rate is undocumented in every source — the 274 K-character API reference never states a number, "Hz" and "frequency" and "latency" appear nowhere as figures, and the SDK's own file named `real_time_control_loop.cpp` is a 1 Hz blocking demo. Yet **every downstream decision hangs on it**: whether F/T feedback can close a meaningful admittance loop at all; whether action chunking or per-step control is right; whether a two-rate cerebrum/cerebellum split is even physically meaningful or whether everything collapses to one ~30 Hz model; how much neck-motion error your latency budget permits (at 30 °/s pan, every 33 ms of latency costs you ~10 mm at 0.6 m reach); and whether off-board inference is tolerable or the Orin is mandatory. Take this measurement before writing a line of policy code.

---

## Unverified — flagged explicitly

**Resolved by my own URDF parse today** (previously unverified in the research): the full link tree and torso-branch structure; leg joints are revolute not prismatic; neck pan/tilt ROM; S–R–S arm structure with 0.35/0.36 m links; head camera absent from all assets; F/T frames and cameras absent from MJCF; MuJoCo macOS performance.

**Still unverified, public sources exhausted:**

- **Arm control loop rate and WBC servo rate** — the single biggest gap. No number exists anywhere.
- **End-to-end latency** from an external PC to arm motion.
- **`*_PVT_BYPASS_CTRL` semantics** — whether "bypass" unlocks torque or gain scheduling. Undocumented; enum names are the only evidence.
- **`SingoriXTarget` field list** — the deepest exposed control channel; capabilities unknown.
- **F/T sensor datasheet**: make, model, range, resolution, noise floor, sample rate, overload limit, mounting frame, and **whether readings are gravity/payload-compensated**.
- **Head camera**: model, native resolution, FOV, stereo baseline, and **the extrinsic to `head_link2`** — the load-bearing number for any head-frame design. Also whether the head produces depth onboard (manual says stereo RGB; MobileH2R says "depth camera" — direct conflict).
- **Whether the SDK can lock/freeze the neck** during manipulation.
- **Which generation ships** (Charlie / Foxtrot / Golf / `G1_V2.2B`) and therefore which asset matches your hardware — joint counts differ between them.
- **Which end effector ships by default, and its handedness** — three sources, three different answers.
- **Whether F/T is universal across all G1 units** (SDK enum is variant-named `GalbotOneFoxtrotSensor`).
- **Arm repeatability.** No figure exists. ⚠️ The widely-quoted "六自由度操作精度误差小于0.5毫米" is a result from the **Open6DOR simulation benchmark**, not a mechanical repeatability spec. **Do not treat 0.5 mm as arm repeatability.**
- **Arm joint velocity / acceleration / torque hardware limits.** The URDF's uniform 1.5 rad/s looks like a placeholder.
- **Safety certification** — no ISO 10218, ISO 13849 PL, ISO/TS 15066, TÜV or CE found in any source.
- **TM01 leader-arm price and whether it can be bought separately**; G1 lead time; non-China purchase and export terms.

**Likely obtainable only under NDA or from a sales contact:** F/T datasheet and compensation semantics; head-camera extrinsic and model; the true control loop rate and real-time guarantees; `PVT_BYPASS` semantics; safety certification status; generation/EE configuration of your specific unit; commercial licensing terms for GraspVLA / GroceryVLA / AstraBrain.

**One caution the research earned the hard way:** third-party English specs for the G1 are unusable. Circulating claims of 47 DoF, a 12-DoF dexterous hand, 85 kg, 10 h battery, "no LiDAR — navigates purely from vision," and "tactile sensors in the hands" all contradict the official manual (21 articulated joints excluding chassis/EE, 92.5 kg, 8 h, 3D LiDAR ×1, and **zero tactile enums in the SDK**). Build against `developer.galbot.com`, the GalbotSDK source, and the Apache-2.0 description repo only — all three are public and unusually complete, so the usual worry about Chinese commercial robots being sales-gated genuinely does not apply here.