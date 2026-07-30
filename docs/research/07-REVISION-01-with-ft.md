# REVISION — Technical Review of the Two-Model Architecture
### Supersedes the prior review. Basis: the developer's confirmed I/O diagram.
**Evidence tags used throughout:** `[M]` measured on this machine/platform · `[PR]` peer-reviewed · `[PP]` preprint · `[V]` vendor spec · `[I]` inference/analysis (mine, not measured or published)

---

## 0. Why this revision exists

The prior review was written against a looser description of the system. The confirmed diagram changes two things that invalidate parts of it:

1. **Wrist F/T is an input to Model 2.** No function of `(target pose, joint angles)` produces a wrench. The prior review's most severe objection — "the low level is a bijective kinematic map, therefore it can only be learning inverse kinematics" — was formed without this fact and is **false as stated**.
2. **Both models consume the *same* head camera frame.** "Two vision encoders" was never a real cost, and the latency figures that drove the prior review's second conclusion turn out to be a *device-placement* artifact, not an architectural property `[M]`.

A third fact emerged only because the diagram is precise: the arms are **7-DoF × 2 = 14, all arm, no gripper anywhere**. The prior review did not catch this, and it outranks everything the prior review did say.

---

## 1. What changed and what did not

| # | Prior claim | Verdict | Reason | Evidence |
|---|---|---|---|---|
| 1a | "Model 2 is inverse kinematics; the input set is degenerate" | **OVERTURNED** | Identical `(pose, q)` can be free space, light contact, or jammed contact. The wrench is the only input in the entire system carrying contact state, object stiffness/weight/CoM, or slip. Input-set degeneracy is retired. | Confirmed diagram + kinematic argument `[I]`; FILIC: "position-only observations miss key task-state information" `[PP]` |
| 1b | "…therefore delete Model 2 / demote it to a residual" | **UPHELD, premise replaced** | The *optimization* degeneracy survives even though the information degeneracy does not. Next-joint-angle MSE is ~99 % explained by `IK(target)`; force control is active in **84 % of in-contact steps but only 2.9 % of free-space steps**, so the wrench is a sparse residual the optimizer will marginalize. Correct justification is now "make the analytic term explicit so the wrench becomes the residual's dominant explanatory input **by construction**," not "it's IK." | FIRST 84 %/2.9 % `[PP]`; ForceVLA naive concat **+2.9 pp** (37.3→40.2) vs **+23.2 pp** structured `[PR]`; FoAR naive force-concat **worse** than vision-only on wiping (0.500→0.475) `[PR]` |
| 1c | "…as a **clipped** residual" | **OVERTURNED** | The entire value of the wrench is corrections during contact that are *large* relative to free-space corrections. A clip bound tuned on free-space residuals suppresses exactly the signal the sensor was added to capture. Use contact-phase gating or contact-conditioned bounds. | `[I]`, consistent with FoAR's contact-gated routing `[PR]` |
| 1d | "learned **worse** than analytic IK" | **UNSUPPORTED — withdraw** | No published work compares learned `(pose, joints, wrench) → joints` against IK + admittance on the same hardware. Evidence for force-as-input points the other way. | FACTR 21.3 % → 61.2 % → 87.5 % on unseen objects `[PP]`; sensorless-bilateral ACT nut-turning 0/5 → 5/5 with force as input only `[PP]`; Bi-ACT eye cream 50 % → 100 % `[PP]` |
| 2a | "Remove RGB from Model 2 — it doubles the most expensive component" | **OVERTURNED** | Same frame, same timestamp; one encoder serves both. Measured saving from sharing is only **14.6 ms/cycle = 6.0 %** of an MPS cycle (9.5 % CPU). The two-encoder cost was never the problem. | `[M]` |
| 2b | "…and caps the fast loop at 4.7 Hz" | **OVERTURNED** | 214/289 ms p50/p99 is **MPS-specific**. Model 2 **with its own ViT-S/16@224** on CPU: **59.5 Hz p50 / 50.4 Hz p99 / 28 Hz worst**, while Model 1 (263.8 M) concurrently holds 4.2 Hz on GPU. Both-on-MPS collapses to 15.5/11.5 Hz — mutual destruction. A **placement bug was diagnosed as an architecture bug**. | `[M]` |
| 2c | "…therefore delete the image" | **OVERTURNED (conclusion, not just reasons)** | Two independent reasons the image is load-bearing in Model 2: (i) it must resolve one redundant DoF per arm — a posture/obstacle question proprioception cannot answer; (ii) wrist F/T is **structurally blind to elbow/forearm collision**, and the elbow sweeps a **0.467 × 0.438 × 0.240 m** box at *fixed* end-effector pose. | `[M]` (elbow box, own numerical experiment); `[I]` |
| 3a | "Absolute 6-DoF pose is the worst-measured action abstraction (8 % vs 94 %)" | **OVERTURNED / MISCITED** | Mazzaglia et al. RA-L 2024 Table I classifies by **final** action. Model 2 outputs 14 joints → it sits in the **96 % oracle column, not the 8 % column**. Those 5 tasks were selected to require full-configuration control (one presses a button *with the elbow*); the baseline was MoveIt + pick_ik at defaults with no posture task; on the same paper's standard 8-task suite the redundancy-aware space is "completely in line with task space." On 6-DoF bimanual hardware, 13 000+ real rollouts: EEF-delta 89.6 % vs joint-delta 88.0 % — a **1.6-point, noise-level gap**. Quoting 8 % here gets the review dismissed. | `[PR]` arXiv:2406.04144 Table I; `[PP]` arXiv:2602.23408 |
| 3b | "Switch to chunk-wise deltas" | **UPHELD and STRENGTHENED** | ACT: **1 % at k=1 vs 44 % at k=100** — and that was at 50 Hz. EEF absolute 69.0 % → EEF delta 89.6 % (+20.6 pp). Chunk-wise beats step-wise by up to 10 pp, O(1) vs O(k) error amplification. Here the loop is 2–5 Hz, so the per-step interface is *far* worse than the published ablation implies. | `[PR]` ACT Fig. 8(a); `[PP]` arXiv:2602.23408 |
| 4 | "Head-relative frame is banned as the interface frame" | **OVERTURNED as stated; replaced by a conditional + an analytic fix** | Camera-frame action grounding **beats** base frame: +13.8 pp sim discrete, +8.0 pp sim continuous, **+10.0 pp real (58.0 → 68.0)** — *conditional on a static camera with known extrinsics*, which OC-VLA never varies. Also, `B_T_H = FK(q)·(H_T_hand)⁻¹` from the robot's own visible hands makes the transform **weakly observable, not absent**. | `[PR]` arXiv:2508.13103 Table II |
| 5 | "Splitting capacity across two models has no supporting evidence" | **OVERTURNED** | *Rate:* heterogeneous placement gives 4.2–4.5 Hz (GPU) + 59.5 Hz p50 (CPU) **concurrently**; no single model does both on this hardware `[M]`. Dual-rate is the norm (Helix S2 7–9 Hz / S1 200 Hz `[V]`; FILIC 25 Hz over a 2 kHz inner loop `[PP]`). *Data reuse:* Model 1's `(image, pose) → pose` trains from human egocentric video with no robot joints (EgoVLA, ~500 k pairs `[PP]`); Model 2 needs joints + wrench, which only robot teleop provides. That asymmetry **is** the reuse justification. | `[M]`, `[PP]`, `[V]` |
| 6a | "The two-model design is **dominated** by single-model + classical spine" | **OVERTURNED** | Domination requires being at least as good on every axis. It isn't: one model on this hardware is either 4 Hz *or* small, and a classical spine cannot exploit implicit object-property inference (ALPHA-α: liquid-filled/irregular objects 50 %→100 %, 50 %→80 %). | `[M]`, `[PP]` |
| 6b | "Build the classical-spine variant first" | **UPHELD, with the baseline raised** | Prior review understated its own comparator. The honest baseline is **IK + sensorless admittance from motor current**, not bare position control: egg-on-bread 40 % → 80 % with no F/T sensor and no learning; 0.69 N force estimation from servo signals. | `[PP]` arXiv:2603.00913 |
| **N1** | *(not raised)* **No gripper command exists anywhere** | **NEWLY RAISED — blocking** | See §3. | `[PR]` |
| **N2** | *(not raised)* **The F/T sensor has no actuation path** | **NEWLY RAISED** | Model 2 emits bare joint positions — no stiffness, no reference wrench, no selection mask. | `[PR]`/`[PP]` |
| **N3** | *(not raised)* **No timing/duration field** | **NEWLY RAISED** | Approach velocity becomes Δpose ÷ inference jitter; approach velocity is linearly proportional to impact force. | `[PR]` arXiv:2106.10969 |
| **N4** | *(not raised)* **No inter-hand relative pose** | **NEWLY RAISED** | 70 % vs 30 % measured, and not recoverable by subtraction. | `[PR]` UMI |
| **N5** | *(not raised)* **6-DoF interface for 7-DoF arms is underdetermined** | **NEWLY RAISED** | See §3(a). | `[M]` + `[PR]` |
| **N6** | *(not raised)* **Model 1's own labels are frame-corrupted** | **NEWLY RAISED** | head-frame-at-t+k ≠ head-frame-at-t. EgoVLA reprojects using world-frame camera poses precisely for this. | `[PP]` |
| **N7** | *(not raised)* **Teleop rig is a precondition** | **NEWLY RAISED** | If unilateral, the wrench channel carries no learnable structure. Check before any modeling work. | `[PP]` |

---

## 2. Revised verdict on the low-level model

**Yes — a learned low-level model is now genuinely justified. It was not, on the prior review's information.** But it is justified for a narrower job than the diagram implies, and only under stated conditions.

**What the wrench buys, decomposed into three parts:**

| Part | Learned or classical? |
|---|---|
| (1) Contact-state estimation / phase detection | **Learned wins.** Classical needs a hand-written state machine. This is where FACTR's gains live. |
| (2) Continuous force regulation along constrained directions | **Classical wins, decisively.** Admittance does this at 1 kHz with stability guarantees. A 3–5 Hz — or even 50 Hz — learned loop cannot. |
| (3) Implicit object-property inference (stiffness, weight/CoM, slip) from wrench history | **Learned wins.** Invisible to vision and kinematics; fixed-gain admittance would apply the same law to every object. ALPHA-α's 50 %→100 % gains are exactly this `[PP]`. |

The learned low level is justified **only to the extent it is capturing (1) and (3)**. Any part of it that is doing (2) is re-implementing a solved problem worse.

**Conditions under which the verdict holds:**

- **Arm command interface — currently unstated and decisive.** If the arms accept **torque or Cartesian impedance** commands, the right design is FILIC-shaped (policy emits pose/stiffness at 25–50 Hz over a 1–2 kHz impedance inner loop) and a joint-position predictor is the wrong abstraction entirely. If the arms accept **joint positions only**, the FACTR / FACTR-2 / FoAR evidence applies, the design is viable — but it delivers force-aware **switching**, not force **regulation**. FACTR's own analysis is explicit: the policy "recognizes contact events and switches strategies, rather than continuously regulating applied force magnitude" `[PP]`. **Answer this before writing code.**
- **Task class.** Justified for contact-rich tasks with unknown/varying object properties (insertion, wiping, placing deformables, lifting objects of unknown weight distribution). **Not** justified for free-space reach-and-place, where analytic IK + a posture task is strictly better and 40 000× cheaper (DLS 6×7 = 5.31 µs `[M]`).
- **Teleop rig.** If the operator did not feel contact, stop. The wrench channel will carry open-loop, demo-inconsistent human forces. ALPHA-α achieved bilateral feedback at $8 951 vs ALOHA's $20 485 `[PP]`, so this is not a cost tradeoff.

**How it must be structured:**

```
q_cmd[t..t+H] = IK_DLS(target_pose_base[t..t+H], q_now)      # analytic, 5.31 µs, exact
               + Δθ(q_now, wrench, target, vision_tokens)     # learned residual, contact-gated
stiffness/mode[t..t+H]                                        # learned or scheduled, per arm
aperture[t..t+H]                                              # learned, continuous
```

Four non-negotiables:

1. **Analytic IK term is explicit.** Removes the ~99 % of the loss that is kinematics, leaving the wrench as the residual's dominant explanatory input by construction rather than by hope.
2. **Contact gating on the force path** — a contact-phase predictor (FoAR), a per-axis selection mask (Force Policy), or FIRST-style pre-contact/contact upsampling (0.818 vs 0.670 for pre-contact vs contact-only upsampling) `[PP]`. Flat concatenation, which is what the diagram draws, is the configuration measured to yield +2.9 pp instead of +23.2 pp `[PR]`.
3. **Chunked output**, not single-step. See §3(d).
4. **An actuation path for the sensed force** — at minimum a 2-level stiffness/mode per arm, plus gripper aperture. Without it the sensor is BOM cost with no output channel.

**Keep the image in Model 2.** The prior review's deletion argument is dead on both stated grounds (`[M]`), and the image is needed for redundancy resolution and for elbow-collision awareness that F/T structurally cannot provide.

---

## 3. Newly-raised defects the diagram exposes, ranked by severity

### #1 — No gripper command exists anywhere (structural impossibility, not a degradation)

Model 1 outputs 6-DoF pose (no spare dimension). Model 2 outputs 14 arm joints. The developer's 7×2 is **all arm**. ACT's superficially identical 14 is **12 arm + 2 gripper** on 6-DoF ViperX arms — verbatim: "joint positions for two robot arms (7+7=14 DoF)" on "ViperX 6-DoF robot arms" `[PR]`. The numerical coincidence is almost certainly why this went unnoticed. Every next-best-pose architecture in this family keeps gripper first-class: HDP `a_high = (a_pose, a_grip)`, PerAct, RVT `[PR]`.

**Failure scenario:** the policy is trained, converges, produces beautiful arm trajectories, and grasping success is **0 %** — not degraded, zero — because no signal in the system can close a hand. No amount of model quality, data, or tuning recovers it.

Two sub-defects that survive even after adding an aperture output:
- **Binary aperture is insufficient.** UMI: "binary gripper actions will be unlikely to meet the precision requirement," and continuous width is *also* the implicit grasp-force channel via series-elastic finger deformation `[PR]`.
- **Gripper *state* must be an input.** A binary commanded state cannot distinguish "grasp close" (object held) from "empty close" (missed). The policy then "incorrectly transitions to post-grasp actions like pull despite lacking a secure grasp" — silently executing the entire remaining trajectory holding nothing. With real feedback: 10–30 % → 100 % under disturbance `[PP]`.
- Timing skew between gripper and arm alone cost UMI **87.5 % → 57.5 %** `[PR]`.

**Open hardware question that changes the fix size by an order of magnitude:** if "양손" means multi-finger hands rather than parallel jaws, the missing field is **24–44 dims** plus a pre-shaping decision that cannot be deferred to a contact heuristic — not 2 scalars.

### #2 — The wrist F/T sensor has no actuation path (the new input is dead)

Model 2's only output is joint positions. It can sense a wrench but can respond only through the servo's fixed, isotropic, uncommandable Kp. Every system in the literature that achieved genuine regulation added an impedance inner loop (FILIC, 2 kHz), a reference wrench + per-axis mask (Force Policy), a hybrid controller (Tactile-VLA), or a hand-coded reactive rule (FoAR, ε = 0.006 m) `[PP]`.

**Failure scenario:** the sensor is bought (~$3.9 k for 2× FT300-S `[V]`), wired, payload-inertia-identified, and the ablation in §5 shows zeroing the wrench changes nothing — because there was never an output channel through which it could matter.

### #3 — The interface is single-step, and there is no duration field (same defect, same fix)

**Chunking:** ACT measured **1 % at k=1 vs 44 % at k=100** `[PR]`. That was on a 50 Hz platform where a missed step costs 20 ms of blind time. Here the full two-stage cycle is ~430 ms p50 / 578 ms p99 under contention if both models sit on MPS — a **1.7–2.3 Hz** loop, ~21× the blind interval `[M]`. Even in the correct CPU/GPU placement, Model 1 is 4.2–4.5 Hz.

**Duration:** a waypoint with no `dt` is not a trajectory. The low level's only available velocity is Δpose ÷ inference inter-arrival interval — i.e. **commanded approach velocity becomes a function of GPU scheduling**. Approach velocity is measured to be linearly proportional to impact force over 0.02–0.16 m/s `[PR]`. Applying the measured jitter: a 10 mm step in 214 ms = 47 mm/s; the same step in 289 ms = 35 mm/s → **~35 % run-to-run contact-force variation on an identical commanded pose** `[I from M]`.

**Compliance cannot rescue this.** Measured: reducing stiffness affects post-impact jerk but has "no significant effect on impact forces, because the robot has to make the contact for the error and stiffness term in the feedback control loop to come into effect" `[PR]`. Duration and stiffness are orthogonal fields solving different problems.

**Failure scenario:** insertion succeeds in the lab and intermittently crushes or bounces off the target in the field, with no change to the commanded pose — the difference is which other process was on the GPU.

### #4 — Model 2 lacks head joint state while consuming a head-frame target and emitting base-frame joints

The prior review's "the transform is missing, therefore unidentifiable" is **too strong**: Model 2 has 14 joints and the head image, the robot's own hands are usually in frame, and `B_T_H = FK_B(q)·(H_T_hand_observed)⁻¹` is markerless eye-hand self-calibration. The transform is **weakly observable**.

The real defect is worse than absence because it is silent. The channel (i) fails exactly when the head moves — hands occluded by the grasped object, or out of a 224×224 FOV; (ii) is worst-conditioned along the optical axis, i.e. the depth direction grasping depends on; (iii) is never requested by any loss term; (iv) loses to the cheaper shortcut of memorizing a constant `B_T_H` from static background cues — a documented failure that "collapses when workspace geometry or camera placement shifts" `[PR]`.

**Quantified:** at the measured 214/289 ms, a modest **30 °/s** head pan injects **56–76 mm** of lateral error at 0.5 m reach (60 °/s → 112–151 mm); DLR needed **3.1 mm** absolute accuracy for reliable manipulation `[M]` + `[PR]`. Static neck calibration adds 8.7 mm per degree at 0.5 m; DLR's uncalibrated whole-chain figure was **21 mm** `[PR]`.

**Also corrupts Model 1's labels, not just Model 2's inputs.** Model 1 predicts a *future* pose in "the head frame," but head-frame-at-t+k ≠ head-frame-at-t. EgoVLA hits precisely this and fixes it by reprojecting future wrist poses using world-frame camera poses `[PP]`.

**Secondary interaction with the new F/T input:** the head frame is not gravity-aligned and tilts with neck pitch, so a head-frame target and a gravity-loaded wrench are in mutually inconsistent frames with a dependence on the unobserved head state `[I]`.

**Failure scenario:** clean training curves, clean in-distribution validation, then progressive drift in the field after a camera bump, a workspace re-layout, or a deployment where operators move their heads more than the training operators did. Worst failure class available: silent.

### #5 — 7-DoF × 2 makes the 6-DoF pose interface provably underdetermined

A 6-DoF pose does not determine a 7-DoF configuration. Measured on a Franka Panda model with the end-effector pose held to 1e-12 m: **joint 1 sweeps 316°, joint 5 sweeps 278°, and the elbow traverses a 0.467 × 0.438 × 0.240 m box on a circle of radius 0.239 m** `[M]`. Half a metre of elbow travel at zero end-effector motion *is* the redundancy problem, stated geometrically.

Two consequences:

- **Model 1 cannot see or command its own elbow.** Two physically distinct 14-DoF states are the same state to the cerebrum; its pose-only state is **not Markov** for the plant; it cannot detect that its own plan drives an elbow into a shelf. All posture selection is silently delegated to Model 2, which must re-infer it from a head image that in a cabinet task frequently does not contain the elbow-relevant obstacle.
- **L2 regression over the self-motion manifold produces off-manifold configurations.** Measured joint-space averaging of two valid IK solutions differing by Δψ: **10° → 1.9 mm, 30° → 17.6 mm, 45° → 38.5 mm, 60° → 50.3 mm, 90° → 96.2 mm** end-effector error `[M]`. Model 2 is conditioned on current `q`, which pins the branch by continuity, so the operative quantity is the **residual ψ spread in your demo data**, not the full 310° manifold.

**The fix is cheap and provably no-cost:** add **one scalar per arm** — the base joint j₁ (ERJ) — to Model 1's input and output. 12 → 14 numbers. Auto-labelled from recorded joints by FK at **~34 µs/sample, ~35 s per 1 M steps** `[M]`, zero human effort. Ablation: odd (twisting) joints work, even (bending) joints cause performance collapse, base joint best "by a large margin" `[PR]`. On non-confined tasks the redundancy-aware space is "completely in line with task space" — **strictly dominant**.

Use ERJ (j₁), not the geometric arm angle: ERA had a higher invalid-action rate and needed a DLS solver to control it. If you do use the arm angle, use the **stereographic SEW** definition — algorithmic singularity is unavoidable (Hairy Ball Theorem), but the conventional fixed-reference form is singular on a *full line through the workspace* and will silently produce discontinuous labels there `[PR]`.

**Also note:** wrist F/T is structurally blind to elbow/forearm contact — a wrist sensor measures only wrenches transmitted through the wrist. In confined 7-DoF operation the elbow is exactly what hits things, so neither the new sensor nor (if out of FOV) the head camera will catch it. The only remaining defence is a hard collision constraint inside the IK.

### #6 — No inter-hand relative pose

Measured **70 % (14/20) vs 30 % (6/20)** on UMI's bimanual cloth fold; described as "a critical ingredient to enable bimanual policy" `[PR]`. The failure mode is *timing* (asynchronous grasp) cured by a *spatial* field — relative pose is what lets the policy represent "both hands are in position now."

**It cannot be recovered by subtracting the two absolute poses.** The decisive argument is the loss, not the arithmetic: an L2 loss over two absolute poses penalizes a harmless common-mode 5 mm error and an object-crushing differential 5 mm error **identically**. The network never receives gradient pressure to preserve the relative constraint `[I]`. (Secondarily: UMI *measured* relative pose in a registered shared frame to 10.1 mm / 0.8°, and the effect was largest when visual overlap was small — an information-availability finding.)

### #7 — No timestamps, no head/torso joint state, no uncertainty

Lowest immediate cost, but timestamps become **mandatory** the moment durations are added (a `dt` is meaningless without a clock both models agree on) and head/torso joint state is required to compute `B_T_H` at *capture* time. Note Helix moves the torso for reach `[V]` — if this platform's torso moves, head joints alone are insufficient; you need the full camera→arm-base chain.

---

## 4. Revised minimal interface spec

For 7-DoF × 2 with wrist F/T. `H` = chunk length; recommend **H = 8** (matches Diffusion Policy's measured optimum Ta=8; ACT's k=100 at 50 Hz scales to ≈4–8 at this platform's rate).

### Model 1 — inputs

| Field | Dim | Status | Why required |
|---|---|---|---|
| Head camera image | 224×224×3 | prior | unchanged |
| Both-hands pose (head frame) | 2 × 7 | prior | unchanged. Keep head frame — OC-VLA's +10.0 pp real gain applies to pose-output models `[PR]` |
| **Redundancy scalar (base joint j₁) per arm** | 2 × 1 | **NEW** | Without it the cerebrum's state is not Markov for a 14-DoF plant and it cannot detect its own infeasible plans (§3.5). Auto-labelled, ~34 µs/sample `[M]` |
| **Gripper aperture, measured, per hand** | 2 × 1 | **NEW** | Cerebrum must know whether the hand is currently holding something to plan the next pose |
| **Grasp-contact flag per hand** | 2 × 1 | **NEW** | Disambiguates "grasp close" from "empty close"; 10–30 % → 100 % under disturbance `[PP]` |
| **Head (and torso) joint state** | n | **NEW** | Required to reproject labels and to compute `B_T_H` (§3.4) |
| **Capture timestamp** | 1 | **NEW** | `B_T_H` must be evaluated at capture time, not inference time |

### Model 1 — outputs (per waypoint, × H)

| Field | Dim | Status | Why required |
|---|---|---|---|
| Both-hands pose (head frame) | 2 × 7 | prior, **now chunked** | 1 % → 44 % from k=1 → k=100 `[PR]`; worse here at 2–5 Hz |
| **`dt` per waypoint** | 1 | **NEW** | Converts a waypoint list into a trajectory; without it approach velocity = Δpose ÷ GPU jitter → ~35 % contact-force variation `[I from M]` |
| **Gripper aperture, continuous** | 2 × 1 | **NEW** | Nothing else in the system can command a hand. Continuous, not binary — it is also the implicit grasp-force channel `[PR]` |
| **Redundancy scalar j₁ per arm** | 2 × 1 | **NEW** | The commandable half of §3.5 |
| **Inter-hand relative pose, separately supervised** | 7 | **NEW** | 70 % vs 30 % `[PR]`; not derivable by subtraction under an L2 loss `[I]` |
| Per-waypoint uncertainty (optional) | 2 × 1 | **NEW, optional** | Lets Model 2 decide when to defer to the analytic path |

**Sizing at H=8:** 8 × (14 pose + 2 aperture + 2 j₁ + 7 relative + 1 dt) ≈ **208 floats**. Against a 60.6 ms vision encoder this is < 0.1 ms `[M]` — there is no performance argument for the current minimal interface.

### Model 2 — inputs

| Field | Dim | Status | Why required |
|---|---|---|---|
| Head camera image (or **shared encoder tokens**) | 196 × d | prior, **kept** | Deletion refuted (§1, item 2c). Sharing costs 14.6 ms/cycle `[M]` |
| **Target pose in BASE frame** (converted analytically from Model 1's head-frame output) | 2 × 7 × H | **CHANGED** | Makes Model 2's input frame match its joint-space output frame; removes the unsupervised self-calibration burden. One SE(3) compose (µs) against a 214 ms budget. This is what EgoVLA does at deployment — *outside* the network `[PP]` |
| Arm joints | 14 | prior | unchanged |
| Wrist F/T | 2 × 6 | confirmed | The only non-kinematic input in the system |
| **Contact-phase flag / predictor output** | 2 × 1 | **NEW** | Gating is what converts +2.9 pp into +23.2 pp `[PR]`. Also hardware-motivated: post-compensation F/T residual is ~0.5 N static but **~2 N in motion** `[V]`, so the wrench is largely garbage during fast free-space motion |
| **Gripper aperture + grasp-contact state** | 2 × 2 | **NEW** | See Model 1 inputs |
| `dt`, relative pose, j₁ | passthrough | **NEW** | Must reach the controller, not stop at the planner |

### Model 2 — outputs (× H)

| Field | Dim | Status | Why required |
|---|---|---|---|
| Arm joints | 14 | prior, **now chunked and residual-structured** | `q = IK_DLS(target, q_now) + Δθ(·)` |
| **Stiffness / coordination mode per arm** (min. 2-level: free-space stiff / in-contact compliant; better: 6-axis mask or reference wrench) | 2 × 1 … 2 × 6 | **NEW** | Otherwise the F/T sensor has no actuation path (§3.2) |
| **Gripper aperture command** | 2 × 1 | **NEW** | Blocking without it |

**If forced to cut to the bone, the irreducible three are: aperture, chunk-with-durations, and relative pose.** Stiffness can start as a fixed schedule keyed to task phase; the redundancy scalar can start as a posture task inside the IK.

---

## 5. Revised diagnostic experiment set

The prior review proposed a two-way ablation (RGB present/absent). With F/T confirmed, **the ablation must be three-way**, because the diagnoses are distinguishable only by the *pattern* across all three channels.

### Protocol

Freeze one trained Model 2. At inference, replace each input with (a) zeros and (b) a version shuffled across trajectories — shuffling controls for "zero is out of distribution." Measure **Δ success rate** and **Δ joint-trajectory MSE**, and report all of them **split by contact vs free-space timesteps** (per FIRST's 84 % / 2.9 % split `[PP]`), and **on held-out object stiffness/weight** (ALPHA-α protocol) as well as training objects. The held-out split is mandatory: FACTR found all methods tie on *training* objects and diverge only on unseen ones `[PP]`.

Symbols below: **0** = statistically indistinguishable from the unablated model; **↓** = significant degradation.

### Decision table

| RGB | F/T | Pose | Diagnosis | Action |
|---|---|---|---|---|
| 0 | 0 | ↓ | **Pure learned IK.** The prior review's original verdict is confirmed after all. | Delete the learned low level. Use `IK_DLS` (5.31 µs `[M]`) + admittance. |
| ↓ | 0 | ↓ | **Visuomotor servo + IK; the wrench is dead.** The ForceVLA/+2.9 pp regime. | Do *not* conclude force is useless. First check the teleop rig (§7 of newly-raised), then add gating/curriculum/reweighting, then re-run. If still 0, drop the sensor and save the BOM. |
| ↓ | ↓ *(contact only)* | ↓ | **Intended behaviour.** Vision for phase/redundancy, force for contact, pose for the target. | Ship this structure. Proceed to the baseline comparison below. |
| ↓ | ↓ *(free space too)* | ↓ | **Wrench is leaking non-force information** — gravity/inertia signature encoding arm configuration or object identity, i.e. a proxy for kinematics. | Fix payload-inertia compensation before believing any force result. Note ~2 N residual in motion `[V]`. |
| 0 | ↓ | ↓ | **Image is redundant given pose + wrench for this task class.** The prior review's "delete RGB" advice is correct *here*. | Delete the image **only after** separately verifying elbow-obstacle and redundancy-resolution cases, which are the two things the image was retained for. |
| ↓ | ↓ | 0 | **The target is being ignored.** Model 2 is running an open-loop visuomotor policy; Model 1 is decorative. | The two-model split has no function. Collapse to a single model, or fix the pose-conditioning path (likely the frame mismatch, §3.4). |
| 0 | 0 | 0 | **Nothing is being used** — trajectory/phase memorization. | Data or eval design is broken. Stop and fix before any architecture decision. |

### Supporting diagnostics (all cheap, run alongside)

- **D2 — Analytic residual.** Compute `r = q_model − IK_DLS(target, q_now)`. If ‖r‖ sits at IK numerical tolerance, the model learned nothing but IK. Then **plot ‖r‖ against ‖wrench‖**: a force-using model shows correlation; a shortcut model shows none.
- **D3 — Sensitivity signature.** `∂q_out/∂wrench` by finite difference, computed *separately* on in-contact and free-space steps. The correct signature is ≈0 in free space, clearly nonzero in contact. A **flat** profile means the wrench is being treated as noise.
- **D4 — Camera/scene shift.** Move the camera or re-layout the background and re-evaluate. Collapse here confirms the `B_T_H`-from-background shortcut `[PR]`.
- **D5 — ψ spread audit.** One FK pass over the existing dataset (~35 s per 1 M steps `[M]`); bin by (target pose, current q) and histogram the arm angle. Spread < 20° → L2 collapse costs < 8 mm, survivable. Spread > 60° → > 50 mm, redundancy scalar becomes mandatory `[M]`.
- **D6 — Beat the honest baseline.** `IK + sensorless admittance from motor current` (egg-on-bread 80 % vs 40 %, 0.69 N force estimation from servo signals, no sensor, no learning `[PP]`). If Model 2 does not beat this, the learned low level is not earning its cost.
- **D7 — Budget for the null.** The observed range of naive force-concat effect across published papers is **−2.5 pp to +39.9 pp**; ForceMimic reported force-input models performing outright poorly once contact began `[PP]`. Decide *in advance* what happens if the measured delta is ~0.

Note that the field's standard practice — attention visualization, used by both FACTR and FACTR-2 — is **strictly weaker evidence** than the zeroing ablation. FACTR itself ran no zeroing ablation `[PP]`.

---

## 6. Does the recommendation still stand?

**Partially. The sequencing preference survives; the specific first build changes, and the baseline gets harder.**

**What changes:**

- **"The two-model design is dominated" is withdrawn.** It is not dominated: measured, one model on this hardware is either 4 Hz *or* small, and the classical spine cannot capture object-property generalization `[M]`, `[PP]`.
- **Build the shared-encoder two-head variant first, not the single-model CIR-1.** It is the single-model code plus one head; it dominates on rate; and it makes the decisive §5 diagnostic cheap because both heads are already wired to the same tokens. The measured cached-token path is the reason: Model 2's head on cached vision tokens runs at **5076 Hz p50 / 4149 Hz p99 / 1965 Hz worst-case on one CPU thread under full GPU load** `[M]` — that is how you get a >1 kHz F/T reflex, not by deleting the image.
- **Placement is a decision, not a default.** M1-on-GPU + M2-on-CPU maximizes Model 2's median (59.5 Hz p50, but 28 Hz worst-case). Both-on-CPU gives better Model 1 rate (5.2 Hz) and a tighter Model 2 tail (32.6 Hz worst) at 41.3 Hz p50 `[M]`. **If worst-case jitter is the binding safety constraint, both-on-CPU is arguably the better engineering choice despite the lower median.** Never both-on-MPS.
- **Raise the go/no-go bar.** The comparator is IK + sensorless admittance, not bare position control (§2, D6).

**What does not change:**

- **The classical spine is still mandatory**, as the baseline every learned variant must beat, and as the fallback controller.
- **Sequencing still starts with the simplest thing that can be measured**, not with the full architecture.

**Two gates that precede all architecture work:**

1. **The gripper channel must exist.** All architecture sequencing is moot while grasping success is structurally 0 %.
2. **Verify the teleop rig gives force feedback to the operator.** If not, the wrench channel carries no learnable structure and every force-related decision above is premature.

**Two decisions to re-open:**

- **Buy vs estimate the F/T sensor.** 2× FT300-S ≈ $3.9 k plus wiring, a startup re-bias routine, mandatory 10-parameter payload-inertia identification, a hard **100 Hz** output ceiling, and ~2 N residual noise during motion `[V]`. NEXT gets **0.018 ± 0.012 Nm** from motor current on a $2 500 arm with 10 min of data and $0 hardware `[PP]`. Buy the sensor only if you need clean 6-axis wrist wrench for low-magnitude fingertip contacts or torque-about-tool-axis (screwing). Note the 100 Hz ceiling independently confirms the fast loop cannot be the learned model.
- **Scope sim-then-real to free-space/kinematic behaviour.** Contact stiffness, friction, and F/T noise/drift are precisely what simulators get wrong; essentially none of the force-policy results in the literature are sim-to-real — nearly all train on real teleop data `[PP]`.

---

## 7. Contradictions this document has not resolved

These are real tensions between the research pass, the adversarial pass, and the measurements. They are stated, not smoothed.

**7.1 — The image in Model 2: "delete it" vs "keep it." Unresolved on the merits.**
The research pass concluded the head image "is now the single worst element of the design and should be deleted," reasoning that Model 2's job is contact-phase reasoning and force-dependent correction, that Model 1 already consumed the image, and that a 38-dim MLP would run at hundreds of Hz. The adversarial pass and the measurements refute both *cost* arguments (`[M]`: sharing saves 6 %, and Model 2 with its own ViT-S hits 50 Hz p99 on CPU) and supply two positive arguments for retention (redundancy resolution; F/T's structural blindness to elbow contact).

**But the research pass's remaining argument survives and is not answered by the measurements:** the image is what the optimizer will latch onto, and every paper that measured it found policies "overfit to using visual modality, effectively disregarding force data," with 80–90 % of attention on vision tokens absent a curriculum `[PP]`. That is an argument for a **modality curriculum or contact gate**, not for deletion — but I cannot prove the gate is sufficient. **This document recommends retention. That recommendation is contingent on the §5 ablation coming back with a nonzero, contact-localized F/T sensitivity.** If it does not, deleting the image becomes live again.

**7.2 — Shared encoder: "don't share" vs "share."**
§1 (item 2a) argues sharing is not worth adopting *as a compute optimization* (6 % saving) and that it pins Model 2's vision to Model 1's 4–5 Hz. §6 recommends building the shared-encoder variant first. These are reconciled only by the cached-token reflex head — which is exactly the configuration that makes the coupling tolerable. If for any reason the reflex head is not built, the shared encoder is a rate liability and Model 2 should get its own ≤25 M-param encoder.

**7.3 — "The wrench justifies a learned low level" vs "the objective will still make it learn IK."**
Both are asserted in this document and both are correct at different levels. The information degeneracy is retired; the optimization degeneracy is not. Nothing in the evidence guarantees the residual structure + gating will actually recover the force behaviour on *this* data. **The §5 ablation is the only thing that settles it, and it should be treated as a gate on further investment in Model 2, not as a post-hoc analysis.**

**7.4 — The prior review's headline numbers were partly wrong, and one of them is a citation error I am correcting rather than defending.**
- The **8 % / 94 % / 96 %** figure is real and reproduced exactly (row-means of RA-L 2024 Table I) but classifies by *final* action; Model 2 outputs joints, so it is in the 96 % column. Quoting 8 % against this architecture is a factual error.
- The **"14 % vs 29 %"** OC-VLA degradation figure used previously **does not exist in the paper**. The token "29" appears nowhere. The actual comparison is 14.0 pp (OC-VLA) vs 21.3 pp (OpenVLA-OFT) / 16.7 pp (base frame) / 16.0 pp (π0) `[PR]`. Any argument sized on a 15-point advantage must be re-sized to ~2.7–7 points.
- The **4.7 Hz** cap that drove the "delete RGB" conclusion — and coloured the conclusions on the split and the sequencing — was an artifact of co-scheduling both models on MPS. **Any future latency claim in this project must state device placement**, because on this machine CPU beats MPS at batch 1 at every Model 2 size tested (ViT-B: **44.7 ms CPU fp32 vs 64.2 ms MPS fp16**), batch-1 inference being dispatch-bound (~179 µs per MPS dispatch vs ~1 µs CPU; 15 % of peak GPU utilization) rather than FLOP-bound `[M]`.

**7.5 — The two largest problems in the design appear nowhere in the prior review, and both outrank every conclusion it reached.** The absent gripper channel and the F/T sensor's missing actuation path are not refinements of the prior analysis; they are things the prior analysis did not see. That is a methodological finding in itself: the prior review reasoned about *model structure* and never audited the *actuator coverage* of the output vector.

**7.6 — Things the evidence does not settle, at all.**
- **The arm command interface** (torque/impedance vs joint-position only) is unstated and changes the correct architecture qualitatively. Nothing here resolves it.
- **Whether the teleop rig is bilateral.** Decisive, unknown, checkable in an afternoon.
- **Whether "양손" is parallel-jaw or multi-finger.** Changes the gripper fix from 2 dims to 24–44.
- **Whether the head is actuated, and whether the torso moves.** The entire §3.4 verdict is conditional on this. Rigidly-mounted head → the design is OC-VLA-optimal as drawn. Live neck → silently biased. Moving torso → head joint state alone is insufficient.
- **How often the hands are visible in the head image during these tasks.** This single number determines whether the self-calibration channel is usable or dead. Directly measurable from existing teleop logs (project both hands via FK into the head camera, count in-FOV unoccluded frames per task phase).
- **No paper runs the comparison this project actually needs** — learned `(pose, joints, wrench) → joints` against analytic IK + admittance + a search primitive, on the same tasks and hardware. Force Policy omits any classical baseline; TER-DAgger has no impedance-only baseline. That gap is itself the argument for running it in-house rather than inferring it.