# Bimanual "Cerebrum/Cerebellum" Split — Decision-Ready Architecture Synthesis

**Bottom line up front.** The instinct to split is right. The proposed *implementation* of the split is wrong in three specific, independently fatal ways: (a) the interface is two independent 6-DoF poses, (b) Model B consumes RGB, (c) the split is justified by capacity. Keep the functional decomposition; change the seam, the interface, the supervision, and the justification. Separately — and this is the constraint that should reorder your quarter — the Mac mini is a good development box and an indefensible product computer, and the "both models must run locally on the Mac mini" premise is silently doing more damage to the design than any ML choice in the plan.

Where the evidence is thin or self-contradictory, it is flagged inline and collected in §10. Read §10 before you quote any number in a board deck.

---

## 1. How each confirmed constraint changes the verdict

The four constraints do not point the same direction. Two argue for the split, two argue against the split *as specified*, and the commercialization lens produces the strongest pro-split arguments in the whole analysis — arguments a pure-ML reviewer would never generate.

### 1.1 BIMANUAL — argues FOR splitting capacity, hard AGAINST the pose interface

**For the split (structural, empirical):** Decoupled per-arm heads beat a monolithic 14-DoF head: 42.6% → 62.4% average. Adding cross-arm interaction modules adds another +16.5 pp → 78.9% (DP3 baseline 55.4%). RDT-1B names the underlying reason: doubling the action space raises the multimodality of the feasible-action distribution. So "give each part of the problem its own capacity" has real backing — *at the network level*.

**Against the interface (geometric, not tunable):** When both grippers rigidly hold one object, `T_left⁻¹ · T_right` is fixed. That is 6 holonomic constraint equations. Two independent 6-DoF pose targets span 12 DoF; 6 of them cannot move the object and can only become internal wrench. No amount of training data makes two independently-regressed poses satisfy a 6-equation constraint to rigid-grasp tolerance. On compliant low-cost arms (~10²–10³ N/m Cartesian stiffness) a 5 mm relative error is 0.5–5 N — the object slips. On stiff cobots (~10⁴–10⁵ N/m) the same 5 mm is 50–500 N — the object is crushed and you take a protective stop. *(Force magnitudes are estimated from stiffness ranges, not measured — see §10.)*

**Against, empirically:** PerAct2 is the closest published instantiation of Model A's proposed output — per-arm 6-DoF pose + gripper flag, realized by a motion planner. Average 16.8% over 13 bimanual tasks; push box 6%, handover 11%, plate pickup 4%. ALOHA-lineage joint-space systems on comparable real tasks sit at 70–95%. The comparison is confounded (keyframes, sim, planner) but the burden of proof is clearly on the split.

**Against, informationally:** Inter-gripper relative pose is the single highest-value bimanual input measured anywhere in this research — UMI cloth folding 70% with it vs 30% without, from one extra 6-D vector. Two independently predicted absolute poses subtracted together *compound both arms' errors into exactly the quantity the closed chain is most sensitive to*.

**Against, temporally:** One model emitting one chunk containing both arms guarantees synchronization by construction. Split the arms and you must re-create it. UMI latency matching moved dynamic tossing 57.5% → 87.5%. InterACT's cross-segment ablation dropped slot insertion 44% → 24%.

> **Verdict shift:** bimanual turns "split the model" from neutral to positive, and turns "per-arm independent pose interface" from plausible to disqualified.

### 1.2 MAC MINI — argues FOR a rate/horizon split, hard AGAINST two image-consuming models

**Against, decisively:** Model B as specified consumes RGB. With 2 cameras that is **four vision-encoder passes per control step** — ~1.0 TFLOP ≈ 200 ms on M4 Pro's ~5.1 TFLOPS effective, a ~5 Hz hard ceiling from that line item alone. Every system that works (π0, GR00T N1, Helix, RDT) shares **one** backbone forward pass between the slow and fast halves via a shared KV cache. Your split takes the single most expensive component and computes it twice. Shared-backbone alternative: ~100 ms → 8–12 Hz. That is 2× for free.

**Against, at the runtime level:** Metal has no compute preemption and no stream priority. Command buffers execute to completion. Two models on one GPU each get roughly 45–50% of solo throughput, plus jitter equal to the other's longest command buffer. A separate bandwidth-only ceiling: A(2B@8-bit) + B(1B@8-bit) = 3 GB read per step ÷ 190 GB/s effective = 63 Hz *ignoring* activations, images, and KV. Real-world ≈ 1/3 of that. *(Derived from Metal's documented execution model, not measured on your workload — benchmark it in week one; this is the single highest-value measurement available to you.)*

**Against, at the OS level:** macOS has no PREEMPT_RT, no `isolcpus`, no cpusets, no `chrt`, no SCHED_FIFO, no core shielding. Mach `THREAD_TIME_CONSTRAINT_POLICY` is the only mechanism and it is soft. Realistic periodic-thread jitter: p50 50–200 µs, p99 1–3 ms, **p99.9 10–50 ms**. A 100 Hz loop with a 50 ms stall has missed five consecutive cycles. A hard ≥100 Hz joint loop on macOS is not defensible and cannot be engineered around — the kernel facilities do not exist.

**For the split:** the Helix/GR00T *rate* split maps cleanly onto this box — slow semantic model at 1–5 Hz, fast chunk policy at 5–10 Hz emitting 1–2 s of 50 Hz trajectory. But note the honest catch: on **one** chip you do not gain throughput from a rate split, because both models share one bandwidth pool. You gain a *bounded fast loop* only if the fast model is genuinely tiny (<50M), pixel-free, and lives on CPU or ANE. MLX will never target the ANE (issue #18, wontfix), so that path is Core ML, with all its silent-fallback pain.

> **Verdict shift:** Mac mini kills "two RGB-consuming transformers." It supports "one shared backbone, two heads, one KV cache," and it forces the real split to be *inference host / real-time controller*, not *Model A / Model B*.

### 1.3 SELF-COLLECTED TELEOP DATA — the strongest argument FOR the split, and the source of the plan's most dangerous flaw

**For, strongly (this should be your headline justification):** Model A can absorb data Model B cannot — simulation, cross-embodiment, human video, procedurally randomized scenes — while only Model B needs your scarce, expensive, on-robot bimanual demos. This is a genuinely good reason to split and it is *better than the capacity argument*, which has no support anywhere in the literature. Anchors: ALOHA Unleashed needed 26,000+ real episodes / 35 operators / 10 robots / 8 months for 70–75% on shirt hanging. Mobile ALOHA got away with 50 demos/task **only** because it co-trained with 825 static demos, worth up to +90 pp. RoboTwin hard setting: from-scratch small policies collapse to 1–2%. You cannot out-collect this; you can only reuse.

**Against, and this is the killer:** if the "desired EE pose" label is `FK(q_{t+k})` computed from the same teleop joint trajectory, it is an **invertible function of the very joint targets Model B must output**. Two failures follow. (1) *Label leakage:* `(q_t, FK(q_{t+k})) → q_{t+k}` is solvable in closed form with zero reference to the image. Model B will learn the IK shortcut and ignore RGB, because that minimizes training loss fastest — textbook causal confusion. At test time, when Model A's pose is imperfect and vision is the only corrective signal, Model B has no learned pathway to use it. (2) *Support mismatch from step one:* Model B only ever trains on poses that are exactly reachable and exactly kinematically consistent; at test it sees Model A's slightly-unreachable, slightly-off-manifold approximations. It is OOD on the first control step and the error compounds.

**Corollary that must be said out loud:** if the interface is a metric EE pose, the exact solution already exists — LeRobot ships `InverseKinematicsEEToJoints`; a bimanual differential-IK QP (Pinocchio + OSQP, or mink) solves it in 0.2–1 ms at 500–1000 Hz with joint limits, self-collision and singularity handling you can certify. Spending model capacity to approximate a function you can compute exactly is a **capacity loss, not a capacity split**. A learned Model B earns its keep only if it has information IK does not: contact compliance, force regulation, human-like redundancy resolution, dynamic feasibility under load. None of those are in a default LeRobot dataset — SO-101 and ALOHA are position-controlled with no F/T channel and no torque/current logged by default.

**Also for the split, on the labeling side:** ALOHA's decisive trick — record the **leader** arm's joint positions as the action, not the follower's — makes applied force implicitly encoded as `Kp · (q_leader − q_follower)`. That is a free, zero-sensor impedance channel the human modulates unconsciously, and it is what makes contact-rich work without F/T sensors. Getting this wrong at collection time is **unrecoverable**. A 6-DoF pose interface throws this channel away, because Model B will correctly drive pose error to zero, which drives applied force to zero.

> **Verdict shift:** data argues for the split at the *training-source* level and against it at the *supervision* level. Both must be honored simultaneously.

### 1.4 COMMERCIALIZATION — gives the best pro-split arguments and the harshest anti-Mac verdict

**Strong FOR — independent update cadence.** This is the single best commercial argument for the split and it is not an ML argument. Regression-testing a contact-level controller on real hardware is expensive and slow; regression-testing a planner is largely offline. Shipping a new task vocabulary without re-validating the layer that touches the object is enormous operational leverage. **Caveat: this only holds if the interface is FROZEN and VERSIONED.** The moment you change the pose representation, both layers need revalidation and the benefit evaporates.

**Strong FOR — per-customer finetuning.** Finetune only the high level on customer-specific task sequences and vocabulary; keep the validated low level fixed. This is a deployment-velocity *and* liability win, and it maps to how Chef and Ambi actually productize (skill libraries over a fixed motion stack). Economics: a LoRA finetune run is ~$50–150 of cloud GPU.

**Strong FOR — a validated, frozen low level is the object your safety case can point at.** A layer with a stable interface, a bounded output space, and a fixed validation suite is something you can put in a technical file. A monolith that changes wholesale every release is not.

**MEDIUM FOR — field debuggability.** Real value, but you get ~90% of it from *logging the intermediate representation*, not from separate weights. π0.5/π0.7 emit language subtasks and visual subgoals precisely for this.

**Strong AGAINST — the commercial metric is MTBI, not success rate, and the split as proposed adds a failure surface without adding recovery.** DYNA-1: 99.4% success, 700–800 napkins in 24 h, **zero interventions**, accepted at only 60% of human throughput *because* it ran unattended. Their named competitor failure mode: "unrecoverable errors after an hour or two." Meanwhile 0.95¹⁰ ≈ 0.60 — to hit 99% on a 10-step bimanual task you need 99.9% per step. Parameter reallocation does not change that exponent; error detection and recovery does. A split whose high level cannot say "I am stuck, re-approach" is commercially worthless regardless of its MSE.

**Strong AGAINST — imitation alone does not reach the bar.** π0.6 out-of-box fully assembles a box **20%** of the time. π*0.6 with Recap (RL on real autonomous experience + a value function) took espresso, laundry and box assembly all to >90% and more than doubled throughput. The last 20→90% was bought with a self-improvement loop, not a better decomposition. Plan for your pure-teleop-imitation stack to land in the 20–80% band.

**Disqualifying AGAINST — Mac mini as product compute.** Not a close call, and it fails on at least six independent grounds: macOS SLA §8E explicitly disclaims suitability where "failure or time delays… could lead to death, personal injury"; §2J forbids redistribution/sublicensing and there is no OEM/embedded program; 10–35 °C, unfiltered intake, no vibe/shock rating, AC-mains only; no PREEMPT_RT; ROS 2 lists macOS as **Tier 3, amd64 only** (Apple Silicon is not a listed target at all); no BMC/IPMI, no serial console, no userspace watchdog, MDM update deferral capped at 90 days; and no lifecycle commitment, no PCN, no last-time-buy — the 64 GB M4 Pro SKU **was discontinued in June 2026 mid-cycle with no notice**. Jetson AGX Thor matches the M4 Pro's 273 GB/s with 128 GB and vastly more compute, at comparable price, on a ROS 2 Tier 1 platform with an RT kernel and DC input. AGX Orin 64 GB is committed to production through Jan 2032.

**Also AGAINST — two arguments you should delete from the pitch.** *Per-embodiment low-level swap:* π0.7 did laundry folding zero-shot on a bimanual UR5e with no training data for that task on that hardware. The industry solved cross-embodiment with co-training + control-modality metadata inside one model, not a swappable module. *Smaller OTA:* two models with two encoders is **bigger**, and you now ship a compatibility matrix. One base + per-customer LoRA is the real answer. A technical DD reviewer who knows π0.7 will mark both as stale.

**Hard AGAINST any learned layer in the safety path.** ISO 10218-1/-2:2025 (published 2025, ISO/TS 15066 fully absorbed into -2, harmonised to ISO 13849-1:2023 PL and IEC 62061:2021 SIL) require quantified dangerous-failure rate, diagnostic coverage, and validated systematic capability *per safety function*. A stochastic, non-exhaustively-testable policy demonstrates none of these. *(No standard says "a neural network cannot implement a safety function" in those words — this is an inference from the PL/SIL requirements plus industry commentary. It is the correct working assumption; it is not a quotation.)*

---

## 2. The single most dangerous assumption

> **That the A→B interface can be supervised from the data they are about to collect.**

Concretely: the assumption that "we'll collect teleop, compute EE pose by forward kinematics, and train Model B to map (pose target, joints, image) → joint commands."

Why this is the most dangerous, above all other flaws in the plan:

1. **It is silent.** Training loss will be excellent. Validation loss will be excellent. Sim will likely look fine, because in sim the poses are exact by construction and calibration error is zero. The failure appears only on hardware, under Model A's real error, months later.
2. **It inverts the plan's own goal.** Model B will learn analytic IK — a function you already have exactly, faster, deterministically, and certifiably. You will have spent your scarcest resource (on-robot bimanual demos) and your scarcest compute (48 GB unified memory) buying a worse version of `InverseKinematicsEEToJoints`. That is not splitting capacity; that is destroying it.
3. **It is unrecoverable at collection time.** The fix requires an *independently measured* EE pose channel (VR/AVP hand tracking logged as its own feature, wrist AprilTags, or mocap) and a *servo current/torque* channel. Both are decided before episode 1 and cannot be reconstructed from an existing dataset. Everything else in this plan can be redone; this one cannot.
4. **It compounds.** Model B trains only on exactly-reachable, exactly-consistent, on-manifold poses. At deployment it is out of distribution on the **first control step**, and every subsequent step is conditioned on a state the demonstrations never covered. This is the classic cascaded shift, and it is what HIRO's off-policy correction exists to fix in the RL setting.

**Two cheap diagnostics that settle it in an afternoon, before you build anything:**
- *Ablation test:* train Model B, then zero out or batch-shuffle its RGB input. If validation loss barely moves, Model B is an IK solver and the split has bought you nothing. **Pass criterion: ablating vision must cost ≥20 pp of rollout success.**
- *Perturbation test:* inject 5–20 mm / 2–5° of noise into Model B's input pose — Model A's realistic error magnitude — and measure joint-command degradation. This is the true test-time regime; validation loss on FK labels systematically overstates it.

**Runners-up, named honestly.** *Most likely to be false:* "splitting capacity across two models improves performance." I found **no evidence for this anywhere** — Helix, GR00T N1 and π0.5 all split explicitly for inference *rate* and data *reuse*, and none claims a capacity-partitioning benefit. It is the one part of the proposal with zero support. *Most expensive:* "the product ships on a Mac mini."

---

## 3. Recommended compute topology

Four tiers, not two. The seam you are missing is not between Model A and Model B — it is between *learned* and *deterministic*, and it is legally mandatory.

```
┌─ TIER 0 ── SENSORS ──────────────────────────────────────────────┐
│ 2× wrist cam + 1–2 scene cam, 640×480 @30 fps (224² at inference)│
│ joint encoders @1 kHz · servo current/torque @1 kHz               │
│ wrist F/T (or current-based external force estimate) @500–1000 Hz │
└──────────────────────────────┬───────────────────────────────────┘
                               │ H1: exposure+readout+USB/GigE+decode  15–30 ms
                               ▼
┌─ TIER 1 ── INFERENCE HOST ───────────────────────────────────────┐
│ DEV: Mac mini M4 Pro / 48 GB   PRODUCT: Jetson AGX Orin 64 GB    │
│                                                                   │
│  ONE shared vision-language backbone  ──────────  10 Hz           │
│    │  H2 preprocess 3–6 ms · H3 backbone 40–70 ms (2 cams @224²)  │
│    ├──► HEAD A "cerebrum"  1–5 Hz  (runs 1-in-N)                  │
│    │      subtask · coordination mode · phase · stiffness ·       │
│    │      inter-gripper target · abort/stuck · latent z           │
│    │      H4a +80–200 ms when it runs      [STOP-GRADIENT here]   │
│    └──► HEAD B "cerebellum"  5–10 Hz, NO PIXELS OF ITS OWN        │
│           consumes: latent z + Head-A block + joints + joint      │
│           velocities + servo current + F/T + T_L→R (measured)     │
│           emits: 100-step joint-position chunk @50 Hz (= 2.0 s)   │
│           4 distilled flow steps · H4b 20–50 ms                   │
│  Heartbeat TX @50 Hz                                              │
└──────────────────────────────┬───────────────────────────────────┘
                               │ H5: serialize + UDP/Ethernet   0.5–2 ms
                               ▼
┌─ TIER 2 ── REAL-TIME CONTROLLER ─────────────────────────────────┐
│ Linux PREEMPT_RT + ros2_control (or STM32/Zynq)     500–1000 Hz  │
│  · chunk buffer + temporal-ensemble blend (w_i = exp(−m·i))      │
│  · interpolate 50 Hz chunk → 1 kHz setpoints                     │
│  · bimanual differential-IK QP (Pinocchio+OSQP / mink) 0.2–1 ms  │
│    — ONLY if a Cartesian target is used; joint chunks bypass it  │
│  · joint / velocity / accel / jerk / torque clamps               │
│  · self-collision + workspace + singularity guard                │
│  · internal-wrench limiter (tightly-coupled mode)                │
│  · HEARTBEAT WATCHDOG: 50–200 ms timeout → ramp-to-stop          │
│           H6: 1–2 ms per cycle, deterministic                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │ H7: EtherCAT 0.25–1 ms  /  CANopen 2–5 ms
                               ▼
┌─ TIER 3 ── DRIVES ───────────────────────────────────────────────┐
│ position PID + current loop  1–10 kHz  (in the servo)            │
│ H8: servo + mechanical response  5–15 ms                          │
└───────────────────────────────────────────────────────────────────┘

┌─ SAFETY CHAIN ── INDEPENDENT, NOT REACHABLE FROM TIERS 0–3 ──────┐
│ dual-channel hardwired e-stop · STO · safety-rated monitored stop │
│ safe speed / safe position limitation · power-and-force limiting  │
│ ISO 13849 Cat-3 PL d minimum, ISO 10218-1/-2:2025                 │
│ Tiers 1–3 are QM devices that emit REQUESTS. This layer DECIDES.  │
└───────────────────────────────────────────────────────────────────┘
```

**Latency budget.**

| Hop | Path | Budget (p50) | Alarm (p99) |
|---|---|---|---|
| H1 | photon → tensor on host | 15–30 ms | 45 ms |
| H2 | preprocess | 3–6 ms | 10 ms |
| H3 | shared backbone (2 cams @224²) | 40–70 ms | 110 ms |
| H4a | Head A (1-in-N cycles) | 80–200 ms | 300 ms |
| H4b | Head B action expert, 4 flow steps | 20–50 ms | 90 ms |
| H5 | host → RT controller | 0.5–2 ms | 5 ms |
| H6 | RT interpolate + clamp + QP | 1–2 ms | 3 ms (hard) |
| H7 | fieldbus | 0.25–5 ms | 2× cycle |
| H8 | servo + mechanics | 5–15 ms | 25 ms |
| **Σ** | **sensor → motion onset** | **90–180 ms** | **250 ms p99.9 = alarm** |

Two budgets, not one. **Planning latency** (above) is hidden by chunking — the robot executes 2 s of trajectory while the next chunk computes. **Reaction latency** is *not* hidden: the RT controller must be able to stop on force threshold or heartbeat loss within 50–200 ms, entirely independent of the policy. Design and test them separately. Track p99.9, never the mean — a policy averaging 40 ms that spikes to 400 ms will hurt someone.

**Decouple the host now.** Put all inference behind a `Policy.infer(obs) → ActionChunk` boundary with MLX and TensorRT implementations, and put the host behind a network interface. Then Mac-for-dev / Jetson-for-product is a config change, not a rewrite. This is the highest-leverage thing you can do this week.

---

## 4. Interface spec (bimanual)

Two channels. The **latent is primary** (Helix/GR00T/π0.x precedent); the **structured block is supervised as an auxiliary head** for interpretability, safety clamping, field triage, and product versioning. Never let the structured block be the sole channel — it cannot express "squeeze harder," "wait for the other hand," "this grasp is slipping," or multimodality.

**One message. Both arms. One shared time base. Emitted at 5 Hz. Never two async per-arm streams.**

### 4.1 Header (per message)

| Field | Type / dim | Notes |
|---|---|---|
| `interface_version` | uint16 | **semver, FROZEN.** Bumping it forces revalidation of BOTH layers. This field is the product asset. |
| `seq` | uint32 | monotonic |
| `t_capture` | int64 ns | monotonic clock, **camera exposure midpoint**, not host receive time |
| `H` | uint8 | horizon steps; **H = 10** |
| `dt` | float | 0.2 s → 2.0 s horizon, matched to Model B's chunk |
| `latency_offset` | float | measured A→B staleness; also injected at training time |

### 4.2 Global intent block (per message)

| Field | Dim | Notes |
|---|---|---|
| `latent_z` | **512** fp16 | primary channel; cross-attended by Head B |
| `subtask` | string ≤64 | language, human-readable; **logged always** — this is your field-triage record |
| `coordination_mode` | 4 (one-hot) | `{INDEPENDENT, GOAL_COORDINATED, LOOSELY_COUPLED, TIGHTLY_COUPLED}` — cheap to auto-label from demos (both grippers closed on same object) |
| `mode_confidence` | 1 | gates the RT layer's internal-wrench limiter |
| `phase_id` | 5 (one-hot) | `{APPROACH, CONTACT, TRANSPORT, MANIPULATE, RELEASE}` |
| `phase_progress` | 1 | 0..1 within phase |
| `T_grasp_frozen` | 9 | inter-gripper transform **latched at contact entry** in TIGHTLY_COUPLED; held constant thereafter |
| `uncertainty` | 9 | per-arm translational σ (3+3) + relative σ (3); RT layer scales speed limit by this |
| `supervisor_flag` | 4 (one-hot) | `{CONTINUE, RETRY_SUBTASK, ABORT_TO_SAFE, REQUEST_TELEOP}` — **this is the MTBI field. Do not ship without it.** |

### 4.3 Per-arm block — for each arm ∈ {L, R}, for each step h ∈ 1..H

| Field | Dim | Frame | Notes |
|---|---|---|---|
| `dp_ee` | 3 | **current EE frame of that arm** | translation delta |
| `dR_ee` | 6 | current EE frame | 6-D continuous rotation. **Not** quaternion, **not** Euler |
| `dp_base` | 3 | `base_link` | same SE(3), re-expressed |
| `dR_base` | 6 | `base_link` | Mixture-of-Frames: up to +15 pp from frame choice, and dynamic switching beats even an oracle fixed frame. Re-expression is ~free |
| `gripper_width` | 1 | metres | **continuous, never binary** |
| `gripper_effort` | 1 | normalized | carries grip force via the same PID-error mechanism |
| `stiffness_trans` | 1 | N/m | |
| `stiffness_rot` | 1 | Nm/rad | |
| `damping_ratio` | 1 | — | |
| `squeeze_force` | 1 | N, signed along grasp axis | **the ALOHA implicit-force channel, made explicit** |
| **subtotal** | **24 / arm / step** | | ×2 arms = **48** |

**ABSOLUTE POSE IS BANNED FROM THIS INTERFACE.** UMI ablation: relative 100% vs absolute 25% (n=20). Surgical Robot Transformer: ~0% with absolute EE pose across three tasks. On a self-built rig, hand-eye calibration error, FK error and backlash will be your dominant error term and they do not shrink with more data. They cancel in relative pose. **Head-relative is banned as the interface frame** — it adds head-joint backlash and a live head-to-base transform to the error budget, and simulation will never show you that penalty because sim extrinsics are exact. Head-relative pose belongs as an *input feature to Head A*, never as the interface.

### 4.4 Inter-arm block — per step h

| Field | Dim | Notes |
|---|---|---|
| `dp_rel`, `dR_rel` | 3 + 6 | target `T_left→right`, **predicted DIRECTLY by the head — never derived by subtracting two absolute pose predictions.** Subtracting compounds both arms' errors into precisely the quantity the closed chain is most sensitive to |
| `internal_wrench_setpoint` | 1 | desired squeeze along the grasp axis in coupled modes |
| `sync_weight` | 1 | 0..1: how strictly the two arms must be simultaneous at this step |
| **subtotal** | **11 / step** | |

### 4.5 Object block — TIGHTLY_COUPLED only, per step h

| Field | Dim | Notes |
|---|---|---|
| `dp_obj`, `dR_obj` | 3 + 6 | object pose delta |
| `obj_squeeze` | 1 | |

In this mode the parameterization becomes **(object pose target) + (FIXED `T_grasp_frozen`) + (squeeze force)**. The 6 excess constraint DoF are *structurally eliminated* rather than regressed. This is the concrete fix for the closed-chain problem and it is nearly free to label.

### 4.6 Size

`H=10 × (48 + 11 + 10) = 690` floats + `~28` global + `9` frozen grasp + `512` latent ≈ **1,239 floats ≈ 2.5 KB fp16, 12.4 KB/s at 5 Hz.** Bandwidth is a non-issue; do not compress at the cost of information.

### 4.7 Model B output spec

| Field | Value |
|---|---|
| Content | **absolute joint position targets**, both arms + continuous gripper |
| Dim / step | 2×(6 or 7) joints + 2 gripper = **14–16** |
| Chunk | **100 steps @ 50 Hz = 2.0 s**, one shared time index covering both arms |
| Emission rate | 5–10 Hz, with temporal ensembling `w_i = exp(−m·i)` across overlaps |
| NOT | joint velocities or torques — for a first product, the ALOHA joint-position→PID pattern is the highest reliability per unit effort and gives implicit force for free |

**Chunk, never per-step.** The ACT ablation is decisive: k=1 → 1% success, k=100 → 44%. A cerebellum reacting per-step to a per-step pose target forfeits the entire benefit chunking exists to provide.

### 4.8 What MUST be in the interface beyond pose — summary

Gripper (continuous width **and** effort, inside Model B's 50 Hz chunk, never routed through the slow side — gripper errors are step-function catastrophic while arm errors self-correct); timing (shared time base + phase + sync weights + explicit latency offset); coordination mode (with the tightly-coupled reparameterization); stiffness/squeeze force (position control has zero force authority by construction — a pose-tracking controller drives force to zero); uncertainty; and the abort/stuck supervisor flag.

---

## 5. Data plan

**Planning constants.** Bimanual teleop throughput **15 ep/hr** (range 10–30; LeRobot's own 60 s episode + 60 s reset defaults hard-cap you at 30 before failures). One operator sustains **4–5 productive hours/day** → ~75 episodes/day, ~1,500/month. **Operator ramp: 4–8 h to usable, 20–40 h to consistent — spend it before episode 1 counts.** Assumed yield 75% *(unverified — if your real yield is 50%, every number below rises 50%)*. Storage 50–150 MB/episode.

### 5.1 Episode budgets

**Stage 1 — decide the architecture (≈2 weeks of collection).**
- Sim: unlimited scripted demos on 8–10 bimanual tasks. Reproduce ACT's published 86% / 32% baseline first to validate the harness.
- Real: **1 task × 3 staging configs × 30 = 90 episodes (~6 h teleop, ~9 h with staging)** plus 100 real eval rollouts per architecture. Purpose is *rank agreement with sim*, not performance.

**Stage 2 — one task to product reliability (≈6–10 weeks, one operator).**
- **20–30 staging configurations × 30–40 demos = 600–1,200 usable.** Budget by diversity, not volume: generalization follows a power law in the *number of environments and objects*, and marginal return of extra demos in the same scene is ≈0 past a threshold. ~1,600 demos across ~32 environments/objects gave ~90% zero-shot on new environment + new object.
- **+2–3 HIL/DAgger rounds × 50 episodes = 100–150.** Budget this as a line item (~+30%), not a contingency.
- Collect ≈**900–1,600** to land 600–1,200 usable. ~60–105 pure teleop hours, ~110–160 h with staging/curation.

**Stage 3 — 3–5 task product (≈3–5 months).**
- **3–5 tasks × 400–500 usable = 1,500–2,500 usable**, ≈2,000–3,300 collected, ~150–220 pure teleop hours, ~300–450 total human-hours. Plus autonomous-rollout data from the flywheel.

**Explicitly out of scope: 50 tasks.** That is ~20,000 usable / ~26,700 collected episodes, ~1,780 pure teleop hours, ~3,000–3,500 total human-hours, 1.3–4 TB, and **$200k–500k DIY / 9–27 months** (or $1M–3M at the $50–150/usable-demo humanoid-program figure). Apply the standard 2–3× re-collection multiplier to whatever you commit. This is a Series-A-scale program, not a side activity. *(Cost figures lean substantially on vendor marketing pages — treat as directional.)*

### 5.2 Collection hardware

**Buy: an active-motor bimanual leader-follower rig.** Prototype: SO-101 bimanual, 4 arms ≈ $800–1,200, plus 4 cameras ≈ $400–600. Validated: ALOHA-2-class ≈ $15–20k. **Product: arms that ship a certified safety controller (UR / Franka / Doosan / Techman / Standard Bots class) — and this decision must precede data collection, because your data is embodiment-specific.**

**Do not buy passive GELLO.** LeRobot's DAgger/HIL loop requires a teleoperator with *active* motors that can enable/disable torque and servo to the follower's pose. HIL is the highest-leverage data type you will collect. A passive leader forecloses it.

**Do not use VR as the primary channel.** With a leader arm the recorded action *is* a joint command with zero IK in the loop. With VR you record hand poses that a solver converted to joints — so the trajectory carries the solver's singularity handling, joint-limit clamping and redundancy choices, and the policy learns those artifacts. **But** VR/AVP is the only option that gives an *independently specified* EE pose signal — which is exactly what breaks the FK label-leakage degeneracy. If you keep a metric pose interface, add VR hand tracking or wrist AprilTags as a **secondary logged feature** alongside the leader arms.

**Cameras: 2 wrist (one per arm) + 1–2 scene, ≥480×640, ~30 fps, fixed mounts, fixed lighting.** A wrist camera on each arm from episode 1 is non-negotiable — the single-global-camera failure (never seeing gripper-object contact) is the most common and most expensive documented mistake, detectable only *after* the data is collected. Note the compute coupling: 2 cameras at 224² is the practical ceiling on M4 Pro; a third costs ~25% of your control rate and 448² costs 4× the vision tokens. **Data is resolution-locked once recorded** — decide before episode 1.

### 5.3 What MUST be logged for the split to be trainable at all

Adopt RDT-1B's 128-dim on-disk schema (per arm: joint pos [0-9], gripper [10-14], joint vel [15-29], eef_pos [30-32], eef 6-D rotation [33-38], eef_vel, eef_angular_vel; left mirrors at [50-94]) — it costs nothing at collection time and makes your data directly finetunable on RDT-1B / π0 / GR00T N1. On top of that, these are the fields without which the split is untrainable or unrecoverable:

1. **Leader joint positions as the action label** (not follower). Applied force ≈ `Kp·(q_leader − q_follower)`. **Getting this wrong is unrecoverable.**
2. Follower measured joints + joint velocities.
3. **Servo motor current / torque per joint.** The Dynamixel/STS servos already report it. This is the *only* information channel that makes a learned Model B better than analytic IK. Without it, you are training a worse IK solver.
4. Gripper commanded width, measured width, **and** current — all three.
5. **Measured `T_left→right` per frame** (fed as input to *both* heads; UMI 70% vs 30%).
6. **An independently measured EE pose channel** (VR/AVP tracker or wrist AprilTags) — if and only if you keep the metric pose interface. Decided at collection time; cannot be reconstructed.
7. Wrist F/T, or a current-based external-force estimate, at ≥500 Hz.
8. **Hardware timestamps per camera stream, all sensors on one monotonic clock**, plus per-episode measured sensor→actuation latency. UMI's latency matching alone was 87.5% vs 57.5%.
9. Coordination-mode label and phase segmentation (auto-derivable: both grippers closed + both in contact with same object).
10. **Intervention / takeover events, success/failure label, and a failure taxonomy.** These become your Recap-style reward signal and your MTBI metric.
11. Scene config ID, lighting ID, object ID, operator ID.
12. **Calibration snapshot per session** (hand-eye, intrinsics, extrinsics). Without it you cannot later separate calibration drift from policy regression.

### 5.4 Avoiding cascaded distribution shift — ranked by cost

1. **Scheduled sampling / student forcing (free).** Train Model B on an annealed mixture of ground-truth interface values and Model A's *own predictions*, 100% → 0% GT over training. Directly targets the train/test input mismatch.
2. **Hindsight relabeling (free).** HIRO's fix, transferred to imitation: label Model B's goal with what was *actually achieved*, not what was nominally commanded, so goal-conditioning is robust to off-manifold goals.
3. **Noise injection sized to Model A's *measured* error (DART, near-free).** Inject 5–20 mm / 2–5° into the interface at training time. DART matches DAgger's final performance at up to 3× lower compute and costs the supervisor only 5% of cumulative reward during collection vs DAgger's 80%.
4. **Latency matching.** Inject the *deployed* A→B staleness as a temporal offset between Head A's conditioning and Head B's targets during training — Helix does exactly this. Because both arms consume the same stale target, the staleness is common-mode and **cancels in the relative term** — an independent argument against absolute pose.
5. **Joint end-to-end finetuning through the interface.** Pretrain the halves separately with a stop-gradient (Knowledge Insulation), then unfreeze and backprop through the interface as final polish. This is what every working dual-system VLA does, and it is what actually kills cascaded shift. Naively bolting a flow-matching action expert onto a pretrained VLM without insulation "significantly harms both training speed and knowledge transfer."
6. **HIL/DAgger rounds on hardware.** ~50 episodes per task per round, 2–3 rounds, with the pause/takeover/recover/return loop recorded as one continuous trajectory (RaC's recovery + correction decomposition). Even large pretrained VLAs need this.
7. **Chunked interface.** The interface carries a *trajectory chunk*, never a single next pose. Non-negotiable.

---

## 6. Sim plan — honest about macOS

### 6.1 What was actually measured on this class of machine

Measured on an M4 Mac mini (10-core, 32 GB, macOS 26.4.1, mujoco 3.10.0 arm64), MuJoCo Menagerie bimanual ALOHA:

- **Physics is not the bottleneck.** Bare ALOHA: 13,481 steps/s = 27× realtime at 2 ms timestep, one core. ALOHA + 5 free objects (nv=46, ~27 contacts): 7,125 steps/s = 14.2× realtime. Multiprocess aggregate: 1 proc 7,053 / 4 procs 25,062 / 8 procs **38,262 steps/s**. 100M state-only env steps ≈ **45 minutes**.
- **Offscreen rendering is a hard wall.** Fixed **13.11 ms per `render()` call, completely independent of resolution** (84², 128², 224², 320² all ≈76–77 fps) — it is GPU-sync/readback overhead, not fill rate. **It does not parallelize:** 1 proc = 38 env-steps/s with 2 cams @224²; 4 procs = 45 total; 8 procs = 44 total. Machine-wide ceiling ≈ **90 rendered frames/s ≈ 40 env-steps/s.**

**Consequences.** Pixel-based RL in sim on this Mac is dead (1M env steps ≈ 6.2 h, 100M ≈ 25 days). Imitation learning from scripted/teleop demos plus rollout evaluation is the only viable local loop. Evaluation budget: a 600-step (20 s @30 Hz) episode costs ~15 s wall → **~240 eval episodes per hour for the entire machine.**

### 6.2 What is unavailable on macOS — confirmed, not workaroundable

| Tool | Status |
|---|---|
| **Isaac Sim / Isaac Lab** | Ubuntu 22.04 x64 / Win 11 x64 only, CUDA 12+, ≥16 GB VRAM. **No macOS.** |
| **MuJoCo Playground / MJWarp** | `jax[cuda12]`; MJWarp needs CUDA ≥12.4. **NVIDIA only.** |
| **RoboTwin 2.0** — *the best-matched bimanual benchmark* (50 dual-arm tasks, 5 embodiments, 731 objects, 100k+ trajectories, 5-axis DR) | Docs verbatim: *"no support for MacOS."* Needs Linux + Vulkan + CUDA 12.1 via SAPIEN. |
| **MJX** | Three independent blockers, all confirmed empirically: `jax 0.11.0` exposes only `CpuDevice` (jax-metal abandoned, issues closed Dec 2025); `mjx.put_model` raises `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_MESH)`; jit of the mesh-heavy bimanual scene at batch=32 was **OOM-killed on 32 GB**. |
| **ManiSkill3** | macOS = CPU sim + render only; docs say macOS is for *"inference, local debugging, and development."* |
| **Genesis** | Claims a Metal backend (Quadrants Taichi fork + PyTorch MPS). **Unverified**; all published throughput is CUDA. Half-day spike, not a bet. |

**Available on macOS:** MuJoCo 3.10 native (excellent), robosuite 1.5 two-arm envs (**only 3 tasks — far too few to test an architectural hypothesis**), RoboCasa-GR1 (24 bimanual tabletop tasks, the better local option if your embodiment is humanoid-torso-like), and your own MJCF.

### 6.3 The plan

**Split the compute geographically, and decide this first.** Mac mini = task authoring, state-only rollouts, the checkpoint-selection harness, latency/memory benchmarking, and the deployment inference target. **A rented Linux + NVIDIA box (L4/A10/4090 class) = RoboTwin 2.0, ManiSkill3 GPU, all synthetic data generation, all training, and the statistically-powered ablation with batched rendering.** LeRobot's own guide: ACT 5 epochs on ~50 episodes = 30–60 min on a 4090 vs **6–14 h on Apple Silicon MPS**, and there is *no MPS row at all* for `diffusion`, `smolvla`, `pi0`, `pi05`. The Mac cannot train your policies regardless of what you ship on.

**Do system identification in week 1, not month 6.** SIMPLER perturbed *only* joint stiffness and damping — nothing visual — and sim's ranking fidelity degraded 2–3× (MMRV 0.031 → 0.070 → 0.100). Until sysID is done, **sim results about Model B are not evidence.** Measure: joint stiffness, damping, friction, actuator lag, control-loop latency and jitter, gripper force curve, and — critically — **your rig's actual Cartesian stiffness**, which determines whether closed-chain violation shows up as slipping (recoverable) or crushing/e-stop (unrecoverable). This means you need a small real dataset even in a sim-first plan.

**Know exactly what sim can and cannot decide.**

*Sim can honestly answer:* does the split learn from fewer demos; does it generalize better to unseen objects/positions/clutter; is the A→B interface information-sufficient (is there a task the monolith solves that the split structurally cannot); does it degrade gracefully when A is wrong; does the stack meet latency/memory budget.

*Sim cannot answer:* whether the split wins on real contact. Evidence: even fully visually-matched + sysID'd rigid tasks leave a **13.6–32.8 pp absolute real-sim success gap**. On contact/deformable tasks in a stock simulator, policy ranking is near-uncorrelated with reality — **Pearson r = 0.237 on rope routing**, 0.649 on T-block pushing; plush-toy grasping **could not be stably simulated at all** even in a purpose-built physics-optimized 3D-Gaussian-Splatting digital twin. A 5–10 pt architectural difference sits well inside that noise band.

*And the risk is bidirectional, which is why a small sim margin carries zero information.* Sim contact is smoother and more forgiving, so a pose bottleneck costs **less** in sim than on hardware where the cerebellum needs slip cues a pose channel discards → **sim overstates the split**. But the monolith sees pixels and joints jointly and can exploit sim-specific dynamics shortcuts → **sim understates the monolith**. Nothing establishes which dominates.

**Therefore:**
- **Pre-register the decision rule before running anything.** Trust a sim verdict only if the effect is **>15–20 pp**, holds across **≥8–10 distinct tasks** and **≥3 seeds**, and **survives a dynamics-randomization sweep** (friction ×[0.5, 2.0]; added control latency 0–100 ms; controller gain error ±30%; gripper force ±30%; hand-eye extrinsic noise 5–10 mm / 1–2°). If the ranking flips anywhere in that sweep, sim cannot answer your question.
- **Insert an early hardware checkpoint.** 2–3 tasks, ~50 real demos each, both architectures, ~100 real eval episodes per architecture. If sim and real agree in *rank*, you have earned the right to trust sim for the rest. If they disagree, sim is a development tool only. **Do not run 6 months of sim before touching hardware.**
- **Plan for co-training, not zero-shot transfer.** The RSS 2025 recipe: 4,000 sim demos + 40–400 real demos → **+37.9% average across 6 tasks / 2 embodiments including a bimanual humanoid**; real-only 31% → 76% with digital cousins + prior sim data; benefit persists even at 400 real demos. Mirror every sim task against a real task with **identical success criteria** (that mattered more than camera-pose matching). Target roughly 10:1 to 100:1 sim:real and tune the ratio explicitly. *(RoboTwin's headline "+367% few-shot" figures are relative gains over a deliberately weak 10-demo baseline — do not budget against them.)*
- **Never select checkpoints by validation loss.** Validation MSE: MMRV 0.375, r 0.308. Simulated rollout success: MMRV 0.056, r 0.924. Build the simulator as a permanent rollout-based regression and checkpoint-selection harness — sim tracks the real success-vs-training-iteration curve and peaks at the same checkpoint. **That is the durable commercial value of your sim, independent of whether it ever settles the architecture question.**
- **Keep every sim rollout state-only where possible.** Physics is 950× cheaper than pixels on this box.

**Statistical power (whole-machine budget, since rendering does not parallelize).** Two-proportion test, α=0.05, power 0.80, 50% baseline:

| Effect to detect | Episodes / arch / task | Mac wall clock, 2 arches |
|---|---|---|
| 5 pt | 1,565 | ~13 h/task |
| 10 pt | 388 | ~3.3 h/task |
| 15 pt | 170 | ~1.5 h/task |
| 20 pt | 93 | ~0.8 h/task |

A 10-task × 3-seed comparison at 10-pt sensitivity ≈ **4 days of continuous Mac mini time**. Affordable once; not affordable to iterate on. Move the sweep to the rented box.

**Biggest unquantified risk in the whole sim plan:** whether a well-tuned MJCF/URDF with accurate inertias, joint limits and gripper geometry exists for your actual arms. Verify this before committing to sim-first. It is a common silent blocker and it gates everything above.

---

## 7. Phased roadmap with GO/NO-GO gates

### STAGE 1 — DECIDE (weeks 1–10)

Weeks 1–2 are a measurement preflight; nothing about the architecture is locked until G0 passes.

**Do:** the three week-one benchmarks (§9); rent the Linux/NVIDIA box; put inference behind `Policy.infer(obs) → ActionChunk`; verify/author the MJCF; reproduce ACT's sim baseline as harness validation; freeze interface spec v0.1 and the logging schema; order the active-motor bimanual rig; pre-register the decision rule; sysID as soon as arms arrive; collect 90 real episodes on one task; run the three-way sim ablation — **(a)** monolithic ACT joint-space, **(b)** the two-model split with FK-derived labels, **(c)** Head A in relative-pose space + analytic QP-IK.

**GATE G0 (end of week 2) — hard, kills the local-both-models premise:**
- Two MLX models concurrent on the GPU: each model's p50/p99/**p99.9** solo vs together. **NO-GO if the fast head's p99.9 exceeds 3× its control period when the slow head is running.**
- 24 h periodic 100 Hz thread, `mach_absolute_time` histogram, Spotlight + Time Machine active. **NO-GO for any macOS-hosted control loop if p99.9 > 10 ms.** (Expect it to fail. That is the point — it produces the written evidence for the two-box architecture.)
- Real vision encoder at real resolution and camera count. **NO-GO on 3 cameras or 448² if it pushes H3 past 110 ms.**

**GATE G1 (end of week 10) — architecture decision:**
- Harness validated: ACT reproduces published sim baselines (86% Transfer Cube / 32% Insertion, 50 scripted demos) within ±10 pp.
- **Vision-ablation on Model B: zeroing/shuffling its non-interface inputs must cost ≥20 pp rollout success. If it costs <5 pp, Model B is a learned IK solver → KILL the learned low level, ship analytic QP-IK.**
- **Pose-perturbation at 10 mm / 3°: ≤10 pp success drop.** If >25 pp, the metric interface is too brittle → move to a latent-primary interface.
- Split beats monolithic ACT by **≥15 pp absolute** on **≥8 tasks**, **≥3 seeds**, and the ranking **survives the full dynamics sweep**.
- **Rank agreement between sim and real on 2–3 tasks × 100 real eval episodes.** Disagreement → sim is demoted to a development tool for the remainder of the program.
- Measured Cartesian stiffness on file; internal-wrench regime classified (slip vs crush).

**NO-GO consequence:** if G1 fails on the ≥15 pp criterion, ship the monolith with a subtask-level high level and analytic QP-IK. That is a *good* outcome — it saves you two training pipelines and a compatibility matrix.

### STAGE 2 — ONE TASK TO PRODUCT RELIABILITY (months 3–6)

**Do:** pick THE task (high labor cost, high mix so fixed automation loses, **low consequence of failure so the safety case is cheap**, fixed controllable workcell); 600–1,200 usable demos across 20–30 staging configs; 2–3 HIL/DAgger rounds; stand up the RT controller + heartbeat + safety chain; build the autonomous-rollout + success-labeling + intervention-logging flywheel; add wrist F/T; buy and read ISO 10218-1:2025 and -2:2025.

**GATE G2:**
- **≥95% per-cycle success over ≥200 consecutive cycles**, unattended.
- **MTBI ≥100 cycles and ≥2 hours** of unattended operation. *(This, not success rate, is the number that gates a sale.)*
- Cycle time ≤2× human. (Reliability buys the right to be slow: DYNA-1 was accepted at 60% of human throughput because it ran unattended.)
- **Zero policy-caused safety-chain trips per 500 cycles**; internal wrench within the measured limit for your rig.
- End-to-end sensor→motion p99.9 ≤250 ms, continuously instrumented.
- The A→B interface has not changed version since G1. (If it has, you do not have the update-cadence benefit you are pitching.)

### STAGE 3 — PRODUCTIZE (months 7–12+)

**Do:** migrate the inference host to Jetson (Orin now, Thor if you need headroom) — the `Policy.infer` boundary makes this a config change; 3–5 tasks; per-customer LoRA on the high level only; RaaS with a teleop-assist fallback from v1 (how 1X and Chef take revenue before the model clears 99%, and it doubles as your data channel); Recap-style RL on autonomous experience; written safety case + CE gap analysis against **EU Machinery Regulation 2023/1230, mandatory 20 Jan 2027.**

**GATE G3:**
- **≥99% per-cycle over ≥700 consecutive cycles** (the DYNA-1 shape), **MTBI ≥4 h unattended**, **on product compute, not the Mac.**
- ≥3 tasks at that bar, with per-customer finetuning demonstrated end-to-end on at least one.
- Safety case reviewed by a notified body or competent external assessor; ISO 10218-1/-2:2025 gap list closed or scheduled.
- Product BOM contains **no component without a published lifecycle commitment.**

---

## 8. Top 5 risks, ranked

**R1 — Label leakage → Model B is a learned IK solver, discovered on hardware.** *(Probability high, impact catastrophic, detectability near-zero without the specific diagnostics.)*
→ Run the vision-ablation and pose-perturbation tests **before building anything** (G1 criteria). Log servo current/torque and wrist F/T so a learned low level has information IK lacks. Add an independent EE-pose channel (VR/AprilTag) at collection time if the metric interface survives. Implement Model B as a **residual on top of analytic QP-IK**, not a replacement. Apply scheduled sampling + hindsight relabeling + DART noise + latency matching + joint finetuning, in that cost order. **Ship v1 with the analytic QP-IK low level and gate the learned version behind a measured win** — this removes the low level from the sim-transfer critical path entirely.

**R2 — Mac mini is not shippable product compute; the team discovers this in month 9.** *(Probability certain, impact = one lost quarter minimum.)*
→ Six independent disqualifiers: EULA §8E injury disclaimer + §2J redistribution ban with no OEM program; 10–35 °C / unfiltered intake / AC-only; no RT kernel; ROS 2 Tier 3, amd64 only; no BMC/watchdog/serial console, 90-day max update deferral; **no lifecycle commitment — the 64 GB SKU already vanished mid-cycle in June 2026.** Mitigation is cheap and immediate: `Policy.infer` boundary + network-separated inference host this week; Jetson AGX Orin 64 GB (275 TOPS, 204.8 GB/s, −25/+80 °C, DC input, ROS 2 Tier 1, PREEMPT_RT in JetPack, **production through Jan 2032**) as the product baseline. Keep the Mac as the dev box — it is genuinely good at that. Do ROS 2 work in arm64 Ubuntu Docker, never natively.

**R3 — Data economics blow up the timeline.** *(Probability high if scope stays at "general-purpose bimanual.")*
→ Rescope to 3–5 tasks, hard. Budget by **diversity** (`scenes × objects × 30–50 demos`), not volume. **Do not train from scratch** — finetune SmolVLA (~450M, Apache-2.0, the only class with a plausible path to useful Hz on Apple Silicon), π0/π0.5 via openpi, or GR00T N1.x. That is the difference between needing 100–500 demos/task and needing thousands. **Verify openpi and GR00T commercial license terms before building a product on them** (SmolVLA, OpenVLA Apache-2.0 and Octo MIT are clean). Write and enforce an operator protocol before episode 1: one canonical grasp strategy per object, fixed mounts, fixed lighting, fixed reset — the scripted-vs-human gap at identical demo counts (86%→50%, 32%→20%) is the measured cost of operator stochasticity, and a documented 60-episode SO-100 failure was caused by mixing top-down and side-pinch grasps on visually identical states.

**R4 — Reliability ceiling: pure imitation plateaus far below the commercial bar.** *(Probability high — this is the industry's stated experience, not speculation.)*
→ Build the flywheel from day one and treat it as higher-priority than the architecture split: autonomous rollouts, success labeling, a value/critic head, and full observation-window capture on every intervention. Make MTBI the north-star metric from sprint 1. Give Head A a first-class `{CONTINUE, RETRY, ABORT, REQUEST_TELEOP}` output. Ship teleop-assist fallback in v1 so you can sell before the model clears 99%. Compound error is the arithmetic: 0.95¹⁰ ≈ 0.60, and no parameter reallocation changes the exponent — only detection and recovery does.

**R5 — Closed-chain internal wrench, and the safety/certification gap behind it.** *(Probability moderate, impact severe and possibly hardware-destroying.)*
→ Measure your rig's Cartesian stiffness in week 1: it determines whether constraint violation manifests as slipping (cheap, compliant arms) or crushing/e-stop (stiff arms) — and those demand different mitigations. Implement the coordination-mode reparameterization so the 6 excess DoF are **structurally eliminated** in TIGHTLY_COUPLED, not regressed. Predict `T_left→right` directly, never by subtraction. Add wrist F/T or current-based external-force estimation as both a policy input and a hard RT-layer threshold — it is the fastest detector of the jam/crush/collision failures that produce unrecoverable errors, and it is required anyway for power-and-force limiting. **Choose arms that ship a certified safety controller before collecting data**, because your data is embodiment-specific and re-collecting it is the most expensive mistake available to you.

---

## 9. STOP / START

### STOP considering — immediately

- **Two independent 6-DoF pose targets as the interface.** Geometry, not tuning: 6 excess DoF that can only become internal wrench.
- **Absolute pose anywhere in the interface.** UMI 100% relative vs 25% absolute; SRT ~0% absolute.
- **Head-relative as the interface frame.** Input feature to Head A only. Sim will never show you the calibration penalty.
- **Model B consuming RGB.** It duplicates the most expensive component and caps you at ~5 Hz. A cerebellum that re-encodes pixels is a second cerebrum.
- **"Split capacity to improve performance" as the justification.** No supporting evidence exists. Replace with the *data* argument (Head A absorbs sim/cross-embodiment/human data; only Head B needs your scarce demos) and the *rate/horizon* argument (proven: Helix, GR00T, π0.5) and the *commercial* arguments (update cadence, per-customer finetuning, logged interpretable interface).
- **Two separately trained models with a permanently frozen non-differentiable pose API.** Use one checkpoint, shared vision encoder, **stop-gradient** at the boundary (Knowledge Insulation), with joint finetuning as the final polish.
- **"Both models must run locally on the Mac mini" as a design constraint.** It is not a requirement; it is an assumption that is actively distorting the architecture.
- **Mac mini as product compute.** And stop planning to train anything on it beyond the first single-task prototype.
- **Per-embodiment low-level swap** and **smaller OTA** as pitch points. Both are stale; a DD reviewer will mark them.
- **50 tasks / general-purpose framing.** ~3,000 human-hours, $200k–500k DIY, 9–27 months.
- **MJX, Isaac Lab, MuJoCo Playground/Warp, RoboTwin locally.** Confirmed unavailable; stop trying.
- **Validation MSE as a checkpoint-selection or decision metric.** MMRV 0.375, r 0.308 — nearly useless.
- **Binary grippers. Per-step (k=1) interfaces. Asynchronous per-arm streams. Passive GELLO leaders. A single global camera.** Each is independently disqualifying.

### START this week — five days, concrete

**Day 1 — the three benchmarks that settle most remaining arguments empirically and cheaply.**
(a) Two MLX models concurrent on the GPU: p50/p99/p99.9 solo vs together. (b) A periodic 100 Hz thread with `mach_absolute_time`, histogram logged over 24 h with Spotlight and Time Machine active — this is your written real-time evidence. (c) Your actual vision encoder at your actual resolution and camera count. Run `PYTORCH_ENABLE_MPS_FALLBACK` **off** in CI from today so missing ops fail loudly.

**Day 1–2 — decouple the hardware decision while it is still free.** Put all inference behind `Policy.infer(obs) → ActionChunk` with MLX and TensorRT implementations, and put the inference host behind a network interface. Rent the Linux + NVIDIA box. Adopt the two-box architecture in simulation *before* touching hardware.

**Day 2–3 — verify the MJCF for your target arms exists and is trustworthy** (inertias, joint limits, gripper geometry). Then reproduce ACT's published sim baseline (86% / 32%) as harness validation. Do not proceed to any ablation until the harness reproduces a known number.

**Day 3 — write and freeze interface spec v0.1 (§4) and the logging schema (§5.3).** RDT's 128-dim layout + servo current/torque + hardware timestamps on one monotonic clock + intervention labels + coordination mode + calibration snapshot. This document is the product asset that makes independent update cadence and per-customer finetuning real. Version it with semver from today.

**Day 4 — commit the hardware.** Active-motor bimanual leader-follower (HIL/DAgger requires torque-controllable leaders — this cannot be retrofitted). Two wrist cameras plus 1–2 scene cameras, fixed mounts, fixed lighting. For the product path, arms that ship a certified safety controller. Order it. In parallel, buy ISO 10218-1:2025 and -2:2025 (~$244 each) and read the functional-safety and power-and-force-limiting sections yourself.

**Day 4–5 — pick THE task,** using the Chef/Dyna filter: high labor cost, high mix (so fixed automation loses), **low consequence of failure** (so the safety case is cheap and a protective stop is acceptable), fixed and controllable workcell. Design the architecture backwards from it.

**Day 5 — pre-register the decision rule in writing before running anything:** which tasks, how many episodes, how many seeds, what effect size counts as a win, and what the dynamics sweep is. Report confidence intervals, not point estimates. Write down in advance that a NO-GO at G1 means shipping the monolith + analytic QP-IK, and that this is an acceptable outcome.

**Also this week:** plan the sysID protocol (joint stiffness, damping, friction, actuator lag, control latency and jitter, gripper force curve, and **Cartesian stiffness**) to execute the day the arms arrive.

---

## 10. Where the research is thin, contradictory, or estimated

**State this section to the team verbatim. Several load-bearing numbers are not measurements.**

**The central question is unanswered in the literature.** There is **no published head-to-head** of "explicit 6-DoF pose interface between two learned models" vs "latent interface" vs "monolithic joint-space policy" on the same bimanual robot and task suite. PerAct2's 16.8% — the strongest negative evidence — is confounded with keyframe discretization, a sampling-based motion planner, single-demo real-world evaluation, and the RLBench2 sim. **Treat the anti-pose verdict as a well-supported prior, not a proven result. It is also the single highest-value experiment you could run, and it is cheap in sim.**

**The research contradicts itself on where the seam goes**, and this must be resolved deliberately rather than averaged. Thread 1 says *keep the split, change the interface to a latent*. Thread 2 says *the split is at the wrong seam — move it to the subtask level and use analytic IK*. Thread 3 says *one backbone, two heads, and build the QP first*. Thread 4 says *split for data reuse but train jointly*. **Resolution adopted here: there are three seams, not one.** (i) A *semantic* seam at 1–5 Hz (subtask, mode, phase, latent) — this is where the measured 2.7× hierarchical win on compositional tasks comes from. (ii) A *rate/horizon* seam at 5–10 Hz emitting 50 Hz chunks — this is Helix/GR00T and it is proven. (iii) A *determinism* seam at the RT controller — this is legally mandatory and non-negotiable. The seam the proposal specified — "where" vs "how" within a single reaching motion — is the one seam with no supporting evidence.

**The strongest unresolved counter-datapoint to the anti-pose argument:** Helix's System 1 **outputs** wrist poses (plus finger flexion/abduction and torso/head orientation) for its 35-DoF upper body. So a pose-like representation *does* appear in a working frontier bimanual system — at the *output* of the fast policy, not as the interface *into* it. I could not determine how Figure resolves the closed-chain problem for two-handed rigid carries, nor whether that wrist-pose output is backed by an impedance or whole-body-control layer. **If it is, that layer is doing work your Model B would also have to do.** This is the most important open question in the whole analysis.

**Estimates, not measurements — do not quote as fact:**
- Internal force magnitudes (0.5–5 N compliant, 50–500 N stiff, for 5 mm relative error) are derived from typical Cartesian stiffness ranges. The mechanism and direction are solid; the magnitudes depend entirely on your arms. **Measure yours.**
- **All Apple Silicon VLA latency figures** (2B VLM at 1.3–3 Hz; 7B at 0.4–0.6 Hz; 300M flow expert at 30–80 ms, ±50%) are derived from llama.cpp-calibrated constants (5.1 TFLOPS effective FP16, 190 GB/s effective on M4 Pro). **No published benchmark of any VLA on Apple Silicon exists.** Accurate to maybe ±20% for prefill-dominated work, worse for the dispatch-bound flow expert.
- **The concurrent-two-model GPU contention figures** (~45–50% throughput each) are derived from Metal's documented command-buffer model, **not measured on your workload**. The entire dual-model-on-one-box question turns on this.
- **macOS jitter distribution** (p99 1–3 ms, p99.9 10–50 ms) is an estimate. The direction is certain; the magnitude is not.
- **The MuJoCo numbers in §6.1 ARE measured** on an M4 Mac mini — those you can quote.

**Data economics are partly vendor marketing.** The $25–50/h operator, 30–60 usable demos/hr, $50–150/demo and $50k–200k production-dataset figures come substantially from data-collection vendor pages. The peer-reviewed anchors (ALOHA $20k hardware; DROID's 50 collectors over 12 months) are consistent with them — reassuring, not confirming. **Episode yield rate is reported nowhere; 75% was assumed. If your real yield is 50%, every collection number rises 50%.** No source gave bimanual-specific per-demonstration cost, so the model likely *understates* bimanual cost.

**Legal and standards inferences, not quotations:**
- No normative standard says "a neural network cannot implement a safety function." That conclusion is inferred from ISO 13849-1:2023 / IEC 62061:2021 / IEC 61508 requirements for quantified failure rates, diagnostic coverage and systematic capability. It is the correct working assumption; get counsel before writing it in a technical file.
- Exact ISO/TS 15066 biomechanical force/pressure limits could not be verified — the tables are inside the paid standard. Commonly cited hand/finger figures (~140 N transient, ~65 N quasi-static) appear in secondary literature and are **unverified here**.
- The legal status of reselling a Mac mini embedded in a larger product is genuinely unsettled. First-sale likely covers the hardware; SLA §2J and §8E create real exposure. **This needs actual counsel.**

**Verify before designing around:**
- **RDT-1B's ALOHA deployment configuration.** I confirmed the 128-dim schema has slots for both joint positions and EEF 6-D pose, but did not find an explicit statement of which subset is populated. The joint-position conclusion is inferred from platform convention. High confidence, not verified.
- **Bi-ACT** (bilateral control ACT, arXiv:2401.17698) predicts joint angles, angular velocities **and forces** of the leader robot — a direct force-carrying extension of ACT, highly relevant to tightly-coupled squeeze tasks. Only the abstract was readable. **If tightly-coupled squeeze is core to the product, read this in full before finalizing the interface.**
- **AWE's bimanual ALOHA waypoint interpolation** — joint space or task space? Unconfirmed. If joint space (likely, given ALOHA's control convention), then the one successful "waypoint-like" bimanual interface is *joint*-space waypoints, further weakening the case for a Cartesian interface.
- **Jetson AGX Thor's operating temperature range and lifecycle commitment** were not stated on NVIDIA's product page. Confirm from the module datasheet. **Do not assume Thor inherits Orin's −25/+80 °C and Jan 2032.**
- **openpi and GR00T commercial license terms.** Unverified. Legal read required before building a product on them.
- **Genesis's Metal backend.** Claimed, unverified. Half-day spike at most.
- **Whether an M5-generation Mac mini exists in your purchasing window** (M5 Pro 307 GB/s / 64 GB and M5 Max 614 GB/s / 128 GB shipped in MacBook Pro only as of Mar 2026). This changes the dev-box calculus, not the product verdict.

**Genuinely unknown and consequential:** the direction of the sim-vs-real inversion for split-vs-monolith. Mechanisms argue both ways (sim's forgiving contact flatters a pose bottleneck; the monolith overfits sim dynamics quirks). No evidence establishes which dominates — **which is precisely why the early hardware checkpoint in Stage 1 is non-optional.**