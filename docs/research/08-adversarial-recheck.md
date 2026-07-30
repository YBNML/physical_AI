# Adversarial re-review against the confirmed I/O spec

Two facts do most of the damage below: **(a) wrist F/T is a non-kinematic input** — no function of `(target pose, joints)` produces it — and **(b) both models consume the *same* head camera frame**, so "two encoders" was never a real cost and the measured latency figures that drove the prior review were a *placement* artifact, not an architectural one.

---

## 1. "Model B is inverse kinematics, learned worse — demote to a clipped residual on analytic IK."

**Premise REFUTED. Prescription survives, but for a different reason, and the word "clipped" is wrong.**

The argument was that `(pose, joints) → joints` is a bijective kinematic map, so the model can only learn IK. Wrist F/T retires that: identical `(pose, q)` can be free space, light contact, or jammed contact. The wrench is the only input in the entire system carrying contact state, object stiffness/weight/CoM, or slip. Input-set degeneracy: gone.

What survives is a *different and weaker* objection — **optimization** degeneracy, not information degeneracy. Next-joint-angle MSE is ~99% explained by `IK(target)`; the wrench contributes a small residual concentrated in a minority of steps (force control active in 84% of in-contact steps vs 2.9% of free-space steps). Every paper that measured it found the policy ignores force unless architecture forces it not to: ForceVLA naive concat **+2.9pp** (37.3→40.2) vs **+23.2pp** structured; FoAR's naive force-concat was *worse* than vision-only on wiping. A flat concatenated vector — exactly what the diagram draws — is the configuration most likely to produce the ~+3pp outcome.

So the residual recommendation is right, but the justification must change from *"it's IK"* to *"the loss is dominated by IK, so make the analytic term explicit and the wrench becomes the residual's dominant explanatory input by construction."*

**"Learned worse" is unsupported.** No published work compares learned `(pose, joints, wrench) → joints` against IK + admittance on the same hardware. The evidence points the other way for the *input*: FACTR 21.3% → 61.2% → 87.5% on unseen objects; sensorless-bilateral ACT nut-turning 0/5 → 5/5 with force as input only; Bi-ACT eye cream 50% → 100%.

**"Clipped" is actively harmful.** The entire value of the wrench is corrections during contact that are large relative to free-space corrections. A clip bound tuned on free-space residuals suppresses exactly the signal you added the sensor to capture. Use a contact-phase gate or contact-conditioned bounds, not a fixed clip.

---

## 2. "Remove RGB from Model B; it doubles the most expensive component and caps the fast loop at 4.7 Hz."

**Both stated reasons REFUTED by measurement. The conclusion itself is probably wrong too.**

*"Doubles the most expensive component"* — both models consume the **same head camera image**, same frame, same timestamp. One encoder can serve both, trivially. And the measured saving from sharing is only **14.6 ms/cycle = 6.0%** of an MPS cycle (9.5% CPU). The two-encoder cost was never the problem, which means removing RGB was never the fix.

*"Caps the fast loop at 4.7 Hz"* — the 214/289 ms p50/p99 figure is **MPS-specific**. Measured on this M4: Model 2 **with its own ViT-S/16@224** on CPU runs **59.5 Hz p50 / 50.4 Hz p99 / 28 Hz worst-case** while Model 1 (263.8M) simultaneously holds 4.2 Hz on the GPU. That is ~11× the claimed cap, *with RGB retained*. Both-on-MPS collapses to 15.5/11.5 Hz — mutual destruction. The prior review diagnosed a **placement bug as an architecture bug**.

The false dichotomy also dissolves: Model 2's head on **cached** tokens costs 0.197 ms — **5076 Hz p50 / 4149 Hz p99 on one CPU thread**. Shared encoder at Model 1's rate + a >1 kHz proprio/F-T reflex head gives you both.

**And RGB is load-bearing in Model 2 for two reasons the prior review didn't consider.** (i) Model 1 hands over a *6-DoF pose for a 7-DoF arm*, so Model 2 must resolve one redundant DoF per arm — that is an obstacle/posture question the image can answer and proprioception cannot. (ii) Wrist F/T is **structurally blind to elbow and forearm collision**, and the elbow sweeps a 0.47 × 0.44 × 0.24 m box at *fixed* end-effector pose. Delete the image and nothing in Model 2 can see the shelf edge the elbow is about to hit.

Correct action: **share the encoder, or give Model 2 its own ≤25M-param encoder on CPU. Do not delete the image.**

---

## 3. "The absolute 6-DoF pose interface is the worst-measured action abstraction; switch to chunk-wise deltas."

**Three claims bundled. One is miscited, one is strengthened, and the real defect is missing.**

*"Worst-measured action abstraction"* — **OVERSTATED to the point of misciting.** The 8%/94%/96% result (Mazzaglia et al., RA-L 2024, Table I) classifies by *final* action. **Model 2 outputs 14 joints, so it sits in the 96% oracle column, not the 8% column.** Further, those five tasks were deliberately selected to require full-configuration control (including pressing a button *with the elbow*), the task-space baseline was MoveIt + pick_ik at defaults with no posture task, and on the same paper's standard 8-task suite the redundancy-aware action space is "completely in line with task space." On genuinely redundancy-free 6-DoF bimanual hardware across 13,000+ real rollouts, EEF-delta 89.6% vs joint-delta 88.0% — a **1.6-point, noise-level gap**. Quoting 8% here will get the whole review dismissed by anyone who opens the paper.

*"Switch to chunk-wise deltas"* — **STRENGTHENED, and more urgent than stated.** EEF absolute 69.0% → EEF delta 89.6% (+20.6pp); chunk-wise beats step-wise by up to 10pp with O(1) vs O(k) error amplification; ACT measured **1% at k=1 vs 44% at k=100** — and that was at 50 Hz. Here the loop is 2–5 Hz, so the per-step interface is far worse than the published ablation implies.

*What the framing misses:* the defect is not that the interface is a **pose**, it is that it is a **6-DoF pose for a 7-DoF arm**. One DoF per arm is unobservable and uncommandable at the cerebrum — Model 1 cannot see its own elbow or detect that its own plan is infeasible, and its pose-only state is not Markov for a 14-DoF plant. The fix is **+1 scalar per arm (base joint j₁), 12→14 numbers**, auto-labeled from recorded joints by FK at ~34 µs/sample (~35 s per 1M steps), with measured zero cost on non-confined tasks. That is dramatically cheaper than re-architecting the interface, and the prior review didn't propose it.

**Caution the prior review missed:** deltas in the *head-image* frame are ill-posed if the head moves. Chunk-wise deltas are necessary but not sufficient without fixing item 4.

---

## 4. "Head-relative frame is banned as the interface frame."

**REFUTED as stated. Under the most likely v1 condition it is backwards.**

OC-VLA measures camera-frame action grounding **beating** robot-base frame: +13.8pp sim discrete, +8.0pp sim continuous, **+10.0pp real (58.0 → 68.0)**, and degrading *less* under novel viewpoint (14.0pp vs 21.3pp OpenVLA-OFT, 16.7pp base-frame). Head/camera frame is the empirically *superior* interface frame — **conditional on a static camera with known extrinsics**, which is the one thing OC-VLA never varies ("the camera remains fixed throughout the evaluation process," recalibrated per placement).

The prior review's reason is also too strong. Model 2 has 14 joints + the head image, and the robot's own hands are usually in frame, so `B_T_H = FK(q)·(H_T_hand)⁻¹` — markerless eye-hand self-calibration. The transform is **weakly observable, not absent**. The real defect is subtler and worse: that channel fails exactly when the head moves (hands occluded by the grasped object, or out of FOV), is worst-conditioned along the optical axis, is never requested by any loss term, and loses to the cheaper shortcut of memorizing a constant `B_T_H` from static background cues — a documented failure that "collapses when workspace geometry or camera placement shifts." That is the silent-failure class: clean validation, field drift.

**Correct claim:** head frame is fine *iff* the head is mechanically fixed or commanded fixed per episode, same at train and deploy. If the neck is live, `B_T_H` is a latent exogenous variable. At the measured 214/289 ms, a modest **30 °/s pan injects 56–76 mm** of lateral error at 0.5 m reach — against a ~3 mm accuracy target. That is the dominant error term, not a rounding term.

**Correct fix — not a ban:** keep head frame for **Model 1** (OC-VLA's observation-action alignment argument applies to pose-output models); insert an **analytic `B_T_H(q_head, t_capture)` between the two models**; hand Model 2 a base-frame target so its input frame matches its joint-space output frame. Cost: one SE(3) compose (µs) against a 214 ms budget. This is exactly what EgoVLA does at deployment ("converted into robot end-effector poses through 3D transformations" *outside* the network).

**Missed entirely:** the frame defect corrupts **Model 1's labels**, not just Model 2's inputs. Model 1 predicts a *future* pose in "the head frame," but head-frame-at-t+k ≠ head-frame-at-t. EgoVLA hits precisely this and fixes it by reprojecting future wrist poses using world-frame camera poses. Without that, Model 1's supervision is already corrupted by head motion during teleop.

---

## 5. "Splitting capacity across two models has no supporting evidence; the split must be re-justified on data reuse and rate."

**REFUTED — both demanded justifications now exist, and one is measured on this exact machine.**

*Rate:* heterogeneous placement gives **Model 1 at 4.2–4.5 Hz on GPU and Model 2 at 59.5 Hz p50 / 50.4 Hz p99 on CPU, concurrently**. No single model does both: a 264M cerebrum cannot run at 50 Hz on this hardware, and a 50 Hz model cannot carry the cerebrum. Shared encoder + cached-token head pushes the fast path to **5076 Hz p50 / 4149 Hz p99 on one thread**. This dual-rate structure is the norm, not an invention (Helix: S2 7–9 Hz / S1 200 Hz; FILIC: 25 Hz policy over a 2 kHz inner loop).

*Data reuse:* Model 1's `(image, pose) → pose` is trainable from human egocentric video with no robot joints at all — EgoVLA does exactly this on ~500k image-action pairs. Model 2 needs joints + wrench, which only robot teleop provides. That asymmetry in data availability **is** the reuse justification the prior review said didn't exist.

*What the prior review got right and should keep:* the split as **drawn** is not justified — because the **interface** destroys nearly everything (no redundancy scalar, no gripper, no duration, no stiffness, no relative pose, no timestamps, no uncertainty, wrong frame). Restate as: **the split is justified; the interface is not.**

---

## 6. "Build the single-model + classical spine variant (CIR-1) first; the two-model design is dominated."

**"Dominated" REFUTED. The sequencing preference survives; the baseline demand should be strengthened.**

"Dominated" requires the single-model variant to be at least as good on every axis. It isn't: on this hardware one model is either 4 Hz *or* small, and a classical spine cannot use the wrench for the two things where the learning wins actually are — contact-phase detection and implicit object-property inference (ALPHA-α: liquid-filled and irregular objects 50%→100%, 50%→80%, precisely the properties invisible to vision and kinematics).

But the prior review **understated its own baseline**, and this part deserves reinforcing. The honest comparator is not "analytic IK" — it is **analytic IK + sensorless admittance from motor current**: Minimalist Compliance Control takes egg-on-bread 40% → 80% with *no F/T sensor and no learning*, and estimates force to 0.69 N from servo signals. Classical force control already solves well-specified insertion (100% at 0.1 mm clearance). Raise the go/no-go bar to that, not to bare position control.

Better sequencing than CIR-1: build the **shared-encoder two-head variant** — it is the single-model code plus one head, it dominates on rate, and it makes the decisive diagnostic cheap. And note that **all architecture sequencing is moot until the gripper channel exists** (below).

---

## What the prior review missed entirely

**1. There is no gripper command anywhere — this is a structural impossibility, not a degradation.** Model 1 outputs 6-DoF pose (no spare dimension); Model 2 outputs 14 arm joints; the developer's 7×2 is **all arm**. ACT's superficially identical 14 is **12 arm + 2 gripper** on 6-DoF ViperX arms — the numerical coincidence is probably why this went unnoticed. Every next-best-pose architecture in this family keeps gripper first-class (HDP: `a_high = (a_pose, a_grip)`; PerAct; RVT). **Grasping success is 0%, regardless of model quality.** Add continuous aperture (not binary — it is also the implicit grasp-force channel), *and* gripper/grasp state as an **input**, since a commanded-state-only policy cannot distinguish "grasp close" from "empty close" and silently executes the whole post-grasp trajectory holding nothing (10–30% → 100% with real feedback). Gripper/arm timing skew alone cost UMI 87.5% → 57.5%.

**2. The wrist F/T is currently a dead input.** Model 2 emits bare joint positions — no stiffness, no reference wrench, no selection mask, no gripper. It can sense a wrench it has no channel to act on except the fixed, isotropic, uncommandable servo Kp. Every system in the literature achieving genuine force *regulation* added an impedance inner loop, a reference wrench + per-axis mask, a hybrid controller, or a hand-coded reactive rule. As drawn this is BOM cost with no actuation path. Either add a stiffness/mode output, or state explicitly that the system does force-aware **switching**, not regulation.

**3. No timing/duration field.** A waypoint with no dt is not a trajectory. Approach velocity becomes Δpose ÷ inference jitter — and approach velocity is **linearly proportional to impact force** (measured, 0.02–0.16 m/s). The 214 vs 289 ms p50/p99 spread yields **~35% run-to-run contact-force variation on an identical commanded pose**. Compliance cannot rescue this: reducing stiffness measurably affects post-impact jerk but has **no significant effect on impact force**, because contact must occur before the stiffness term acts.

**4. No inter-hand relative pose.** Measured 70% vs 30% on UMI's bimanual fold. It **cannot** be recovered by subtracting the two absolute poses: an L2 loss over two absolute poses penalizes a harmless common-mode error and an object-crushing differential error *identically*, so the network never receives gradient pressure to preserve the relative constraint.

**5. No timestamps or head/torso joint state.** Required to compute `B_T_H` at capture time; also, "duration" is meaningless without a clock the two models agree on.

**6. The teleop rig is a precondition, not a detail.** If the rig is unilateral, the operator never felt contact, their applied forces were open-loop and inconsistent across demos, and the wrench channel carries no learnable structure — no architecture recovers a force policy from that. Every large force gain in the literature came from a bilateral or force-feedback rig. ALPHA-α achieved this at $8,951 vs ALOHA's $20,485, so it is not a cost tradeoff. **Check this before any modeling work.**

**7. The decisive diagnostic is one eval run.** Freeze the trained Model 2; zero the wrench, and separately shuffle it across trajectories. If success rate and joint MSE are statistically unchanged, the model took the IK shortcut and conclusion 1's prescription becomes mandatory. Pair with `‖q_model − IK_DLS(·)‖` plotted against `‖wrench‖`, `∂q/∂wrench` split by contact vs free space, and evaluation on **held-out object stiffness/weight** — force's payoff is generalization and does not appear on training objects.

**8. Re-open buy-vs-estimate on the sensor.** Two FT300-S is ~$3.9k plus wiring, mandatory payload-inertia identification, a hard 100 Hz ceiling, and **~2 N residual noise during motion** (which exceeds many contact forces of interest, and is an independent hardware-level reason to gate force on contact phase rather than concatenate it flat). NEXT gets 0.018 ± 0.012 Nm from motor current on a $2,500 arm with 10 min of data and $0 hardware.

**9. Sim-then-real should be scoped.** Contact stiffness, friction, and F/T noise/drift are precisely what simulators get wrong; essentially none of the force-policy results in the literature are sim-to-real. Scope the sim phase to free-space/kinematic behavior.

**10. Methodological note.** The 4.7 Hz figure that drove conclusion 2 — and colored 5 and 6 — was an artifact of co-scheduling both models on MPS. Any future latency claim should specify device placement, because on this machine CPU beats MPS at batch 1 at every Model 2 size tested (ViT-B: 44.7 ms CPU fp32 vs 64.2 ms MPS fp16), batch-1 inference being dispatch-bound rather than FLOP-bound.

---

## Net

Of the six conclusions, **one survives intact** (3's delta/chunking half), **two survive with the premise replaced** (1, 6), and **three are refuted or materially overstated** (2, 4, 5). The two largest problems in the design — the absent gripper channel and the fact that the newly-confirmed F/T sensor has no actuation path — appear nowhere in the prior review, and both outrank every conclusion it did reach.