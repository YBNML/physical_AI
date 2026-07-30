# Empirical Plan: Does the A/B Split Actually Beat a Monolith on *Your* Setup?

**Bottom line up front.** Your hypothesis is really four hypotheses wearing one coat, and they have wildly different prior support. Separate them before spending a day of compute, because three of them are cheap to test and one of them is the only one you're actually claiming.

| # | Hypothesis | Prior evidence | Cost to test |
|---|---|---|---|
| **H1 — Capacity** | At fixed total params, data, init, and command rate, two specialized models beat one. | **Negative.** HiRT real quasi-static: 70.0 split vs 71.3 mono. Orchestration study short-horizon: 69.57 vs 69.63 (tie). MoE scaling laws: routing gains diminish with scale, reasoning saturates. **No paper runs this controlled experiment.** | Days (E0), weeks (E1) |
| **H2 — Latency** | The split raises achievable command rate. | **Confounded.** SmolVLA got equal success, ~30% faster completion, 2× throughput from async chunking on *one* model. OpenVLA-OFT: 26× faster on a 7B monolith, no split. On a single Mac GPU the two models *serialize*. | 1 week (E3), measured |
| **H3 — Data provenance** | The split lets the high level eat cheap non-teleop data the low level can't. | **Strongly positive.** RT-Affordance predictor: 77% → 24% without ~750 hand-annotated images, → 11% without web co-training. HAMSTER: 1.2M off-domain high-level samples vs 320 in-domain low-level episodes. | 2 weeks (E2c) |
| **H4 — Inspectability** | A pose interface is debuggable, unit-testable, human-correctable. | **Positive and underrated.** RT-H's largest single number was human correction (40% → 63%), not the hierarchy (+15%). | ~Free, falls out of E1 |

**H1 is what you stated. H1 is the one nobody has ever demonstrated.** The plan below is designed to kill H1 in under a week if it's false, and to route you to H3/H4 (which are probably true and probably worth more) if it is.

---

## Pre-flight (hours, not days) — do these before any training

**P1. Hardware capability gate (30 min).** Determine what your arm's command interface actually accepts.
- Joint position only (SO-100/SO-101, Koch, most hobby arms): **the compliance justification for Model B is void.** A learned Model B has exactly the expressive power of IK + a position servo on contact tasks. Scope your target tasks to free-space pick/place/reach in writing, now.
- Torque / Cartesian impedance / variable stiffness (Franka FCI 1 kHz, KUKA FRI, UR force mode): the contact justification is testable, and you must add a stiffness channel to the interface.
- Record the answer. It gates E4/E5 task selection.

**P2. Concurrency measurement on the actual box (2 hours).** Run two processes: one ViT-B/16-class encoder in a tight loop, one 500M–1B model in a tight loop. Log p50/p95/p99/max for both, solo and concurrent. On an M4-base, the reference measurement is: ViT-B/16 fp16 batch-1 = **62.75 ms p50 solo → 214.29 ms p50 / 288.91 ms p99 under contention** (4.7 Hz). If your numbers look like that, **H2 is dead on this hardware before you write a line of model code**, and Model B must lose its RGB input or move to CoreML/ANE.

**P3. Camera rate ceiling (5 min).** Model B's visual reactivity is capped at camera frame rate. A 30 fps camera = 33 ms floor. "200 Hz cerebellum on RGB" cannot exist. Write down the number.

**P4. Calibration error budget (half day).** Measure hand-eye extrinsic error on your rig (AprilTag grid, AX=XB). Typical is ~2.1 mm RMS / ~3.2°. Any task with tolerance below ~2× that number cannot be done with an *absolute* pose interface, full stop, regardless of model quality.

---

## E0 — THE CHEAPEST POSSIBLE FALSIFICATION
**3–5 days · laptop · simulation · zero new teleop · 3 training runs**

### The experiment

**Train a low-level policy on the *oracle* interface and see whether it beats a monolith.** The oracle interface is the hindsight-labeled ground-truth future EE pose, extracted for free from the demonstration data. This is an **upper bound on the entire architecture**: no Model A you ever build can be better than ground truth.

**Venue:** LIBERO-Spatial (10 tasks) + LIBERO-Object (10 tasks). Built on robosuite/MuJoCo, runs on macOS CPU, ships 50 demos/task = **1,000 demos, all provided**. Chosen because (a) zero data collection, (b) EE pose is in the state vector so hindsight labels are a `numpy` slice, (c) robosuite lets you swap `OSC_POSE` ↔ `JOINT_POSITION` controllers with a config flag, which gives you the IK ablation for free.

**Three arms, ~25M params each, 3 seeds each:**

| Arm | Definition |
|---|---|
| **C0 — MONO** | One net: RGB (2 views) + proprio → action chunk. 25M params. |
| **C6 — SPLIT-ORACLE** | Low-level net only: hindsight GT target EE pose (t+k) + joints + RGB → action chunk. Params ≤ 25M. Model A does not exist. |
| **C8 — ORACLE-IK** | Hindsight GT target EE pose → robosuite `OSC_POSE` controller. **Zero learned parameters.** |

Hold constant: dataset, augmentation, optimizer, schedule, action representation, chunk horizon H=16, evaluation seeds, initial-state distribution.

**Evaluation:** 25 episodes × 20 tasks = **500 episodes per arm per seed**, 1,500 per arm total. At n=500, minimum detectable difference at 80% power ≈ **8.9 points** (two-proportion, p≈0.5). At n=1,500, ≈ **5.1 points**.

**Laptop feasibility:** MuJoCo offscreen render at 224×224 on an M4 runs a 300-step LIBERO episode in ~5–10 s. 1,500 episodes ≈ 2–4 h per arm, overnight. Training 25M on 1,000 demos on M4 GPU: ~8–16 h per arm. Total ≈ 4 days wall-clock if you pipeline.

### Decision rules

**KILL H1 (stop, pivot to H3/H4) if:**
```
C6 − C0  ≤  +5 points,  or the 95% CI of (C6 − C0) contains 0
```
Rationale: C6 has *privileged hindsight information* that no deployed system has. GHOST measured oracle-vs-deployed at **90% vs 36.7%** (−53 pts); RT-Affordance at **76% vs 68%** (−8 pts). If your ceiling is only +5 over the monolith, Model A's real prediction error will put the deployed split *below* the monolith with certainty. There is nothing to build.

**KILL MODEL B (pivot to "learned A + classical IK") if:**
```
C8  ≥  C6 − 5 points
```
Rationale: a zero-parameter analytic controller matched a 25M network. The whole PerAct/RVT/RVT-2 keypose family works exactly this way and gets precise manipulation from ~10 demos/task. If this fires, your project becomes "Model A + TRAC-IK + impedance," which is a *better* project, ships in a weekend, and frees your entire parameter budget for the half where GHOST says the error actually lives.

**GO to E1 if:**
```
C6 − C0  ≥  +10 points  AND  C6 − C8  ≥  +5 points
```

**PIVOT-REGIME if `+5 < C6 − C0 < +10`:** the ceiling is thin. Re-run E0 on **LIBERO-Long (10 long-horizon tasks)** and **CALVIN ABC→D** only. The orchestration study found hierarchy at +8.6 pts short-horizon but **+41.8 long-horizon and +30.0 reasoning-heavy**. If the ceiling is thin on short-horizon and fat on long-horizon, your project is a long-horizon project and you should say so in writing before continuing.

> **Why this is the right single kill-shot:** it costs 3 training runs and no data collection; it tests the *ceiling* of the architecture rather than one instantiation of it, so a negative result is not rescuable by "better Model A" or "more data"; and it simultaneously runs the Model-B-necessity test on the same harness for free.

---

## E1 — Isolating THE SPLIT from every confound
**2–3 weeks · sim · rent a 4090/A100 spot instance (~$100–250 total) or run 2 weeks on the Mac**

E0 established a ceiling. E1 asks whether the *deployed, separately-trained* split beats the *best* control — not the worst one.

### The control ladder (name these explicitly in your writeup)

All arms at **matched total parameters P**, **matched demonstration set D**, **matched initialization**, **matched effective command rate R**, **matched action representation**.

| ID | Name | What it isolates |
|---|---|---|
| **C0** | **MONO** — one net, RGB+state → action chunk, P params | The baseline you must beat |
| **C0s** | **MONO-SMALL** — one net at P_A params, and again at P_B params | The parameter-count scaling curve. Without this you cannot tell a "split gain" from "the effective model got bigger" |
| **C1** | **MONO-AUX** — C0 + auxiliary head predicting future EE pose during training; head unused at runtime | **The π0.5 "implicit HL" control.** Isolates *decomposition-as-training-signal* from *decomposition-as-architecture*. OpenHelix's single largest gain (avg len 3.45→4.01) came from exactly this auxiliary head |
| **C2** | **MONO-2PASS** — same weights as C1, but at runtime actually runs pose-then-action two-stage inference | Isolates *runtime hierarchy* from *parameter separation* |
| **C3** | **SPLIT-JOINT** — two nets, P_A + P_B = P, trained end-to-end with gradients through a continuous interface | Isolates *two networks* from *two independently-trained networks* (the Helix/GR00T recipe) |
| **C4** | **SPLIT-SEQ** — **your proposal.** A and B trained separately; non-differentiable pose interface; B trained on demo GT poses | The thing you actually want to build |
| **C5** | **SPLIT-SEQ+DAGGER** — C4 + one DAgger relabeling round (roll A out, relabel, retrain B on A's actual output distribution) | Isolates the **cascaded covariate-shift tax**. GHIL-Glue measured this at +25% on CALVIN and 54%→70% real |
| **C6** | **SPLIT-ORACLE** — from E0, carried forward | The architecture ceiling; the denominator for failure attribution |
| **C7** | **SPLIT-IK** — learned A → analytic IK + OSC/impedance, zero learned low level | Isolates *whether Model B needs to be learned at all* |

### Crossed confound factors (each is a separate axis, not a fixed setting)

| Confound | Levels | Why crossed, not held fixed |
|---|---|---|
| **Pretraining** | {random init, DINOv2/SigLIP-pretrained vision, full VLM init} | Random→pretrained is **~20 points** (77.5% → 97.8% on LIBERO in a controlled study) — larger than any short-horizon hierarchy premium. If you hold init fixed you may be measuring pretraining and calling it hierarchy. Crossing it also tests whether the split's value is init-dependent (it probably is, and *negatively*: from-scratch local training forfeits the split's only real advantage) |
| **Total parameters** | P ∈ {10M, 25M, 60M, 150M} for C0, C4, C7 | MoE scaling laws say split gains *diminish with scale*. You need the curve, not a point |
| **Action chunk horizon** | H ∈ {1, 16, 50} × {C0, C4} | Chunking alone buys most of what people attribute to hierarchy. If C0@H=50 ≥ C4@H=50, the "split" was chunking |
| **High-level update interval** | k ∈ {1, 10, 30, 60} steps for C4 | OpenHelix swept this and got 94/97/95/95/95/95/95 — essentially flat. Verify on *your* setup; if flat, your Model A can run at 1 Hz and H2's premise changes |
| **Effective command rate** | {native, rate-matched to 30 Hz via chunking for all arms} | If the split's advantage vanishes at matched rate, you measured latency, and SmolVLA-style async gets it for ~50 lines of code |
| **Action representation** | {EE absolute, EE chunk-wise delta, joint absolute, joint chunk-wise delta} × 6D rotation | Measured head-to-head on real hardware: **EE-absolute 69.0%, joint-absolute 77.3%, joint-delta 88.0%, EE-delta 89.6%.** Your specified interface is the *worst* of the four. This ablation alone may be worth more than the split |
| **Interface encoding** | {raw 7-vector, chunk of 8–16 waypoints with gripper+dt, GMM rendered as image-plane heatmap, pose + learned latent} | GHOST used a GMM heatmap; RT-Affordance renders poses onto the image; RoboDual passes pose *and* latent. Nobody who succeeded passed a bare vector |
| **Demo count** | N ∈ {10, 25, 50, 100, 200} per task, for C0, C1, C4, C7 | Produces the sample-efficiency curves and the crossover point |

**Practical scoping:** the full cross is thousands of runs. Do it as a **staged screen**: (1) fix everything at the literature-best setting and run the C0…C7 ladder once (8 arms × 3 seeds = 24 runs); (2) take the top-2 arms and sweep only the confounds where the ladder was close; (3) sweep demo count last, on 3 arms only.

**Evaluation:** ≥1,000 episodes/arm (detects ~6.3 pts), ≥2,000 for the final head-to-head (detects ~4.4 pts). Report 3 seeds; seed variance in BC on LIBERO is commonly ±3–5 pts, so any claim under 5 pts is noise.

### Decision rules

**GO to E2 only if:**
```
C5  −  max(C0, C1, C2, C7)  ≥  +8 points   on ≥3 of 4 benchmark suites,
with non-overlapping 95% CIs
```

**NO-GO with named verdicts:**

| Condition | Verdict | What you actually learned |
|---|---|---|
| `C1 ≥ C5 − 3` | **The benefit was the training signal.** | Ship C1. One model, one forward pass, auxiliary pose head. Half the latency, no interface, no covariate shift, no second dataset. This is π0.5's own finding |
| `C7 ≥ C5 − 3` | **Model B is unnecessary.** | Ship A + TRAC-IK + impedance. Move all parameters to A |
| `C3 ≥ C5 + 8` | **Separate training is the problem, not the split.** | Move to joint training through a continuous latent (Helix), or freeze-A + pre-aligned projector (OpenHelix — without pre-alignment, *every* configuration scored 0/0/0/0/0) |
| `C5 − C4 ≥ +10` | **Covariate shift is your dominant error term.** | DAgger on the interface is mandatory, permanently, after every Model A retrain |
| Split advantage vanishes when init is crossed to pretrained | **You measured pretraining.** | Use a pretrained VLM for A or abandon the architecture |
| Split advantage vanishes at matched command rate | **You measured latency.** | Implement async chunking on the monolith; delete the second model |
| `C0` at P ≈ `C0` at P/2 (flat scaling curve) | **You're not capacity-limited at all.** | The whole "split the parameter budget" framing is moot; you're data-limited. Go to E2c |

---

## E2 — Regime tests: where hierarchy is *allowed* to win
**1–2 weeks · sim · run only if E1 passed**

E1 was run on the regime where the literature says hierarchy ties. E2 tests the regimes where it wins, and the generalization axes you care about.

### E2a — Dynamic tasks (the one regime with a large published effect)
HiRT: static tasks 70.0 vs 71.3 (mono ahead); **dynamic tasks (objects moving at 1 cm/s) 75 vs 48** (split ahead by 27). OpenHelix CALVIN-D: single-system RoboFlamingo scored 100% static / **0% on all four dynamic conditions**.

- Venue: **CALVIN-D** (objects move during episode), **ManiSkill3** dynamic variants, **MuJoCo Playground** with scripted object velocity.
- Manipulation: object translation at {0, 1, 3, 10} cm/s during the episode.
- Metric: success vs object speed. Report the *slope*, not just the mean.
- **GO if the split's advantage grows monotonically with object speed.** If it's flat in speed, the split is not buying reactivity and H2 is dead.

### E2b — Perturbation recovery
- Scripted disturbances applied at t = T/2: object nudge {2, 5, 10} cm lateral; arm push (external wrench 5 N for 200 ms); 0.5 s camera occlusion; gripper slip (release + re-grasp required).
- Metric: **recovery rate** = fraction of episodes that reach success *after* the perturbation, conditioned on being on-track before it.
- Secondary: time-to-recovery, and whether the high level re-queries (this is where the missing switching/termination policy will bite — the orchestration study found ablated hierarchies collapsing from ~95% to near 0 on this axis).

### E2c — The data-provenance test (H3 — probably your real result)
This is the mechanism that actually made every pose-interface system work, and it is **not tested by any arm above**.

- **C4+ARM (split with off-domain high-level data):** Model A additionally trained on data Model B structurally cannot consume — sim-rendered scenes from other embodiments, ~750–2,000 hand-annotated still frames (RT-Affordance's number: 1 h to collect, 2 h to annotate), point-labeled web images, action-free human video with hand-pose extraction.
- **C0+ARM (monolith with the same extra data):** the honest control — can the monolith absorb it too? (Usually only partially, via auxiliary losses.)
- **GO signal:** `C4+ARM − C4 ≥ +15 points` while `C0+ARM − C0 < +5`. That is the split earning its existence through *data access*, which is the claim the literature supports. RT-Affordance's ablation is the shape to expect: 77% → 24% without the annotated images, → 11% without web co-training.

### E2d — Generalization axes (report each separately, never averaged)
Hold the policy fixed; vary one factor at a time, 200 episodes each:

| Axis | Manipulation | Suites |
|---|---|---|
| Novel object instance | Same category, unseen mesh/texture | LIBERO-Object, ManiSkill3 |
| Novel object category | Unseen category entirely | ManiSkill3, SimplerEnv |
| Novel position | Initial pose sampled *outside* the training convex hull | Any (custom reset dist) |
| Novel lighting | Ambient ±50%, directional angle ±45°, color temp 3000–7000 K | robosuite renderer |
| Novel camera pose | Extrinsic perturbed 5 cm / 10° | robosuite; also the OC-VLA frame test |
| Novel distractors | +3 unseen objects in scene | LIBERO-Spatial |
| Instruction rephrasing | CALVIN-E enriched instructions | CALVIN-E |

**Also run E2d with the oracle interface (C6).** The gap `C6 − C4` on each axis is your **failure attribution**: it tells you what fraction of each generalization failure is Model A's pose error versus Model B's execution. GHOST used exactly this and found 40% → 90% from swapping only the high level, which is what proved the bottleneck was upstream.

### E2e — Sim-to-real correlation check (cheap insurance)
Before spending real-robot months, run your top-2 arms on **SimplerEnv** (Bridge + Google Robot real-to-sim). If your sim ranking doesn't correlate with SimplerEnv's, you have no reason to believe the sim conclusion transfers. Google ran **>90% of Gemini Robotics 1.5 development evals in simulation** — but they validated the correlation first.

---

## E3 — Rate and systems measurement on the actual target box
**1 week · your hardware · no training**

This is where H2 lives or dies, and it is a *measurement* task, not a modeling task.

**Measure, do not extrapolate:**

1. **MONO + chunking + async, on the Mac mini.** One model, H=50 chunks, SmolVLA-style RobotClient/PolicyServer with `chunk_size_threshold ≈ 0.6`, weighted-average temporal ensembling. Record: effective command rate, p50/p95/p99/max photon-to-command latency, queue-underrun rate.
2. **SPLIT (two processes).** Same, with A and B. Record the same.
3. **SPLIT with Model B stripped of RGB** (interface + joints + wrench only).
4. **SPLIT with A in MLX/Metal (GPU) and B compiled to CoreML pinned to `cpuAndNeuralEngine`, GPU explicitly excluded.** Verify ANE residency with `powermetrics` and the CoreML performance report — CoreML silently falls back to GPU on unsupported ops and will recreate the contention you're trying to avoid.

**Reference numbers to compare against (M4-base, measured):** ViT-B/16 fp16 b1 = 62.75 ms p50 solo → 214.29 p50 / 288.91 p99 concurrent. 805M-param model over 1024 visual tokens = 467 ms p50 solo → 500 p50 / 628 p99 concurrent. Proprio-scale MLP = 2.18 ms p50 / 2.80 p99. Total GPU throughput is conserved: **two models on one GPU cost the sum of their latencies.**

**Decision rules:**
```
KILL H2 if:  MONO+chunking achieves ≥30 Hz effective command rate with p99 < 2×p50
GO on H2 only if:  SPLIT achieves ≥2× MONO's effective rate AND fast-loop p99 < 50 ms
```
Also record: `kern.sched` (Apple's `edge` scheduler migrates threads across P/E cores), whether `thread_time_constraint_policy` is in use, and the jitter ratio p99/p50 under camera capture + display load. **If the ≥500 Hz servo loop is planned to run on macOS, stop and move it** — to the arm's own controller or a Linux SBC over wired Ethernet (measured cost of that hop in the literature: ~0–4 ms, i.e. free). That single change moots every macOS real-time objection while keeping the Mac as an inference server, which it's good at.

---

## E4 — Real-robot pilot
**3–4 weeks · run only if E1 passed with ≥15 points**

**Why the 15-point bar:** on a real arm you can realistically afford ~400 rollouts per arm, which detects ~10 points at 80% power. HiRT's split-vs-mono gap on real quasi-static tasks was **1.3 points**. TRI needed **50 rollouts per task per policy per condition, 1,800 total trials, blind randomized A/B, Bayesian posteriors, Bonferroni correction** to separate policies. If E1 gave you 8 points in sim, the real robot cannot resolve it and E4 is a waste of months.

### Hardware options by budget

| Platform | ~Cost | Control interface | What it can test |
|---|---|---|---|
| **SO-101 (×2, leader/follower)** | $250–500 | Serial-bus servo, **position only** | Free-space pick/place, generalization, sample efficiency. **Cannot test compliance.** LeRobot native, MPS-supported, async inference built in |
| **Koch v1.1** | ~$400 | Position only | Same class as SO-101 |
| **ALOHA / bimanual SO-101** | $1k–30k | Position, 50 Hz | Bimanual coordination, long-horizon. Tests the relative-transform interface question |
| **UR5 / UR5e (used)** | $10k–35k | RTDE 500 Hz, `servoJ`, force mode | Rate decoupling, basic compliance |
| **Franka Panda / FR3** | $20k–30k | **FCI 1 kHz torque + Cartesian impedance, joint torque sensing** | The only one where the contact/stiffness justification for Model B is testable at all |
| Add-on: Robotiq FT-300 / Bota | $3k–5k | 6-axis wrench | Contact-force metric; enables ForceVLA-class comparisons |
| Poor-man's proxy | $0 | Servo current draw (SO-101 exposes it) | Crude contact detection, adequate for a stall/jam metric |

**Recommendation for a hobbyist/lab budget doing this honestly: SO-101 pair (~$400) for E4, and rent/borrow a Franka for one week only if E2 showed the split winning on contact-rich tasks.** Do not buy a Franka to test H1.

### Teleop demonstration budget

| Stage | New teleop demos | Wall-clock |
|---|---|---|
| E0, E1, E2a/b/d | **0** (LIBERO/CALVIN/ManiSkill ship demos) | — |
| E2c off-domain data for A | 0 teleop; **~750–2,000 annotated still frames** | ~1 h collect + 2–3 h annotate |
| **E4 pilot** | **5 tasks × 50 demos = 250** | **6–10 h** including resets |
| E4 DAgger round | 0 teleop (autonomous rollouts + hindsight relabel) | ~3 h robot time |
| E5 powered comparison | **0 new demos** — same 250 | — |
| If sample-efficiency curve needed on real | +5 tasks × 150 = 750 more | +20–30 h |

**Realistic totals:** ~250 demos to get a working pilot; ~1,000 demos if you need real-robot sample-efficiency curves. For reference: π0 fine-tunes on 50–200 bimanual demos/task; OpenVLA shows in-distribution gains from as few as 10; SmolVLA reports 60.0% average at 200/task; the keypose+planner family gets precise manipulation from **~10 demos/task**.

**Critical scheduling constraint:** Model B's dataset **cannot be collected until Model A exists**, because B must be trained on A's error distribution, not on demo ground truth. The schedule is strictly `train A → roll out A → hindsight relabel → train B`. Every Model A retrain invalidates B's dataset. Budget a DAgger round after *every* A retrain, and pin `(A-version, B-version, calibration-version)` as a single deployable artifact.

### E4 arms (pilot, 3 arms only)
`C0` (mono), `C5` (split + DAgger), `C7` (learned A + TRAC-IK/OSC). 5 tasks, 20 rollouts each = 100/arm. This is **underpowered by design** — it is a debugging and instrumentation pass, not a comparison. Its purpose is to shake out staleness, frame errors, safety trips, and NaNs before you spend on E5.

### Mandatory instrumentation before E4 (build this *first*)
- **Non-learned reflex layer, in code, below both models:** joint position/velocity/acceleration/jerk clamps; torque or motor-current threshold with auto-retract; workspace bounding box; self-collision + reachability rejection; finite/in-range assertion on both models' outputs; step-magnitude limiter (reject any joint command >N mrad from current measured config); deadman watchdog (hold-last-good, then ramp velocity to zero after K missed deadlines).
- **Typed, timestamped, versioned interface message:** SE(3) target (**6D continuous rotation**, never Euler or raw quaternion) + gripper command + duration/velocity scale + capture timestamp + head/base joint state **at capture** + frame id + sequence number + validity horizon. Convert to `base_link` at A's output using the capture-time head state, never inside B. Stale policy: beyond the horizon, hold and decay to zero — never extrapolate.
- **Oracle harness:** (a) Oracle-A — replay hindsight GT poses into B; (b) Oracle-B — swap B for TRAC-IK+impedance, feed A's live poses. Without these two you cannot attribute a single real-world failure to A vs B vs interface vs staleness vs calibration, and each hypothesis costs a week to chase.
- **Offline replay regression test:** replay a fixed logged interface trace through B, assert command bounds + smoothness. Catches the silent A-retrain-breaks-B regression in CI instead of on the robot.

---

## E5 — Powered real-robot comparison
**4–6 weeks · run only if E4 was clean and E1 gave ≥15 points**

- **Arms:** `C0` (mono, matched params, matched init, matched effective rate via chunking) vs `C5` (split + DAgger) vs `C7` (A + IK). Three arms.
- **Design:** 5 tasks × **80 rollouts per arm per task = 400 per arm**, 1,200 total. Blind (operator does not know which policy is loaded), randomized order, fixed reset protocol with marked object positions, fixed lighting log.
- **Statistics:** two-proportion tests with Bonferroni correction across tasks; Beta/Dirichlet posteriors with credible intervals; Compact Letter Display for the multi-arm comparison. Report per-task, never a single average.
- **Minimum detectable effect:** n=400 → ~9.9 points. n=250 → ~12.5. n=180 → ~14.8. State this in your writeup.

---

## Metrics — full definitions

**Primary**
1. **Task success** — binary, per predefined criterion, judged from video by a rater blind to condition.
2. **Progress score** — partial credit over predefined subgoals (grasp achieved / lifted / transported / placed). Success rate alone will not separate policies at these effect sizes.

**Control & timing** (log every cycle; report distributions, never means)
3. **Control rate achieved** — commands/s at the servo input, p50 and p5 (worst-case sustained).
4. **End-to-end latency** — photon (camera timestamp) → joint command at the servo. Report **p50 / p95 / p99 / max**.
5. **Jitter ratio** — p99/p50. A healthy edge stack looks like the Jetson reference: 150.5 ms mean, **0.13 ms std**, range 150.4–151.0. If yours is >2×, you have a scheduling or contention problem, not a model problem.
6. **Interface staleness** — age of the Model A output at the moment Model B consumes it. p50/p99.
7. **Queue underrun rate** — fraction of control cycles with no action available.

**Physical accuracy**
8. **Pose tracking error** — ‖achieved EE pose − commanded target‖: translation (mm) and rotation (deg), RMS and p95, reported *separately per task phase* (free-space approach vs contact).
9. **Interface error** — ‖A's predicted pose − hindsight GT pose‖. This is the single most diagnostic number in the whole system.
10. **Terminal placement error** — final object pose vs goal pose.
11. **Path smoothness** — RMS joint jerk; count of sign reversals in joint velocity (catches IK branch-flipping and chunk-boundary discontinuity).

**Contact**
12. **Peak contact force / wrench** during contact phases (F/T sensor, or MuJoCo contact force in sim, or motor current as proxy).
13. **RMS contact force** during sustained contact (wiping, insertion).
14. **Jam/stall rate** — episodes terminated by current limit or force threshold.

**Robustness**
15. **Recovery rate** — success after scripted perturbation, conditioned on being on-track pre-perturbation. Report per perturbation type and magnitude.
16. **Time-to-recovery**.
17. **Livelock rate** — episodes where the same failed action repeats ≥3 times (this is the missing switching/termination policy showing up).

**Generalization** — success on each of the 7 axes in E2d, reported separately, plus the `C6 − C4` oracle gap per axis as attribution.

**Sample efficiency**
18. **Success vs N demos** curve at N ∈ {10, 25, 50, 100, 200}, with **crossover N*** (the demo count at which split overtakes mono, if ever). If split only wins at large N, it is not the sample-efficiency story you wanted.

**Safety / liveness** (per 100 rollouts)
19. Joint-limit trips · velocity-clamp trips · IK-infeasible rejections · workspace-bound rejections · finite/range rejections · watchdog trips · human interventions.

**Attribution**
20. **Oracle gap** `C6 − C4` — fraction of the remaining error owned by Model A.
21. **Classical gap** `C7 − C4` — how much (or how little) the learned Model B is worth.

---

## GO / NO-GO summary table

| Gate | Cost | GO | NO-GO / KILL | Verdict on NO-GO |
|---|---|---|---|---|
| **P1 hardware** | 30 min | Torque/impedance interface exists | Position-only arm | Compliance justification void; restrict tasks to free-space, in writing |
| **P2 concurrency** | 2 h | Fast loop p99 < 50 ms under contention | Fast loop p99 > 150 ms | H2 dead on this box; strip RGB from B or move B to ANE |
| **E0 ceiling** | 3–5 d | `C6 − C0 ≥ +10` | `C6 − C0 ≤ +5` or CI∋0 | **KILL H1.** Ceiling below monolith; no Model A can rescue it |
| **E0 IK** | (same) | `C6 − C8 ≥ +5` | `C8 ≥ C6 − 5` | **KILL MODEL B.** Ship A + TRAC-IK/OSC |
| **E1 ladder** | 2–3 wk | `C5 − max(C0,C1,C2,C7) ≥ +8` on ≥3/4 suites | `C1 ≥ C5 − 3` | Benefit was the training signal → ship the auxiliary-head monolith |
| | | | `C7 ≥ C5 − 3` | Model B unnecessary |
| | | | Advantage vanishes at pretrained init | You measured pretraining |
| | | | Advantage vanishes at matched rate | You measured latency → async chunking, ~50 lines |
| | | | `C3 ≥ C5 + 8` | Separate training is the defect → joint training or frozen-A + pre-aligned projector |
| **E2a dynamic** | 1 wk | Split advantage grows with object speed | Flat in speed | H2 dead; the split isn't buying reactivity |
| **E2c data** | 2 wk | `C4+ARM − C4 ≥ +15` while `C0+ARM − C0 < +5` | Both gain equally | The monolith absorbs the extra data too; H3 doesn't need a split |
| **E3 rate** | 1 wk | Split ≥2× mono effective rate, fast-loop p99 <50 ms | Mono+chunking hits ≥30 Hz | **KILL H2.** Delete the latency justification |
| **E4 pilot** | 3–4 wk | Clean instrumentation, <1 safety trip / 100 rollouts | Repeated staleness/frame/NaN faults | Fix systems before spending on E5 |
| **E5 powered** | 4–6 wk | `C5 − best control ≥ +15` at n=400/arm, Bonferroni-corrected | `< +15` or CI∋0 | Underpowered to distinguish; report the tie honestly |

**Secondary gates at E5 (any failure → investigate before declaring GO):** pose tracking p95 ≤ 5 mm / 3°; safety trips ≤ 1 per 100 rollouts; interface staleness p99 ≤ 100 ms; recovery rate from 5 cm nudge ≥ 50% and strictly > monolith.

---

## What to build instead, on each NO-GO

These are not consolation prizes — several are better projects than the one you proposed.

1. **`C1` wins → the auxiliary-head monolith.** One model, one forward pass, target-pose prediction as an auxiliary loss. Half the latency, no interface, no covariate shift, no second dataset, no MLX queue contention. This was π0.5's own second-best configuration and the authors' conclusion was verbatim that *"a significant portion of that benefit is already obtained simply by including subtask prediction data in the training mixture."*

2. **`C7` wins → learned A + TRAC-IK + Cartesian impedance.** Zero learned parameters below the pose. Ships in a weekend. TRAC-IK: >99.8% query success, sub-millisecond, versus a learned pose→joint map at mm-to-cm error and 3 orders of magnitude slower — and IK returns a *typed infeasibility signal* ("no solution", "at joint limit", "near singularity") that a regression network structurally cannot produce. This is what every deployed VLA (OpenVLA, π0, RT-2) actually does.

3. **H2 dead → async chunked monolith.** LeRobot's PolicyServer/RobotClient with `chunk_size_threshold ≈ 0.6` and weighted-average temporal ensembling, plus Real-Time Chunking for smooth chunk boundaries. Equal success, ~30% faster completion, 2× throughput, no retraining. Chunk length is nearly free (5→250 actions costs ~11% end-to-end latency) while denoising steps are expensive (10→50 costs 5×) — so use long chunks and few flow-matching steps, and never an autoregressive token action head (102× penalty).

4. **H3 confirmed → keep the split, but for the right reason and in the right shape.** Lopsided (A ≫ B; Helix is 7B/80M, RoboDual 7B/20M trainable — *nobody splits 50/50*), pretrained frozen high level, chunk-wise SE(3) **delta** with 6D rotation + gripper + duration + (7-DoF) swivel angle + compliance flag, delivered to B as a **rendered image-plane heatmap** rather than a raw vector, and fed by cheap non-teleop supervision. Model B as a **clamped residual** on TRAC-IK+impedance, so residual→0 recovers a working classical stack.

5. **H4 confirmed → ship the interface as the product.** Pose targets are inspectable in RViz, correctable with a 6-DoF mouse, unit-testable, and loggable. RT-H's largest single number was human correction (40% → 63%, beating IWR by 50%), not the hierarchy itself. If your deliverable is a debuggable, human-in-the-loop-correctable robot, the pose interface earns its keep even at zero success-rate gain — and you'll have measured that honestly rather than claimed it.

---

## Total budget

| | Wall-clock | Compute | Teleop | Money |
|---|---|---|---|---|
| Pre-flight | 1 day | laptop | 0 | $0 |
| **E0 (the kill-shot)** | **3–5 days** | **laptop** | **0** | **$0** |
| E1 | 2–3 weeks | rented 4090/A100 spot | 0 | $100–250 |
| E2 | 2–3 weeks | rented GPU | 0 (+3 h annotation) | $100–200 |
| E3 | 1 week | target hardware | 0 | $0 |
| E4 | 3–4 weeks | laptop + arm | **250 demos / 6–10 h** | $400 (SO-101 pair) |
| E5 | 4–6 weeks | laptop + arm | 0 new | $0 |

**You can falsify the central claim for $0 in under a week.** Everything after E0 is contingent on E0 passing, and E0 tests the *ceiling*, so a negative there is not rescuable by more data, a better Model A, or a bigger budget. That is the entire point of running it first.