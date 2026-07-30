# Two-Model "Cerebrum/Cerebellum" Manipulation Architecture — Decision-Ready Technical Review

**Scope:** evaluation of a proposed local dual-model design — Model A (RGB + current EE pose → target EE pose) and Model B (target pose + joint values + RGB → joint commands) — for a physical-AI manipulation robot running on an Apple-silicon Mac mini.

**Evidence-strength convention used throughout:**
- **[M]** = measured on the user's actual machine during this review (Apple M4 base, 10 CPU / 10 GPU cores, 32 GB unified, PyTorch 2.11/MPS, batch 1).
- **[P]** = peer-reviewed or well-established published result.
- **[X]** = 2025–2026 arXiv preprint, retrieved but not independently verified or replicated. Treat as directional.
- **[V]** = vendor/company blog, no ablation published. Architectural details reliable, performance claims not.
- **[I]** = inference/reasoning by the reviewer, not a measured or cited result.

---

## 1. Verdict

**What is right:** the instinct to decompose is sound, the choice of split point (Cartesian, between "where the hand goes" and "how the arm gets there") is the same split point 40 years of robotics and every keyframe-based manipulation system converged on, and a pose-shaped intermediate representation has the strongest evidence in the entire hierarchical-VLA literature (RT-Affordance: 68% with predicted poses vs 28% for flat RT-2 at matched backbone and data [X]; HDP 80.18% vs 15–18% for flat Diffusion Policy/ACT [P]). Giving the low level its own fresh perception is also correct and is literally the definitional criterion that separates a genuine dual-system from a single model with two rates. **What is wrong:** three specific things, in descending severity. (a) The stated *goal* — split a fixed parameter budget across two models to improve task success — is the one claim the literature does not support anywhere: no published work runs the controlled experiment, the best-controlled proxy has the monolith marginally *ahead* on real quasi-static tasks (HiRT 71.3 vs 70.0 [X]), a naive hierarchy ties a flat VLA exactly on short-horizon manipulation (69.63% vs 69.57% [X]), and the one lab that directly ablated runtime hierarchy vs hierarchy-shaped training data found most of the benefit lives in the data (π0.5 "implicit HL" [X]). (b) Model B as specified — target pose + joints → joint commands — is inverse kinematics plus a servo loop, which is solved exactly in **5.31 µs** on this machine **[M]** against a realistic 25–60 Hz for a learned network, and which returns a *typed infeasibility signal* ("no solution", "at joint limit", "near singularity") that a regression network structurally cannot produce; on a redundant arm it is additionally an underdetermined command, and the fix that recovers 8% → 94% success is one extra scalar, not a neural network **[X]**. (c) On this specific hardware the split *inverts* the rate story rather than decoupling it: two models on one Metal GPU cost the **sum** of their latencies — a ViT-B/16 encoder, the minimum Model B needs to consume RGB, measures 62.75 ms p50 solo and **214 ms p50 / 289 ms p99 under contention from a concurrent cerebrum-scale model [M]**, i.e. the "fast" loop lands at 4.7 Hz, slower than the "slow" loop was supposed to be.

**The single biggest correction:** *demote Model B from the command path to a bounded residual on top of analytic IK + impedance control, and re-justify the split on latency, cheap high-level data, and inspectability rather than on capacity allocation.* Everything else on this list is a refinement of that one move.

---

## 2. Claim Scorecard

| # | Claim as stated | Verdict | Evidence |
|---|---|---|---|
| C1 | Splitting a fixed parameter budget across two specialized models improves task success | **CONTRADICTED** | No paper runs the controlled experiment (fixed params, data, init, rate). HiRT real quasi-static: hierarchical 70.0 vs monolithic 71.3 **[X]**. Naive hierarchy vs flat VLA short-horizon: 69.57% vs 69.63% — exact tie **[X]**. MoE scaling laws (the only rigorous study of splitting capacity): routing gains follow a power law that *diminishes with model size* (Clark et al. 2022 **[P]**); at fixed active params, more experts improve memorization while reasoning *saturates*, and there exist problems no number of experts solves that a slightly wider dense model does (Mixture of Parrots, NeurIPS 2024 **[P]**) |
| C2 | Splitting improves generalization | **PARTIALLY SUPPORTED — but the mechanism is not capacity** | Real gains exist: RT-H +8–12% on novel scenes **[X]**; hierarchy +41.8 pts long-horizon, +30.0 reasoning-heavy **[X]**. But the attributed mechanism is always (a) shared reusable intermediate primitives, (b) off-domain data the high level can consume, or (c) internet pretraining. HAMSTER trains its high level on 1.2M off-domain samples and its low level on 320 in-domain episodes **[X]** |
| C3 | Splitting improves control rate | **CONTRADICTED on this hardware; supported only with dedicated accelerators** | Helix gets 7–9 Hz / 200 Hz by running S1 and S2 on **dedicated GPUs** **[V]**; Gemini Robotics puts the backbone in the **cloud** **[X]**. On one Metal GPU: ViT-B/16 62.75 → 214 ms p50 (6.3×) and 289 ms p99 (9.2×) under contention **[M]**; the 805M cerebrum-scale model went 467 → 500 ms p50 / 628 ms p99 **[M]**. Total GPU throughput is conserved. Counterfactual: SmolVLA got equal success, ~30% faster completion (9.7 s vs 13.75 s), 2× throughput (19 vs 9 cycles) from async chunking on **one** model **[X]**; OpenVLA-OFT made a 7B monolith 26× faster at action generation with 3× lower latency and then scored 97.1% on LIBERO **[X]** |
| C4 | A pose is the right *kind* of intermediate representation | **SUPPORTED** | RT-Affordance 68% predicted / 76% oracle vs RT-2 28% and goal-image-conditioned 24% with a *larger* backbone **[X]**; non-grasping tasks 70% vs 3% **[X]**. HDP 80.18% vs 57.72% (PerAct+planner) and 15–18% (flat DP/ACT) **[P]**. GHOST fold-onesie 80% vs 10% flat DP **[X]**. RT-Trajectory 63% vs 29% on 41 unseen tasks **[X]**. This is the strongest-supported element of the proposal |
| C5 | An *absolute SE(3) waypoint* is the right pose encoding | **CONTRADICTED** | Only clean head-to-head available (real single-arm, ACT-regression, avg progress): EE-absolute **69.0%**, joint-absolute 77.3%, joint-delta 88.0%, EE-delta **89.6%** **[X]**. Chunk-wise delta beats step-wise by ~10 pts with an O(1) vs O(k) noise-amplification argument **[X]**. TRI's LBM converged on relative SE(3) with 6D rotation **[X]**; GR00T N1.6 explicitly abandoned absolute joint angles for state-relative chunks **[X]**. Absolute also permanently bakes in the ~2.1 mm RMS / 3.2° hand-eye calibration bias, which deltas cancel |
| C6 | A *bare 6/7-vector* is a sufficient interface payload | **CONTRADICTED** | Nobody who succeeded passed a raw vector. GHOST projects a **GMM over poses** into each camera plane as a dense distance-field heatmap and found heatmaps beat single-pixel masks **[X]**; RT-Affordance **renders** the gripper outline onto the image and explicitly notes it would pass text tokens only when calibration is unavailable **[X]**; RoboDual passes latents **and** a discretized action **[X]**. HDP's published interface is (a_trans, a_rot, **a_grip**) — the gripper bit is not optional **[P]**. Missing from the proposal entirely: gripper aperture, duration/velocity, compliance, redundancy scalar, uncertainty |
| C7 | Model B (learned pose → joint commands) is a well-motivated module | **CONTRADICTED** | This is IK. Measured on this box: damped-least-squares 6×7 solve = **5.31 µs (188 kHz)** **[M]**. IKFast closed-form 5–50 µs; TRAC-IK >99.8% solve rate sub-ms vs ~96% for stock KDL **[P]**. Learned IK is *less* accurate: IKFlow mm-level translation, up to ~1.5° rotation **[P]**. No published hierarchical VLA asks a learned low level to do this: RT-1/RT-2/RT-A output Cartesian deltas, HiRT outputs EE deltas, TRI's LBM outputs relative SE(3), GR00T N1.6 abandoned joint angles. Additionally, IK seeded with q_current and selecting argmin‖q − q_prev‖ gives a **hard branch-continuity guarantee**; an L2-trained regressor over a bimodal (elbow-up/elbow-down) target converges to the arithmetic mean, whose FK is nowhere near the commanded pose **[I]** |
| C8 | Model B needs RGB | **PARTIALLY SUPPORTED — one legitimate justification of three** | Legitimate: closed-loop visual servoing to cancel Model A's systematic calibration bias — but then it must take an **image-space** goal (not a metric pose that already carries the bias), run at camera rate, and be benchmarked against classical uncalibrated IBVS with online Jacobian estimation, which already solves calibration-free servoing without learning **[P]**. Not legitimate: collision-aware redundancy (needs depth/geometry, not RGB; cuRobo/QP-IK do it with completeness guarantees) or contact adaptation (void if the arm accepts only joint positions). Cost is severe: two vision encoders on the compute-bound platform, and OpenHelix's probing found the latent in real dual systems is "largely insensitive to changes in visual information" **[X]** |
| C9 | The low level should have its own fresh perception | **SUPPORTED** (as a category) | OpenHelix uses exactly this as the definition of a dual system, and uses it to *exclude* π0 and GR00T N1 from the category **[X]**. The proposal is correct in kind; the objection in C8 is about cost and encoder size, not about the principle |
| C10 | Two models can be trained separately and composed | **CONTRADICTED as specified; SUPPORTED with fixes** | OpenHelix Table 6: **without projector pre-alignment, every configuration scored 0/0/0/0/0**; with it, 94–96% first-task success **[X]**. GHIL-Glue: fixing only the handoff (subgoal filtering + de-synchronized augmentation) = **+25% CALVIN, 54% → 70% real robot [X]**. GHOST oracle swap with the low level frozen: **36.7% → 90% [X]**. Helix backprops S1→S2 through the latent **[V]**; GR00T N1 is "tightly coupled and jointly optimized" **[X]**; FiS-VLA's entire contribution is *undoing* two-separate-models because separateness "limits System 1 from fully leveraging the rich pretrained knowledge" **[X]** |
| C11 | Running two models locally on a Mac mini is feasible | **PARTIALLY SUPPORTED** | Memory is fine (~2.5–7 GB of 32 GB) **[M/I]**. Compute is the binding constraint: **3.78 TFLOPS fp16 sustained, 104 GB/s achieved** **[M]** — roughly 1/43 of an RTX 4090 on compute, 1/10 on bandwidth. A 3B trunk prefill over 600 tokens measures **3.74 s (0.27 Hz) [M]** — π0.5/GR00T-class is simply not deployable here. MLX has a single Metal dispatch queue per process; concurrent models from separate threads crash (ml-explore/mlx #3078, closed unfixed) **[X]**. macOS: no PREEMPT_RT, no isolcpus, `kern.sched=edge` migrates threads across P/E cores **[M]**, no SocketCAN, no EtherCAT master, ROS 2 is REP-2000 **Tier 3** (source-only, no CI) **[P]** |
| C12 | The cerebrum/cerebellum naming is apt | **CONTRADICTED** | See §4. Ablate a real cerebellum → ataxia/dysmetria; ablate Model B → paralysis. The cerebellum has *no* direct motoneuron projection **[P]** |
| C13 | Model A (RGB + state → target pose) is a well-motivated module | **SUPPORTED** | This is RT-Affordance's affordance generator, GHOST's π_hi, and the PerAct/RVT keypose family's core. It is also where every oracle ablation says the remaining error lives (GHOST 36.7% deployed vs 90% oracle **[X]**; RT-Affordance 68% predicted vs 76% oracle **[X]**) |
| C14 | Local from-scratch training of both halves is acceptable | **CONTRADICTED** | Random-init vs pretrained VLM: **77.5% → 97.8% on LIBERO, ~20 pts [X]**; 15–25 pts typical across studies. That is *larger than every short-horizon hierarchy premium in the literature* (0–9 pts) and it is available to a monolith. Helix's headline is that internet pretraining let it work from ~500 h of teleop, "<5% of prior VLA datasets" **[V]** |

---

## 3. Prior-Art Map

### 3.1 Comparison table

| System | High level | Low level | Interface | Rates | Train coupling | Evidence quality |
|---|---|---|---|---|---|---|
| **Proposal** | RGB + EE pose → target SE(3) pose | target pose + joints + RGB → joint commands | **absolute SE(3) pose, raw vector** | unspecified | **separate** | — |
| **RT-1** [P] | none (monolithic) | — | — | 3 Hz, <100 ms budget | — | 130k demos, 3k trials |
| **RT-2** [P] | none (monolithic) | — | — | 1–3 Hz (55B), ~5 Hz (5B) | — | 6k real evals |
| **RT-H** [X] | π_h → language motion | π_l → 7-DoF action | **natural language phrase**, into encoder | ~RT-2 rate; async 1-step-ahead | **same model, 2 prompts** | Offline MSE 30.2→24.9 (oracle 17.9); novel objects 55→65; interventions 40→63 |
| **RT-Trajectory** [X] | human / VLM / hindsight | RT-1-class policy | **2D/2.5D sketch drawn on image** | ~3 Hz | separate (HL is not a net) | 63% vs 29% on 41 unseen; 2.5D 38% vs 2D 0% |
| **RT-Affordance** [X] ← *closest to proposal* | PaLM-E 1B, image+lang → **EE poses at keyframes** | PaLM-E 1B, Cartesian EE actions | **SE(3) poses, RENDERED onto the image** as colored gripper outline | not reported; replan "fixed or adaptive" | separate predictor, shared backbone family | 68% pred / 76% oracle vs RT-2 28%; predictor 77%→24% w/o 750 imgs, →11% w/o web |
| **HAMSTER** [X] | VLM finetuned on 1.2M off-domain | 3D-aware policy, 320 in-domain eps | **coarse 2D pixel path** + gripper flags | HL ~once/episode | separate | >2× OpenVLA; authors concede path "cannot communicate force or rotation" |
| **HDP** [P] ← *architecturally closest* | PerAct → next-best EE pose | RK-Diffuser → **joint** trajectory, FK-distilled | **(a_trans, a_rot, a_grip)** | keyframe-triggered | distillation via differentiable FK | 80.18% vs 57.72% planner, 15–18% flat; **pose-diffuser 24.55% IK error rate** |
| **GHOST** [X] | DINOv3 + transformer → **GMM over 3D EE poses** | Diffusion Policy, goal-conditioned | **3D pose projected to each camera as a dense heatmap** | 15 Hz, chunk 16 | separate | 80% vs 10% flat DP; **oracle HL 90% vs deployed 36.7%** |
| **PerAct / RVT / RVT-2** [P] | keypose prediction | **classical motion planner, 0 learned params** | SE(3) keypose + gripper | sub-Hz HL, servo-rate LL | n/a | RVT-2 ~81.4% RLBench from ~10 demos/task |
| **HiRT** [X] ← *best controlled test* | InstructBLIP-7B + LoRA | 35M (sim) / 150M (real), EE deltas | **single latent, MAP-pooled last layer**; FiLM + cross-attn + prefix | 9.8 Hz vs 4.1 Hz mono; VLM async, cached | joint | **Static: 70.0 split vs 71.3 mono. Dynamic: 75 vs 48** |
| **RoboDual** [X] | OpenVLA-7B, LoRA | ~20M DiT + 16M sensory encoders | **latents AND discretized action**, cross-attn | generalist 3.9 Hz, specialist 28.6 Hz, system 15 Hz | joint | +26.7% real over OpenVLA; strong at 5% of demos |
| **OpenHelix** [X] ← *only systematic study* | LLaVA variants | 3D Diffuser Actor | surveys all: `<ACT>` token, last-layer, mid-layer, MaxPool, action+lang latents | swept 1→60 steps/query | studied both | **HL freq sweep is flat (94/97/95/95/95/95/95); no pre-alignment → 0/0/0/0/0; pretrained LL 3.53 vs scratch 2.85; aux pos/rot head 3.45→4.01** |
| **π0** [X] | — (one transformer) | 300M action expert (separate *weights*, same forward pass) | **self-attention inside one transformer — no serialized interface** | 50 Hz via chunking, ~10 flow steps | one graph | no split-vs-mono ablation |
| **π0.5** [X] ← *most damaging ablation* | same weights, 1st query | same weights, flow expert | **natural-language subtask string**; proprio as text tokens | 50 Hz LL; HL "lower frequency" | one model | **"implicit HL" (no runtime HL, subtask data in training) is 2nd best; full π0.5 beats a human HL oracle; GPT-4 as HL is worse** |
| **GR00T N1 / N1.5 / N1.6** [X] | Eagle-2 VLM 1.34B | DiT ~0.86B | **layer-12 hidden states via cross-attention** | 10 Hz / 120 Hz claimed; 63.9 ms per 16-action chunk measured on L40 | "tightly coupled, jointly optimized" | **no split-vs-mono ablation published.** N1→N1.5 gains (43.3%→83.0% real) attributed to *freezing the VLM*, an adapter/LayerNorm fix, and FLARE — not the split. N1.6 doubled the *low-level* DiT |
| **Helix (Figure)** [V] | 7B VLM, 6 cameras | **80M** visuomotor policy | **single continuous latent**, projected into S1 token space | **7–9 Hz / 200 Hz, DEDICATED GPUs** | **gradients backprop S1→S2 through the latent** | **no quantitative results published at all** |
| **Gemini Robotics 1.5** [X] | GR-ER 1.5 orchestrator | GR 1.5 multi-embodiment VLA | **open-vocabulary natural language** (the VLA is a "tool") | not given for control; ER at 5 Hz for success detection | separate models, agentic | Fig. 4 attributes gains to Motion Transfer + multi-embodiment data, **not** the hierarchy. >90% of dev evals in sim |
| **FiS-VLA** [X] | VLM | **last 2 blocks of the same VLM** | **shared parameters — no serialized interface** | 117.7 Hz, best async ratio 1:4 | co-trained; removing "slow loss" 69%→62% | Motivation stated as: separate models block S1 from VLM pretraining |
| **SmolVLA** [X] ← *key counterfactual* | none (450M monolith) | — | action queue only | 30 fps env cycle; async | — | **87.3% LIBERO vs OpenVLA-7B 76.5%; async = equal success, ~30% faster, 2× throughput** |
| **OpenVLA-OFT** [X] | none (7B monolith) | — | — | **26× faster generation, 3× lower latency** than OpenVLA | — | **97.1% LIBERO, beating π0, Octo, DP, MDT** |

### 3.2 How every real system's interface differs from the proposal

The proposal passes a **raw absolute SE(3) vector between two independently-trained networks, with a joint-space learned low level.** Sorting the literature by interface:

- **Learned continuous latent** — Helix, HiRT, GR00T N1 (layer-12), RoboDual, LCB (`<ACT>` token), DP-VLA. *This is what every large-scale production dual system uses.* All are jointly trained with gradients crossing the boundary.
- **Natural language** — RT-H ("move arm forward"), π0.5 subtask string, Gemini Robotics 1.5 orchestrator→VLA. Cheapest and most debuggable; used for *long-horizon decomposition*, never for closed-loop motion.
- **Image-space spatial goal** — RT-Trajectory (2.5D sketch), HAMSTER (2D path), GHOST (pose→heatmap), RT-Affordance (pose→rendered outline). *Every system that uses poses renders them into the image rather than passing numbers.*
- **Shared parameters / no serialized interface** — π0, π0.5, FiS-VLA.
- **Hybrid** — RoboDual passes latents *and* a coarse discretized action, and its ablation says both carry distinct information.

**Three specific divergences, in order of importance:**

1. **Nobody passes a bare numeric pose vector.** The two systems whose interface *is* a pose (RT-Affordance, GHOST) both re-express it in image space so the low level's visual encoder can attend to it spatially and in the same coordinate frame as its observations. GHOST additionally passes a *distribution* (GMM), not a point estimate. RT-Affordance notes passing tokenized text values is the fallback "when camera calibration is unavailable."
2. **Nobody has a learned low level that outputs joint commands from a Cartesian target.** RT-Affordance's LL is Cartesian; HiRT outputs EE deltas; TRI's LBM outputs relative SE(3) with 6D rotation; GR00T N1.6 explicitly moved *away* from absolute joint angles. HDP does output joints — but only by distilling a pose diffuser through differentiable forward kinematics, and its ablation shows the pose-only variant had a **24.55% IK error rate**, which is the honest headroom estimate for a learned pose→joint map. π0.5 commands joint targets, tracked by "simple PD controllers... without any additional trajectory planning or collision detection."
3. **Nobody trains the two halves independently and bolts them together.** Three working recipes exist: end-to-end backprop through a latent (Helix), freeze HL + pre-align a projector + prompt-tune (OpenHelix), or freeze the VLM entirely and train only the adapter + action model (GR00T N1.5). Skipping alignment measured **0/0/0/0/0**.

**Capacity ratios, for calibration:** Helix 7B/80M (**87:1**), RoboDual 7B/20M trainable (**350:1**), π0 3B/300M (**10:1**), GR00T N1 1.34B/0.86B (~1.6:1, and it is one graph). **Nobody splits 50/50.** When NVIDIA wanted more performance in N1.6, they *doubled the low-level DiT* and *froze more of the high level*.

---

## 4. The Neuroscience Analogy: Where It Helps, Where It Misleads

### 4.1 Where it genuinely helps

- **Multi-rate hierarchy is real and is the analogy's best contribution.** Biology runs three tiers, not two: monosynaptic stretch reflex ~20–40 ms; transcortical long-latency reflex ~50–100 ms; visually-guided voluntary correction ~150–250 ms **[P, order-of-magnitude]**. This *is* the correct argument for a hierarchy, and it maps to: fixed servo/impedance at 500–2000 Hz, learned reactive policy at 20–50 Hz, semantic policy at 0.5–10 Hz.
- **Model A ≈ parietal AIP → premotor F5 is a fair mapping.** Extracting grasp-relevant object affordances from vision and selecting a hand configuration is precisely what that circuit does **[P]**.
- **Capacity allocation intuition is inverted, usefully.** Human cerebellum ≈69B neurons vs cerebral cortex ≈16B (Herculano-Houzel **[P]**) — 4× more units in the "low level," but organized as one enormous *wide, shallow, sparse* expansion layer with a single learned readout, hence cheap per inference. If taken seriously this argues for a **wide, shallow, cheap** low level, which is exactly the right shape for a high-rate local policy — and the opposite of a second image-consuming transformer. *(Caveat: neuron count is a weak proxy for parameter count; granule cells are near-random expansion features. Treat as a heuristic **[I]**.)*
- **The low level should get abstracted, not raw, perception.** Cerebellar input arrives via corticopontine → pontine nuclei → mossy fibers — already processed by cortex. There is no retina→cerebellum projection. *(Slight over-simplification: visual info does reach the cerebellum via pontine relays and retinal slip reaches the inferior olive for VOR/OKR. The accurate statement is that cerebellar visual input is low-dimensional and motion/error-related, not object-recognition-grade **[P]**.)* This argues for **sharing Model A's visual features** rather than duplicating the encoder.

### 4.2 Where it actively misleads

- **Model B is not a cerebellum; it is the corticospinal tract plus spinal cord.** The cerebellum has **no direct output to motoneurons.** Its only outputs are deep cerebellar nuclei → (a) thalamus → cortex and (b) red nucleus / reticular formation → spinal interneurons **[P, textbook]**. It biases commands generated elsewhere. Model B *generates* the commands.
- **The ablation test settles it.** Remove a cerebellum → ataxia, dysmetria, intention tremor (~3–5 Hz, the signature of a delayed feedback loop with the predictive element removed), hypotonia. Movement remains; strength is preserved **[P]**. Remove Model B → the robot is inert. A module whose ablation produces paralysis rather than incoordination is the command path.
- **The single most robustly attributed cerebellar function is entirely absent: prediction.** The cerebellum is a **forward model** — efference copy + state → predicted future state / predicted sensory consequences, compensating ~100–200 ms of sensory delay. TMS over lateral cerebellum makes reaches be planned from a hand position **~138 ms out of date** (Miall et al., PLoS Biol 2007 **[P]**); healthy subjects show ~60 ms *predictive lead* vs ~172 ms *lag* in cerebellar ataxia **[P]**. Model B has no efference copy, no prediction, no prediction-error signal. The one thing that most makes something a cerebellum is what was left out.
- **Even the most controller-like theory makes it additive, never exclusive.** Kawato's feedback-error-learning: the cerebellar inverse model is placed *on top of* an existing crude feedback controller, its output **summed with** the feedback controller's, and the feedback controller's own output serves as the climbing-fiber teaching signal **[P]**. The baseline never disappears.
- **The topology is wrong.** Cerebro-cerebellar circuits are **closed loops**: the cerebellar regions receiving input from a given cortical area are the same ones projecting back to it, replicated for M1, premotor, oculomotor, prefrontal, and parietal (Kelly & Strick, J. Neurosci. 2003 **[P]**). Kelly & Strick frame this explicitly as contrasting with "the traditional view of the cerebellum as merely executing commands from higher brain centers" — which is exactly the proposal's strictly-A→B topology. There are also *many parallel loops*, not two boxes in series.
- **Even the "high level" is a feedback controller.** The transcortical long-latency reflex loop runs through M1 (Scott, Nat. Rev. Neurosci. 2004 **[P]**, influential theory rather than settled fact). A high level that emits a pose and then goes silent is not what cortex does.
- **A third functional block is missing entirely: basal ganglia.** Action selection by disinhibition, and *when to switch or abort* **[P]**. In a robot this is the subtask/termination/switching policy — which the proposal does not specify at all, and which the systematic orchestration study found to be one of the decisive design variables (naive hierarchy 40.6% vs best 67.1% on long-horizon **[X]**).

### 4.3 Honest caveats on the neuroscience

The forward-model vs inverse-model question for the cerebellum is explicitly flagged as **unresolved** in the current literature, and Ivry's *timing hypothesis* is a live competing framework **[P, contested]**. Marr–Albus LTD-as-memory is also no longer consensus — mice with impaired parallel-fiber LTD still learn motor tasks (Schonewille et al., Neuron 2011 **[P]**), and the field has moved to "plasticity at multiple sites." **None of this rescues the naming:** every competing theory makes the cerebellum about prediction and/or timing, and none makes it the sole generator of joint commands from images.

**Practical recommendation:** either rename Model B ("low-level policy" / "motor controller"), or add the function that defines the name — a forward-model head predicting next joint state and next-step visual features from its own efference copy. That head is free supervision from the same demonstrations, it regularizes the representation, it lets you roll predictions forward to compensate inference latency (the actual cerebellar function), and its prediction error becomes a natural anomaly trigger for re-querying Model A.

---

## 5. Top Objections, Deduplicated and Ranked

### O1 — FATAL: Model B is inverse kinematics, learned worse

**Objection.** "Target EE pose + current joints → joint commands" is the IK signature plus a servo. Measured on this box: DLS 6×7 solve = **5.31 µs (188 kHz) [M]**. IKFast closed-form 5–50 µs; TRAC-IK >99.8% sub-ms vs KDL ~96% **[P]**. A learned Model B is 3 orders of magnitude slower and mm-to-cm accurate where analytic is sub-micron. Worse, IK **returns a typed failure** ("no solution", "at joint limit", "near singularity") you can branch on; a regressor always returns a confident, plausible, wrong number with no runtime signal. Compounding: IK is multi-modal (up to 8 branches on 6-DoF, a 1-D null-space manifold on 7-DoF), and an L2-trained regressor over a bimodal target converges to the **arithmetic mean of two valid branches**, whose forward kinematics is nowhere near the target; a stochastic head instead flips branch between consecutive timesteps. Analytic IK seeded with q_current and taking argmin‖q − q_prev‖ gives a **hard continuity guarantee** that no imitation objective provides. This failure is invisible in offline MSE and gets *worse* with more data.

**Concrete failure.** "Pick the mug on the far left, place it far right." Crossing the workspace midline forces a configuration flip; demos contain both branches at nearly identical poses. Model B either emits a mid-configuration 15 cm off with the gripper sideways, or slews the wrist ~180° in one 33 ms step.

**Mitigation.** Do not learn the pose→joint map. Build `Model A → TRAC-IK (joint-limit + singularity avoidance, null-space posture bias) → impedance/PD at 500–1000 Hz` first — a weekend of integration, zero training data, and the configuration OpenVLA/π0/RT-2 deployments actually use. Only reintroduce a learned Model B as a **clipped additive residual**: `q_cmd = IK(target, seed=q_prev) + clip(f_θ(·))`. If learning must touch kinematics, restrict it to *selecting among exact solutions* (which branch / what swivel angle) — the kinematics stays exact, the network only expresses preference.

---

### O2 — FATAL: no safety envelope, and Model B is the sole command generator

**Objection.** No joint/velocity/acceleration/jerk clamp, no torque threshold, no workspace bound, no reachability check, no NaN assert, no watchdog, and — critically — **no baseline controller to degrade to**. Model A's failure rate in the published oracle ablations is the *normal operating regime*, not the tail: RT-Affordance's predictor drops to 24% without 750 cheap images and 11% without web co-training **[X]**; GHOST measured its deployed high level at 36.7% against a 90% oracle **[X]**. Roughly a third to half of high-level outputs being wrong is baseline.

**Concrete failure.** On a novel scene Model A regresses a target 40 cm below the table. Model B extrapolates and the arm drives into the surface at commanded speed. Nothing can interrupt it. Secondary version on 7-DoF: the elbow swings through the null space into whatever is beside the workspace, because a 6-DoF pose does not determine the arm.

**Mitigation.** Build a **fixed, non-learned reflex layer below both models, in code not weights, before either model exists**: joint pos/vel/acc/jerk clamps; torque or motor-current threshold with auto-retract; workspace AABB; self-collision + **reachability filter applied to Model A's pose before Model B ever sees it** (returns a typed rejection that re-triggers A); finite-and-in-range asserts at three points with reject-and-hold; step-magnitude limiter; deadman watchdog that holds last-good then ramps velocity to zero. Then restructure Model B as a residual so residual→0 recovers a working robot — matching the biological phenotype (dysmetria, not paralysis).

---

### O3 — FATAL on this hardware: the split inverts the rate story

**Objection.** Measured **[M]**: ViT-B/16 (the minimum Model B needs for RGB) = 62.75 ms p50 solo → **214.29 ms p50 / 288.91 ms p99** under contention from a concurrent 805M cerebrum-scale model, i.e. **4.7 Hz**. The cerebrum itself degrades 467 → 500 ms p50 / 628 ms p99. Total GPU throughput is conserved; two models cost the **sum**. Helix gets its 25× decoupling by using **dedicated GPUs [V]**; Gemini Robotics uses the **cloud [X]**. Additionally: a camera-fed Model B is hard-capped at camera frame rate (30–60 Hz) regardless of accelerator speed, so "200 Hz reflex layer on RGB" cannot exist. And MLX has a single Metal dispatch queue per process, with concurrent-thread crashes (#3078, closed unfixed) **[X]**.

**Concrete failure.** Budgeted A at 3 Hz and B at 100 Hz; measured A at ~2 Hz and B at 4.7 Hz p50 / 3.5 Hz p99. Photon-to-joint-command lag ≈ 700–850 ms including staleness. An object slipping at 0.2 m/s is 15 cm gone before the correction lands.

**Mitigation, in order.** (a) **Delete RGB from Model B** — feed it the interface + joints + velocities + wrench; a proprio-scale net measures **2.18 ms p50 / 2.80 ms p99 [M]**, and a 3.2M proprio net on CPU with 2 threads measures **1.17 ms p50 / 1.78 ms p99 (858 Hz) while the GPU was saturated [M]**. (b) **Put Model B on the CPU, not MPS** — the same models on MPS measured 23.6–43 ms p50 with 106–129 ms p99, i.e. 4× slower with a 16× worse tail, *and* they steal throughput from A **[M]**. This one-line change (`device='cpu'`, `set_num_threads(2)`) converts the split from fiction into a real 25–140× rate ratio. (c) **Before building anything, buy the rate the cheap way**: action chunking + Real-Time Chunking + async execution on one model gave equal success, ~30% faster completion, 2× throughput **[X]** for ~50 lines. Chunk length is nearly free (5→250 actions = +11% latency) while denoising steps are expensive (10→50 = 5×), and an autoregressive token action head carries a ~102× penalty **[X]**. (d) Never run the ≥500 Hz servo loop on macOS — push it to the arm's controller or a $60–150 Linux SBC over wired Ethernet (~0–4 ms, effectively free **[X]**). This single move moots every macOS real-time objection.

---

### O4 — MAJOR: the interface is not an action

**Objection.** A bare SE(3) pose carries no gripper command, no duration/velocity, no compliance, no redundancy scalar, no uncertainty. Consequences: (i) approach-with-gripper-open and close-gripper occur at nearly identical poses with nearly identical wrist images, so Model B's learning problem is *formally ill-posed* — it will emit a half-closed gripper or silently re-infer task phase from RGB, i.e. re-do Model A's job with a smaller model; (ii) a waypoint with no time is not a trajectory, and approach speed *is* impact force; (iii) on 7-DoF the pose does not determine the arm — measured 96% (joint) / 94% (ERJ = pose + one redundancy scalar) / **8% (pure task space)**, with 0% on "take cup out of cabinet" **[X]**; (iv) no compliance channel means no contact task — force-aware VLAs measure **+23.2%** (ForceVLA over π0) to **+38.0 pts** (FAVLA over vision-only π0) **[X]**; (v) a point estimate mode-averages a bimodal affordance (frying pan: handle or rim) into a pose in mid-air, invisible in offline MSE. HAMSTER's authors concede their own interface "cannot communicate nuances such as force or rotation" **[X]**.

**Mitigation.** Widen to the minimum executable unit: **chunk of 8–16 waypoints × (Δp(3), 6D-rotation(6), gripper aperture(1), dt(1))**, plus a header of stiffness (6 or a 3-level flag), swivel angle ψ (1), confidence/entropy (1), phase (1), and optionally a 60-dim latent side-channel (the RoboDual pattern). Deliver it to Model B by **rendering it into the RGB Model B already consumes** as a heatmap/overlay (GHOST, RT-Affordance) rather than concatenating a vector. Emit a **mixture**, not a point.

---

### O5 — MAJOR: absolute pose is the worst-measured abstraction and bakes in calibration bias

**Objection.** EE-absolute **69.0%** vs joint-absolute 77.3% vs joint-delta 88.0% vs **EE-delta 89.6%** **[X]**; chunk-wise beats step-wise by ~10 pts. Absolute also inherits the entire calibration chain as a systematic, unobservable offset: ~2.1 mm RMS translation / ~3.2° rotation for hand-eye, plus robot positioning error, plus posture-dependent compliant deflection **[P]**. Model B cannot see it because the same bias was baked into both label and observation during training. Deltas anchored on the *measured* EE pose cancel most of it because both endpoints share it.

**Concrete failure.** Pressing a 12 mm button: ~4–5 mm posture-dependent offset means it works at the near edge of the workspace and reliably misses at the far edge. Presents as a data problem; is a frame problem.

**Mitigation.** Chunk-wise SE(3) **delta** relative to the pose at chunk start, with **6D continuous rotation** (never Euler, never raw quaternion — the wrap-around discontinuity is a free source of regression error). This is the single highest-value, lowest-cost edit to the whole proposal. *Caveat: the 69.0/77.3/88.0/89.6 table is from a 2026 preprint I could not verify — re-check before quoting **[X]**.*

---

### O6 — MAJOR: cascaded distribution shift at a non-differentiable seam

**Objection.** Model B trained on hindsight ground-truth poses and deployed on Model A's error distribution is textbook covariate shift, with no gradient, reward, or signal by which either side can correct. During training every (pose, RGB) pair is mutually consistent; at deployment the pose is 10–20 mm off the object in the image, and B has *never seen an inconsistent pair*, so its learned prior is "the target is correct, servo to it." Measured costs: GHIL-Glue fixed only the handoff for **+25% CALVIN, 54% → 70% real [X]**; GHOST's oracle swap with the LL frozen was **36.7% → 90% [X]**; OpenHelix without projector pre-alignment scored **0/0/0/0/0 across every configuration [X]**. The HRL literature names the same thing formally (HIRO's off-policy correction exists precisely for this).

**Mitigation.** (a) Noise-augment the interface during B's training using A's **measured** held-out error covariance, not a guessed Gaussian. (b) Mandatory **DAgger round**: after A converges, roll it out, log real interface messages *including failures*, relabel, retrain B on that distribution. (c) A GHIL-Glue-style progress/feasibility filter. (d) Train B on **stale** targets from day one (HiRT caches the latest latent and trains the LL against it); measured p50→p99 spread on A is ~128 ms **[M]**, so staleness is *jittering*, not fixed. (e) Best available: make the seam differentiable — pass a continuous latent alongside the pose and backprop through it, or freeze A and pre-align a projector.

---

### O7 — MAJOR: the interface has no timestamp, frame stamp, or staleness policy

**Objection.** A pose "relative to the head/camera frame" is meaningless without the head extrinsic *at image-capture time*. At a measured 500 ms p50 / 628 ms p99 A-period **[M]**, a head panning 20°/s produces ~6° ≈ **63 mm** of lateral target error at 0.6 m, and the target silently slides with no error raised anywhere. Camera-frame actions *do* measurably win when extrinsics are known and static (+8.0 sim-continuous, +13.8 sim-discrete, +10.0 real fixed-camera, and roughly half the degradation under novel viewpoints: 14% vs 29% **[X]**) — but that result never studies calibration-error sensitivity and presumes a static T.

**Concrete failure.** Grasps fail *intermittently* — only when the head happened to be moving. Worst possible debugging signature; will be misdiagnosed as perception for weeks.

**Mitigation.** Make the interface a **typed, timestamped, versioned message**: capture_stamp, head/base joint state **at capture**, frame_id, seq, validity_horizon, and a hash of (A-version, B-version, calibration-version) that B refuses on mismatch. Model A predicts in camera frame (keeping the viewpoint-invariance benefit) and **its own output stage** converts to base_link using the capture-time head state. Model B only ever sees base_link. Beyond the horizon: hold and decay velocity to zero, **never extrapolate**.

---

### O8 — MAJOR: capacity is being spent on the half that is already saturated

**Objection.** Every oracle ablation says the residual error is upstream: GHOST **36.7% → 90%** from swapping only the high level with the LL frozen **[X]**; RT-Affordance 68% predicted vs 76% oracle **[X]**; the entire PerAct/RVT/RVT-2 family achieves precise manipulation from ~10 demos/task with a **zero-parameter** low level **[P]**. Meanwhile every working split is violently lopsided (87:1, 350:1, 10:1) with an internet-pretrained top, and a 50/50 from-scratch local split additionally forfeits the ~20-point pretraining lever (LIBERO random-init 77.5% vs pretrained 97.8% **[X]**) — larger than every short-horizon hierarchy premium in the literature.

**Mitigation.** Make the split 25:1 or more, freeze an internet-pretrained VLM as Model A, and inject the cheap non-teleop data that is the split's *actual* mechanism: ~750–1500 hand-annotated stills (~1 h to collect, ~2 h to label; RT-Affordance's predictor fell 77% → 24% without them) plus web co-training (→11% without) **[X]**.

---

### O9 — MODERATE: no switching, termination, or retry policy

**Objection.** A pose says "go here"; it cannot say "you already tried this and failed." After a failed grasp the scene is visually unchanged, so A emits the same pose and B — a deterministic function — emits the same command. **Livelock.** π0.5 and Hi Robot both build explicit failure detection and replan; the systematic orchestration study found switching/termination to be a decisive variable, with ablated hierarchies collapsing from ~95% to near zero **[X]**. This is the basal-ganglia-shaped hole in §4.

**Mitigation.** Specify triggers explicitly (OpenHelix says the *rate* barely matters — 94/97/95/95/95/95/95 from every-step to once-per-episode **[X]** — so spend the design effort on triggers, not frequency): chunk 60% consumed; gripper state transition; tracking error > τ; forward-model prediction error saturating; IK-infeasible rejection; operator interrupt. Add an `attempt=k` field to the interface and an upward status channel from B so A can sample a different mode on attempt two.

---

### O10 — MODERATE: no failure-attribution mechanism, and the effect size is below the noise floor

**Objection.** Two separately-trained models + a non-differentiable interface + no oracle harness means every failure has four indistinguishable explanations (A's pose bias, B's execution, staleness, calibration drift). Compounding: the effect being hunted is tiny where the tasks are — HiRT's split-vs-mono gap on real quasi-static tasks was **1.3 points, in the monolith's favor [X]**; naive hierarchy vs flat VLA on short-horizon was an exact tie **[X]**. TRI needed **50 rollouts per task per policy per condition, 1,800 total trials, blind randomized A/B, Bayesian posteriors, Bonferroni correction** to separate policies at all **[X]**.

**Mitigation.** Build the harness **before** the models: **Oracle-A** (replay hindsight ground-truth poses into B — upper bound on B) and **Oracle-B** (swap B for IK + impedance, feed A's live poses — upper bound on A). Log the interface with full timestamps and replay traces offline. Evaluate on **dynamic** and **long-horizon** tasks, the only regimes where any published hierarchy advantage reliably appears. Budget ≥50 rollouts per condition or state up front that the experiment cannot conclude.

---

### O11 — MODERATE: no finiteness/liveness contract, on a backend where deadline misses are the norm

**Objection.** fp16 overflow in attention or normalization is routine; a NaN pose from A propagates into B, whose input LayerNorm converts it to a *finite but garbage* value — worse than a NaN, because a naive check won't trip. Measured p99s (628 ms cerebrum, 289 ms ViT contended **[M]**; even a 12M proprio MLP measured 0.80 ms p50 but 38.7 ms max, a ~48× tail **[M]**) mean missed deadlines are normal operating behavior, so "what happens when the message is late" must be *designed*, not discovered.

**Mitigation.** ~50 lines: finite-and-in-range asserts at A's output, B's output, and the servo input, with reject-and-hold rather than pass-through; step-magnitude limiter; deadman timer ramping velocity to zero after K missed deadlines; log every rejection with the input that caused it.

---

### O12 — MODERATE: the data cost is sequential and re-entrant, not 2×

**Objection.** B's dataset cannot be collected until A exists (see O6), forcing `train A → roll out A → relabel → train B`. Every A retrain — more data, changed frame convention, hand-eye recalibration — silently invalidates B's fit, with no shared CI to catch it.

**Concrete failure.** Six weeks in, you improve A and switch its output frame. A's standalone metrics improve; end-to-end success on insertion drops 55% → 30%. Both models look individually correct in the logs, because the interface is human-readable poses.

**Mitigation.** Pin `(A-version, B-version, calibration-version)` as **one deployable artifact** that cannot be mixed, enforced by a hash in the message header. Add an offline CI test that replays a fixed logged interface trace through B and asserts command bounds and smoothness.

---

### O13 — MODERATE: the compliance rationale may be void on your hardware

**Objection.** If the arm accepts only joint position commands (SO-100/SO-101, Koch, most hobby arms, many industrial position-mode interfaces), then a learned Model B has *exactly* the expressive power of IK + a position servo on contact tasks, the stiffness channel is a decorative field, and every contact-rich justification evaporates. The contact literature is unambiguous: "when the clearance between peg and hole is small, relying solely on velocity or position controllers is insufficient, as the tolerance and the robot's precision are at the same level (sub-millimeter)" **[P]**.

**Mitigation.** Determine the arm's command interface **first, in 30 minutes**. If position-only: scope target tasks to free-space pick/place/reach **in writing**, and delete the compliance half of the design rather than carrying it as an unstated assumption.

---

## 6. Three Alternative Architectures, Head to Head

### 6.1 The candidates

**ΔChunk — "keep the split, fix everything else."** Model A = SmolVLA-450M (frozen SmolVLM backbone + LoRA), emitting a 256-float stamped message: 16 waypoints × (Δp, 6D-rot, gripper, dt) + stiffness + swivel + entropy + phase + a 60-dim latent side-channel, at 2–4 Hz event-triggered on the GPU. Model B = 3.2M (v0) → 18M (v1) **clipped residual** on top of TRAC-IK + impedance, **on the CPU**. Non-learned reflex layer below both, servo off-Mac.

**CIR-1 — "delete Model B."** One 450M pretrained VLA emitting the same chunk-wise delta interface, handed to TRAC-IK + Cartesian impedance at 1 kHz on the robot's own controller, plus exactly one learned thing at the bottom: a **0.18M-parameter, camera-free, hard-clipped force residual** (2-layer GRU over joints/velocities/wrench/pose-error/stiffness/phase → 6-dim wrench correction, ±8 N clip) summed onto the impedance controller. Target-pose and subtask prediction survive as **auxiliary training losses**.

**TRISYS-500 — "one model, two weight sets, three clock domains."** π0.5/GR00T shape. System 2 = **frozen** SmolVLM2-500M (first 16 of 32 decoder layers), whose **layer-16 hidden states** (176×720, via a LayerNorm-Linear-GELU-Linear-LayerNorm adapter) are the interface — not a pose. System 1 = ~100M flow-matching DiT in the **same forward graph**, cross-attending to the cached prefix, with its own 128×128 wrist crop through a 3M conv stem, emitting 50-step chunk-wise SE(3)-delta chunks at 4 flow steps. Analytic spinal layer below. Target-pose and subtask heads kept as auxiliary losses off the prefix.

### 6.2 Head-to-head

| Dimension | ΔChunk | CIR-1 | TRISYS-500 |
|---|---|---|---|
| Preserves the user's stated architecture | **Yes** (two models, pose interface) | No (one model + classical spine) | Partly (two weight sets, one graph; pose demoted to aux loss) |
| Interface | pose chunk + 60-d latent (hybrid) | pose chunk (pure) | **learned latent** (176×720) |
| Interface inspectable / human-correctable | **Yes** | **Yes** | Only via aux heads decoded for logging; **not correctable** |
| Learned params below the interface | 3–18M residual | **0.18M** residual | ~100M action expert |
| Total learned params | ~468M | ~450M | ~370M executed / 600M checkpoint |
| Split ratio | 25–140:1 | **2500:1** | ~2.7:1 (but one graph) |
| GPU contention on this box | **None** (A on GPU, B on CPU) **[M]** | **None** (one model) | **None** (one process, one graph) |
| Measured fast-loop latency | 6.29 ms p50 / 7.92 p99 (18M + crop, CPU, GPU saturated) **[M]** | **0.05 ms p50 / 0.06 p99** (1 CPU thread) **[M]** | ~60–72 ms per 50-action chunk (14–17 Hz generation) **[I from M roofline]** |
| Effective command rate | 100–160 Hz setpoints | 200 Hz retimed, 1 kHz force | 30–50 Hz smooth via chunking + RTC |
| Cascaded-shift exposure | Moderate (mitigated by DAgger) | **Near-zero** (classical block is correct for every reachable target by construction) | **Zero** (one graph, joint training) |
| Joint-training-collapse risk (0/0/0/0/0) | Moderate — needs Stage-4 partial joint finetune | **None** | Mitigated by mandatory adapter pre-alignment stage |
| Can be trained on the Mac | Stages 1–2 need a rented GPU | A needs ~1 GPU-day rented; residual trains in **20 min on the Mac CPU** | **No** — needs 24–48 h on A100/4090 |
| Time to a working robot | ~4–6 weeks | **~2 weeks** (residual off is already a shippable system) | ~4–5 weeks |
| Time to first honest number | ~5–6 weeks | **~2–3 weeks** | ~5 weeks |
| Capability ceiling | SmolVLA-class | SmolVLA-class | SmolVLA-class (3B is 3.74 s prefill on this box **[M]** — undeployable) |
| Contact-rich capability | Good if hardware allows | Good if hardware allows; residual is *specifically* the force-tracking corrector | Good; stiffness channel in the chunk |
| Cross-embodiment / arm swap | Good (Cartesian interface) | Good | Weaker (latent is embodiment-entangled without embodiment-specific encoders) |
| Forecloses end-to-end gradients | Partly (Stage 4 optional) | **Yes, by design** | **No** — this is the path that keeps them |
| Degrades gracefully | Yes (residual→0) | **Yes** (residual→0 recovers full classical stack) | Partly (S1 fault is still the command path, mitigated by the spinal layer) |

### 6.3 RECOMMENDATION

**Build CIR-1 first. Keep TRISYS-500 as the explicit upgrade target. Do not build ΔChunk unless human-in-the-loop pose correction is your primary product goal.**

**Why CIR-1 first — four reasons, in order:**

1. **It is the mandatory baseline anyway.** Every one of the three designs needs the same spine — something has to convert a Cartesian target into joint motion and keep the arm safe. CIR-1 *is* that spine plus one pretrained VLA. Building it is not a detour; it is the first two weeks of any of the three plans, with the difference that CIR-1 stops there and measures.
2. **It is the honest control the evaluation plan demands.** The decision rule "if `Model A + classical IK` comes within 5 points of the split, delete Model B" cannot be evaluated without building `Model A + classical IK`. Building it first converts a hypothetical into a measurement, and — given the oracle ablations (GHOST 36.7% vs 90%; RT-Affordance 68% vs 76%; PerAct/RVT at ~10 demos/task with a zero-parameter low level) — the honest prior is that it wins or ties.
3. **It buys the most success-per-engineering-hour by a wide margin.** ~2 weeks to a working robot vs ~4–6. One dataset, one training run, one model, no interface-alignment problem, no GPU contention problem, no second sequentially-blocked dataset, no re-entrant DAgger maintenance tax.
4. **Its one learned component has a job that fits in a sentence and a matching classical baseline.** The 0.18M residual cancels the tracking error the impedance controller fails to close, using force and proprioception. It is not doing IK, not doing planning, not doing visual servoing. Its label is the feedback controller's own output — Kawato's climbing-fiber analogue, zero human labeling. It measures **0.05 ms p50 [M]**, so it runs inside the 1 kHz loop with 20× headroom. Ablate it and you get sloppy tracking, not paralysis.

**Why TRISYS-500 is the upgrade, not the starting point.** TRISYS is architecturally the *most defensible* design in the list — it matches what every production dual system converged on (frozen pretrained VLM, mid-layer hidden states via cross-attention, one forward graph, action expert, aux subtask/pose losses, chunk-wise deltas, analytic spine below). It also structurally eliminates O3, O6, O10-partial, and O12 at once. But it costs a rented GPU and 24–48 h of joint training, it gives up interface inspectability (the one advantage a pose has over a latent, and the one that produced RT-H's largest single number: interventions 40% → 63% **[X]**), and — decisively — **it does not obviously beat CIR-1 on this hardware**, because both are capped at SmolVLA-class capability by a 3.78 TFLOPS accelerator. Its own strongest control ("same model, aux pose head, no runtime hierarchy") may tie it, which is exactly what π0.5's ablation found.

**Why ΔChunk is dominated.** It is the most faithful to the original idea and it fixes every specific defect correctly. But it spends 4–6 weeks of engineering — two processes, a versioned schema, a shared-memory transport, a DAgger round, a partial joint finetune, two attribution harnesses — to preserve a hypothesis (C1) that the evidence contradicts, and its 3–18M residual sits on the half of the system that every oracle ablation says is already saturated. The honest expected v1 result is "the residual did nothing measurable; the classical core did the work." **The exception:** if your deliverable is a *debuggable, human-correctable* robot — a pose you can render in RViz, drag with a 6-DoF mouse, unit-test, and fold corrections back into training — then ΔChunk's inspectable interface is the product, and it is the right choice regardless of success rate. That is a legitimate goal and it should be stated explicitly if it is yours.

**The upgrade path is a config change, not a rewrite.** CIR-1 and TRISYS-500 share the entire spine, the safety layer, the interface schema below the model, and the evaluation harness. Moving from CIR-1 to TRISYS is: swap the single VLA for a frozen-VLM + action-expert pair, and let the aux pose head keep doing what it was already doing. That is the single strongest argument for the ordering.

---

## 7. Empirical Plan (Condensed)

**Separate four hypotheses before spending a day of compute.** Three are cheap; one is what you actually claimed.

| | Hypothesis | Prior | Cost to test |
|---|---|---|---|
| H1 | At fixed params/data/init/rate, two models beat one | **Negative** | Days (E0) |
| H2 | The split raises achievable command rate | **Confounded** — SmolVLA got it from async chunking on one model | 1 week (E3), measured |
| H3 | The split lets the high level eat cheap non-teleop data | **Strongly positive** | 2 weeks (E2c) |
| H4 | A pose interface is inspectable and human-correctable | **Positive, underrated** | ~free |

### Pre-flight (hours)
- **P1 (30 min):** What does the arm's command interface accept? Position-only → O13 fires; scope to free-space, in writing.
- **P2 (2 h):** Two-process concurrency measurement. Reference on M4-base: ViT-B/16 **62.75 ms solo → 214/289 ms p50/p99 contended [M]**. If yours matches, **H2 is dead before you write model code**.
- **P3 (5 min):** Camera frame rate = Model B's visual reactivity ceiling. Write it down.
- **P4 (half day):** Measure hand-eye extrinsic error (AprilTag grid, AX=XB). Any task with tolerance below ~2× that number cannot use an absolute pose interface.

### E0 — the cheapest possible falsification (3–5 days, laptop, zero new teleop)

Train a low-level policy on the **oracle interface** (hindsight ground-truth future EE pose, a numpy slice of the LIBERO state vector). This is the **ceiling of the entire architecture** — no Model A you ever build can beat ground truth.

Venue: LIBERO-Spatial + LIBERO-Object (20 tasks, 1000 demos shipped, runs on macOS CPU, robosuite lets you flip `OSC_POSE` ↔ `JOINT_POSITION` by config). Three arms at ~25M params, 3 seeds, 500 episodes each (MDE ≈ 8.9 pts at n=500, ≈ 5.1 at n=1500):

- **C0 MONO** — RGB + proprio → action chunk
- **C6 SPLIT-ORACLE** — GT target pose + joints + RGB → action chunk
- **C8 ORACLE-IK** — GT target pose → `OSC_POSE`, **zero learned params**

**Kill rules:**
- `C6 − C0 ≤ +5` or CI∋0 → **KILL H1.** Oracle information is privileged; GHOST's oracle-to-deployed drop was −53 pts, RT-Affordance's −8. A thin ceiling means the deployed split lands *below* the monolith with certainty.
- `C8 ≥ C6 − 5` → **KILL MODEL B.** Zero parameters matched 25M. Project becomes CIR-1, ships in a weekend, frees the whole budget for Model A.
- `+5 < C6 − C0 < +10` → re-run on **LIBERO-Long** and **CALVIN ABC→D** only. If the ceiling is thin short-horizon and fat long-horizon, this is a long-horizon project — say so in writing before continuing.

### E1 — isolating the split from every confound (2–3 weeks, ~$100–250 rented GPU)

Ladder, all at matched total params P / demos D / init / effective rate R / action representation:

| ID | Arm | Isolates |
|---|---|---|
| C0 | MONO | baseline |
| C0s | MONO at P_A and at P_B | the parameter-scaling curve — without it you can't distinguish "split gain" from "effective model got bigger" |
| **C1** | **MONO-AUX** — C0 + auxiliary future-pose head, unused at runtime | **the π0.5 "implicit HL" control.** Decomposition-as-training-signal vs as-architecture |
| C2 | MONO-2PASS — C1 weights, actual two-stage runtime inference | runtime hierarchy vs parameter separation |
| C3 | SPLIT-JOINT — two nets, gradients through a continuous interface | two nets vs two *independently trained* nets |
| **C4** | **SPLIT-SEQ — the proposal as stated** | the thing you want to build |
| C5 | SPLIT-SEQ + one DAgger round on the interface | the covariate-shift tax |
| C6 | SPLIT-ORACLE (carried from E0) | architecture ceiling + attribution denominator |
| **C7** | **SPLIT-IK — learned A → analytic IK + OSC** | whether Model B needs to be learned at all |

Crossed confounds (staged screen, not a full cross): **pretraining** {random, pretrained vision, full VLM} — this alone is ~20 pts and may be what you're measuring; **total params** {10, 25, 60, 150M}; **chunk horizon** {1, 16, 50}; **HL update interval** {1, 10, 30, 60}; **matched vs native command rate**; **action representation** {EE-abs, EE-chunk-delta, joint-abs, joint-chunk-delta}; **interface encoding** {raw vector, waypoint chunk, GMM heatmap, pose+latent}; **demo count** {10, 25, 50, 100, 200}/task.

**GO only if** `C5 − max(C0, C1, C2, C7) ≥ +8` on ≥3 of 4 suites with non-overlapping 95% CIs. **Named NO-GO verdicts:** `C1 ≥ C5−3` → benefit was the training signal, ship C1. `C7 ≥ C5−3` → Model B unnecessary. Advantage vanishes at pretrained init → you measured pretraining. Advantage vanishes at matched rate → you measured latency. `C3 ≥ C5+8` → separate training is the defect. `C5 − C4 ≥ +10` → DAgger is permanently mandatory.

### E2 — regimes where hierarchy is *allowed* to win (1–2 weeks)
- **E2a dynamic:** CALVIN-D / ManiSkill3 with object velocity {0, 1, 3, 10} cm/s. Report the **slope**, not the mean. HiRT's dynamic gap was 75 vs 48; static was 70.0 vs 71.3. Flat in speed → H2 dead.
- **E2b perturbation recovery:** object nudge {2, 5, 10} cm, 5 N arm push for 200 ms, 0.5 s occlusion, gripper slip. Metric: recovery rate conditioned on being on-track pre-perturbation. Also: **livelock rate**.
- **E2c data provenance (H3 — probably your real result):** `C4+ARM` (A additionally trained on ~750–2000 annotated stills + web/sim data B cannot consume) vs `C0+ARM` (the honest control — can the monolith absorb it too?). **GO signal:** `C4+ARM − C4 ≥ +15` while `C0+ARM − C0 < +5`.
- **E2d generalization, 7 axes reported separately, never averaged:** novel instance / novel category / novel position outside training convex hull / lighting / camera pose / distractors / instruction rephrasing. **Run each with C6 too** — the `C6 − C4` gap per axis *is* your failure attribution.
- **E2e:** SimplerEnv correlation check before spending real-robot months.

### E3 — rate and systems measurement on the target box (1 week, no training)
Measure four configurations: (1) MONO + chunking + async; (2) SPLIT, two processes; (3) SPLIT with B stripped of RGB; (4) SPLIT with A in MLX/Metal and B in CoreML pinned to `cpuAndNeuralEngine`, GPU explicitly excluded — **verify ANE residency with `powermetrics`**, because CoreML silently falls back to GPU. **KILL H2 if** MONO+chunking hits ≥30 Hz with p99 < 2×p50. **GO only if** SPLIT achieves ≥2× MONO's effective rate with fast-loop p99 < 50 ms.

### E4/E5 — real robot (3–4 then 4–6 weeks; **only if E1 gave ≥15 pts**)
The 15-point bar is not arbitrary: ~400 rollouts/arm is what a real arm affords, detecting ~10 pts at 80% power, and HiRT's real quasi-static gap was **1.3 points**. If E1 gave 8 points in sim, the real robot cannot resolve it.

**Teleop budget:** E0/E1/E2a/b/d need **zero** new demos (benchmarks ship them). E2c needs ~750–2000 *stills* (~1 h + 2–3 h annotation), zero teleop. E4 pilot: **5 tasks × 50 demos = 250, ≈ 6–10 h** including resets. E5: **zero new demos**. Real-robot sample-efficiency curves, if needed: +750.

**Hardware:** SO-101 pair (~$400, position-only — cannot test compliance) is sufficient for E4/E5 on H1/H3/H4. Rent/borrow a Franka for one week **only if** E2 showed the split winning on contact-rich tasks. Do not buy a Franka to test H1.

**Mandatory before E4:** the reflex layer, the typed timestamped interface, the Oracle-A/Oracle-B harness, and the offline replay regression test. Without the harness, every real-world failure has four indistinguishable causes at a week each.

**Metrics (report distributions, never means):** success + progress score; control rate p50/p5; photon-to-command latency p50/p95/p99/max; **jitter ratio p99/p50** (healthy edge reference: 150.5 ms mean, **0.13 ms std**); interface staleness p50/p99; queue underrun rate; pose tracking error translation/rotation RMS+p95 **split by task phase**; interface error ‖A's pose − GT‖ (the single most diagnostic number); terminal placement error; RMS joint jerk + joint-velocity sign reversals (catches IK branch-flipping and chunk-boundary jerk); peak and RMS contact force; jam/stall rate; recovery rate and time; livelock rate; per-axis generalization; success-vs-N-demos with crossover N*; safety trips per 100 rollouts (joint-limit, velocity, IK-infeasible, workspace, NaN, watchdog, human intervention); **oracle gap C6−C4** and **classical gap C7−C4**.

**Total: you can falsify the central claim for $0 in under a week (E0). Everything after is contingent on E0 passing, and E0 tests the ceiling — a negative there is not rescuable by more data, a better Model A, or a bigger budget.**

---

## 8. Open Questions Only You Can Answer

| # | Question | Why it is decisive | How each answer changes the recommendation |
|---|---|---|---|
| **Q1** | **What does the arm's command interface accept — joint position only, or torque / Cartesian impedance / variable stiffness?** | Gates the entire compliance argument, and therefore whether a learned low level can do *anything* IK cannot | **Position-only** → delete the stiffness channel and the contact rationale; scope to free-space pick/place/reach in writing; CIR-1's residual has almost nothing to correct — ship CIR-1 minus the residual, i.e. `VLA + IK + retimer`. **Torque/impedance (Franka FCI, KUKA FRI, QDD)** → the residual is genuinely load-bearing; CIR-1 as specified; contact tasks enter scope; buy a wrist F/T sensor ($3–5k) as the single highest-value sensor purchase |
| **Q2** | **6-DoF or 7-DoF arm?** | On 7-DoF an EE pose is *underdetermined* — measured 8% (pure task space) vs 94% (pose + one redundancy scalar) vs 96% (joint space) on confined-space tasks | **6-DoF** → the redundancy issue is minor (branch selection only, handled by IK seeding); interface can drop the swivel field. **7-DoF** → the swivel/arm-angle scalar becomes **mandatory** in the interface, and null-space posture bias becomes mandatory in the IK call. This is a 7-numbers fix, not a network |
| **Q3** | **What is the target task class — free-space pick-and-place, contact-rich (insertion/wiping/doors), deformables, or long-horizon multi-stage?** | The measured hierarchy premium is **an exact tie on short-horizon** (69.57 vs 69.63) and **+30 to +42 pts on long-horizon/reasoning**. The pose-waypoint interface is proven for quasi-static keyframe tasks and documented to fail on continuous/contact tasks | **Short-horizon pick-and-place** → the split is measured to buy nothing; build the monolith (CIR-1) and stop. **Long-horizon / multi-stage** → hierarchy is genuinely justified; also add an explicit switching/termination policy (O9). **Contact-rich** → Q1 becomes a hard prerequisite; add force to the interface *and* to Model B's inputs. **Deformables** → the hard part is Model A, not Model B; put everything into Model A |
| **Q4** | **Is a moving head/base part of the design, or is the camera static during manipulation?** | A camera-frame pose is meaningless without the head extrinsic *at capture*; at 500 ms latency and 20°/s pan that is ~63 mm of silent error | **Static camera** → camera-frame prediction is safe and worth +8 to +14 pts; the timestamp contract is good hygiene but not urgent. **Moving head** → the capture-time head-state stamp and immediate base_link conversion become **mandatory and load-bearing**; consider freezing the head during the manipulation phase as a v1 simplification |
| **Q5** | **Is the Mac mini a hard constraint or a preference?** | The measured ceiling is **3.78 TFLOPS / 104 GB/s [M]**; a 3B trunk prefill is **3.74 s [M]**. A used RTX 3090/4090 on the same wired switch runs the backbone ~8–10× faster with genuine priority streams and TensorRT int8, for roughly the price of an M4 Pro upgrade, at ~0–4 ms network cost | **Hard constraint** → cap Model A at ~450M–1B, one model on the GPU, everything learned in the fast path on the CPU, servo off-Mac. CIR-1 or TRISYS-500 at SmolVLA scale. **Preference** → buy the GPU box, run a π0.5-class model, and note that at 32–57 Hz monolithic the entire latency rationale for a split evaporates |
| **Q6** | **Is training a pretrained VLM acceptable, or must everything be trained locally from scratch?** | Random-init vs pretrained is **~20 pts (77.5% → 97.8% on LIBERO)** — larger than every short-horizon hierarchy premium in the literature | **Pretrained acceptable** → freeze a pretrained VLM as Model A, LoRA the rest; all three alternatives are viable. **From-scratch mandated** → you forfeit ~20 points, which is more than any architecture choice can recover; the split's strongest supporting mechanism (internet-pretrained frozen top) is unavailable, and the honest recommendation becomes "build a monolith and don't expect hierarchy to help" |
| **Q7** | **Are you willing to collect ~750–2000 hand-annotated still frames (≈1 h collect + 2–3 h annotate)?** | This is the *actual* mechanism behind every pose-interface success. RT-Affordance's predictor: 77% → **24%** without them, → **11%** without web co-training | **Yes** → H3 is testable and is probably your real result; the split earns its existence through data access; run E2c. **No** → the split's only well-evidenced performance benefit is unavailable, you pay every cost for none of the benefit, and CIR-1 (one model) is unambiguously correct |
| **Q8** | **Is the goal a research result, a working robot, or a debuggable/human-correctable robot?** | These three goals point at three different architectures | **Working robot** → CIR-1, ~2 weeks. **Research result** → note that the *only* genuinely untested configuration in this space is the original proposal's learned pose→joint low level, and it is untested because it does not work; the interesting untested question is instead "explicit pose vs learned latent at matched everything," which no paper runs. **Debuggable/correctable** → ΔChunk; the pose interface is the product, RT-H's interventions (40% → 63%) are the evidence, and a success-rate tie is an acceptable outcome |
| **Q9** | **How many real-robot rollouts can you actually afford per condition?** | HiRT's split-vs-mono gap was **1.3 points**; TRI needed 1,800 trials with Bonferroni correction to separate policies | **<20/condition** → you cannot distinguish any of these designs; do not run a real-robot comparison, run E0/E1 in sim and ship whichever is simpler. **50–100/condition** → you can detect ~10–15 pts; set the E1 GO bar at ≥15 pts accordingly. **≥400/condition** → E5 as specified is meaningful |
| **Q10** | **Single arm or bimanual?** | Two independent absolute targets with 5 mm error each compose into up to 10 mm *relative* error, which for a rigid two-handed grasp is squeeze (internal force, stalled joint) or gap (drop) | **Single** → everything above applies. **Bimanual** → the **relative transform** must be a first-class interface field, not two independent poses, and the IK/impedance formulation must be coordinated. This is genuinely more engineering and should be explicitly deferred to v2 |

---

## 9. Phased Roadmap

### Phase 0 — Decide and measure (1 week, $0)
1. Answer Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q10 **in writing**. Several of them delete entire branches of this plan.
2. Run P1–P4: hardware interface probe, two-process concurrency measurement on the actual box, camera-rate ceiling, hand-eye calibration error budget.
3. **Gate:** if P2 reproduces the 6.3×/9.2× contention penalty **[M]**, H2 is dead on this hardware; the split must be CPU/GPU-partitioned or single-graph from here on.

### Phase 1 — The spine (days 1–5 of the build, zero learning, zero data)
URDF → TRAC-IK → Ruckig jerk-limited retimer → Cartesian impedance (or position servo) on the arm's own controller or a Linux SBC over wired Ethernet → the **non-learned reflex layer** (clamps, thresholds, workspace AABB, reachability + self-collision filter, NaN asserts, step-magnitude limiter, deadman watchdog) → the **typed, timestamped, versioned interface schema** with its replay tooling.

**Acceptance test:** teleop the arm *through the interface* with a 6-DoF mouse writing chunk messages at 3 Hz. If you cannot do that, no amount of model training will help.

**Deliverable:** a robot that moves safely and a schema you can log, replay, and unit-test.

### Phase 2 — The cheap falsification (E0, 3–5 days, laptop, $0)
Three LIBERO training runs: MONO, SPLIT-ORACLE, ORACLE-IK. 1,500 episodes each. **This is the highest-value week in the entire project** — it tests the *ceiling* of the architecture, so a negative result is not rescuable by a better Model A or more data.

**Gates:** `C6 − C0 ≤ +5` → kill H1, go straight to Phase 3 as CIR-1 and stop worrying about the split. `C8 ≥ C6 − 5` → kill Model B permanently.

### Phase 3 — Model A + spine, residual off (weeks 2–4)
- Teleop: **250–400 episodes** across 8–12 tasks (~6–10 h).
- Fine-tune **from `lerobot/smolvla_base`** — never from scratch. LoRA on the LM, full training of the action expert. ~1 GPU-day rented (~$20–60). Do not fine-tune on the Mac.
- Data mixture: ~60% your teleop, ~15% the 750–1500 annotated stills, ~15% web VQA/grounding (anti-forgetting), ~10% auto-extracted subtask labels (gripper-flip keyframes, zero human labeling).
- Auxiliary losses that cost almost nothing and carry most of the decomposition benefit: **target-pose head** (OpenHelix's single largest gain, 3.45 → 4.01) and **subtask-language head** (π0.5's "implicit HL" finding).
- Interface: chunk-wise SE(3) deltas, 6D rotation, gripper aperture, dt, stiffness, swivel, confidence, phase.
- Async chunked execution: H=50, replan at ~60–70% queue drain, Real-Time Chunking prefix freezing, weighted-average temporal ensembling.
- Build **Oracle-A and Oracle-B harnesses now**, not later.

**Deliverable: a working robot at end of week 4, with the residual switched off.** This is the baseline everything else must beat.

### Phase 4 — Measure honestly (weeks 4–6)
E3 (rate/jitter on the target box) + E1 ladder in sim + E2a/E2b/E2d. Report per-task, per-axis, with p99s and not means. Decide, using the named NO-GO verdicts, whether the split has anything left to justify it.

**Branch point.** If `C1 ≥ C5 − 3` or `C7 ≥ C5 − 3`, you are done: ship Phase 3, write up the negative result honestly, and stop. That is a legitimate and useful outcome.

### Phase 5 — The residual (week 7, ~2 days)
Only after Phase 3 is deployed. Roll out A + spine with **residual = 0**, log at 200 Hz, label = the error the impedance controller failed to close (the feedback controller's own output is the teaching signal — zero human labeling). Train the 0.18M GRU in ~20 minutes on the Mac CPU. One DAgger round because the residual changes its own input distribution. Hard-clip to ±8 N and add the L1 magnitude penalty so graceful degradation is a training objective, not a hope.

### Phase 6 — E2c, the data-provenance test (weeks 7–9)
The experiment that tests H3, which is probably your real result. `C4+ARM` vs `C0+ARM`. **This is where the split either earns its existence or is confirmed unnecessary.**

### Phase 7 — Conditional: real-robot powered comparison (weeks 9–15)
Only if E1 gave ≥15 points. E4 pilot (3 arms × 100 rollouts, instrumentation shakedown) then E5 (3 arms × 400 rollouts, blind randomized, Bonferroni-corrected). Budget honestly: at n=400 the minimum detectable effect is ~9.9 points.

### Phase 8 — Conditional upgrade: TRISYS-500 (weeks 12–17)
Only if Phase 4 showed a genuine capability ceiling on Model A that a larger/differently-structured high level would lift. Swap the single VLA for frozen-VLM + adapter + flow-matching action expert in one graph, with **mandatory adapter pre-alignment as a separate stage** (OpenHelix: skipping it = 0/0/0/0/0), joint co-training, and staleness conditioning in the last 20% of training. The spine, safety layer, schema, and evaluation harness are unchanged — this is why the ordering matters.

### Explicit off-ramps
- **Q6 = from-scratch mandated** → skip Phases 5–8; you have forfeited more than any architecture can recover; build the monolith and set expectations accordingly.
- **Q1 = position-only arm** → skip Phase 5; ship `VLA + IK + retimer`; restrict the task list in writing.
- **Q3 = short-horizon pick-and-place only** → skip Phases 6–8; the split is measured to buy nothing in this regime.
- **Q8 = debuggable/correctable robot is the goal** → after Phase 4, pivot to ΔChunk and build the intervention tooling (RViz target rendering, 6-DoF-mouse correction, correction-replay into training). Accept a success-rate tie as the correct outcome and measure the intervention benefit instead (RT-H's 40% → 63% is the number to try to reproduce).

---

## Residual Uncertainties — stated plainly

1. **The decisive experiment does not exist.** No published work runs a fixed-total-parameter, fixed-data, matched-rate, matched-init comparison of monolith vs pose-interface hierarchy vs latent-interface hierarchy. My conclusion that "the split does not buy accuracy per parameter" is an *inference* from adjacent evidence (MoE scaling laws, π0.5's implicit-HL ablation, HiRT's monolith margin, FiS-VLA's parameter-sharing result, the pretraining ablations), not a direct refutation. E0/E1 is designed to close exactly this gap on your own setup.

2. **Several load-bearing numbers are unverified 2026 preprints.** The action-space table (69.0 / 77.3 / 88.0 / 89.6), the LIBERO pretraining gap (77.5% → 97.8%), the orchestration study's success rates, GHOST's oracle swap, HDP's 24.55% IK error rate, and the ER/ERJ 8%/94%/96% triple all warrant re-verification against primary sources before being quoted in a final document. I have marked them **[X]** throughout.

3. **π0.5's Figure 13 numeric values could not be extracted** — I report only the authors' stated ordering (full > implicit HL > no HL) and their verbatim conclusion. Since this is the single most important piece of counter-evidence in the whole review, read arXiv 2504.16054's ablation section directly before making a build/no-build decision on its basis.

4. **Helix publishes no quantitative results at all.** The 7B/80M, 7–9 Hz/200 Hz, single-latent, dedicated-GPU, and end-to-end-backprop details are all from Figure AI's own blog. Architectural description reliable; performance claims unverified.

5. **GR00T N1 publishes no split-vs-monolith ablation.** The system the proposal most resembles rhetorically has, as far as I can verify, no evidence that its split beats a monolith. Its design is justified by citing Kahneman.

6. **No published study measures how VLA-class policies degrade as a function of hand-eye calibration error.** The 2.1 mm / 3.2° figures come from calibration-methods papers, not from policy-performance studies. The inference that calibration error meaningfully limits an absolute-pose interface is my reasoning from the peg-in-hole tolerance argument **[I]**, not a directly measured result.

7. **The Apple-silicon numbers I report are measured [M], but on an M4 *base*, not an M4 Pro.** An M4 Pro would be roughly 2× on both compute and bandwidth. Also note a directly contradictory data point I could not resolve: one source claims ResNet-50 runs only ~3× slower on Apple Silicon than an RTX 4090 — almost certainly a large-batch throughput comparison rather than batch-1 latency, but it is the largest single source of uncertainty in the extrapolations. Measure, don't trust either number.

8. **I did not investigate training feasibility in depth** — only inference and deployment. Training two coupled policies jointly (which is what Helix and GR00T actually do) is a substantially larger compute problem, and the Mac mini is almost certainly not viable for it. All three alternatives assume a rented GPU for the main training run.

9. **I did not verify the MLX concurrency issue's resolution status.** Issue #3078 appears closed without a maintainer fix, but MLX moves fast. A 30-minute experiment — two MLX processes, one running a 3B model in a loop, one running a 50M model at 100 Hz, logging the small model's p99 — would settle it definitively and should be run before designing around it.