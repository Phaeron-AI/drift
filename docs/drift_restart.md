# Living Portraits — Project Brief & Research Map

*Privacy-first, on-device photorealistic motion for photographers.*

> **Scope note.** This document deliberately stops short of architecture. It covers the idea, how it makes money, what to watch out for, a general execution roadmap, and the research literature to read. The full mathematical walkthrough of the model — the part with the derivations — is the **next** document, written once a concrete approach is chosen. Section 5 flags exactly where that math will live so you can start reading toward it now.

---

## 1. The Idea (privacy is the wedge, not a feature)

The product takes a single photograph and brings part of it to life as **photorealistic video motion** — not illustrated/cartoon animation. A portrait where the couple stays sharp and still while the bride's veil drifts, the dress flows, candle flames flicker, steam rises off the chai, and leaves move behind them. The output is a seamless 3–5 second loop: a **cinemagraph**. The face is locked; the environment breathes.

The default behavior is one click: *lock the subject, animate the surroundings.* Beyond that, the user can **selectively** choose which regions move (a motion-brush style control), and do light cleanup in service of the loop (e.g. remove a distracting background object before animating). That selective-region motion is the same "animate only this object" capability explored earlier — here it finds its natural home and its buyer.

**The wedge is privacy, and it is structural, not cosmetic.** The entire I2V market — Runway, Kling, Sora, Veo — runs in the cloud, because that is their business model. That means there is an entire class of user those companies *cannot* serve: anyone contractually or ethically barred from uploading their imagery to a third party. A photographer holds whole galleries of private clients under contract and trust, and clients increasingly ask whether their faces are being uploaded or used to train someone's model. A tool that runs **entirely on the photographer's own machine — images never leave the device** — competes on a different axis from the incumbents. Not "is it the best motion?" but *"are you even allowed to use this?"* On that axis, the cloud giants have no answer.

**Why the cinemagraph framing is the right shape:**
- It is an **established premium deliverable**, not a novelty you'd have to teach the market. Photographers have sold cinemagraphs to luxury, fashion, and real-estate clients for over a decade; the reason they charge for them is the traditional workflow (hand-masking, hours per image) is brutal.
- **Locking the face is the quality fix, not a compromise.** Faces are exactly where I2V breaks — animate one and you get uncanny warping that's unsellable to a wedding client. Freezing the subject and animating only the environment sidesteps the worst failure mode of the whole field *and* lands on the tasteful aesthetic photographers actually want. The constraint and the feature are the same thing.

**Initial buyer (India-first):** wedding and portrait photographers — an enormous, physically reachable market, with a resale model (they charge clients a premium per cinemagraph). **Expansion buyer:** confidential creative/product work under NDA (design studios, product-viz teams) where privacy is a *hard compliance wall* rather than a strong preference — higher contract value, harder to reach as a solo founder, so it comes second.

---

## 2. Monetization Plan

**Guiding principle:** in India especially, consumer subscription willingness-to-pay is low, so the buyer must pay because the tool *makes or saves them money*, not because it's cool. A cinemagraph is a premium item the photographer **resells**, which gives you a value-linked story: you charge the photographer; the photographer charges the couple more than they pay you.

**The local-first model is also an economic advantage, not just a privacy one.** Because generation happens on the customer's GPU, *you have no per-generation compute cost.* That means you can offer flat, predictable pricing with high margins — no metering, no cloud bill scaling with usage — which is precisely the opposite of every cloud I2V competitor's cost structure.

**Pricing structures to weigh (pick after validation):**
- **Per-seat Pro subscription** (monthly/annual) — recurring revenue; unlimited local generation since it costs you nothing. Likely the core model.
- **Perpetual license + paid major upgrades** — aligns beautifully with the "it's your machine, your software" privacy ethos and appeals to buyers wary of subscriptions; weaker recurring revenue.
- **Studio / multi-seat tier** — higher ACV for photography studios with several shooters.
- **Add-on packs** — motion presets, style packs, seasonal looks; small incremental revenue, high perceived value, near-zero marginal cost.

**Funnel:** free trial (watermarked or limited exports) → **Pro** (per-seat, unlimited local generation) → **Studio** (multi-seat) → later **White-label / SDK** for the NDA/product-viz segment (highest ACV, B2B).

**Sequencing:** price for the Indian photographer market first to win the wedge; introduce higher global/agency tiers once the NDA expansion is real. Resist the urge to chase the high-ACV enterprise deal before the photographer product is loved — the enterprise sale is slow and will stall a solo founder.

---

## 3. Things to Watch Out For

**The identity/consistency failure is the #1 risk — and it's the one that already bit you.** Last time, the source contained one object and the output contained a *different object entirely*. That is the canonical generative-I2V failure mode, and understanding *why* tells you how to avoid it:

- **Root cause:** end-to-end generative I2V *resynthesizes* pixels. When image conditioning is weak and/or a text prompt's guidance is too strong, the model "drifts" — it regenerates content from its prior instead of preserving the source, so the object's identity changes.
- **The governing principle of the fix: do not regenerate what should not change.** Everything that isn't supposed to move should be *copied from the source*, not re-synthesized. Concretely, the cinemagraph framing solves most of this *by construction*: animate only a masked region, composite that motion over a **locked static plate** of the original image, and the untouched 99% of the frame is literally the source pixels. Identity can't drift in a region you never regenerated.
- **Where it's still hard:** the boundary between the moving region and the static plate (seams, halos), and **disocclusion** — when something moves, it reveals pixels behind it that must be filled plausibly. This is the genuinely hard sub-problem and a natural research focus (see §5).

**Other risks to track:**
- **Face warping / uncanny output** — mitigated by the lock-the-face default; never let the subject be the thing that moves.
- **Temporal artifacts** — flicker, shimmer, and a visible "jump" where the loop restarts. You need *seamless looping* (the loop endpoints must match), which is a real engineering requirement, not an afterthought.
- **Privacy claims must be literally true, end to end.** This is a *trust* product. If you market "images never leave your machine," there can be **no** telemetry, crash logs, or analytics that ship image data anywhere. One leak — or one credible accusation of one — kills the entire value proposition. Treat the privacy guarantee as a hard architectural and legal commitment, and be ready to prove it (offline operation, open inspection, audited builds).
- **Base-model licensing.** Open I2V weights carry varied commercial terms — some are Apache-2.0 (cleanest for a product), others carry territorial or monthly-active-user restrictions, and training-data provenance varies. Read each license before you build on it; this constrains which model you can ship.
- **Hardware reach (your biggest go-to-market friction).** The product needs a capable GPU. Many photographers are on Macs or laptops without a strong NVIDIA card. Decide early how you handle this: a clear minimum spec, an Apple-Silicon path, or an optional local-network "appliance" — but don't pretend the friction isn't there.
- **Scope creep — guard v1 ruthlessly.** "Editing capabilities" is an open door to building a second product (a Photoshop competitor). Allow *only* editing that serves the cinemagraph (cleanup before animating). Everything a photographer would do in their existing editor stays in their existing editor.

---

## 4. Breakdown & Roadmap (general)

A staged plan that keeps validation ahead of building, so you never sink months into something a buyer won't pay for.

1. **Phase 0 — Buyer validation (2 weeks, before serious building).** Put 3 example cinemagraphs and a price in front of real photographers. You are hunting for one specific *yes*: *"I'd pay for this, and it mattering that it stays on my machine is real for me."* The only thing allowed to kill the idea here is "no one will pay." This is testable with mockups — you do **not** need the working pipeline to find out.
2. **Phase 1 — Feasibility spike.** Get an open I2V model running locally on your GPU. Prove the core loop: region mask → motion only inside the mask → composite over the static plate → seamless loop. Confirm the lock-the-face default looks genuinely good. This de-risks the technical heart before you build product around it.
3. **Phase 2 — MVP pipeline.** The minimal end-to-end flow: load portrait → auto-segment → one-click "living portrait" → export loop. Smallest possible UI. This is the demoable, sellable core.
4. **Phase 3 — Selective control.** Motion-brush region selection, a small library of motion presets, basic in-service-of-loop cleanup. Resist anything beyond this.
5. **Phase 4 — Packaging & trust.** Installer, licensing/activation, the privacy guarantee made real and demonstrable, minimum-spec handling, graceful behavior on weak hardware.
6. **Phase 5 — Pilot & price.** A handful of paying photographers. Iterate on quality and workflow, settle pricing against observed willingness-to-pay.
7. **Phase 6 — Expand.** The NDA/product-viz segment; white-label / SDK; higher tiers.

**Parallel research track (runs alongside, not instead).** Every phase generates the material for a benchmark and a paper — an identity-preservation/region-leakage metric, the disocclusion method, on-device efficiency results. Keep the research as a *byproduct of building rigorously*, so it strengthens the product instead of competing with it.

---

## 5. Research Literature & Possible Contributions

Reading is organized into clusters, ending with where your math/physics derivations will actually live. Items are grouped roughly easy-entry → foundational.

### A. Cinemagraph & single-image animation — *the most relevant cluster, and the most mathematically rich*
This is the lineage your product sits in, and it directly addresses your consistency problem, because these methods **move real pixels (warping) rather than regenerating them** — identity is preserved by construction.
- **Chuang et al., "Animating Pictures with Stochastic Motion Textures," SIGGRAPH 2005.** The origin: layered scene, motion synthesized via a spectral method. Good for intuition about layering and looping.
- **Endo et al., "Animating Landscape: Self-Supervised Single-Image Animation," SIGGRAPH Asia 2019.** Learned motion + appearance prediction from a single image.
- **Holynski, Curless, Seitz, Szeliski, "Animating Pictures with Eulerian Motion Fields," CVPR 2021.** *Read this one closely.* A still becomes a looping video via a **time-invariant, constant-velocity Eulerian flow field**, applied through **Euler integration** to produce displacement fields, then **deep warping with symmetric splatting** to handle the holes that warping opens up. This is the mathematical and physical heart of the whole approach (see derivations note below).
- **Mahapatra & Kulkarni, "Controllable Animation of Fluid Elements in Still Images," CVPR 2022.** Adds interactive user control over the motion direction — i.e. your motion-brush idea, in the literature.
- **Mahapatra et al., "Text-Guided Synthesis of Eulerian Cinemagraphs" (Text2Cinemagraph), ACM TOG 2023.** Eulerian cinemagraphs from text; bridges the classical motion-field approach with modern generative priors.
- **3D Cinemagraphy (Li et al., 2023)** and follow-on single-image dynamic-scene work — joint animation + parallax, if you ever want subtle camera motion.

### B. Image-to-video models you'll actually build on
- **Stable Video Diffusion (Blattmann et al., 2023)** — the canonical open I2V baseline.
- **AnimateDiff (Guo et al., 2023)** — motion modules over a frozen image model.
- **CogVideoX, HunyuanVideo, LTX-Video, Wan** — current open models that run on consumer GPUs; study their I2V conditioning and licensing.

### C. Region & motion control (your selective-animation feature)
- **MOFA-Video (Niu et al., ECCV 2024)** — *especially relevant*: controllable image animation via generative **motion-field** adaptations in a frozen I2V diffusion model. This is the cleanest bridge between the classical Eulerian-field world (Cluster A) and modern diffusion.
- **DragAnything (ECCV 2024), MOFA, MotionCtrl, DragNUWA, Tora** — the trajectory/region-control family; read for how motion intent gets injected as conditioning.
- **SAM / SAM 2 (Kirillov et al., 2023; 2024)** — segmentation, for getting your masks (SAM 2 handles video).

### D. Identity preservation & "don't regenerate what shouldn't change"
- **AnyDoor (Chen et al., 2023)** — object identity preservation under relocation; note its successes *and* its identity-drift failure cases.
- **ObjectDrop (Winter et al., ECCV 2024)** — counterfactual data for physically-consistent insertion/removal; the data-as-moat idea.
- **ObjectMover (2025)** — single-image object movement framed as sequence-to-sequence with a video prior; enumerates exactly the sub-problems (relighting, disocclusion, shadow/reflection sync, identity) you'll face.
- **ControlNet, IP-Adapter, DDIM inversion** — the standard toolkit for strong image conditioning and source anchoring (the antidote to your "different object entirely" failure).

### E. Generative foundations — *where your derivations begin*
The math you want to work through, in rough reading order:
- **DDPM (Ho et al., 2020)** — denoising diffusion; the variational objective.
- **Score-Based Generative Modeling through SDEs (Song et al., 2021)** — the continuous-time view; forward/reverse SDEs, the probability-flow ODE. This is the richest single source for derivations.
- **Latent Diffusion (Rombach et al., 2022)** — why generation happens in latent space.
- **EDM (Karras et al., 2022)** — a clean, well-parameterized treatment of diffusion design choices.
- **Flow Matching (Lipman et al., 2023) / Rectified Flow** — the formulation most current video models actually use; conceptually cleaner than score-matching for many derivations.

### Possible research contributions

**Incremental (lower risk, very shippable as a first paper):**
- A **region-locked / static-plate-composited I2V method** that *guarantees* identity preservation for cinemagraphs, plus a **benchmark and metric** for identity preservation and "region leakage" (motion bleeding outside the intended mask) in single-image motion — the field lacks a standard one, and you must build it internally anyway.
- **On-device efficiency** for cinemagraph I2V (quantization/distillation/scheduling to run well on consumer GPUs) — a systems contribution that is *also* your moat.

**More novel (higher ceiling, plays directly to your math/physics strength):**
- A **hybrid warping + diffusion** method: use **Eulerian motion fields** (Cluster A — physically grounded, mathematically elegant) for structure-preserving motion of fluids/hair/fabric where warping excels, and use diffusion *only* for **disocclusion and refinement** where warping fails. This is the principled marriage of the two lineages, it preserves identity by construction in the warped regions, and the derivation is genuinely deep — optical flow, Euler integration of a velocity field, splatting as a differentiable operation, and the diffusion SDE/flow-matching objective for the inpainting term.

### Where the derivations will live (next document)
Your full mathematical walkthrough naturally splits into three derivable pieces, all of which you can start reading toward now:
1. **The motion model** — the Eulerian velocity field, Euler integration to displacement fields `F₀→t`, and the looping/symmetric-splatting construction (from Holynski et al. and MOFA-Video).
2. **The warping/compositing operator** — splatting as a differentiable map, the disocclusion/hole term, and seam blending against the static plate.
3. **The generative refinement** — the diffusion or flow-matching objective for the regions warping can't handle, from the Cluster E foundations.

Once you pick the approach (warping-first, diffusion-first, or the hybrid), the next document derives each of these end to end.

---

*Prepared as a living document — extend the research clusters and roadmap as Phase 0 validation comes back.*