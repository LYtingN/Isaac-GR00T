# Force / Tactile VLA References

Last checked: 2026-09-21

This note records references relevant to tactile-conditioned GR00T. It stores links and concise design takeaways only; no PDF files are kept in this repository.

## 1. ForceVLA

- Link: https://sites.google.com/view/forcevla2025
- Focus: force modality integration for contact-rich manipulation.
- Method: projects real-time 6-axis force into force tokens and fuses them with post-VLM visual-language features through a force-aware Mixture-of-Experts module before the flow-based action head.
- Key takeaway: prefer post-VLM / action-decoder-side fusion. The reported ablation indicates that early force fusion can disturb pretrained VLM representations.
- Relevance here: a direct reference for injecting tactile tokens into GR00T's action context rather than modifying the visual-language backbone.
- Priority: **high — current architecture**.

## 2. ForceVLA2

- Link: https://sites.google.com/view/force-vla2/home
- Focus: hybrid force-position control for contact-rich manipulation.
- Method: combines long-horizon force-aware reasoning with a short-horizon reactive force pathway. Its outputs include motion chunks, force objectives, control mode, and task progress.
- Key takeaway: force can be used both as an observation and as an explicit control target; a fast reactive pathway complements slower semantic planning.
- Relevance here: useful when extending the current SONIC action space with force targets or compliance/control-mode commands.
- Priority: **medium — later control extension**.

## 3. Being-H0.8

- Link: https://research.beingbeyond.com/being-h08
- Focus: scalable tactile representation and slow-fast tactile-aware action generation.
- Method: embeds variable-length tactile measurements and uses a query-based Perceiver to resample them into a fixed number of tactile tokens. A learned missing-touch representation keeps the interface stable when tactile data is unavailable. Its slow-fast action expert refreshes short action segments using recent state and tactile feedback.
- Key takeaway: fixed-token resampling is a strong reference for handling different body regions, sensor densities, and missing tactile zones.
- Relevance here: reference for `tactile_encoder.py`; zone/taxel inputs can be converted into a fixed token count before GR00T cross-attention.
- Source note: the project page states that the paper is forthcoming, so architectural claims should currently be treated as project-reported descriptions.
- Priority: **high — encoder design**.

## 4. TaF-VLA

- Paper: https://arxiv.org/abs/2601.20321
- PDF: https://arxiv.org/pdf/2601.20321
- Title: *TaF-VLA: Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation*.
- Focus: tactile encoding aligned with physical force rather than treating tactile observations as ordinary images.
- Method: the TaF-Adapter extracts discretized latent information from sequential tactile observations and aligns it with synchronized 6-axis force/torque and force maps.
- Key takeaway: temporal tactile encoding and tactile-force alignment can encourage representations that capture physical dynamics and resist sensor noise.
- Relevance here: useful if raw tactile tokens prove insufficient and an auxiliary tactile-force alignment or discretized-latent objective is introduced.
- Priority: **high — encoding and auxiliary supervision**.

## 5. LIFT / Never Too Late for Force

- Paper: https://arxiv.org/abs/2607.14236
- PDF: https://arxiv.org/pdf/2607.14236
- Title: *Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection*.
- Focus: adding force after a VLA has already been pretrained.
- Method: attaches a reactive action expert, injects recent 6D force through causal force memory and zero-initialized cross-attention, and combines offline training with online DAgger corrections.
- Key takeaway: zero-initialized force injection can preserve the original policy at initialization; causal force memory is valuable for reactive contact behavior.
- Relevance here: relevant if tactile is introduced through a second post-training stage after the basic GR00T fine-tune is working.
- Priority: **low for now — post-training phase**.

## 6. Gated Memory Policy

- Project: https://gated-memory-policy.github.io/
- Paper: https://arxiv.org/abs/2604.18933
- Focus: learning when and what historical context should affect the current action.
- Method: uses a learned memory gate and lightweight cross-attention over cached historical tokens; irrelevant history can be suppressed instead of always being passed to the policy.
- Key takeaway: longer context is not automatically beneficial. A gate can selectively activate force/tactile history only during contact-dependent phases.
- Relevance here: reference for a later `gated_force_context` module that attends over recent tactile tokens to support adaptive behavior.
- Priority: **medium — temporal tactile context after the current-frame baseline**.

## Recommended implementation order

1. **ForceVLA-style integration:** encode tactile after the VLM and expose it to the GR00T action head through context tokens.
2. **Being-H0.8 / TaF-VLA-style encoding:** convert heterogeneous zones into fixed tactile tokens; add temporal or force-alignment objectives only if needed.
3. **Gated Memory Policy-style context:** add selective access to recent tactile history after the current-frame model is validated.
4. **ForceVLA2 / LIFT extensions:** add force targets, compliance modes, fast reactive experts, or a separate post-training stage later.

## Current project mapping

| Reference idea | Planned module |
|---|---|
| ResNet spatial tactile tokens | `gr00t_tactile/gr00t_tactile/model/tactile_encoder.py` |
| Post-VLM cross-attention context | `gr00t_tactile/gr00t_tactile/model/gr00t_tactile.py` |
| Shared offline/online preprocessing | `gr00t_tactile/gr00t_tactile/data/tactile_image.py` |
| Gated tactile history | future temporal tactile-memory module |
| Hybrid force-position commands | future SONIC/action-space extension |
