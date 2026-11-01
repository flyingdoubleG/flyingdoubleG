I audit open-source **RL / post-training frameworks, Triton kernels and interpretability libraries** for silent correctness bugs: wrong gradients, mis-normalized losses, off-by-one indexing, out-of-bounds kernel reads. Every fix comes with a minimal reproduction and a test that fails before and passes after.

**44 merged PRs across 15 repositories** (as of 2026-11).

---

#### RL & post-training frameworks
- [modelscope/twinkle#334](https://github.com/modelscope/twinkle/pull/334): chunked cross-entropy backward wrote gradients into the logits in place, so the default accuracy metric always read ≈ 0
- [modelscope/twinkle#335](https://github.com/modelscope/twinkle/pull/335): Liger fused CE normalized twice in mean mode and crashed under micro-batching
- [modelscope/twinkle#336](https://github.com/modelscope/twinkle/pull/336): InfoNCE hard-negative upsampling offset copied the positive in as a negative
- *In review:* verl [#8363](https://github.com/verl-project/verl/pull/8363) (SGLang silently ignores `logprobs_mode`), [#8364](https://github.com/verl-project/verl/pull/8364), [#8365](https://github.com/verl-project/verl/pull/8365), [#8366](https://github.com/verl-project/verl/pull/8366); Trinity-RFT [#636](https://github.com/agentscope-ai/Trinity-RFT/pull/636) (SAPO gradient), [#640](https://github.com/agentscope-ai/Trinity-RFT/pull/640) (JSD advantage points away from −∇JSD), [#641](https://github.com/agentscope-ai/Trinity-RFT/pull/641) (k× gradient under the verl ≥ 0.8 engine); ROLL [#509](https://github.com/alibaba/ROLL/pull/509) (`agg_loss` ignores `loss_mask`)

#### Linear-attention kernels · [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [#1620](https://github.com/fla-org/flash-linear-attention/pull/1620): DeltaFormer backward walked chunks from the wrong origin, so every gradient was wrong when T is not a multiple of the chunk size
- [#1618](https://github.com/fla-org/flash-linear-attention/pull/1618): gated DeltaNet's default `beta` was read out of bounds under GVA
- [#1627](https://github.com/fla-org/flash-linear-attention/pull/1627): ABC read the last partial chunk out of bounds, corrupting the recurrent cache
- [#1577](https://github.com/fla-org/flash-linear-attention/pull/1577), [#1619](https://github.com/fla-org/flash-linear-attention/pull/1619), [#1621](https://github.com/fla-org/flash-linear-attention/pull/1621), [#1629](https://github.com/fla-org/flash-linear-attention/pull/1629): Comba reference decay order, Oja GVA guard, a test that never asserted, RWKV-7 single-layer crash

#### Interpretability & data attribution
- [TransformerLens#1953](https://github.com/TransformerLensOrg/TransformerLens/pull/1953): residual-stream decomposition on post-norm models (Gemma 2/3, OLMo 2, …); relative error 0.7 → 2e-7, checked across 38 architectures
- [TransformerLens#1941](https://github.com/TransformerLensOrg/TransformerLens/pull/1941): Jacobian-lens ablation now projects out the whole subspace instead of one token at a time
- [bergson#565](https://github.com/EleutherAI/bergson/pull/565): MoE expert K-FAC covariances were normalized by all tokens instead of routed tokens (influence inflated by ~1/p); [#567](https://github.com/EleutherAI/bergson/pull/567): learning-rate history off by one step
- [delphi#182](https://github.com/EleutherAI/delphi/pull/182), [#183](https://github.com/EleutherAI/delphi/pull/183), [#184](https://github.com/EleutherAI/delphi/pull/184): explainer/scorer example leakage, IntruderScorer crash, stale KV cache in SurprisalScorer
- [shap#5261](https://github.com/shap/shap/pull/5261), [shapiq#643](https://github.com/mmschlk/shapiq/pull/643): DeepExplainer elementwise activations; CHII Monte Carlo weights

#### Inference-time compute, evolutionary search, model merging
- [optillm#348](https://github.com/algorithmicsuperintelligence/optillm/pull/348): DeepConf aligned with the paper (4 discrepancies); [#349](https://github.com/algorithmicsuperintelligence/optillm/pull/349): top-p removed the most likely tokens instead of the tail
- [openevolve#524](https://github.com/algorithmicsuperintelligence/openevolve/pull/524), [#528](https://github.com/algorithmicsuperintelligence/openevolve/pull/528): island migration and novelty-rejected programs left in the database
- [mergekit#720](https://github.com/arcee-ai/mergekit/pull/720), [#721](https://github.com/arcee-ai/mergekit/pull/721): inverted Procrustes rotation; padding leaking into MoE gate vectors
- [LLaMEA#157](https://github.com/XAI-liacs/LLaMEA/pull/157), [#159](https://github.com/XAI-liacs/LLaMEA/pull/159), [pytorch_optimizer#660](https://github.com/kozistr/pytorch_optimizer/pull/660), EMI-Group [evox](https://github.com/EMI-Group/evox/pulls?q=is%3Apr+author%3AflyingdoubleG+is%3Amerged) (5) and [evomo](https://github.com/EMI-Group/evomo/pulls?q=is%3Apr+author%3AflyingdoubleG+is%3Amerged) (6)

#### Evaluation
- [modelscope/evalscope](https://github.com/modelscope/evalscope/pulls?q=is%3Apr+author%3AflyingdoubleG+is%3Amerged): BIRD-SQL, TableBench, DS-1000, TAT-QA, FinQA
