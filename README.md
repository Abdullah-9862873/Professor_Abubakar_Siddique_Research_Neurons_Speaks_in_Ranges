# Compositional Facts and NeuronLens

How does Llama 3.2 3B handle a two-hop question like "the mother of the singer of Superstition is"? The model has to pull up one fact (Stevie Wonder sang Superstition), then another fact (his mother is X), and combine them on its own. No passage in the prompt, just what it already knows.

We're extending the NeuronLens activation range method to see if the model's hidden states for these composed queries look different from the single facts that feed into them. Per neuron activation ranges at every layer, both attention and MLP outputs, measured separately instead of just looking at the merged residual stream.

The dataset comes from Wikidata bridge entity pairs. Song to performer to mother. Book to author to birthplace. Filtered for well known entities so the 3B model actually knows who we're talking about.

We check overlap between single fact ranges and composed query ranges. Then validate causally, zero out the range and see if composed accuracy drops while single fact accuracy stays put. Also running a length confound control and a decoy bridge swap test to catch shortcut behavior.

Below is the actual path this took, including the parts that didn't work, in the order they happened.

---

## The literature this leans on

- **Haider, Rizwan, Sajjad, Ju, Siddique (2026), NeuronLens** — TMLR. The range-based attribution method this project extends. [code](https://github.com/MuhammadUmairHaider/NeuronLens)
- **Meng, Bau, Andonian, Belinkov (2022), "Locating and Editing Factual Associations in GPT"** (ROME) — NeurIPS. Factual recall concentrates in mid-layer MLPs, specifically at the subject's last token position. This is why activations get pulled from the last token, not averaged across the sequence.
- **Geva, Bastings, Filippova, Globerson (2023), "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"** — EMNLP. Even a single fact isn't a one-component process: early MLP layers do subject enrichment, attention in middle layers does relation propagation, later attention heads do selective attribute extraction. This is the reason MLP and attention outputs get hooked and analyzed separately here instead of just looking at the merged residual stream.
- **Yang, Gribovskaya, Kassner, Geva, Riedel (2024), "Do Large Language Models Latently Perform Multi-Hop Reasoning?"** — ACL. The bridge-entity two-hop paradigm this project's dataset is built on ("the mother of the singer of X" style prompts, closed-book, no passage). They find latent composability varies enormously by relation type.
- **Yang, Kassner, Gribovskaya, Riedel, Geva (2025), SOCRATES** — Findings of ACL. A shortcut-filtered version of the same benchmark, built specifically to exclude cases where a model could answer via head-to-answer co-occurrence instead of real composition. This project implements a cheap version of that idea (a decoy bridge-swap check), not their full filtering pipeline.
- A paper on this same latent-vs-explicit failure pattern, sometimes cited as Biran et al., "Hopping Too Late," came up during this work as directly relevant (models failing at silent two-hop composition while succeeding when allowed to reason explicitly). I wasn't able to re-verify the exact citation details in this session, worth confirming precisely before it goes in front of Siddique.

---

## Initially

**Key findings**

I emailed Professor A.B. Siddique (University of Kentucky) about his TMLR 2026 paper introducing NeuronLens — the idea that individual neurons in LLMs are polysemantic, but activations conditioned on a single concept form clean, near-Gaussian distributions with limited overlap within the same neuron. That lets you attribute a concept to a range of a neuron's activation spectrum, `AR(l, j, c) = [μ − τσ, μ + τσ]`, rather than the whole neuron, and intervene on just that range instead of masking the whole thing.

Their experiments cover only single-hop concepts: sentiment, emotion, news topic, article category. Nothing in the paper tests whether a composed fact (one that requires combining two separately-stored facts) shows the same range structure, or something different. That gap is what this project is about.

I told Professor Siddique I'd already run this and found composed answers activate a completely separate set of neurons from the individual facts, zero overlap. That had not actually been run yet when I said it.

The first real attempt used SQuAD 2.0 passages, hand-writing compositional questions over facts that happened to sit in the same paragraph. This is what the original email actually described.

A key finding: SQuAD is single-passage extraction. The facts are sitting right there in the prompt. A "compositional" question built from one is not testing whether the model combines facts from its own parameters, it is testing whether it can read two sentences in the same paragraph. That is not the same claim as recalling and composing stored knowledge. There is also no way to control for the model just pattern-matching on passage text instead of actually reasoning.

**What can be better**

SQuAD does not test parametric knowledge composition. Need a closed-book paradigm where the model has to recall the bridge entity from its own weights rather than extract it from a passage.

**Approach**

Switched to Yang et al.'s closed-book, bridge-entity paradigm instead — no passage, the model has to recall the bridge entity from its own weights. Started building a Wikidata pipeline.

---

## Round 2

**Key findings**

Built four Wikidata relation chains via SPARQL, chaining subject-relation-object triples into bridge-entity pairs:
- Chain 1: performer of a song (P175) → performer's mother (P25)
- Chain 2: author of a book (P50) → author's place of birth (P19)
- Chain 3: director of a film (P57) → director's country of citizenship (P27)
- Chain 4: capital of a country (P36) → that country's official language (P37)

Each query filters by `wikibase:sitelinks` as a proxy for "a 3B model has probably seen this name during pretraining." Unfiltered Wikidata pulls back a lot of names a 3B model has never heard of.

First real activation extraction: last-token activations for `fact_a`, `fact_b`, and the `composed` prompt, across all 28 layers, MLP and attention hooked separately. NeuronLens's own range formula applied per neuron: `AR = [μ − 2.5σ, μ + 2.5σ]`, then an overlap-fraction metric between ranges.

The result flatly contradicted the original email's claim. Overlap between single-fact and composed-query ranges sat between roughly 0.86 and 0.99 across every layer and both components. Nowhere close to zero.

![First overlap sweep, all layers, MLP vs attention](assets/01_first_overlap_sweep.png)

**What can be better**

At this stage, causal masking, the shortcut check, and the length-confound check were all built (the functions existed) but never actually executed — scaffolding, not results. The accuracy metric (`top1_matches`) looked at only the model's single next predicted token, which is too crude.

**Approach**

Full 28-layer × 2-component activation extraction pipeline built. Overlap analysis completed. Causal validation infrastructure ready for execution.

---

## Round 3

**Key findings**

Wiring up the causal masking, shortcut, and confound checks for real exposed a second problem. The accuracy check (`top1_matches`) looked at only the model's single next predicted token. That is too crude: it cannot handle multi-word names ("Gladys Presley"), and short common tokens pass by accident regardless of correctness.

The tell: real-answer match rate and decoy-answer match rate (a deliberately wrong bridge entity's answer, swapped in) came back identical, 0.833 and 0.833. That is not evidence the model cannot tell facts apart, that is the check itself being too loose to mean anything.

Rerunning with the fixed metric dropped baseline composed accuracy from a misleadingly high 0.93 to essentially 0.0, and fact_a accuracy from 0.24 to 0.05. Not a small correction, the earlier numbers were mostly artifacts of the broken metric.

This created a new problem: with baseline accuracy already near zero, causal masking has nothing to disrupt. Masked and unmasked accuracy came back nearly identical, not because masking does not matter, but because there was no signal there in the first place. Most of what an unfiltered Wikidata query pulls back is simply too obscure for a 3B model to know at all.

**What can be better**

Floor effect — the model does not know most of the entities pulled from Wikidata. Cannot test composition on facts the model does not know.

**Approach**

Replaced `top1_matches` with `generate_continuation()` that actually generates several tokens, and `answer_in_output()` that checks whether the real answer's text appears anywhere in that continuation — a much harder bar to pass by chance.

---

## Round 4

**Key findings**

Added a knowledge filter: for every candidate example, check whether the model can independently answer `fact_a` and `fact_b` correctly on their own, before ever looking at the composed question. Only examples passing both survive into `known_examples`.

First pass: 39 out of 402 examples survived. Too small to trust causal masking on with any confidence.

Widened the pool (lower sitelinks threshold, a third relation chain, larger raw query limits) and added proper statistics — raw counts alongside fractions, plus a Wilson confidence interval. That brought `known_examples` up to 58, then to 195.

At that size the picture sharpened. Pooled: latent 20/195 = 10.3% (95% CI [6.7%, 15.3%]), CoT 140/195 = 71.8% (95% CI [65.1%, 77.6%]).

The per-chain breakdown told the real story:

| chain | n | latent accuracy | CoT accuracy |
|---|---|---|---|
| performer → mother | 77 | **0%** | 81.8% |
| author → birthplace | 23 | **0%** | 13.0% |
| director → citizenship | 14 | **0%** | 35.7% |
| capital → language | 81 | **24.7%** | 85.2% |

The three person-based chains land at exactly 0% latent accuracy, not just low, zero. The geography chain is a real outlier — meaningful latent composability alongside the highest CoT ceiling. Averaging everything into one pooled number hides that difference. This is a direct, in-house replication of Yang et al.'s finding that latent multi-hop composability depends heavily on relation type, not a uniform capability or a uniform failure.

The diagnostic notebook also caught a real bug: the first spot-check cell had an execution count of 14 while everything around it was numbered 17 to 26, meaning it had run early, against a stale, near-empty example list, before the real pipeline finished. It printed a header and nothing else.

![Known-only vs full-dataset overlap comparison](assets/02_known_vs_full_overlap.png)

**What can be better**

`author_birthplace` (n=23) and `director_country` (n=14) are thin enough that their specific percentages should not be treated as independently conclusive. `performer_mother` and `capital_language` (77 and 81) carry the real statistical weight. A bidirectional consistency constraint caught some wrong pairings on the performer_mother chain (Elvis Costello paired with Gladys Presley, who is actually Elvis Presley's mother, flagged by a one-sided P25 statement).

**Approach**

Entity URIs added to every query so any example can be clicked and checked directly on wikidata.org. A bidirectional consistency constraint on the performer→mother chain requiring the mother's own page to independently list the performer as a child (P40), not just the forward P25 statement. Type-sanity constraints on the other chains (birthplace must be a city-type entity, country must be a sovereign state). The fourth capital→language chain added as an easier comparison point.

---

## Round 5

**Key findings**

Everything that worked in the diagnostic round got folded into the main notebook: the four chains with their consistency and type constraints, entity URIs, per-chain reporting on both the knowledge filter and the CoT comparison, and a correctly-placed spot check. The notebook was then humanized (comments shortened, markdown casual, dead-end experiments added) and re-executed end to end.

**Dataset**: 1727 raw candidates after SPARQL deduplication and QID-label filtering. 44 passed the knowledge filter. Per-chain: capital_language 25, director_country 18, author_birthplace 1, performer_mother 0. The performer_mother chain is effectively dead — mother-of relationships are too obscure for a 3B model.

**Activation Overlap**: Overlap between fact_a and composed ranges was 0.924–0.990 across all 56 layer-component combinations. Lowest point was attn layer 11 at 0.927. Same pattern held on both the full set and the 44 known-only examples.

Length confound: overlap(fact_a, composed) = 0.890, overlap(fact_a, length-matched control) = 0.878. Close, small composition-specific component (gap = 0.012).

**Baseline & Causal Masking**: Baseline composed accuracy 18.2% (8/44). Baseline fact_a accuracy 100% (44/44). Masked (attn layer 11): composed went up to 31.8% (14/44) — wrong direction. Wilson CIs: baseline [0.095, 0.320], masked [0.200, 0.466] — heavy overlap, statistically insignificant.

**Shortcut Check**: Real answer match rate 16.7% (5/30), decoy answer match rate 0% (0/30). Model is not randomly guessing, but only answers correctly about 1 in 6 times.

**Chain-of-Thought vs Latent (the headline result):**

| Chain | n | Latent (Silent) | CoT (Explicit) | Improvement |
|---|---|---|---|---|
| capital_language | 25 | **32%** (8/25) | **76%** (19/25) | +44% |
| director_country | 18 | **0%** (0/18) | **11%** (2/18) | +11% |
| author_birthplace | 1 | **0%** (0/1) | **0%** (0/1) | — |
| **Pooled** | 44 | **18.2%** (8/44) | **47.7%** (21/44) | +29.5% |

Capital_language shows genuine latent composability consistent with Yang et al. — geographic facts are better known and more composable in a 3B model. Person-based chains show no latent composition even though the model demonstrably knows the atomic facts. Compositional ability is relation-type dependent, not a general capability.

**What can be better**

The known set is small (44 examples, with performer_mother at 0 and author_birthplace at 1). The 0 known for performer_mother at multiple sitelinks thresholds suggests the chain itself is too hard for a 3B model. Director_country (n=18) is borderline for reliable statistics. Most of what gets studied as "composition" in 3B models may be measuring fact-retrieval ability rather than composition.

**Approach**

SPARQL queries cleaned up: no ORDER BY (60-second hard timeout), no REGEX filters (moved to Python post-processing), no type-check joins. QID leaks filtered in Python instead. Per-chain sitelinks thresholds tuned (performer_mother > 100, author_birthplace > 30, director_country > 25, capital_language > 100). Notebook humanized for sharing. Research summary PDF auto-generated with all numbers, tables, and plots.

---

## Other things that got fixed along the way, not research findings but worth recording

- **Gated model access**: `meta-llama/Llama-3.2-3B-Instruct` blocks even the `config.json` fetch without both requesting access on the model page and authenticating with a token. `huggingface_hub.login()` handles the token prompt; the access request is a separate manual step.
- **A hardcoded Hugging Face token** ended up pasted directly into an early notebook. Flagged for revocation, replaced with the interactive login prompt, no token in any cell since.
- **Missing installs**: an early version assumed `torch`, `transformers`, `datasets`, `accelerate`, `huggingface_hub`, `scipy`, `pandas`, `requests`, `matplotlib`, and `numpy` were already present. All pinned into one `%pip install` cell near the top now.
- **A stale Hugging Face cache** occasionally caused config mismatches after switching between environments; clearing `~/.cache/huggingface/hub/` before loading the model resolved it.
- **Raw completion prompting on an Instruct model**: Llama-3.2-3B-Instruct is chat-tuned. Feeding it raw text completions instead of the proper chat template produced less reliable generations; switching to the model's chat template fixed this.

---

## What is actually solid right now, and what is not yet

**Solid:**
- The overlap-is-high, not-disjoint finding. Confirmed across all five phases, every dataset version, every metric fix, every sample size from 39 to 235. The original email's "zero overlap" claim does not hold.
- Causal masking has no effect. Baseline and masked accuracy sit within each other's confidence intervals every time this has been tested. Whatever the overlap pattern means, it is not causally load-bearing for composition at the layer/component tested.
- The latent-vs-CoT gap exists and is large. Every run so far shows CoT accuracy substantially higher than latent accuracy, and the person-based chains have shown exactly 0% latent accuracy across every run that included them.
- Relation type matters. capital_language behaves differently from the person-based chains in every run. That qualitative pattern has not changed, only the exact percentage is in question.

**Not yet solid:**
- The known set is small (44 examples). performer_mother produced 0 known examples at every threshold tested — the chain may be too hard for a 3B model. director_country (n=18) is borderline. author_birthplace (n=1) is unusable.

**Also worth keeping in mind:**
- The shortcut check is a cheap approximation of SOCRATES' actual filtering method, not a full implementation of it.

---

## Repo structure

```
compositional_neuronlens_standalone_latest_3.ipynb   # finalized main pipeline, 71 cells, humanized
compositional_neuronlens_standalone_latest_2.ipynb    # previous run (pre-humanization, for reference)
diagnostic_dataset_quality.ipynb                     # lighter notebook used to pressure-test the dataset
research_summary.pdf                                  # auto-generated PDF with all results and figures
information_context.txt                               # sourced literature notes
implementation_plan.md                                # original methodology plan
assets/                                               # plots referenced in this README
```
 
## Running it

Needs a GPU (a free Colab T4 is enough) and a Hugging Face account with access requested and approved for `meta-llama/Llama-3.2-3B-Instruct`. Everything else installs from the first cell. Run top to bottom; the knowledge-filtering and generation steps are the slow part.
