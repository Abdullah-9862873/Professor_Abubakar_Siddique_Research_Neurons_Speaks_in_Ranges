# Compositional Facts and NeuronLens

How does Llama 3.2 3B handle a two hop question like "the mother of the singer of Superstition is"? The model has to pull up one fact (Stevie Wonder sang Superstition), then another fact (his mother is X), and combine them on its own. No passage in the prompt, just what it already knows.

We're extending the NeuronLens activation range method to see if the model's hidden states for these composed queries look different from the single facts that feed into them. Per neuron activation ranges at every layer, both attention and MLP outputs, measured separately instead of just looking at the merged residual stream.

The dataset comes from Wikidata bridge entity pairs. Song to performer to mother. Book to author to birthplace. Filtered for well known entities so the 3B model actually knows who we're talking about.

We check overlap between single fact ranges and composed query ranges. Then validate causally, zero out the range and see if composed accuracy drops while single fact accuracy stays put. Also running a length confound control and a decoy bridge swap test to catch shortcut behavior.

Below is the actual path this took, including the parts that didn't work, in the order they happened.

---

## Why this project exists

I emailed Professor A.B. Siddique (University of Kentucky) about his TMLR 2026 paper, *Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution* (Haider, Rizwan, Sajjad, Ju, Siddique), which introduces NeuronLens. The core idea of that paper: individual neurons in LLMs are polysemantic, they respond to more than one concept, but activations *conditioned on a single concept* form clean, near-Gaussian distributions with limited overlap within the same neuron. That lets you attribute a concept to a *range* of a neuron's activation spectrum, `AR(l, j, c) = [μ − τσ, μ + τσ]`, rather than the whole neuron, and intervene on just that range instead of masking the whole thing.

Their experiments cover single-hop concepts: sentiment, emotion, news topic, article category. Nothing in the paper tests whether a *composed* fact (one that requires combining two separately-stored facts) shows the same range structure, or something different. That gap is what this project is about.

I told Professor Siddique I'd already run this and found composed answers activate a completely separate set of neurons from the individual facts, zero overlap. That hadn't actually been run yet when I said it. Everything below is the process of making it real.

He replied with three specific questions: was the analysis done at a single-instance or full-dataset level, which layer, and which component's neurons. Those questions shaped the whole methodology, not just the eventual reply.

---

## The literature this leans on

- **Haider, Rizwan, Sajjad, Ju, Siddique (2026), NeuronLens** — TMLR. The range-based attribution method this project extends. [code](https://github.com/MuhammadUmairHaider/NeuronLens)
- **Meng, Bau, Andonian, Belinkov (2022), "Locating and Editing Factual Associations in GPT"** (ROME) — NeurIPS. Factual recall concentrates in mid-layer MLPs, specifically at the subject's last token position. This is why activations get pulled from the last token, not averaged across the sequence.
- **Geva, Bastings, Filippova, Globerson (2023), "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"** — EMNLP. Even a single fact isn't a one-component process: early MLP layers do subject enrichment, attention in middle layers does relation propagation, later attention heads do selective attribute extraction. This is the reason MLP and attention outputs get hooked and analyzed separately here instead of just looking at the merged residual stream.
- **Yang, Gribovskaya, Kassner, Geva, Riedel (2024), "Do Large Language Models Latently Perform Multi-Hop Reasoning?"** — ACL. The bridge-entity two-hop paradigm this project's dataset is built on ("the mother of the singer of X" style prompts, closed-book, no passage). They find latent composability varies enormously by relation type.
- **Yang, Kassner, Gribovskaya, Riedel, Geva (2025), SOCRATES** — Findings of ACL. A shortcut-filtered version of the same benchmark, built specifically to exclude cases where a model could answer via head-to-answer co-occurrence instead of real composition. This project implements a cheap version of that idea (a decoy bridge-swap check), not their full filtering pipeline.
- A paper on this same latent-vs-explicit failure pattern, sometimes cited as Biran et al., "Hopping Too Late," came up during this work as directly relevant (models failing at silent two-hop composition while succeeding when allowed to reason explicitly). I wasn't able to re-verify the exact citation details in this session, worth confirming precisely before it goes in front of Siddique.

---

## Attempt 1: SQuAD 2.0 (abandoned)

The first real attempt used SQuAD 2.0 passages, hand-writing compositional questions over facts that happened to sit in the same paragraph. This is what the original email actually described.

It doesn't work, for a structural reason: SQuAD is single-passage extraction. The facts are sitting right there in the prompt. A "compositional" question built from one isn't testing whether the model combines facts from its own parameters, it's testing whether it can read two sentences in the same paragraph. That's not the same claim as recalling and composing stored knowledge. There's also no way to control for the model just pattern-matching on passage text instead of actually reasoning.

This is what led to Yang et al.'s closed-book, bridge-entity paradigm instead, no passage, the model has to recall the bridge entity from its own weights.

---

## Building the real dataset: Wikidata bridge entities

Two hand-checked examples got the shape right first (the Stevie Wonder one, straight from the Yang et al. paper, since it's independently verifiable). Two examples isn't a dataset, so the rest came from Wikidata via SPARQL, chaining subject-relation-object triples into bridge-entity pairs:

- **Chain 1**: performer of a song (P175) → performer's mother (P25)
- **Chain 2**: author of a book (P50) → author's place of birth (P19)
- **Chain 3**: director of a film (P57) → director's country of citizenship (P27)
- **Chain 4**, added later: capital of a country (P36) → that country's official language (P37)

Each query filters and sorts by `wikibase:sitelinks`, how many Wikipedia language editions have an article on an entity, as a proxy for "a 3B model has probably seen this name during pretraining." Unfiltered Wikidata pulls back a lot of names a 3B model has never heard of.

---

## Round 1: does the overlap even exist

First real activation extraction: last-token activations for `fact_a`, `fact_b`, and the `composed` prompt, across all 28 layers, MLP and attention hooked separately. NeuronLens's own range formula applied per neuron: `AR = [μ − 2.5σ, μ + 2.5σ]`, then an overlap-fraction metric between ranges.

The result flatly contradicted the original email's claim. Overlap between single-fact and composed-query ranges sat between roughly 0.86 and 0.99 across every layer and both components. Nowhere close to zero.

![First overlap sweep, all layers, MLP vs attention](assets/01_first_overlap_sweep.png)

At this stage, causal masking, the shortcut check, and the length-confound check were all built (the functions existed) but never actually executed, scaffolding, not results.

---

## Round 2: the accuracy metric was broken

Wiring up the causal masking, shortcut, and confound checks for real exposed a second problem. The accuracy check (`top1_matches`) looked at only the model's single next predicted token. That's too crude: it can't handle multi-word names ("Gladys Presley"), and short common tokens pass by accident regardless of correctness.

The tell: real-answer match rate and decoy-answer match rate (a deliberately wrong bridge entity's answer, swapped in) came back **identical**, 0.833 and 0.833. That's not evidence the model can't tell facts apart, that's the check itself being too loose to mean anything.

Fix: `generate_continuation()` actually generates several tokens, and `answer_in_output()` checks whether the real answer's text appears anywhere in that continuation, a much harder bar to pass by chance.

---

## Round 3: floor effect

Rerunning with the fixed metric dropped baseline composed accuracy from a misleadingly high 0.93 to essentially **0.0**, and fact_a accuracy from 0.24 to 0.05. Not a small correction, the earlier numbers were mostly artifacts of the broken metric.

But this created a new problem: with baseline accuracy already near zero, causal masking has nothing to disrupt. Masked and unmasked accuracy came back nearly identical, not because masking doesn't matter, but because there was no signal there in the first place. Most of what an unfiltered Wikidata query pulls back is simply too obscure for a 3B model to know at all.

---

## Round 4: filtering to facts the model actually knows

Added a knowledge filter: for every candidate example, check whether the model can independently answer `fact_a` and `fact_b` correctly on their own, *before* ever looking at the composed question. Only examples passing both survive into `known_examples`.

First pass: 39 out of 402 examples survived. Too small to trust causal masking on with any confidence.

Widened the pool (lower sitelinks threshold, a third relation chain, larger raw query limits) and added proper statistics, raw counts alongside fractions, plus a Wilson confidence interval so "0%" is a defensible claim rather than a number that might just be small-sample noise. That brought `known_examples` up to 58.

At that size, the picture sharpened:

- Baseline (latent) composed accuracy: **0/58 = 0.0%**, 95% CI [0.0, 0.062]
- CoT-prompted composed accuracy: **34/58 = 58.6%**, 95% CI [0.458, 0.704]

Non-overlapping confidence intervals. The chain-of-thought test exists specifically to separate two explanations for the 0% latent result: either the model can't compose these facts at all, or it can't do it *silently*, in one shot, but can if allowed to reason step by step. That distinction is exactly what Yang et al. and the "Hopping Too Late" line of work are about.

![Known-only vs full-dataset overlap comparison](assets/02_known_vs_full_overlap.png)

---

## Round 5: the diagnostic notebook, checking whether any of this was noise or bad data

Two things needed checking before trusting the 0% → 58.6% result: whether the Wikidata facts themselves were clean, and whether widening the pool further would change the picture. Ran this as a separate, lighter notebook (no activation hooks, no 28-layer extraction, just dataset construction and the model's actual answering behavior) so it would iterate fast.

**What it added:**

- **Entity URIs, not just labels**, in every query, so any example that looks wrong can be clicked and checked directly on wikidata.org.
- **A bidirectional consistency constraint** on the performer→mother chain: requiring the mother's own page to independently list the performer as a child (`P40`), not just the forward `P25` statement. This exists because a spot check turned up a real mismatch, a song correctly identified as performed by Elvis Costello, paired with "Gladys Presley" as the expected mother, who is actually Elvis *Presley's* mother. A one-sided statement produced a wrong pairing; requiring both directions to agree filters that out.
- **Type-sanity constraints** on the other chains (birthplace must actually be a city-type entity, country must actually be a sovereign state).
- **A fourth, "easier" relation chain**: capital of a country → official language. Geographic facts are generally much better known than person-level facts, useful as a comparison point.

One real bug caught in this round: the first spot-check cell had an execution count of 14 while everything around it was numbered 17 to 26, meaning it had run early, against a stale, near-empty example list, before the real pipeline finished. It printed a header and nothing else. Worth calling out because it's an easy mistake to miss, a cell can run without erroring and still be checking the wrong thing.

**Results, at n = 195 (up from 58), broken down by chain:**

| chain | n | latent accuracy | CoT accuracy |
|---|---|---|---|
| performer → mother | 77 | **0%** | 81.8% |
| author → birthplace | 23 | **0%** | 13.0% |
| director → citizenship | 14 | **0%** | 35.7% |
| capital → language | 81 | **24.7%** | 85.2% |

Pooled: latent 20/195 = 10.3% (95% CI [6.7%, 15.3%]), CoT 140/195 = 71.8% (95% CI [65.1%, 77.6%]).

This is the actual headline finding. The three person-based chains land at *exactly* 0% latent accuracy, not just low, zero. The geography chain is a real outlier, meaningful latent composability alongside the highest CoT ceiling. Averaging everything into one pooled number, the way the 58-example run had to, hides that difference. This is a direct, in-house replication of Yang et al.'s finding that latent multi-hop composability depends heavily on relation type, not a uniform capability or a uniform failure.

`author_birthplace` (n=23) and `director_country` (n=14) are thin enough that their specific percentages shouldn't be treated as independently conclusive, they're directionally consistent with the other two person-based chains, not proof on their own. `performer_mother` and `capital_language` (77 and 81) carry the real statistical weight here.

---

## Merging back into the main notebook

Everything that worked in the diagnostic round got folded into `compositional_neuronlens_standalone.ipynb`: the four chains with their consistency and type constraints, entity URIs, per-chain reporting on both the knowledge filter and the CoT comparison, and a correctly-placed spot check (fixing the stale-execution bug). The overlap analysis, causal masking, shortcut check, and confound check all carry over automatically onto the larger, better-characterized dataset, they weren't rebuilt, just fed better data.

As of this writing, the merged notebook hasn't been executed end to end yet, the 195-example, 4-chain version of the full activation-extraction and plotting pipeline is still pending a run. The results section in the notebook is a template for exactly that reason.

---

## Other things that got fixed along the way, not research findings but worth recording

- **Gated model access**: `meta-llama/Llama-3.2-3B-Instruct` blocks even the `config.json` fetch without both requesting access on the model page and authenticating with a token. `huggingface_hub.login()` handles the token prompt; the access request is a separate manual step.
- **A hardcoded Hugging Face token** ended up pasted directly into an early notebook. Flagged for revocation, replaced with the interactive login prompt, no token in any cell since.
- **Missing installs**: an early version assumed `torch`, `transformers`, `datasets`, `accelerate`, `huggingface_hub`, `scipy`, `pandas`, `requests`, `matplotlib`, and `numpy` were already present. All pinned into one `%pip install` cell near the top now.
- **A stale Hugging Face cache** occasionally caused config mismatches after switching between environments; clearing `~/.cache/huggingface/hub/` before loading the model resolved it.
- **Raw completion prompting on an Instruct model**: Llama-3.2-3B-Instruct is chat-tuned. Feeding it raw text completions instead of the proper chat template produced less reliable generations; switching to the model's chat template fixed this.

---

## What's actually solid right now, and what isn't yet

**Solid:**
- The overlap-is-high, not-disjoint finding. Confirmed across every run, every dataset version, every metric fix. The original email's "zero overlap" claim does not hold.
- The latent-vs-CoT gap for person-based relations. 0% latent accuracy across three independent chains (n=77, 23, 14), each jumping substantially with explicit reasoning.
- The relation-type dependence. Geography composes better than person-facts, latently, matching the literature.

**Not yet solid:**
- The 195-example, 4-chain dataset hasn't had the full range/causal/shortcut/confound pipeline run on it yet, only the lighter diagnostic version.
- `author_birthplace` and `director_country` are individually too thin to lean on hard.
- The shortcut check is a cheap approximation of SOCRATES' actual filtering method, not a full implementation of it.

---

## Running it

Needs a GPU (a free Colab T4 is enough) and a Hugging Face account with access requested and approved for `meta-llama/Llama-3.2-3B-Instruct`. Everything else installs from the first cell. Run top to bottom; the knowledge-filtering and generation steps are the slow part.