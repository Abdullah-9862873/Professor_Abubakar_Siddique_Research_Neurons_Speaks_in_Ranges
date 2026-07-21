# Compositional Facts and NeuronLens

How does Llama 3.2 3B handle a two hop question like "the mother of the singer of Superstition is"? The model has to pull up one fact (Stevie Wonder sang Superstition), then another fact (his mother is X), and combine them on its own. No passage in the prompt, just what it already knows.

This extends the NeuronLens activation range method to see if the model's hidden states for these composed queries look different from the single facts that feed into them. Per neuron activation ranges at every layer, both attention and MLP outputs, measured separately instead of just looking at the merged residual stream.

Below is the actual path this took, in the order it happened, including the parts that didn't work, ending with the final, most-verified results.

---

## Where this started

Read A.B. Siddique's TMLR paper, *Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution* (Haider, Rizwan, Sajjad, Ju, Siddique), which introduces NeuronLens. Core idea: individual neurons in LLMs are polysemantic, they respond to more than one concept, but activations *conditioned on a single concept* form clean, near-Gaussian distributions with limited overlap within the same neuron. That lets you attribute a concept to a *range* of a neuron's activation spectrum, `AR(l, j, c) = [μ − τσ, μ + τσ]`, rather than the whole neuron, and intervene on just that range instead of masking the whole thing.

Their experiments cover single-hop concepts: sentiment, emotion, news topic, article category. Nothing in the paper tests whether a *composed* fact (one that requires combining two separately-stored facts) shows the same range structure, or something different. That gap is what this project is about: does a composed answer's activation range look distinct from the ranges of the individual facts that produced it?

Read further to figure out how to test this properly:

- **Meng, Bau, Andonian, Belinkov (2022), "Locating and Editing Factual Associations in GPT"** (ROME) — NeurIPS. Factual recall concentrates in mid-layer MLPs, specifically at the subject's last token position. This is why activations get pulled from the last token here, not averaged across the sequence.
- **Geva, Bastings, Filippova, Globerson (2023), "Dissecting Recall of Factual Associations in Auto-Regressive Language Models"** — EMNLP. Even a single fact isn't a one-component process: early MLP layers do subject enrichment, attention in middle layers does relation propagation, later attention heads do selective attribute extraction. This is why MLP and attention outputs get hooked and analyzed separately here, not the merged residual stream.
- **Yang, Gribovskaya, Kassner, Geva, Riedel (2024), "Do Large Language Models Latently Perform Multi-Hop Reasoning?"** — ACL. The bridge-entity two-hop paradigm the dataset here is built on ("the mother of the singer of X" style prompts, closed-book, no passage). They find latent composability varies enormously by relation type.
- **Yang, Kassner, Gribovskaya, Riedel, Geva (2025), SOCRATES** — Findings of ACL. A shortcut-filtered version of the same benchmark, built to exclude cases where a model could answer via head-to-answer co-occurrence instead of real composition. This project implements a cheap version of that idea (a decoy bridge-swap check), not their full filtering pipeline.

A paper describing this same latent-vs-explicit failure pattern came up as relevant during this work, but its exact citation was never independently verified, so it's deliberately left out of the reference list here rather than guessed at.

---

## Attempt 1: SQuAD 2.0 (abandoned)

The first real attempt used SQuAD 2.0 passages, hand-writing compositional questions over facts that happened to sit in the same paragraph.

Doesn't work, for a structural reason: SQuAD is single-passage extraction. The facts sit right there in the prompt. A "compositional" question built from one isn't testing whether the model combines facts from its own parameters, it's testing whether it can read two sentences in the same paragraph. There's also no way to control for the model just pattern-matching on passage text instead of actually reasoning.

This led directly to Yang et al.'s closed-book, bridge-entity paradigm instead, no passage, the model has to recall the bridge entity from its own weights.

---

## Building the dataset: Wikidata bridge entities

Two hand-checked examples got the shape right first (the Stevie Wonder one, straight from the Yang et al. paper, since it's independently verifiable). The rest came from Wikidata via SPARQL, chaining subject-relation-object triples into bridge-entity pairs:

- **Chain 1**: performer of a song (P175) → performer's mother (P25)
- **Chain 2**: author of a book (P50) → author's place of birth (P19)
- **Chain 3**: director of a film (P57) → director's country of citizenship (P27)
- **Chain 4**, added later: capital of a country (P36) → that country's official language (P37)

Each query filters by `wikibase:sitelinks` (how many Wikipedia language editions have an article on an entity) as a proxy for "a 3B model has probably seen this name during pretraining."

---

## Round 1: does the overlap even exist

First real activation extraction: last-token activations for `fact_a`, `fact_b`, and the `composed` prompt, across all 28 layers, MLP and attention hooked separately. NeuronLens's range formula applied per neuron: `AR = [μ − 2.5σ, μ + 2.5σ]`, then an overlap-fraction metric between ranges.

The overlap came back consistently high, roughly 0.86 to 0.99 across every layer and both components. Not the clean separation the initial framing of the question implied might be possible.

At this stage, causal masking, the shortcut check, and the length-confound check were all built but not yet actually executed.

---

## Round 2: the accuracy metric was broken

Wiring up causal masking, shortcut, and confound checks for real exposed a problem: the accuracy check (`top1_matches`) looked at only the model's single next predicted token. Too crude, can't handle multi-word names, and short common tokens pass by accident.

The tell: real-answer match rate and decoy-answer match rate (a deliberately wrong bridge entity's answer, swapped in) came back **identical**, 0.833 and 0.833. Not evidence the model can't tell facts apart, evidence the check was too loose to mean anything.

Fix: `generate_continuation()` actually generates several tokens, `answer_in_output()` checks whether the real answer's text appears anywhere in that continuation.

---

## Round 3: floor effect

Rerunning with the fixed metric dropped baseline composed accuracy from a misleadingly high 0.93 to essentially **0.0**, and fact_a accuracy from 0.24 to 0.05. Most of what an unfiltered Wikidata query pulls back is simply too obscure for a 3B model to know at all, so causal masking had nothing to disrupt, no signal there in the first place.

---

## Round 4: filtering to facts the model actually knows, first CoT test

Added a knowledge filter: for every candidate, check whether the model can independently answer `fact_a` and `fact_b` correctly on their own, before ever looking at the composed question. Only examples passing both survive into `known_examples`.

First pass: 39 out of 402. Too small to trust.

Widened the pool (lower sitelinks threshold, a third relation chain, proper statistics with raw counts plus a Wilson confidence interval) brought `known_examples` to 58. At that size:

- Baseline (latent) composed accuracy: **0/58 = 0.0%**, 95% CI [0.0, 0.062]
- CoT-prompted composed accuracy: **34/58 = 58.6%**, 95% CI [0.458, 0.704]

Non-overlapping confidence intervals. The chain-of-thought test exists specifically to separate two explanations for the 0% latent result: either the model can't compose these facts at all, or it can't do it *silently*, in one shot, but can if allowed to reason step by step.

---

## Round 5: a diagnostic detour, catching a real data bug

Ran a separate, lighter notebook (no activation hooks, just dataset construction and answering behavior) to pressure-test the dataset before trusting it further.

A spot check turned up a real problem: a song correctly identified as performed by Elvis Costello, paired with "Gladys Presley" as the expected mother, who is actually Elvis *Presley's* mother. Wikidata's label service falls back to the raw entity ID when an item lacks an English label, and this had produced a wrong pairing somewhere in the chain.

Fixes made in this round:
- Entity URIs kept alongside labels, so anything suspicious can be clicked and checked on wikidata.org directly.
- A bidirectional consistency constraint on the performer→mother chain (`P40`, requiring the mother's own page to independently list the performer as a child, not just the forward `P25` statement).
- Type-sanity constraints on the other chains (birthplace must be a city-type entity, country must be a sovereign state) — later removed in Round 7 for performance reasons, see below.
- A fourth chain, capital → official language, as a comparison point, since geographic facts are generally better known than person-level facts.

At n = 195, broken down by chain, the pattern sharpened:

| chain | n | latent accuracy | CoT accuracy |
|---|---|---|---|
| performer → mother | 77 | 0% | 81.8% |
| author → birthplace | 23 | 0% | 13.0% |
| director → citizenship | 14 | 0% | 35.7% |
| capital → language | 81 | 24.7% | 85.2% |

The three person-based chains landed at *exactly* 0% latent accuracy. The geography chain was a real outlier. Pooling everything into one number, the way the 58-example run had to, hid that difference.

---

## Round 6: merged into the main pipeline, two bugs found

Folded everything that worked in the diagnostic round back into the main notebook: the four chains with constraints, URIs, per-chain reporting, and a corrected spot check (an earlier version of it had accidentally run against a stale, near-empty example list due to an out-of-order cell execution, caught and fixed here).

Full run at n = 235 known examples surfaced two real problems:

**1. A generation-length inconsistency.** The baseline accuracy cell used the function's default of 16 tokens; the per-chain breakdown cell explicitly overrode it to 8. Same prompts, same model, very different accuracy depending purely on this, backing out the math, capital_language's latent accuracy could be read as anywhere from 13% to 77% depending only on token budget.

**2. `author_birthplace` returned zero known examples.** Not because the model doesn't know author birthplaces, a raw Wikidata QID ("Q112554") was leaking into a prompt in place of a real title, a symptom of the same missing-label problem from Round 5, this time affecting a different chain's data.

Fixes: standardized the token budget to 8 everywhere an answer gets checked, and added a QID-pattern guard (filtering out any row where a label looks like a raw entity ID) across all four chains, not just the one that broke.

---

## Round 7: infrastructure problems, unrelated to the research itself

Two environment issues surfaced and got fixed separately from the methodology:

- **Model downloads occasionally failed mid-transfer** (`ChunkedEncodingError` / `IncompleteRead` on the ~5GB model shard), a Colab network hiccup, not a version problem. Fixed with `hf_transfer` (a more resilient downloader) and a retry loop that resumes from where the download left off rather than restarting.
- **Wikidata queries started timing out.** `ORDER BY` combined with `LIMIT` forces the public endpoint to sort the entire filtered result set before returning anything, and transitive property paths (the type-check filters using `P279*`) are separately expensive. Both are classic causes of WDQS timeouts. Removed `ORDER BY` from all four queries and the transitive/type-check joins from chains 2 and 3, keeping the QID guards (implemented as a Python post-filter checking that no label starts with "Q", not a SPARQL `REGEX`), which were the fix that actually mattered for data quality.

---

## Round 8: removing ORDER BY had a real side effect

Rerunning after the timeout fix surfaced a new problem: `known_examples` collapsed from 235 to **20**. Chain breakdown: director_country 17, capital_language 2, author_birthplace 1, performer_mother 0.

Reason: `FILTER(?sitelinks > 25)` alone is a floor, not a ceiling. With `ORDER BY DESC(?sitelinks)` in place, the query always returned the most famous qualifying entities. Without it, the endpoint returns an arbitrary sample that merely clears a fairly low bar, most of which turn out to be moderately known, not famous enough for a 3B model to reliably know facts about them.

Fix: raised the sitelinks threshold on a per-chain basis to compensate for losing the sort, calibrated individually rather than with one global number:

| chain | threshold | reason |
|---|---|---|
| performer_mother | 25 → 150 | collapsed 100→0 known, needs household-name fame |
| capital_language | 25 → 100 | collapsed 105→2 known |
| author_birthplace | 25 → 40 | already data-starved, modest bump only |
| director_country | 25 (unchanged) | held up fine, left alone |

---

## Round 9: the final, most-verified run

This is the run every number below has been triple-checked against, cell by cell, in the actual executed notebook, not estimated or carried over from an earlier round.

**Dataset**: 1729 total examples (2 hand-written plus 1727 from Wikidata: performer_mother 495 raw, capital_language 489 raw, director_country 395 raw, author_birthplace 348 raw). Knowledge filter brought that down to **44 known examples**:

| chain | raw candidates | known (passed filter) |
|---|---|---|
| capital_language | 489 | 25 |
| director_country | 395 | 18 |
| author_birthplace | 348 | 1 |
| performer_mother | 495 | 0 |

`performer_mother` still produced zero known examples even at the raised threshold of 150. The most likely explanation is three constraints compounding at once: the entity has to be famous enough, the Wikidata `P40`/`P25` bidirectional consistency check has to actually hold (many mother entities lack a complete "child" statement even when the reverse exists), and the model has to specifically know that person's mother's name. None of these individually is rare, but requiring all three together may be enough to crush this chain's yield toward zero. Treated here as an open, honestly-reported limitation, not smoothed over.

**Gaussianity**, checked at three depths (early, mid, late layer) in both components, before trusting any range statistics:

| component | layer | skew | kurtosis |
|---|---|---|---|
| MLP | 0 | 0.170 | 2.374 |
| MLP | 14 | -0.050 | 2.484 |
| MLP | 27 | -0.008 | 2.687 |
| Attention | 0 | -0.114 | 2.430 |
| Attention | 14 | -0.072 | 2.554 |
| Attention | 27 | -0.035 | 3.069 |

Skew stayed close to 0 throughout, consistent with the original paper. Kurtosis ran a bit lower than the paper's reported 3.2–4.0 range (2.37 to 3.07 here), flatter-tailed than their numbers, but the same general ballpark, close enough to proceed.

**Overlap sweep**: across all 56 layer-component combinations (28 layers × 2 components), every single `fact_a`-vs-`composed` overlap value on the full 1727-example dataset fell between **0.924 and 0.990**. No layer or component showed the kind of separable, low-overlap range that a dedicated compositional circuit would predict. The known-only subset (44 examples) tracked the same pattern as the full dataset within a couple of percentage points at every layer, confirming the high overlap wasn't an artifact of including examples the model doesn't actually know.

**Causal masking**: the lowest-overlap point in the known-only sweep was attention layer 11, at 0.927. Masking that range during a composed-question forward pass:

| condition | composed accuracy | fact_a accuracy | 95% CI (composed) |
|---|---|---|---|
| baseline (no mask) | 18.2% (8/44) | 100% (44/44) | [0.095, 0.320] |
| masked, attn layer 11 | 31.8% (14/44) | 77.3% (34/44) | [0.200, 0.466] |

Composed accuracy went *up*, not down, after masking, and the confidence intervals overlap heavily. If this range were causally necessary for composition, masking should have hurt composed answers specifically. It didn't, consistent with every earlier round's masking result.

**Shortcut check**: on a 30-example slice, real-answer match rate was 16.7%, decoy-answer match rate (a deliberately wrong bridge entity's answer) was 0.0%. The model isn't randomly guessing, but it also isn't reliably composing.

**Chain-of-thought vs latent, pooled**: latent (silent) composed accuracy 18.2% (8/44) vs CoT-prompted 47.7% (21/44), 95% CI [0.338, 0.621] for the CoT figure. A large jump, but the pooled number hides the real story:

| chain | n | latent (silent) | chain-of-thought |
|---|---|---|---|
| capital_language | 25 | 32% (8/25) | 76% (19/25) |
| director_country | 18 | 0% (0/18) | 11.1% (2/18) |
| author_birthplace | 1 | 0% (0/1) | 0% (0/1) |

`capital_language` shows genuine silent composability that more than doubles with explicit reasoning, consistent with a retrieval bottleneck rather than a missing capability. `director_country` shows almost nothing even with reasoning scaffolding, despite the model knowing the individual facts, a different and harder failure mode. `author_birthplace` has only one example and isn't conclusive on its own. `performer_mother` couldn't be evaluated in this chain at all, zero known examples.

**Length confound**: overlap(fact_a, composed) = 0.890, overlap(fact_a, length-matched control) = 0.878. Close together, still consistent with prompt length accounting for some, though clearly not all, of the high-overlap pattern (the small residual gap of 0.012 is present but modest).

---

## What's actually solid, and what isn't

**Solid, held up across every round, every dataset version, every metric fix, from the first 402-example run through the final 1727-example one:**
- Overlap between single-fact and composed-query activation ranges is consistently high, never disjoint, at any layer, in any component, in any round.
- Causal masking has shown no clear effect in any round. Baseline and masked accuracy have overlapped within each other's confidence intervals every single time this has been tested, and in the final round masking even improved composed accuracy rather than hurting it.
- A latent-vs-CoT gap exists and is large, every round that included both measurements has shown CoT accuracy substantially higher than latent accuracy.
- Relation type matters. `capital_language` has behaved differently from the person-based chains in every round it's appeared in, with genuine latent composability that the person-based chains never show.

**Not fully solid:**
- Total known-example count (44) is thinner than the Round 6 peak (235), a direct, still not fully recovered consequence of fixing the Wikidata timeout problem in Round 7-8.
- `performer_mother` currently contributes zero known examples, likely from fame, Wikidata-consistency, and model-knowledge constraints compounding together, not yet resolved.
- `author_birthplace` has been thin in every round it's appeared (n = 1 to 23), never large enough to trust its specific percentage independently.
- The shortcut check remains a cheap approximation of SOCRATES' actual filtering method, not a full implementation of it.
- Everything here used a single 3B model, a single GPU, one run's configuration; the `answer_in_output` substring check can misfire on short or common tokens; causal masking tested only the single lowest-overlap layer, not cascading effects elsewhere in the network.

---

## Repo structure

```
compositional_neuronlens_standalone.ipynb   # main pipeline: dataset, extraction, range analysis, causal validation
diagnostic_dataset_quality.ipynb            # lighter notebook used to pressure-test the dataset before merging fixes back
information_context.txt                     # sourced literature notes, what's verified vs not
implementation_plan.md                      # original methodology plan
generate_research_summary.py                # generates the formatted PDF research summary from these same results
```

## Running it

Needs a GPU (a free Colab T4 is enough) and a Hugging Face account with access requested and approved for `meta-llama/Llama-3.2-3B-Instruct`. Everything else installs from the first cell. Run top to bottom; the knowledge-filtering and generation steps are the slow part.
