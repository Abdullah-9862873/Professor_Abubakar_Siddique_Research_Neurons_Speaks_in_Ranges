# Compositional Facts + NeuronLens

This project investigates how Llama-3.2-3B-Instruct internally represents
compositional (multi-hop) factual knowledge. We extend the NeuronLens framework
(activation range attribution) to study whether the model's hidden states for a
two-hop query — e.g., "the mother of the singer of 'Superstition' is" — occupy
a distinct activation regime compared to the two single facts that compose it.

## Approach

- Extract per-neuron activation ranges (μ ± τσ, τ=2.5) for MLP and attention
  outputs across all 28 layers, for single facts and composed queries
- Measure overlap between single-fact ranges and composed-query ranges per
  layer and component (attention vs. MLP)
- Validate causally: zero out the identified activation range and check that
  composed-answer accuracy drops while single-fact accuracy is preserved
- Control for length confounds and shortcut behavior via decoy-bridge swaps
- Dataset is generated from Wikidata bridge-entity pairs (song→performer→mother,
  book→author→birthplace), filtered for entity notability

## Status

Pipeline complete — results are being finalized.
