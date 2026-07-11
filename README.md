# Compositional Facts and NeuronLens

How does Llama 3.2 3B handle a two hop question like "the mother of the singer of Superstition is"? The model has to pull up one fact (Stevie Wonder sang Superstition), then another fact (his mother is X), and combine them on its own. No passage in the prompt, just what it already knows.

We're extending the NeuronLens activation range method to see if the models hidden states for these composed queries look different from the single facts that feed into them. Per neuron activation ranges at every layer, both attention and MLP outputs, measured separately instead of just looking at the merged residual stream.

The dataset comes from Wikidata bridge entity pairs. Song to performer to mother. Book to author to birthplace. Filtered for well known entities so the 3B model actually knows who we're talking about.

We check overlap between single fact ranges and composed query ranges. Then validate causally zero out the range and see if composed accuracy drops while single fact accuracy stays put. Also running a length confound control and a decoy bridge swap test to catch shortcut behavior.

Pipeline is done, results are coming together now.
