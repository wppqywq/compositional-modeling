# Compositional Program Learning and Bayesian Reconstruction in Cultural Transmission

This project works as an exploration of cultural evolution in structured task domains using ideas from:

* Program induction / compositional representations (the COSMOS tutorial “Learning to communicate about shared procedural abstractions”).
* Information-theoretic compression (information bottleneck, complexity–accuracy trade-offs).
* Synergy and higher-order structure (partial information decomposition as a possible longer-term direction).

We are interested in questions about:

1. How agents can represent tasks as programs in a domain-specific language (DSL), discover reusable abstractions.
2. How these representations might change across generations in transmission.
3. How to connect this to Bayesian modeling of communication(as in the COSMOS notebbok3).

The notebook4 is a new integrated attempt, to address all three themes in a single set of models.
Below is a detailed intro of `/notebooks_new/4_transmission_chain.ipynb`.

## Models

1. Noise-only transmission chains(baseline): programs are passed down with random mutations. 
    
    Goal: Baseline without preference.

2. Selection-based chains (token-level IB): at each generation, multiple candidate noisy programs are generated and a selection step chooses the one that best trades off compression and sequence accuracy. 

    Quantify: 
    Complexity: expanded program length (number of primitive operations).
    Accuracy: token-level sequence accuracy vs. the ground-truth.

    Goal: approximating an information-bottleneck pressure; creates an empirical complexity–accuracy frontier

3. AST-based IB chains with fragment learning: programs are represented as abstract syntax trees (ASTs) in a compositional DSL; agents can discover and reuse fragments (sub-programs) and are subject to an IB-style loss defined on program complexity and structural accuracy.



4. Bayesian listener chains: agents receive noisy observations and use a Bayesian MAP parser. We combine a length-based prior and a similarity-based likelihood to reconstruct programs. This model introduces an explicit Bayesian reconstruction step while still allowing compression and abstraction to emerge.

This progression is designed to make visible, in a controlled way, how adding structure (DSL, ASTs, fragments) and Bayesian inference changes the dynamics of cultural transmission.

## Experiment Design

### Block-building domain

All models operate on a simple block-building domain inspired by the COSMOS tutorial. A “scene” is a tower-like arrangement of blocks on a discrete grid. The ground-truth scene is encoded as a program in a small DSL:

* Horizontal moves, vertical moves, and right/left shifts.
* Block placements at specified locations.
* Simple repetition operators (e.g., `*2` for repeated moves).

In the fourth notebook, we designed a longer structured target program (~40 tokens) that contains obvious repeated motifs. This makes it possible to study whether the models can:

* Compress.
* Discover recurring fragments.
* Maintain structural accuracy.

### Program representations

Programs at two levels:

* Token strings, as in the original tutorial, for backward compatibility and visualization.
* Abstract syntax trees (ASTs), with nodes representing DSL primitives and composition operators.
For the reason of  fragment discovery and for defining structural similarity metrics between programs.

---

## Implement Details

### AST-based IB chains with fragment learning

Idea: Introduce compositional structure and abstraction explicitly. Programs are represented as ASTs, and agents can:

* Detect frequently occurring subtrees.
* Promote them to a fragment library.
* Replace repeated subtrees with fragment calls, reducing description length.

Agents are guided by an IB-style loss:

* Complexity: description length of the AST under the current fragment library.
* Accuracy: structural similarity to the ground-truth AST (1 – normalized tree-edit distance).

Outcomes in notebook 4:

* Fragment usage (fraction of CALL nodes) increases across generations and across chains, showing that agents learn to reuse subprograms.
* Complexity drops substantially as fragments are adopted.
* Accuracy declines more than in the Bayesian listener model, making this model a useful “strong compression” contrast.

This model demonstrates that, given only IB pressure and structural noise, the system learns real abstractions but may over-compress, sacrificing accuracy.

### Bayesian listener chains

Idea: Add an explicit Bayesian reconstruction step, closer in spirit to the COSMOS work. Each generation:

1. Receives a noisy token sequence (the observation).
2. Generates a small candidate set of ASTs via local transformations (including fragment-based rewrites).
3. Evaluates each candidate with a prior and likelihood:

   * Prior: prefers shorter programs and higher fragment reuse (compression bias).
   * Likelihood: measures similarity between the candidate program and the observed token sequence (sequence and/or structural similarity).
4. Selects the MAP program (maximum a posteriori) and passes it to the next generation.

Key design choice: The ground-truth program is not fixed across generations, unlike in the COSMOS speaker–listener models. Instead, agents continually reconstruct and possibly transform the program, creating a genuine program-level cultural evolution process.

---

## Results from the Bayesian listener model

### 5.1 Single-chain dynamics

In a typical Bayesian listener chain:

* Initial program complexity is high (around the original 40+ tokens).
* Over the first several generations, complexity decreases quickly as the listener adopts shorter programs that still explain the noisy observations well.
* Fragment usage rises from near zero to a substantial fraction (often 0.4–0.7 CALL nodes in the AST), indicating that the listener is actively exploiting the fragment library to compress programs.
* Structural accuracy (as measured by the IB-style accuracy metric) decreases from near 1.0 but stabilizes at an intermediate value (roughly 0.5–0.6), rather than collapsing to zero.

This pattern supports the following interpretation:

* The Bayesian listener is not simply collapsing to trivial programs; instead, it is finding compressed but still structurally informative reconstructions.
* Fragment learning and Bayesian reconstruction interact to produce stable abstract programs that retain substantial information about the original scene.

### 5.2 Multi-chain patterns and IB frontier

Across multiple independent chains with different random seeds:

* We obtain a cloud of (complexity, accuracy) points colored by generation, analogous to the IB scatter plots from the token-level models.
* The empirical frontier (maximum accuracy at each complexity level) shows that Bayesian reconstruction yields efficient programs—they lie near the frontier, combining relatively high accuracy with substantial compression.
* Fragment usage, averaged across chains, increases with generation and then plateaus, indicating a robust trend toward abstraction.

Compared to the pure AST-based IB model without Bayesian reconstruction:

* The Bayesian listener model maintains higher accuracy at similar complexity levels.
* Fragment usage remains high but is less associated with catastrophic loss of structure.

This aligns more closely with the COSMOS intuition that communication can become more abstract while preserving shared procedural content.

---

## Goal

1. Program induction and compositional representations

   * Programs are represented in a compositional DSL as ASTs.
   * Fragment discovery algorithms identify reusable subprograms, and the fragment library is explicitly tracked.
   * The Bayesian listener uses these abstractions in reconstruction, mirroring how humans might reuse learned subroutines.

2. Cultural evolution and transmission chains

   * All models are formulated as multi-generation chains, making explicit how representations drift or stabilize over time.
   * The Bayesian listener model shows how a population of agents can converge on shorter, more abstract programs while still preserving functional structure.
   * The contrast with the noise-only and IB-only models highlights the specific contribution of structured inference.

3. Information-theoretic compression (information bottleneck)

   * Complexity and accuracy are measured in every model, and empirical frontiers are plotted.
   * The AST-based IB and Bayesian listener models make the compression–fidelity trade-off explicit via their losses and priors.
   * The results show regimes where abstraction improves efficiency without fully destroying performance.

4. Bayesian modeling of communication

   * The Bayesian listener is a first step toward a full Bayesian communication model in this domain.
   * Although there is not yet an explicit pragmatic speaker or lexicon uncertainty, the model already captures a core aspect of COSMOS: recovering structured programs from noisy or underspecified observations under priors favoring simplicity.
   * This provides a natural foundation for future work adding a speaker model and a small artificial lexicon.

---

## 7. Limitations and next steps

The current notebook is deliberately minimal in several respects:

* Listener-only Bayesian model: There is no explicit speaker model or pragmatic reasoning. Agents do not choose utterances; they only interpret them. Extending this to a simple Rational Speech Act (RSA) speaker–listener pair would move the model closer to the COSMOS experiments on communication.
* Observation model: The “utterance” is currently the noisy program string itself, rather than a separate linguistic signal. A natural extension would be to introduce a small artificial lexicon that names fragments, bringing the model closer to the COSMOS lexicon-learning setting.
* Human data: The current results are purely synthetic. A future step would be to connect this to behavioral data from human participants performing similar block-building or communication tasks.

Despite these limitations, the current implementation already provides a coherent, end-to-end demonstration that:

* Compositional program representations and fragment learning can be embedded within a cultural transmission framework.
* Information-theoretic pressures and Bayesian reconstruction jointly shape the evolution of these representations.
* Different modeling assumptions (pure noise, selection-only, IB-only, Bayesian listener) lead to qualitatively different cultural dynamics that can be compared and interpreted.

