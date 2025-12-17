# Discussion and Future Directions

This project studies how communicative conventions and compositional abstractions stabilize under iterated cultural transmission.

In the current experiments (Notebook 5), underlying meanings (towers) are fixed across generations.
This design choice isolates changes in communicative abstractions from changes in semantics.
<!-- The system can be described as $p(z \mid x; \theta)$ under fixed $x$, where $z$ denotes program abstractions and $\theta$ denotes speaker and listener policies. -->

This controlled setting establishes a baseline, 
and allows us to analysis three distinct forces in communication: rational inference, interaction-level commitment, and population-level drift.

Drift-based policies test whether inference is necessary for convention formation, while pact-aware policies test whether inference alone is sufficient to account for the stability and inertia. Both mechanisms predict convergence in performance.

Base on this, a few future steps can be discussed.

### 1. Abstraction Structures Identifiability

A natural next step is to examine whether the learned abstraction structures themselves are identifiable under fixed meanings. 

While accuracy and compression may converge, we still can't answer whether different runs converge to the same abstraction system or to multiple equivalent but distinct solutions. 

Possible method: running same communication chain with different (i) random seeds (ii) parameters (iii) speaker's policies. Or, try other fragment discovery algorithms other than dream coder. Comparing the resulting abstraction libraries at a fixed generation (e.g., generation 50). Agreement across runs can be quantified using set-based overlap measures (e.g., Jaccard similarity over chunk sets) or representation-level similarity metrics that account for functional equivalence. 
<!-- todo -->

Goal: distinguishes behavioral convergence from representational convergence.


### 2. Varying Meanings

More general transmission settings can be explored in which meanings themselves are allowed to vary. 

Relaxing this constraint introduces additional degrees of freedom, and require a clearer baseline understanding of abstraction stability from direction 1. Otherwise, we may confuse semantic changes with the representation instability.

(must within known distrbution $p(z \mid x;\theta)$ under representatoin $z$, policy $\theta$ and fixed meanning $x$, one can compute $p(z_t \mid x_t;\theta)$, $x_t \sim p(x_{t-1})$)


### 3. Complex System for Social dynamics

Finally, the current task setting may limit the emergence of more complex population-level phenomena, such as early lock-in, persistent divergence, or failure modes in cultural dynamics. These are all possible circumstances to explore.

Both pact-aware and drift-based policies predict such phenomena under different conditions:

- Pact-aware policies predict strong path dependence and inertia following early ratification.
- Drift-based policies predict middle stage neutral turnover driven by frequency-dependent copying.

To add complexity, we can design richer task distributions, increased perceptual or communicative noise, or larger joint task sets that require agents to share and update priors across multiple communication chains. 
