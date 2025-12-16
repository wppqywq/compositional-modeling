## Discussion and Future Directions

In notebook 5, we set the underlying meanings (tower programs) to be fixed, allowing us to isolate how compositional abstractions evolve under iterated cultural transmission. 

This restriction is deliberate: when meanings are allowed to drift simultaneously with representations, it becomes difficult to attribute changes in abstraction to either transmission dynamics or semantic change.

On this basis, a few future directions can be discussed:

### 1. abstraction structures

A natural next step is to examine whether the learned abstraction structures themselves are identifiable under fixed meanings. 

While accuracy and compression may converge, we still can't answer whether different runs converge to the same abstraction system or to multiple equivalent but distinct solutions. 

Possible method: running the communication chain multiple times with different random seeds and comparing the resulting abstraction libraries at a fixed generation (e.g., generation 50). Agreement across runs can be quantified using set-based overlap measures (e.g., Jaccard similarity over chunk sets) or representation-level similarity metrics that account for functional equivalence.

### 2. Varying Meanings

More general transmission settings can be explored in which meanings themselves are allowed to vary. 

Relaxing this constraint introduces additional degrees of freedom that require a clearer baseline understanding of abstraction stability before meaningful results. (have to verify $p(z \mid x;\theta)$ under $\theta$ and fixe $x$, before computing $p(z_t \mid x_t;\theta)$, $x_t \sim p(x_{t-1})$ )

Validate identifiability under fixed meanings (direction 1) is a necessary foundation for such extensions.


## 3. Complex System for culture dynamics

Finally, the current task setting may limit the emergence of more complex population-level phenomena, such as early lock-in, persistent divergence, or failure modes in cultural dynamics. These are all possible circumstances to explore.

To add complexity, we can design richer task distributions, increased perceptual or communicative noise, or larger joint task sets that require agents to share and update priors across multiple communication chains. 
