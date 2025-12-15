# Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, please see [`README_original.md`](./README_original.md) in this repository.

### Extended modules

Main notebooks(Clone this repo and execute directly.):
- [`notebooks_new/5_communication_chain.ipynb`](notebooks_new/5_communication_chain.ipynb): Communication system evolution (based on agents from notebook3). Meanings are fixed from empirical trials; conventions and chunk usage evolve across generations.
- [`notebooks_new/6_social_conventions.ipynb`](notebooks_new/6_social_conventions.ipynb): The final generation of the transmission chain (e.g. Generation 50) is evaluated on a new set of held-out trials, without further learning or chunk promotion (frozen evaluation).

New modular components:
- `model/transmission/transmission_chain.py`: evolution chain runner (`run_comm_chain_bayes_rsa`).
- `model/convention_formation/`: Notebook3 convention formation code (lexicon + priors + Bayesian update) reused by notebook5.

---

## Review of Notebook 5 & 6

Build an iterated transmission-chain setting, to study how compositional abstractions and lexical conventions stabilize under repeated cultural transmission.

**Future Directions**: Please check [`future_works.md`](future_works.md).

### Settings

Meaning:
- Meanings are fixed across generation
- Means represent as tower-building programs in DSL, in multiple program-level(fully decompositions -> reuseable chunks).
> Assume: abstraction arises from choosing among alternative descriptions of the same underlying structure, not new meanings.

---

Communication:
- Speakers choose: (i) program representation to express; (ii) utterance to use for each step.
- Softmax balance: (i) expected listener success, (ii) message length / chunk usage (cost)

---

transmission chain:
- At each generation: one speaker–listener pair communicates over a fixed set of trials (e.g. 12 towers in `data/model/programs_for_you/programs_ppt_1.json`).
- Each trial produces a set of steps(programs), (msg_len=#programs).
- Each step produces (utterance, intention, listener response) observations.
- End of each generation: both agents update their lexicon beliefs from observation (iterated learning)

> Question: Does too fast abstraction cause early lock-in?
>
>Here we specify abstract chunks must earn their sharedness through communicative use, to distinguish from individual-level interaction to social conventions. Result in Notebook5 shows that adding this gate do slow down the processe but no effect on the final communicative competence.

---

Generalization:

freeze the final lexicon posterior and active chunks of agent learned for 50 generations, evaluate on held-out participants (same data distribution, unseen trials/programs per participant)
- Track entropy during training (program-choice entropy; lexeme-mapping posterior entropy) to evaluate stababilization.

Details of metrics please refer to the notebooks.

