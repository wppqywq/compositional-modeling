## Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, please see [`README_original.md`](./README_original.md) in this repository.

### Extended modules

The main notebook:
- [`notebooks_new/5_communication_chain.ipynb`](notebooks_new/5_communication_chain.ipynb): Communication system evolution (based on 3 agents from notebook3). 
Meanings are fixed from empirical trials; conventions and chunk usage evolve across generations.


New modular components:
- `model/transmission/transmission_chain.py`: evolution chain runner (`run_comm_chain_bayes_rsa`).
- `model/convention_formation/`: Notebook3 convention formation code (lexicon + priors + Bayesian update) reused by notebook5.

Only for study purpose, no specific hypothesis:
- ~~[`notebooks_new/4_transmission_chain_sim.ipynb`](notebooks_new/4_transmission_chain_sim.ipynb): Transmission chain simulation with fragment learning, and IB objectives. base on notebook2. No utterance, no fixed meaning, just program drift.~~

---

# Review of Notebook 5

### Goal
Question: compositional representations stabilize through repeated transmission under constraints. Does shared abstraction really require population-level agreement, or is local success sufficient?

### Settings

Meaning:
- Meanings are fixed across generation
- Means represent as tower-building programs in DSL, in multiple program-level(fully decompositions -> reuseable chunks).
> Assume: abstraction arises from choosing among alternative descriptions of the same underlying structure, not new meanings.

Communication:
- Speakers choose: (i) program representation to express; (ii) utterance to use for each step.
- Softmax balance: (i) expected listener success, (ii) message length / chunk usage (cost)

transmission chain:
- At each generation: one speaker–listener pair communicates over a fixed set of trials (e.g. 12 towers in `data/model/programs_for_you/programs_ppt_1.json`).
- Each trial produces a set of steps(programs), (msg_len=#programs).
- Each step produces (utterance, intention, listener response) observations.
- End of each generation: both agents update their lexicon beliefs from observation (iterated learning)

In the original paper, abstract fragments are assumed to be shared components of the DSL. Here we specify abstract chunks must earn their sharedness through communicative use.

The aim is to distinguish from individual-level interaction to social conventions.

Gated Promotion Rule:
Candidate abstraction can becom shared chunk only if they:
- be used often
- achieve a minimum listener success rate at step level
- per generation cap is not reached
