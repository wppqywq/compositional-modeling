## Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, please see [`README_original.md`](./README_original.md) in this repository.

### Extended modules (AST-based transmission chains)

The main notebook:
- `notebooks_new/5_communication_chain.ipynb.ipynb`: Communication system evolution (based on 3 agents from notebook3). 
Meanings are fixed from empirical trials; conventions and chunk usage evolve across generations.


New modular components:
- `model/transmission/transmission_chain.py`: evolution chain runner (`run_comm_chain_bayes_rsa`).
- `model/convention_formation/`: Notebook3 convention formation code (lexicon + priors + Bayesian update) reused by notebook5.

Only for study purpose:
- `notebooks_new/4_transmission_chain_sim.ipynb`: Transmission chain simulation with fragment learning, and IB objectives. base on notebook2. No utterance, no fixed meaning, just program drift.
