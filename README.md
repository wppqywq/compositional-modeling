## Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, please see [`README_original.md`](./README_original.md) in this repository.

### Extended modules (AST-based transmission chains)

The main notebook:
- `notebooks_new/notebook5_2.ipynb`: Communication system evolution (based on notebook3). Meanings are fixed from empirical trials; conventions and chunk usage evolve across generations.


For study purpose:
- `notebooks_new/4_transmission_chain_sim.ipynb`: Transmission chain simulation with abstract-syntax-tree programs, fragment learning, and information-bottleneck objectives. base on notebook2.

New modular components:
- `model/transmission/transmission_chain.py`: evolution chain runner (`run_comm_chain_bayes_rsa`). Also AST chain with IB loss in notebook4.
- `model/convention_formation/`: Notebook3 convention formation code (lexicon + priors + Bayesian update) reused by notebook5.
- `model/dsl/`: AST representation, token-AST parser, fragments and library.
- `model/program_induction/`: Fragment discovery via frequent subsequence mining.
