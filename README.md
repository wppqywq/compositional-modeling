## Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, please see [`README_original.md`](./README_original.md) in this repository.

### Extended modules (AST-based transmission chains)

- `notebooks_new/4_transmission_chain_sim.ipynb`: Transmission chain simulation with AST-structured programs, fragment learning, and information-bottleneck objectives.

New modular components:
- `model/dsl/`: AST representation, token↔AST parser, fragments and library
- `model/program_induction/`: Fragment discovery via frequent subsequence mining
- `model/eval/`: Tree edit distance, IB-style loss functions
- `model/transmission/`: Multiple chain types (noise-only, token-selection, AST-based with IB loss)