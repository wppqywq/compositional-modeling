# Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, see [`README_original.md`](./README_original.md).

**Update 2025-12-27**: Notebook 5 and 6 chains use observation-only cultural transmission (no posterior inheritance) with adaptive prior and likelihood weighting.

---

## Extended Modules

### Notebook 5: Communication Chains

Iterated transmission chain studying how program abstractions and lexical conventions stabilize through cultural transmission.

- **Notebook**: [`notebooks_new/5_communication_chains.ipynb`](notebooks_new/5_communication_chains.ipynb)
- **Documentation**: [`5_chain.md`](5_chain.md)


### Notebook 6: Speaker Policies

Two alternative speaker policies for program selection, testing different hypotheses about convention formation.

A drift-based policy provides a minimal baseline to reduce inference, while a pact-aware policy models history-sensitive commitments inspired by conceptual pacts in human dialogue. One reduces complexity, while the other increases.

- **Notebook**: [`notebooks_new/6_speakers_policy.ipynb`](notebooks_new/6_speakers_policy.ipynb)
- **Documentation**: [`6_speaker_policies.md`](6_speaker_policies.md)
- Additional Module: [`model/speaker_policies.py`](model/speaker_policies.py)

---

## Future Directions

See [`future_works.md`](future_works.md).
