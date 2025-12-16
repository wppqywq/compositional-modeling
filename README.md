# Compositional Abstractions Tutorial (Personal Copy)

This repository is a personal working copy of the original project [`cogtoolslab/compositional-abstractions-tutorial`](https://github.com/cogtoolslab/compositional-abstractions-tutorial).

For the original documentation, please see [`README_original.md`](./README_original.md) in this repository.

### Extended modules

Main notebooks(Clone this repo and execute directly.):
- [`notebooks_new/5_communication_chains.ipynb`](notebooks_new/5_communication_chains.ipynb): Communication system evolution (based on agents from notebook3). Meanings are fixed from empirical trials; conventions and chunk usage evolve across generations. 


---

## Review of Notebook 5

Build an iterated transmission-chain setting, to study how compositional abstractions and lexical conventions stabilize under repeated cultural transmission.

**Future Directions**: Please check [`future_works.md`](future_works.md).

### Settings

Meaning:
- Meaning is the scene (tower arrangement), fixed across generations.
- Each scene has multiple program representations in a DSL (from fully decomposed steps to chunk-reusing descriptions).

> Assumption:
> Abstraction in this setting arises from choosing among alternative descriptions of the same underlying structure, not from inventing new meanings.

---

Communication:
- Speakers choose: (i) program representation to express; (ii) utterance to use for each step.
- Softmax balance: (i) expected listener success, (ii) message length / chunk usage (cost)

---

Transmission chain:
- At each generation: one speaker-listener pair communicates over a fixed distribution of trials.
- Training:
  - Random select one `ppt_id` each generation.
  - Use only the first 5 tower types in the manual ordering (`CL`, `CPi`, `PiC`, `LPi`, `LC`), which correspond to 10 trials.
  - The remaining tower type (`PiL`) is held out as a small test task (2 trials per participant).
- Message length `msg_len`: the number of steps / trial.
- Each step produces (utterance, intention, listener response) observations.
- End of each generation: both agents update their lexicon beliefs from observation (iterated learning)

> Question: Does too fast abstraction cause early lock-in?
>
> Here we specify abstract chunks must earn their sharedness through communication (high usage_freq, high correction_rate), to distinguish from individual-level interaction to social conventions. 
>
> Result in Notebook5 shows that adding this gate do stabilize the accuracy, and slow down the compression.

---

Generalization:

Freeze the final lexicon posterior and active chunks learned during training, then evaluate on held-out participants.
- Test:
  - Use the held-out tower type (`PiL`) only (2 trials per participant).
  - Evaluate on a random subset of 10 participants.

> Question: Does the agents successed in first-seem task, given lexicon prior knowledge learned from 50 generations?
>
> All agents remian the high accuracy across trials, RSA agent achieve a compression level as in the training trails.

Details of metrics please refer to the notebook.

