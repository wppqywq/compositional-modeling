# Communication evolution over fixed scenes (iterated learning)

Notebook 5 builds an baseline iterated transmission chain to study how program abstractions (chunks) and lexical conventions stabilize through repeated cultural transmission, rather than within a single interaction.

**Future Directions**: Please check [`future_works.md`](future_works.md).

### Settings

**Meaning representation** (Reuse Dreamcoder from Notebook 2)
- Meaning is the scene (tower arrangement), fixed across generations.
- Each scene has multiple program representations in a DSL (base primitives + DreamCoder-learned chunks).

> **Assumption**:
> Abstraction in this setting arises from choosing among alternative descriptions of the same underlying structure, not from inventing new meanings.

---

**Communication Agents** (from Notebook 3)

Two roles: Speaker (Architect) and Listener (Builder). Both update lexicon beliefs using Bayesian belief update.
- Speakers choose: (i) program representation to express; (ii) utterance to use for each step.
- Softmax balance: (i) expected listener success, (ii) message length / chunk usage (cost)

---

**Transmission chain:**
- At each generation: one speaker-listener pair communicates over a fixed distribution of trials.
- Training:
  - Random select one `ppt_id` each generation.
  - Use only the first 5 tower types in the manual ordering (`CL`, `CPi`, `PiC`, `LPi`, `LC`), which correspond to 10 trials.
  - The remaining tower type (`PiL`) is held out as a small test task (2 trials per participant).
- Message length `msg_len`: the number of steps / trial.
- Each step produces (utterance, intention, listener response) observations.
- End of each generation: **cultural transmission is observation-only**. The next generation does **not** inherit the previous generation's full posterior. Instead, it receives a bottlenecked set of transmitted observations (intention, utterance; with minimal trial metadata like `trial` and `step_idx`) and re-inferrs beliefs starting from `LexiconPrior`.
- Belief update uses adaptive weights: $\log posterior \propto \alpha_t \log prior + \beta_t \log likelihood$, where $\alpha_t,\beta_t$ depend on transmission fidelity (e.g., communication accuracy).

> **Question**: Does too fast abstraction cause early lock-in?
>
> Here we specify abstract chunks must earn their sharedness through communication (high usage_freq, high correction_rate), to distinguish from individual-level interaction to social conventions. 
>
> Result in Notebook5 shows that adding this gate do stabilize the accuracy, and slow down the compression.

---

**Generalization**:

Freeze the final lexicon posterior (re-inferred from accumulated transmitted observations) and active chunks learned during training, then evaluate on held-out participants.
- Test:
  - Use the held-out tower type (`PiL`) only (2 trials per participant).
  - Evaluate on a random subset of 10 participants.

> **Question**: Does the agents successed in first-seem task, given lexicon prior knowledge learned from 50 generations?
>
> All agents remian the high accuracy across trials, strategic agent achieve a compression level as in the training trails.

Details of metrics please refer to the notebook.

---

**Metrics**

- Accuracy: success rate

- Message length: steps per trial

- Fragment rate: percentage of steps using chunks

- Program choice entropy: speaker uncertainty about how to describe a scene

- Lexeme mapping entropy: listener uncertainty about what a word means
