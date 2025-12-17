# Speaker Policies for Program Communication

This document describes two speaker policies implemented for studying how conventions emerge in program communication tasks.

## Baseline: RSA Speaker

The Rational Speech Act (RSA) speaker selects programs and utterances to maximize listener understanding while minimizing cost.

### Hypothesis
Convention Stabilization, Abstraction and Compression rise from rational reasoning.

**Program utility:**
$$
U_{\text{base}}(p \mid s) = (1 - \beta) \cdot \mathbb{E}[\log P_L(a = \text{step} \mid u)] - \beta \cdot |p|
$$

where $|p|$ is program length (number of steps), and the expectation is over steps and utterance choices.

**Program choice:**
$$
P(p \mid s) \propto \exp(\alpha_p \cdot U_{\text{base}}(p \mid s))
$$

**Utterance choice (per step):**
$$
P(u \mid \text{step}) \propto \exp(\alpha_u \cdot \mathbb{E}_L[\log P_L(a = \text{step} \mid u)])
$$

---

## Policy A: Pact-Aware Speaker

Based on Brennan & Clark (1996), this policy models how speakers form and maintain conceptual pacts with partners.

As a supplement of RSA momentary rational inference, Pact policy introduce interaction-level memory and inertia.

### Mechanism

1. **Proposal**: Speaker's first use of a program is a tentative conceptualization
2. **Ratification**: Listener succeeds without clarification -> pact forms
3. **Repair**: Listener fails or needs clarification -> pact weakens
4. **Inertia**: Once ratified, speakers resist switching even if shorter alternatives exist

### Pact Memory (per partner)

For each conceptualization key $k$ (= program string), maintain:
- $n_k$: usage count
- $r_k$: ratification count  
- $q_k$: repair count
- $k^*$: active pact (last ratified key)

### Pact Strength

$$
\text{strength}(k) = \log(1 + r_k) - \log(1 + q_k)
$$

### Switching Cost

$$
C_{\text{switch}}(k \mid k^*, M) = 
\begin{cases}
0 & \text{if } k = k^* \text{ or } k^* = \text{None} \\
\eta \cdot \text{strength}(k^*) & \text{otherwise}
\end{cases}
$$

### Pact-Aware Utility

$$
U_{\text{pact}}(p \mid s, M) = U_{\text{base}}(p \mid s) - \gamma \cdot C_{\text{switch}}(p \mid k^*, M)
$$

### Update Rules

After each trial with chosen program $k$:
- Always: $n_k \leftarrow n_k + 1$
- If ratified (success, no repairs): $r_k \leftarrow r_k + 1$, $k^* \leftarrow k$
- If repair occurred: $q_k \leftarrow q_k + 1$ (do not update $k^*$)

### Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `PACT_GAMMA` | $\gamma$ | Weight of switching cost in utility |
| `PACT_ETA` | $\eta$ | Scaling factor for pact strength |

### Reference

Brennan, S. E., & Clark, H. H. (1996). Conceptual pacts and lexical choice in conversation. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 22(6), 1482-1493.

---

## Policy B: Drift-Biased Speaker

Based on Bentley et al. (2007), this policy models **neutral cultural drift** through random copying from observed variants.

### Hypothesis (against RSA)
Convention Stabilization, Abstraction and Compression can also arise from random drift.

### Mechanism

1. **Copying channel**: With probability $\varepsilon$, copy from historical distribution
2. **Rational channel**: With probability $1-\varepsilon$, use RSA program choice
3. **Innovation**: Within copying, small probability $\mu$ of random selection

### Cultural Pool

Maintain two frequency counts for each variant key $k$:
- $H_{\text{prev}}(k)$: count from previous generation
- $H_{\text{ppt}}(k)$: count from current ppt's history

**Mixed distribution:**
$$
H_{\text{mix}}(k) = w \cdot H_{\text{prev}}(k) + (1-w) \cdot H_{\text{ppt}}(k)
$$

**Copying distribution:**
$$
\pi_{\text{copy}}(k) = \frac{H_{\text{mix}}(k)^\tau}{\sum_j H_{\text{mix}}(j)^\tau}
$$

### Program Choice

$$
P(p \mid s) = (1 - \varepsilon) \cdot P_{\text{RSA}}(p \mid s) + \varepsilon \cdot P_{\text{copy}}(p)
$$

where:
$$
P_{\text{copy}}(p) = 
\begin{cases}
\mu \cdot \text{Uniform}(p) + (1-\mu) \cdot \pi_{\text{copy}}(p) & \text{if } p \in \text{pool} \\
\mu \cdot \text{Uniform}(p) & \text{otherwise innovation}
\end{cases}
$$

### Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `DRIFT_EPSILON` | $\varepsilon$ | Probability of copying channel |
| `DRIFT_TAU` | $\tau$ | Frequency exponent (1.0 = pure random copying) |
| `DRIFT_MU` | $\mu$ | Innovation probability |
| `DRIFT_POOL_WEIGHT_PREV_GEN` | $w$ | Weight of previous-gen vs within-partner pool |

### Reference

Bentley, R. A., Lipo, C. P., Herzog, H. A., & Hahn, M. W. (2007). Regular rates of popular culture change reflect random copying. *Evolution and Human Behavior*, 28(3), 151-158.



---

## Implementation

See `model/speaker_policies.py`.

