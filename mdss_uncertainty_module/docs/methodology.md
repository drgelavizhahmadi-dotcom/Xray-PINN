# Uncertainty Quantification Methodology

## Overview

We implement Monte Carlo dropout for uncertainty estimation in medical X-ray analysis.

## Uncertainty Decomposition

**Total Uncertainty**: Predictive entropy from MC samples

**Aleatoric Uncertainty**: Expected entropy (data noise)

**Epistemic Uncertainty**: Mutual information (model uncertainty)

## Risk Levels

| Level | Uncertainty | Confidence | Action |
|-------|-------------|------------|--------|
| LOW | < 0.10 | > 0.85 | Autonomous |
| MEDIUM | < 0.25 | > 0.70 | Monitor |
| HIGH | > 0.25 | < 0.70 | Human review |

## AI Act Compliance

- Article 14: Human oversight with automated triggers
- Article 15: Uncertainty quantification for accuracy
