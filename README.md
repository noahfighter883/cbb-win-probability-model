
---

## CBB Predictor (Java)

Repo: `cbb-win-probability-model`

```markdown
# College Basketball Win Probability Model

Win-probability estimator using NET, SOR, adjusted efficiencies, and quadrant records.

## Overview

This model estimates matchup win probabilities by blending advanced team metrics and strength-of-schedule indicators.

## Inputs

- NET ranking
- Strength of Record (SOR)
- Adjusted offensive efficiency
- Adjusted defensive efficiency
- Quadrant records

## Modeling Approach

Weighted composite score generates expected win probability between two teams.

## Compile

```bash
javac CBBPredictor.java
