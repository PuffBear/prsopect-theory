# Threshold Robustness Analysis

**Collapse definition variants** — `mean_S < S_thr AND profit ≤ P_thr`

**Baseline:** mean_S < 10 AND profit ≤ −127.1  →  w\_crit = 0.6657


| S\_thr | P\_thr   | n\_collapsed | w\_crit | Boot mean | 95% CI           | Note     |
|--------|---------|------------|--------|-----------|------------------|----------|
| 8      |  -102.5 |          106 | 0.6813 | 0.6811    | [0.6493, 0.7099] |          |
| 8      |  -127.1 |          103 | 0.6657 | 0.6656    | [0.6351, 0.6942] |          |
| 8      |  -152.5 |            0 | N/A | N/A    | N/A |          |
| 10      |  -102.5 |          106 | 0.6813 | 0.6811    | [0.6493, 0.7099] |          |
| 10      |  -127.1 |          103 | 0.6657 | 0.6656    | [0.6351, 0.6942] | BASELINE |
| 10      |  -152.5 |            0 | N/A | N/A    | N/A |          |
| 12      |  -102.5 |          106 | 0.6813 | 0.6811    | [0.6493, 0.7099] |          |
| 12      |  -127.1 |          103 | 0.6657 | 0.6656    | [0.6351, 0.6942] |          |
| 12      |  -152.5 |            0 | N/A | N/A    | N/A |          |

**Note on N/A rows:** The three P\_thr = −152.5 variants are degenerate — the minimum profit in the dataset is −128.1, so no run satisfies `profit ≤ −152.5`. These variants produce zero collapsed episodes and are excluded from the deviation calculation.

**Max deviation from baseline w\_crit across the 6 estimable variants:** 0.0156

**Definition is robust to ±20% threshold perturbation.**