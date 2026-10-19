# C_p table: critical time step of an uncut SCM cell

`cp_table.py` computes C_p in

    dt_crit = C_p * h / c

for one square, uncut Spectral Cell Method (SCM) cell (scalar wave equation, 2D, explicit CDM).

## Sources

- **[T]** C. Faßbender, *Time Integration for the Spectral Cell Method with Application to the Full Waveform Inversion*, MSc thesis, TUM, 2023.
- **[P]** C. Faßbender, T. Bürchner, P. Kopp, E. Rank, S. Kollmannsberger, *Implicit-explicit time integration for the immersed wave equation*, Comput. Math. Appl. 163 (2024) 1–13, doi:10.1016/j.camwa.2024.02.049.
- **[H]** T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover, 2000. This is ref. [5] in [T] and ref. [17] in [P].

## Method

1. **Critical time step for CDM:** dt_crit = 2 / omega_max ([T] Eq. 3.4, [P] Eq. 1; both cite [H]).
2. **omega_max:** the square root of the largest eigenvalue of det(K_e − omega² M_e) = 0, solved at cell level ([T] Eq. 3.5 and Sec. 3.2.2, [P] Eq. 2).
3. **Basis:** Lagrange polynomials on the p+1 Gauß-Lobatto-Legendre (GLL) points. The GLL points are ±1 plus the roots of P_p' ([T] Eqs. 2.34–2.37, [P] Eqs. 5–6). The 2D basis is the tensor product ([T] Eq. 2.38).
4. **Quadrature in an uncut cell** ([T] Secs. 2.3 and 2.5, [P] p. 4):
   - **Mass:** GLL quadrature on the nodes, giving a diagonal (nodally lumped) M_e ([T] Eq. 2.40).
   - **Stiffness, default `stiffness="gll"`:** GLL quadrature on the p+1 nodes. This is what SCM_cuda does (confirmed in its C++ source, 2026-10-06). It is NOT the rule of [T]/[P].
   - **Stiffness, `stiffness="gl"`:** Gauß-Legendre quadrature with p+1 points, the rule of [T] Sec. 4.1.1 (p_int = p+1) and [P] p. 4. Kept only to reproduce the published values.
   - **Weights:** the formula w_i = 2 / (p(p+1) P_p(xi_i)²) is the standard Lobatto rule. Neither [T] nor [P] states it. The script checks it numerically instead: exact integration of monomials up to degree 2p−1, consistent with [T] Eq. 2.41.
5. **Element matrices:** [T] Eqs. 2.28–2.29 with rho = 1. The 2D matrices are assembled from 1D blocks with Kronecker products:
   - M_e = M_GLL ⊗ M_GLL
   - K_e = c² (K ⊗ M_q + M_q ⊗ K), where M_q is the 1D mass on the stiffness quadrature (equal to M_GLL for `"gll"`)

## Why dt_crit is exactly proportional to h

This is a scaling argument, not a fit:

- M_e ∝ h² (cell area).
- K_e ∝ (1/h)² · h² = h⁰ in 2D (two gradients times the area).
- So omega_max² ∝ c²/h², which gives dt_crit = C_p · h/c.

The script checks this at three mesh sizes; the spread of C_p is about 1e-16.

## Validation

**GLL stiffness (default) against SCM_cuda runs**, pD=2, ne=20, h=0.1, c=1, duration=6:

| Run | dt | Script dt_crit = 2.8868e-2 predicts | Observed |
|---|---|---|---|
| A5 (nsteps 211) | 2.8436e-2 | bounded | bounded |
| A6 (nsteps 205) | 2.9268e-2 | diverges | diverges |

The growth rates of the diverging runs (A3, A3h, A4h, A6) give C_2 = 0.2883–0.2887, using the CDM amplification g² − (2 − ω²dt²)g + 1 = 0 from [P] Eqs. 24–25. The ne=40 runs flip at the same dt·c/h, confirming the scaling with h. Only p = 2 is tested.

**GL stiffness (`stiffness="gl"`) against the published values:**

| Case | Script | Published |
|---|---|---|
| p=5, h=0.25, c=1 | 1.8131e-2 | 1.8131e-2 ([T] Eq. 4.14; [P] Table 1: 18.13 ms) |
| p=5, h=1/6, c=1 | 1.2088e-2 | 1.2088e-2 ([T] Eq. 4.26) |
| p=2, h=2.5 mm, c=6000 | 1.5528e-7 | 1.5528e-7 ([T] Eq. 5.10) |

## Validity limits

- **Uncut, square cells only.** The script uses the same element count in x and y.
- **Cut cells** have a much smaller dt_crit: a factor r = 16 to 27 in [T] Eqs. 4.16, 4.28 and 5.12, and [P] Table 1. It depends on the fill ratio and cannot be expressed through (h, p) alone.
- **Newmark IMEX (cut cells implicit):** the global dt_crit is about equal to the cell-wise uncut value ([P] Table 1: 18.14 vs 18.13 ms; [P] p. 9 calls it "a good and cheap estimate").
- **Explicit CDM over all cells with a uniform material:** the global omega_max does not exceed the largest element omega_max. This is the element eigenvalue inequality, discussed in [H]; the exact section was not verified.
- **Margin:** the limit is sharp. 0.99·dt_crit is stable and 1.01·dt_crit diverges ([P] Fig. 9). Any safety factor is the user's choice.
- **Other stiffness quadrature:** if your code uses a different stiffness quadrature order for uncut cells, C_p changes. Re-run the script with your rule.

## Results (p = 1..10)

GLL-lumped M, GLL-integrated K (use this for SCM_cuda):

    0.707106781186548, 0.288675134594813, 0.164054624686693, 0.104442278311696,
    0.071432315609361, 0.051655014512142, 0.039005419539448, 0.030464365739356,
    0.024438904279170, 0.020034146407630

GLL-lumped M, Gauß-Legendre K (thesis/paper rule; overestimates dt_crit for SCM_cuda, which caused the 2026-10-06 p=2 failures):

    1.000000000000000, 0.372677996249965, 0.182574185835055, 0.108425405035419,
    0.072525190378972, 0.052026033161093, 0.039153969663832, 0.030531704731319,
    0.024472468861462, 0.020052169824593
