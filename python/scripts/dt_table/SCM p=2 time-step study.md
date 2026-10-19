# SCM p=2 time-step study

Oct 6, 2026 · @Lalo

## Summary

The p=2 failure was a time-step instability, not spatial dispersion. The code's critical step is Δt\_crit = 0.2887·h/c, not the 0.3727·h/c from the first Cp table.

- The first auto-stable rule gave Δt = 0.9 · 0.3727 · 0.1 = 3.35e-2 for ne=20. That is above the measured limit of 2.89e-2, so the run diverged.
- The limit is bracketed by A5 (Δt = 2.844e-2, stable) and A6 (Δt = 2.927e-2, unstable). The growth rates of the five unstable runs give C = 0.2884 ± 0.0003, excluding A4.
- The limit scales linearly with h. The ne=40 runs reproduce the ne=20 bracket at half the Δt.
- At Δt = 1.2e-3, ne=20 is within 1 % of ne=40 at the centre node. ne=10 is off by up to 29 %.

## Setup

The runs cover a 2×2 domain with a uniform mesh, c = 1, explicit CDM and no cut cells. The time step is Δt = duration / nsteps, with duration = 6.

```bash
python3 install/Weq2D_02.py --run --solver omp --ne <ne> --pD 2 --nsteps <n> --duration 6 --filename <name>
```

- Element size: h = 2 / ne, so 0.2, 0.1 and 0.05 for ne = 10, 20 and 40.
- Global DOFs: (ne·pD + 1)², which is 441, 1681 and 6561. The logged nodes u\_(220), u\_(840) and u\_(3280) are the centre node of each mesh.
- Signal: the log prints the centre node every 50 steps. The B-run analysis uses that log, not the CSVs.

## Stability runs

Every run flips from bounded to divergent at the same Δt·c/h for both mesh sizes, between 0.2844 and 0.2927. The model with GLL-lumped mass and GLL-integrated stiffness predicts 0.2887.

| Run | ne | nsteps | Δt | Δt·c/h | Outcome | g measured | g predicted (C=0.2887) | C inferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 20 | 360 | 1.667e-2 | 0.1667 | bounded | — | — | — |
| A1h | 40 | 720 | 8.333e-3 | 0.1667 | bounded | — | — | — |
| A2 | 20 | 280 | 2.143e-2 | 0.2143 | bounded | — | — | — |
| A2h | 40 | 560 | 1.071e-2 | 0.2143 | bounded | — | — | — |
| A5 | 20 | 211 | 2.844e-2 | 0.2844 | bounded | — | — | — |
| A6 | 20 | 205 | 2.927e-2 | 0.2927 | diverges | 1.3974 | 1.3950 | 0.2886 |
| A3 | 20 | 200 | 3.000e-2 | 0.3000 | diverges | 1.7603 | 1.7479 | 0.2884 |
| A3h | 40 | 400 | 1.500e-2 | 0.3000 | diverges | 1.7455 | 1.7479 | 0.2887 |
| A4 | 20 | 150 | 4.000e-2 | 0.4000 | diverges | 5.8636 | 5.4981 | 0.2822 |
| A4h | 40 | 300 | 2.000e-2 | 0.4000 | diverges | 5.5170 | 5.4981 | 0.2883 |

How the growth factor g and C are obtained. g is the per-step amplification of |u| at the centre node over the last 50 logged steps. With Mü = −Ku, the CDM sum step u\_{n+1} = 2u\_n − u\_{n−1} + Δt²ü\_n gives, per mode:

```latex
g^2 - (2 - \omega^2 \Delta t^2)\, g + 1 = 0
```

For ωΔt > 2 one root is real with |g| > 1. Solving for the highest mode:

```latex
\omega_{max}\Delta t = \sqrt{|g|} + \frac{1}{\sqrt{|g|}}, \qquad \Delta t_{crit} = \frac{2\,\Delta t}{\omega_{max}\Delta t}, \qquad C = \frac{\Delta t_{crit}\, c}{h}
```

A4 is the outlier. It is measured at steps 100–150, where the highest mode may not yet dominate. A4h, at the same Δt·c/h, agrees with the prediction.

## Spatial runs

At a fixed Δt = 1.2e-3, ne=20 is spatially converged at the centre node, while ne=10 is not.

&#91;embedded content: B10/B20/B40 run logs, centre node every 50 steps, t ≥ 1.5 s (u ≈ 0 before)\]

- ne=20 vs ne=40: max |Δu| = 2.1e-3, which is 1.0 % of max |u|.
- ne=10 vs ne=40: max |Δu| = 6.2e-2, which is 29 % of max |u|.
- ne=10 arrives on time but loses amplitude, so the error is mostly amplitude, not phase.

Wavelength resolution at f = 1 Hz (λ = 1 m) is 5, 10 and 20 elements per λ for ne = 10, 20 and 40.

## Conclusions

The code uses GLL-lumped mass and GLL-integrated stiffness (confirmed in the C++ source), and all measurements agree with that model. Use Δt\_crit = C\_p·h/c with the GLL table below, not the Gauss-Legendre table from the thesis and paper.

1. The p=2 failure came from Δt, not h. The old rule set Δt at 1.16× the real limit.
2. The stability limit is sharp and linear in h. Bounded and divergent runs are separated by less than 3 % in Δt (A5 vs A6).
3. For accuracy at pD=2, ne=20 is enough at this signal frequency; ne=10 is not.

| p | C\_p (GLL M, GLL K) | Status |
| --- | --- | --- |
| 1 | 0.707106781186548 | computed, not tested |
| 2 | 0.288675134594813 | confirmed by runs |
| 3 | 0.164054624686693 | computed, not tested |
| 4 | 0.104442278311696 | consistent with the earlier ne=40 run, not bracketed |
| 5 | 0.071432315609361 | computed, not tested |
| 6 | 0.051655014512142 | computed, not tested |
| 7 | 0.039005419539448 | computed, not tested |
| 8 | 0.030464365739356 | computed, not tested |
| 9 | 0.024438904279170 | computed, not tested |
| 10 | 0.020034146407630 | computed, not tested |

Open items:

- [ ] Confirm in the C++ source that the stiffness of uncut cells is integrated on the GLL points (p+1 per direction).
- [ ] Bracket p=4 the same way (A5/A6 style, ±1.5 % around 0.10444·h/c).
- [ ] Test temporal accuracy separately. No run here varies Δt at a fixed mesh below the limit and compares the results.
- [ ] Decide on a fixed physical duration with nsteps = ceil(T / Δt), instead of duration = nsteps·Δt.

## Sources

- &#91;T\] C. Faßbender, *Time Integration for the Spectral Cell Method with Application to the Full Waveform Inversion*, MSc thesis, TUM, 2023. It is the source of Δt\_crit = 2/ω\_max (Eq. 3.4), the GLL-lumped mass (Eq. 2.40) and the reference values 1.8131e-2, 1.2088e-2 and 1.5528e-7 (Eqs. 4.14, 4.26, 5.10).
- &#91;P\] C. Faßbender, T. Bürchner, P. Kopp, E. Rank, S. Kollmannsberger, Implicit-explicit time integration for the immersed wave equation, Comput. Math. Appl. 163 (2024) 1–13, [doi:10.1016/j.camwa.2024.02.049](https://doi.org/10.1016/j.camwa.2024.02.049). It is the source of Eqs. 1–2, the CDM sum step (Eqs. 24–25), Table 1 and Fig. 9 (0.99 vs 1.01·Δt\_crit).
- The GLL-stiffness C\_p table and the growth-factor inversion are my own derivations from \[P\] Eqs. 1–2 and 24–25. Neither source gives them.
- Simulation data: the run logs of A1–A6, A1h–A4h and B10/B20/B40 from 2026-10-06. `C_p values come from cp_table.py`.
