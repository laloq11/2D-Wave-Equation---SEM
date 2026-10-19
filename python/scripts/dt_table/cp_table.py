import numpy as np
from numpy.polynomial import legendre as leg


def gll_points_weights(p):
    Pp = leg.Legendre.basis(p)
    xi = np.concatenate(([-1.0], np.sort(Pp.deriv().roots().real), [1.0]))
    w = 2.0 / (p * (p + 1) * Pp(xi) ** 2)
    return xi, w


def lagrange_on(nodes, x):
    p = len(nodes) - 1
    C = np.linalg.inv(leg.legvander(nodes, p))
    N = leg.legvander(x, p) @ C
    dN = leg.legvander(x, p - 1) @ leg.legder(C, axis=0)
    return N, dN


def dt_crit_uncut(n_elem, p, Lx, Ly, c, stiffness="gll"):
    hx, hy = Lx / n_elem, Ly / n_elem
    xi_gll, w_gll = gll_points_weights(p)
    xi_q, w_q = (xi_gll, w_gll) if stiffness == "gll" else leg.leggauss(p + 1)
    N, dN = lagrange_on(xi_gll, xi_q)

    def mats_1d(h):
        M_lumped = np.diag(w_gll) * h / 2
        M_q = (N.T * w_q) @ N * h / 2
        K = (dN.T * w_q) @ dN * 2 / h
        return M_lumped, M_q, K

    Mx_l, Mx_q, Kx = mats_1d(hx)
    My_l, My_q, Ky = mats_1d(hy)

    Me = np.kron(Mx_l, My_l)
    Ke = c**2 * (np.kron(Kx, My_q) + np.kron(Mx_q, Ky))

    d = 1.0 / np.sqrt(np.diag(Me))
    omega_max = np.sqrt(np.linalg.eigvalsh(d[:, None] * Ke * d[None, :]).max())
    return 2.0 / omega_max


def check_gll(p):
    xi, w = gll_points_weights(p)
    exact = lambda k: (1 - (-1) ** (k + 1)) / (k + 1)
    err = max(abs(w @ xi**k - exact(k)) for k in range(2 * p))
    return err


def cp_table(p_max=10, L=2.0, meshes=(4, 10, 37), stiffness="gll"):
    rows = []
    for p in range(1, p_max + 1):
        cp = [dt_crit_uncut(n, p, L, L, 1.0, stiffness) * n / L for n in meshes]
        rows.append((p, cp[0], max(cp) - min(cp), check_gll(p)))
    return rows


if __name__ == "__main__":
    print("C_p, GLL-lumped M, GLL-integrated K (SCM_cuda)")
    print("p   C_p                 spread(h)   GLL quad err (deg<=2p-1)")
    for p, cp, spread, err in cp_table():
        print(f"{p:<3} {cp:.15f}  {spread:.1e}     {err:.1e}")

    print("\nValidation against SCM_cuda runs (pD=2, ne=20, duration=6):")
    print(f"dt_crit p=2 h=0.1 c=1 : {dt_crit_uncut(20, 2, 2, 2, 1):.4e}  (A5 6/211=2.8436e-2 bounded, A6 6/205=2.9268e-2 diverges)")

    print("\nValidation of GL-stiffness variant against Fassbender MSc thesis (2023):")
    print(f"Eq. 4.14  p=5 h=0.25  c=1    : {dt_crit_uncut(40, 5, 10, 10, 1, 'gl'):.4e}  (1.8131e-2)")
    print(f"Eq. 4.26  p=5 h=1/6   c=1    : {dt_crit_uncut(60, 5, 10, 10, 1, 'gl'):.4e}  (1.2088e-2)")
    print(f"Eq. 5.10  p=2 h=2.5mm c=6000 : {dt_crit_uncut(40, 2, 0.1, 0.1, 6000, 'gl'):.4e}  (1.5528e-7)")
