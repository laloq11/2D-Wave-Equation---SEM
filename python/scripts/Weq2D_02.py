"""Weq2D_02.py: solve the 2D wave equation with pyspectralcm and save u(x, y, t) as a GIF.

    python3 Weq2D_02.py --run --solver cu_pDegreeThread --ne 20 --pD 4 --nsteps 3000
    python3 Weq2D_02.py --filename wave20 --ne 20 --pD 4 --nsteps 3000 --writefreq 200   # plot existing wave20.csv
"""

import argparse
import contextlib
import ctypes
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

NUM_CORES = multiprocessing.cpu_count()
SAMPLE_EVERY = 50   # solver writes u every 50 steps (plotfreq_ in params.cpp)
GPU_MAX_PD = 5      # MAX_pD in cuontheFly.cu

# C_p = dt_crit * c / h of an uncut square cell, p = 1..10 (same table as w_eq_2d_autostable in spectralcm.cpp)
CP = (0.707106781186548, 0.288675134594813, 0.164054624686693, 0.104442278311696,
      0.071432315609361, 0.051655014512142, 0.039005419539448, 0.030464365739356,
      0.024438904279170, 0.020034146407630)

SOLVERS = {"omp": "ontheFly_omp_write",
           "cu_singleThread": "cu_singleThread",
           "cu_pDegreeThread": "cu_pDegreeThread"}


@dataclass(slots=True)
class case_setup:
    eps: float
    nelements: np.ndarray
    lengths: np.ndarray
    center: np.ndarray
    nsteps: int
    polDegree: int
    duration: float
    radius: float
    freq: float
    sigmaS: float
    E: float
    rho: float
    domain: float
    alpha: float
    c: float


case_base = case_setup(eps=1e-12,
    nelements=np.array([5, 5]),
    lengths=np.array([2.0, 2.0]),
    center=np.array([1.0, 1.0]),
    nsteps=1000,
    polDegree=2,
    duration=12.0,
    radius=0.4,
    freq=1.0,
    sigmaS=0.06,
    E=1.0,
    rho=1.0,
    domain=1.0,
    alpha=1.0,
    c=1.0)


def parsargs(argv=None):
    p = argparse.ArgumentParser(description="Run (or only plot) one 2D wave-equation case and save it as a GIF.")
    p.add_argument("--run", action="store_true", help="solve; without it only plot <filename>.csv")
    p.add_argument("--solver", default="omp", choices=list(SOLVERS))
    p.add_argument("--cpus", "--cputhreads", type=int, default=NUM_CORES, help="OpenMP threads (omp only)")
    p.add_argument("--ne", type=int, help="elements per direction")
    p.add_argument("--pD", type=int, help="polynomial degree")
    p.add_argument("--nsteps", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--filename")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true", help="C++ output to <name>.log")
    p.add_argument("--reps", type=int, default=1, help="only used in the default file name")
    p.add_argument("--writefreq", type=int, help=f"solver steps between frames, multiple of {SAMPLE_EVERY}")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--vmax", type=float, help="colour limit |u|")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def createcase(args):
    overrides = {"nelements": None if args.ne is None else np.array([args.ne, args.ne]),
                 "polDegree": args.pD, "nsteps": args.nsteps, "duration": args.duration}
    return replace(case_base, **{k: v for k, v in overrides.items() if v is not None})


def critical_dt(case):
    """C_p * h / c for uncut cells, or None when p is outside the C_p table."""
    if case.polDegree > len(CP):
        return None
    h = min(case.lengths[0] / case.nelements[0], case.lengths[1] / case.nelements[1])
    return CP[case.polDegree - 1] * h / case.c


def free_name(base, exts):
    name, k = base, 2
    while any(os.path.exists(name + e) for e in exts):
        name, k = f"{base}_{k}", k + 1
    return name


@contextlib.contextmanager
def stdout_to(path):
    """Redirects fd 1 (C++ stdout) to path."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(path, "w") as f:
        os.dup2(f.fileno(), 1)
        try:
            yield
        finally:
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved, 1)
            os.close(saved)


def node_coords(params):
    lagrangeCoords = np.array(params.lagrangeCoords())
    x = [params.mapX(ielement, r) for ielement in range(params.nelements(0)) for r in lagrangeCoords[:-1]]
    y = [params.mapY(jelement, s) for jelement in range(params.nelements(1)) for s in lagrangeCoords[:-1]]
    x += [params.mapX(params.nelements(0) - 1, lagrangeCoords[-1])]
    y += [params.mapY(params.nelements(1) - 1, lagrangeCoords[-1])]
    return np.meshgrid(x, y, indexing='ij')


def make_animation(u, params, frames, dt, header, command, gif, vmax, dpi):
    shape = (params.ndofdirection()[0], params.ndofdirection()[1])
    vmax = vmax or float(np.abs(u).max()) or 1.0
    lx, ly = params.lengths()[0], params.lengths()[1]
    t_end = params.nsteps() * dt
    X, Y = node_coords(params)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))   # taller than default: same plot size, room for the header
    fig.subplots_adjust(bottom=0.10, top=0.78)
    mesh = ax.pcolormesh(X, Y, np.reshape(u[frames[0]], shape), shading='gouraud',
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for xe in np.linspace(0, lx, params.nelements(0) + 1):
        ax.axvline(xe, color='0.6', lw=0.4)
    for ye in np.linspace(0, ly, params.nelements(1) + 1):
        ax.axhline(ye, color='0.6', lw=0.4)
    ax.add_artist(plt.Circle(params.center(), params.radius(), fill=False, ls='--', lw=0.9))
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect('equal')
    fig.colorbar(mesh, ax=ax, label='u')
    fig.text(0.02, 0.955, "2D wave equation", fontsize=12, weight='bold')
    fig.text(0.02, 0.915, header[0], fontsize=8)
    fig.text(0.02, 0.885, header[1], fontsize=8, color='0.4')
    if command:
        fig.text(0.02, 0.015, command, fontsize=6, family='monospace', color='0.4')

    def animate(i):   # row i holds step (i+1)*SAMPLE_EVERY
        mesh.set_array(np.reshape(u[i], shape))
        step = (i + 1) * SAMPLE_EVERY
        ax.set_title(f"t = {step * dt:6.2f} / {t_end:.2f} s    step {step}/{params.nsteps()}",
                     fontsize=10, family='monospace')
        return mesh,

    anim = animation.FuncAnimation(fig, animate, frames, interval=100)
    anim.save(gif, writer="pillow", fps=10, dpi=dpi)
    return anim


def main(argv=None):
    args = parsargs(argv)
    case = createcase(args)
    dt = case.duration / case.nsteps
    if args.run and args.solver != "omp" and case.polDegree > GPU_MAX_PD:
        sys.exit(f"error: CUDA solvers support p <= {GPU_MAX_PD}")
    dt_crit = critical_dt(case)
    if args.run and dt_crit is not None and dt > dt_crit:
        print(f"warning: dt = {dt:.2e} s is above the estimated stable limit {dt_crit:.2e} s; "
              f"the run will probably blow up. Use --nsteps {math.ceil(case.duration / (0.9 * dt_crit))} or more.",
              file=sys.stderr)

    os.environ["OMP_NUM_THREADS"] = str(args.cpus)   # must be set before pyspectralcm is loaded
    import pyspectralcm as scm

    if args.filename is None:   # same pattern as benchmark_solver.py
        name = f"SEM_{args.solver}_ne{case.nelements[0]}_pd{case.polDegree}_cpu{args.cpus}_run{args.reps}_dt{dt:.0e}"
    else:
        name = args.filename[:-4] if args.filename.endswith((".csv", ".gif", ".log")) else args.filename

    if args.run:
        base = name if args.overwrite else free_name(name, (".csv", ".gif", ".log"))
        csv = base + ".csv"
        if os.path.exists(csv):
            os.remove(csv)   # the C++ writer appends
    else:
        csv = name + ".csv"
        if not os.path.exists(csv):
            sys.exit(f"error: {csv} not found, add --run")
        base = name if args.overwrite else free_name(name, (".gif",))

    params = scm.SCM_ParametersPatch(case.nelements, case.lengths, case.center,
                                     case.E, case.rho, case.domain, case.alpha, case.c,
                                     case.duration, case.radius, case.freq, case.sigmaS,
                                     case.nsteps, case.polDegree)
    ndof = params.ndofdirection()[0] * params.ndofdirection()[1]
    p = case.polDegree
    threads = {"omp": f"{args.cpus} OpenMP threads",
               "cu_singleThread": "1 GPU thread per element",
               "cu_pDegreeThread": f"{(p + 1) ** 2} GPU threads per element"}[args.solver]
    solver_line = f"{args.solver}, {threads}" if args.run else os.path.basename(name)
    case_line = (f"{case.nelements[0]}×{case.nelements[1]} elements  ·  p = {p}  ·  "
                 f"{ndof:,} DoFs  ·  {case.nsteps:,} steps of dt = {dt:.1e} s")
    command = None

    if args.run:
        print(f"solving with {args.solver} -> {csv}")
        t0 = time.perf_counter()
        with stdout_to(base + ".log") if args.quiet else contextlib.nullcontext():
            params.solve(SOLVERS[args.solver], csv)
        elapsed = time.perf_counter() - t0
        solver_line += f"  ·  solved in {elapsed:.2f} s"
        command = (f"python3 Weq2D_02.py --run --solver {args.solver}"
                   + (f" --cpus {args.cpus}" if args.solver == "omp" else "")
                   + f" --ne {case.nelements[0]} --pD {p} --nsteps {case.nsteps} --duration {case.duration:g}")

    u = np.atleast_2d(np.loadtxt(csv, delimiter=","))
    if u.shape[1] != ndof:
        sys.exit(f"error: {csv} has {u.shape[1]} values per row, case has {ndof}; pass the --ne/--pD of that run")

    stride = max(1, round(args.writefreq / SAMPLE_EVERY)) if args.writefreq else 1
    frames = list(range(0, len(u), stride))
    if frames[-1] != len(u) - 1:
        frames.append(len(u) - 1)

    gif = base + ".gif"
    result = [f"read      {len(u)} snapshots × {ndof:,} DoFs, u from {u.min():+.4f} to {u.max():+.4f}",
              f"gif       {os.path.basename(gif)}, {len(frames)} frames"]
    if args.run:
        result.insert(0, f"solved    {elapsed:.2f} s (wall time of the solve call, set-up included)")
        with open(base + ".log", "a") as f:   # written before rendering, so it is there even if rendering fails
            f.write("\n--- result ---\n" + "\n".join(result) + "\n")
    print("\n".join(result))
    print(f"rendering {len(frames)} frames -> {gif}")
    anim = make_animation(u, params, frames, dt, (solver_line, case_line), command, gif, args.vmax, args.dpi)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()