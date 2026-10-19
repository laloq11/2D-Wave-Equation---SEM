#!/usr/bin/env python3
import os
import argparse
import subprocess
import multiprocessing
import csv
from dataclasses import dataclass

'''
This script is meant to be used for comparing omp and cuda solvers

The goal is to compute at different
    1 number of elements,
    2 polynomial degrees

Based on this numbers we are going to be able to
    define future parallelism efforts

Updates 05.10.2026.

--run --solver omp cu_pDegreeThread --ne 5 --pD 2 3 --cpus 8 --gpus 1 --reps 1 --filename test_SEM.csv


'''

HERE = os.path.dirname(os.path.abspath(__file__))
# installed: build/install/lib/benchmark_solver.py | source: python/scripts/benchmark_solver.py
EXE_CANDIDATES = [
    os.path.join(HERE, "..", "..", "roofline", "benchmark_solver"),
    os.path.join(HERE, "..", "..", "build", "roofline", "benchmark_solver"),
]


@dataclass(slots=True)   # slots=True requires Python >= 3.10
class roofline_setup:
    run: bool
    solver: list[str]
    ne: list[int]
    pD: list[int]
    cpus: list[int]
    gpus: list[int]
    reps: int
    warmup: int
    duration: float
    nsteps: list[int]
    filename: str
    exe: str


# Cp = [1.000000000000000, 0.372677996249965, 0.182574185835055, 0.108425405035419,
#             0.072525190378972, 0.052026033161093, 0.039153969663832, 0.030531704731319,
#             0.024472468861462, 0.020052169824593]
Cp = [ 0.707106781186548, 0.288675134594813, 0.164054624686693, 0.104442278311696,
            0.071432315609361, 0.051655014512142, 0.039005419539448, 0.030464365739356,
            0.024438904279170, 0.020034146407630 ]
safety = 0.9
c=1.0

def default_threads():
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads is not None:
        return int(omp_threads)
    # Fallback to the number of physical/logical cores
    return multiprocessing.cpu_count()


def default_exe():
    for exe in EXE_CANDIDATES:
        if os.path.isfile(exe):
            return os.path.normpath(exe)
    return None

# defaults of benchmark_solver (--duration, --nsteps)
SOLVER_DURATION = 12.0
SOLVER_NSTEPS = 1000

SOLVERS = ["omp", "cu_singleThread", "cu_pDegreeThread"]   # -d values of benchmark_solver

def run_case(exe, device, ne, pD, threads, reps, warmup, nsteps, simdata_filename):
    global Cp, safety, c
    """Runs one case and returns (mean, median, min, max) in s of the timed repetitions,
    as printed by benchmark_solver --stats 1."""
    h = 2.0 / ne
    dt = safety * Cp[pD - 1] * h / c
    duration = dt * nsteps
    cmd = [exe, "-d", device, "-n", str(ne), "-p", str(pD),
           "--threads", str(threads), "--num_reps", str(reps), "--warmup", str(warmup), "--stats", "1",
           "--filename", simdata_filename, "--duration", str(duration), "--nsteps", str(nsteps)]
    if os.path.exists(simdata_filename):
        os.remove(simdata_filename)   # the C++ writer appends; start from an empty file
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
    out = result.stdout.strip()
    if not out:
        raise RuntimeError(f"no output from: {' '.join(cmd)} (unknown device?)")
    return tuple(float(x) for x in out.splitlines()[-1].split())



def FULL_roofline_setup(args):
    # full_solver = ["omp"]
    full_solver = ["omp","cu_pDegreeThread", "cu_singleThread"]
    # full_gpu= [8,16]
    full_gpu= [1]

    # blocksize_cuda=[32,64,128,256]

    full_cpu=[36,16,8,4]
    # full_cpu=[1]
    # blocksize_omp=[1,4,8,16]

    # strategies=["st_shmem"]
    # sizes=[2**x for x in range(4, 15, 2)]

    full_pd = [2,3,4]
    full_ne = [5,10,20,30,80]

    return roofline_setup(args.run, full_solver, full_ne, full_pd, full_cpu, full_gpu, args.reps, args.warmup, args.duration, args.nsteps, args.filename, args.exe)

def createcase(args):
    if args.full:
        return FULL_roofline_setup(args)
    return roofline_setup(args.run, args.solver, args.ne, args.pD, args.cpus, args.gpus, args.reps, args.warmup, args.duration, args.nsteps, args.filename, args.exe)

def parsargs():
    parser = argparse.ArgumentParser(description="Run benchmark_solver for omp/cuda and store timings in a csv.")
    parser.add_argument("--run", action="store_true", help="Run the benchmark. Without it, only prints the csv.")
    parser.add_argument("--full", action="store_true", help="Use the full roofline setup (FULL_roofline_setup) instead of --solver/--ne/--pD/--cpus/--gpus")
    parser.add_argument("--solver", nargs="+", default=["omp", "cu_pDegreeThread"], choices=SOLVERS, help="Solvers to run")
    parser.add_argument("--ne", nargs="+", type=int, default=[10, 20, 30, 40], help="Number of elements per direction")
    parser.add_argument("--pD", nargs="+", type=int, default=[2, 3, 4], choices=range(1, len(Cp) + 1), metavar="PD", help="Polynomial degrees (1 to 10)")
    parser.add_argument("--cpus", nargs="+", type=int, default=[default_threads()], help="CPU thread counts (omp)")
    parser.add_argument("--gpus", nargs="+", type=int, default=[1], help="GPU thread counts (cu_singleThread, cu_pDegreeThread)")
    parser.add_argument("--reps", type=int, default=3, help="Timed repetitions per case (time_s is the mean; median, min and max are stored too)")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed runs before the timed repetitions of each case")
    parser.add_argument("--duration", type=float, default=SOLVER_DURATION, help="Simulated time")
    parser.add_argument("--nsteps", nargs="+", type=int, default=[SOLVER_NSTEPS], help="Time steps (dt = duration/nsteps)")
    parser.add_argument("--filename", type=str, default="benchmark_solver.csv", help="CSV file name for output or input.")
    parser.add_argument("--exe", type=str, default=default_exe(), help="Path to the benchmark_solver executable")
    args = parser.parse_args()
    if max(args.pD) > 5 and any(s != "omp" for s in args.solver) and not args.full:
        parser.error("the CUDA solvers support --pD up to 5")
    return args




def maindef():
    global Cp, safety, c
    args = parsargs()
    case=createcase(args)
    if case.run:
        if case.exe is None or not os.path.isfile(case.exe):
            raise SystemExit(f"[error] benchmark_solver not found, pass --exe. Tried: {EXE_CANDIDATES}")
        print(f"[info] running benchmark with {case.exe}")

        with open(case.filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["device", "ne", "pD", "threads", "reps", "duration", "nsteps", "dt","time_s",
                             "time_median_s", "time_min_s", "time_max_s", "warmup"])
            for d in case.solver:
                for threads in (case.cpus if d == "omp" else case.gpus):
                    for ne in case.ne:
                        for pD in case.pD:
                            for nsteps in case.nsteps:
                                h = 2.0 / ne ## standard size also in cpp files. 
                                dt = safety * Cp[pD - 1] * h / c
                                duration = dt * nsteps
                                simdata_filename = f"SEM_{d}_ne{ne}_pd{pD}_cpu{threads}_run{case.reps}_dt{dt:.0e}.csv"
                                t, t_median, t_min, t_max = run_case(case.exe, d, ne, pD, threads, case.reps, case.warmup, nsteps, simdata_filename)
                                print(f"[info] {d:5s} ne={ne:3d} pD={pD} threads={threads:2d} dt={dt:.2e} -> {t:.6f} s"
                                      f" (median {t_median:.6f}, min {t_min:.6f}, max {t_max:.6f})")
                                writer.writerow([d, ne, pD, threads, case.reps, duration, nsteps,dt, t,
                                                 t_median, t_min, t_max, case.warmup])
                                csvfile.flush()

    with open(case.filename) as csvfile:
        print(f"[info] {case.filename}:")
        print(csvfile.read())


if __name__ == "__main__":
    print("[info] Welcome.")
    maindef()
    print("[info] Goodbye.")
