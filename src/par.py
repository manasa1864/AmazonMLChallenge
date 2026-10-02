"""Parallel map that survives intermittent Windows/loky worker crashes (BrokenProcessPool): retry, then fall back to serial.

Set the environment variable AML_JOBS (e.g. 8) to cap the number of worker processes / LightGBM threads, which keeps a
laptop cooler at the cost of a longer run."""
import os
import time
from joblib import Parallel, delayed

JOBS_CAP = int(os.environ.get("AML_JOBS", "0")) or None       # None = no cap


def par_map(fn, arg_tuples, n_jobs: int = 20, tries: int = 3):
    arg_tuples = list(arg_tuples)
    if JOBS_CAP:
        n_jobs = min(n_jobs, JOBS_CAP)
    for i in range(tries):
        try:
            return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(*a) for a in arg_tuples)
        except Exception as e:                      # BrokenProcessPool, OSError(WinError 6), ...
            print(f"  [par_map] attempt {i + 1}/{tries} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(3)                           # joblib rebuilds its worker pool on the next call
    print("  [par_map] falling back to serial execution", flush=True)
    return [fn(*a) for a in arg_tuples]
