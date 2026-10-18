"""Opt-in C++ backend for SuperJUNO's FSMP random preparation.

Drop-in replacement for the numeric core of upstream
``src/analysis/fsmp/prepare.py::prepare_random`` (upstream commit 82f25ea).
For each ``(EventID, channel)`` it reproduces, bit for bit, the NumPy
Generator/PCG64/SeedSequence stream

    rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([seed, 1, int(eid), int(ch)])))
    istar   = rng.random(TRIALS)
    erandom = rng.random(TRIALS)
    wanders = rng.standard_normal(TRIALS, np.float64)
    accepts = np.log(rng.random(TRIALS))

by calling the C++17 library ``_sjrng_core.so``. The final ``np.log`` is
applied here with NumPy on the bit-identical raw uniforms, so no libm
substitution can perturb it. Threads only split independent streams, so the
result does not depend on the thread count.

Version scope: the port was written from the NumPy 2.4.6 sources
(numpy/random is dual NCSA/BSD-3-Clause; verbatim notices in
third_party/NUMPY_LICENSE.txt) and verified bit-for-bit against the NumPy
installed beside
it. NumPy does not promise Generator/PCG64/SeedSequence *distribution*
bitstreams across releases, so any other NumPy version must re-run
cpp/test_sjrng.py (or the root verify_juno_supervisor.py) and pass before
SUPERJUNO_SJRNG is enabled there.

The module never changes random-value ownership: the arrays it returns are
byte-for-byte what the upstream numpy loop would produce (verified locally by
cpp/test_sjrng.py). Any error here propagates so the caller can fall back to
the untouched upstream path.
"""

import ctypes
import os
import sys
import threading

import numpy as np

# One-line audit trail per process, on stderr (stage stderr goes to the
# Slurm .err file; stdout/timing stays untouched).
_notice_lock = threading.Lock()
_notice_shown = False


def _notice() -> None:
    global _notice_shown
    with _notice_lock:
        if not _notice_shown:
            _notice_shown = True
            print(
                "[SUPERJUNO] sjrng C++ random backend active "
                f"({_lib_path()})",
                file=sys.stderr,
            )


def _lib_path() -> str:
    override = os.environ.get("SJRNG_LIB")
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_sjrng_core.so")


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    path = _lib_path()
    lib = ctypes.CDLL(path)
    lib.sjrng_build_id.restype = ctypes.c_char_p
    lib.sjrng_generate.restype = ctypes.c_int
    lib.sjrng_generate.argtypes = [
        ctypes.c_longlong,  # seed
        ctypes.c_longlong,  # stream (upstream WAVEFORM_STREAM = 1)
        ctypes.c_longlong,  # n_streams
        ctypes.POINTER(ctypes.c_longlong),  # eids
        ctypes.POINTER(ctypes.c_longlong),  # chs
        ctypes.c_longlong,  # trials
        ctypes.POINTER(ctypes.c_double),  # istar (trials, n_streams)
        ctypes.POINTER(ctypes.c_double),  # erandom
        ctypes.POINTER(ctypes.c_double),  # wanders
        ctypes.POINTER(ctypes.c_double),  # accept_uniform
        ctypes.c_int,  # n_threads
    ]
    _lib = lib
    return lib


_STATUS_TEXT = {
    1: "negative n_streams or trials",
    2: "null id pointers",
    3: "negative seed/stream/eid/ch (SeedSequence entropy must be >= 0)",
    4: "null output pointers",
    5: "n_streams*trials exceeds addressable scratch",
    6: "out of memory",
    7: "internal worker error",
    8: "thread creation failed",
}


def _strict_int(name: str, value) -> int:
    # Reject floats/bools/objects: int(3.7) would silently truncate and a
    # shifted stream must fail loudly, not map to a different seed.
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    return int(value)


def _int64_column(requester, field: str):
    """Contiguous int64 copy of a requester field, without silent truncation.

    Structured-array fields are strided views: even when the dtype is already
    int64, ``astype(np.int64, copy=False)`` keeps the stride and handing the
    raw pointer to C++ would read neighbouring fields as EventIDs. A copy via
    ascontiguousarray is always required. Non-integer dtypes and uint64
    values above INT64_MAX are rejected instead of wrapped.
    """
    col = requester[field]
    if col.ndim != 1:
        raise ValueError(
            f"requester field '{field}' must be 1-D, got shape {col.shape}"
        )
    if col.dtype.kind not in "iu":
        raise ValueError(
            f"requester field '{field}' must have an integer dtype, "
            f"got {col.dtype} (silent conversion refused)"
        )
    if col.dtype.kind == "u" and col.dtype.itemsize >= 8 and col.size:
        if (col > np.iinfo(np.int64).max).any():
            raise ValueError(
                f"requester field '{field}' has values above INT64_MAX; "
                "SeedSequence entropy would be truncated (refused)"
            )
    return np.ascontiguousarray(col, dtype=np.int64)


def prepare_random(seed: int, trials: int, requester, stream: int = 1):
    """Return (istar, erandom, wanders, accepts) as upstream prepare_random.

    ``requester`` is the upstream structured array with ``eid`` and ``ch``
    fields; column i of every output belongs to requester row i.
    """
    seed = _strict_int("seed", seed)
    trials = _strict_int("trials", trials)
    stream = _strict_int("stream", stream)
    if seed < 0 or stream < 0 or trials < 0:
        raise ValueError("seed, stream and trials must be non-negative")

    eids = _int64_column(requester, "eid")
    chs = _int64_column(requester, "ch")
    if eids.shape != chs.shape:
        raise ValueError("requester eid/ch must be aligned 1-D arrays")
    if eids.size and (eids.min() < 0 or chs.min() < 0):
        raise ValueError("eid/ch must be non-negative for SeedSequence")

    n_streams = eids.size
    shape = (trials, n_streams)
    istar = np.empty(shape, dtype=np.float64)
    erandom = np.empty(shape, dtype=np.float64)
    wanders = np.empty(shape, dtype=np.float64)
    accept_uniform = np.empty(shape, dtype=np.float64)

    threads = os.environ.get("SJRNG_THREADS")
    n_threads = int(threads) if threads else 0  # 0 = library default

    lib = _load()
    status = lib.sjrng_generate(
        seed,
        stream,
        n_streams,
        eids.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        chs.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        trials,
        istar.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        erandom.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wanders.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        accept_uniform.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_threads,
    )
    if status != 0:
        raise RuntimeError(
            f"sjrng_generate failed with status {status} "
            f"({_STATUS_TEXT.get(status, 'unknown')})"
        )

    _notice()
    # np.log on the raw uniforms, exactly like upstream's fourth draw step.
    accepts = np.log(accept_uniform)
    return istar, erandom, wanders, accepts


def build_id() -> str:
    return _load().sjrng_build_id().decode()
