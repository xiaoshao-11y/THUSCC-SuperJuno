// sjrng_core.cpp — bit-exact C++17 reimplementation of the NumPy random
// streams consumed by SuperJUNO upstream `src/analysis/fsmp/prepare.py::
// prepare_random` (upstream commit 82f25ea).
//
// For every (EventID, channel) the upstream code builds
//     rng = np.random.Generator(np.random.PCG64(
//         np.random.SeedSequence([seed, 1, int(eid), int(ch)])))
// and draws, in this exact order and float64 dtype:
//     istar   = rng.random(TRIALS)                      # (TRIALS,)
//     erandom = rng.random(TRIALS)
//     wanders = rng.standard_normal(TRIALS, np.float64) # ziggurat
//     accepts = np.log(rng.random(TRIALS))              # log applied by NumPy
// This library reproduces the three RNG arrays bit-for-bit and returns the
// *raw uniform* of the fourth draw; the caller applies np.log with NumPy so
// that step stays bit-identical to upstream regardless of libm versions.
//
// Spec sources — ported from the NumPy 2.4.6 sources (the ported files all
// live in numpy/random/, dual NCSA/BSD-3-Clause; the verbatim notices from
// the locally installed NumPy 2.4.6 are reproduced in
// third_party/NUMPY_LICENSE.txt, see also README.md). This is a port of
// one inspected version, NOT a
// cross-version contract: NumPy documents PCG64's raw integer stream, but
// the Generator distribution/float layers above it have no documented
// bitstream stability promise across NumPy releases. The port was verified
// bit-for-bit against the NumPy that will run next to it (see
// cpp/test_sjrng.py), and any deployment on a different NumPy version must
// repeat that comparison before enabling SUPERJUNO_SJRNG:
//   SeedSequence  numpy/random/bit_generator.pyx   (hashmix/mix/generate_state)
//   PCG64         numpy/random/src/pcg64/pcg64.{h,c} (XSL-RR 128/64)
//   next_double   numpy/random/_pcg64.pyx uint64_to_double: (u >> 11) * 2^-53
//   normal        numpy/random/src/distributions/distributions.c
//                 random_standard_normal (ziggurat, 256 levels) + tables in
//                 numpy/random/src/distributions/ziggurat_constants.h
//
// Numerics rules honored here and enforced by ../Makefile (no -ffast-math,
// no -Ofast, no -march=native reassociation):
//   * all seeding/mixing is 32/64/128-bit integer arithmetic (exact);
//   * standard_normal uses only *, +, -, compare, std::log1p, std::exp with
//     the same operands in the same order as numpy's C code;
//   * log1p/exp resolve to the same glibc symbols numpy's npy_log1p/npy-exp
//     wrappers call on Linux, so results are bit-identical on the same
//     libm; test_sjrng.py verifies this exhaustively against local numpy.

#include "sjrng_ziggurat_tables.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#if defined(__SIZEOF_INT128__)
using u128 = unsigned __int128;
#else
#error "sjrng_core.cpp requires a 128-bit integer type (unsigned __int128)"
#endif

namespace {

// ---------------------------------------------------------------- SeedSequence
// numpy/random/bit_generator.pyx
constexpr std::uint32_t kInitA = 0x43b0d7e5u;    // INIT_A
constexpr std::uint32_t kMultA = 0x931e8875u;    // MULT_A
constexpr std::uint32_t kInitB = 0x8b51f9ddu;    // INIT_B
constexpr std::uint32_t kMultB = 0x58f38dedu;    // MULT_B
constexpr std::uint32_t kMixMultL = 0xca01f9ddu; // MIX_MULT_L
constexpr std::uint32_t kMixMultR = 0x4973f715u; // MIX_MULT_R
constexpr std::uint32_t kXShift = 16;            // uint32 itemsize * 8 / 2

inline std::uint32_t hashmix(std::uint32_t value, std::uint32_t &hash_const) {
    // hash_const is in-out, matching the Cython `uint32_t * hash_const`.
    value ^= hash_const;
    hash_const *= kMultA;
    value *= hash_const;
    value ^= value >> kXShift;
    return value;
}

inline std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = kMixMultL * x - kMixMultR * y;
    result ^= result >> kXShift;
    return result;
}

// _int_to_uint32_array: non-negative integer -> little-endian base-2^32 words.
inline std::vector<std::uint32_t> int_to_uint32_words(std::int64_t value) {
    std::vector<std::uint32_t> words;
    if (value == 0) {
        words.push_back(0);
        return words;
    }
    std::uint64_t n = static_cast<std::uint64_t>(value); // caller rejects < 0
    while (n > 0) {
        words.push_back(static_cast<std::uint32_t>(n & 0xFFFFFFFFu));
        n >>= 32;
    }
    return words;
}

// SeedSequence(pool_size=4).generate_state-equivalent: fill `pool` (the mixed
// 4-word entropy pool) from the entropy word list, then emit `n_words` words.
inline void seedsequence_pool(std::uint32_t pool[4],
                              const std::vector<std::uint32_t> &entropy) {
    const std::size_t n_ent = entropy.size();
    std::uint32_t hash_const = kInitA;

    // mix_entropy: first pass up to the pool size.
    for (std::size_t i = 0; i < 4; ++i) {
        pool[i] = hashmix(i < n_ent ? entropy[i] : 0, hash_const);
    }
    // Mix all pool words together so late entropy affects early words.
    for (std::size_t i_src = 0; i_src < 4; ++i_src) {
        for (std::size_t i_dst = 0; i_dst < 4; ++i_dst) {
            if (i_src != i_dst) {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], hash_const));
            }
        }
    }
    // Entropy beyond the pool size (happens when an id needs > 1 word and the
    // total entropy exceeds 4 words) is mixed in word by word.
    for (std::size_t i_src = 4; i_src < n_ent; ++i_src) {
        for (std::size_t i_dst = 0; i_dst < 4; ++i_dst) {
            pool[i_dst] = mix(pool[i_dst], hashmix(entropy[i_src], hash_const));
        }
    }
}

inline void seedsequence_generate4(std::uint64_t out[4],
                                   const std::uint32_t pool[4]) {
    // generate_state(4, dtype=uint64) -> 8 uint32 words, then little-endian
    // pairing into uint64: u64[i] = w[2i] | (w[2i+1] << 32).
    std::uint32_t w[8];
    std::uint32_t hash_const = kInitB;
    for (int i = 0; i < 8; ++i) {
        std::uint32_t data_val = pool[i % 4]; // itertools.cycle(self.pool)
        data_val ^= hash_const;
        hash_const *= kMultB;
        data_val *= hash_const;
        data_val ^= data_val >> kXShift;
        w[i] = data_val;
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<std::uint64_t>(w[2 * i]) |
                 (static_cast<std::uint64_t>(w[2 * i + 1]) << 32);
    }
}

// ------------------------------------------------------------------ PCG64
// numpy/random/src/pcg64/pcg64.h (native 128-bit branch).
constexpr u128 kPcgMultiplier =
    (static_cast<u128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;

inline std::uint64_t rotr64(std::uint64_t value, unsigned int rot) {
    return (value >> rot) | (value << ((64u - rot) & 63u));
}

struct PCG64 {
    u128 state;
    u128 inc;

    inline void seed(std::uint64_t seed_words[4]) {
        // pcg64_set_seed + pcg_setseq_128_srandom_r.
        const u128 initstate =
            (static_cast<u128>(seed_words[0]) << 64) | seed_words[1];
        const u128 initseq =
            (static_cast<u128>(seed_words[2]) << 64) | seed_words[3];
        state = 0;
        inc = (initseq << 1) | 1;
        step();
        state += initstate;
        step();
    }

    inline void step() { state = state * kPcgMultiplier + inc; }

    inline std::uint64_t next64() {
        step();
        const std::uint64_t hi = static_cast<std::uint64_t>(state >> 64);
        const std::uint64_t lo = static_cast<std::uint64_t>(state);
        return rotr64(hi ^ lo, static_cast<unsigned int>(hi >> 58));
    }

    // uint64_to_double (numpy/random/_pcg64.pyx): (u >> 11) * 2^-53.
    inline double next_double() {
        return static_cast<double>(next64() >> 11) *
               (1.0 / 9007199254740992.0);
    }
};

// --------------------------------------------- ziggurat standard normal
// numpy/random/src/distributions/distributions.c :: random_standard_normal,
// double variant, statement-for-statement.
inline double standard_normal(PCG64 &g) {
    for (;;) {
        std::uint64_t r = g.next64();
        const int idx = static_cast<int>(r & 0xff);
        r >>= 8;
        const std::uint64_t sign = r & 0x1;
        const std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
        double x = static_cast<double>(rabs) * wi_double[idx];
        if (sign & 0x1) {
            x = -x;
        }
        if (rabs < ki_double[idx]) {
            return x; // 99.3% fast path
        }
        if (idx == 0) {
            // Tail block: Marsaglia-Tsang, using 1.0 - U to avoid log(0).
            for (;;) {
                const double xx =
                    -ziggurat_nor_inv_r * std::log1p(-g.next_double());
                const double yy = -std::log1p(-g.next_double());
                if (yy + yy > xx * xx) {
                    return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                                               : (ziggurat_nor_r + xx);
                }
            }
        } else {
            // Wedge between levels: uniform comparison against exp(-x^2/2).
            if ((fi_double[idx - 1] - fi_double[idx]) * g.next_double() +
                    fi_double[idx] <
                std::exp(-0.5 * x * x)) {
                return x;
            }
        }
    }
}

// ------------------------------------------------------------------ driver
struct StreamArgs {
    std::int64_t seed;
    std::int64_t stream;
    std::int64_t eid;
    std::int64_t ch;
};

// Generate one stream into four contiguous per-stream rows of the scratch
// buffers tmp[k][stream * trials + t] (row-major (n_streams, trials)).
// Contiguous per-stream fill avoids the (trials, n_streams) strided-column
// writes the upstream Python loop performs; a blocked transpose below moves
// the bits into the final layout unchanged.
void generate_stream(const StreamArgs &args, std::int64_t trials,
                     std::size_t stream_index, double *tmp_istar,
                     double *tmp_erandom, double *tmp_wanders,
                     double *tmp_accept) {
    std::vector<std::uint32_t> entropy;
    {
        std::vector<std::uint32_t> w;
        w = int_to_uint32_words(args.seed);
        entropy.insert(entropy.end(), w.begin(), w.end());
        w = int_to_uint32_words(args.stream);
        entropy.insert(entropy.end(), w.begin(), w.end());
        w = int_to_uint32_words(args.eid);
        entropy.insert(entropy.end(), w.begin(), w.end());
        w = int_to_uint32_words(args.ch);
        entropy.insert(entropy.end(), w.begin(), w.end());
    }

    std::uint32_t pool[4];
    seedsequence_pool(pool, entropy);
    std::uint64_t seed_words[4];
    seedsequence_generate4(seed_words, pool);

    PCG64 rng;
    rng.seed(seed_words);

    const std::size_t base = stream_index * static_cast<std::size_t>(trials);
    for (std::int64_t t = 0; t < trials; ++t) {
        tmp_istar[base + t] = rng.next_double();
    }
    for (std::int64_t t = 0; t < trials; ++t) {
        tmp_erandom[base + t] = rng.next_double();
    }
    for (std::int64_t t = 0; t < trials; ++t) {
        tmp_wanders[base + t] = standard_normal(rng);
    }
    for (std::int64_t t = 0; t < trials; ++t) {
        // Upstream applies np.log after the draw; the Python wrapper does the
        // same with the returned raw uniforms so libm choice cannot matter.
        tmp_accept[base + t] = rng.next_double();
    }
}

// dst(t, s) = src(s, t); each thread transposes the trial-row strip
// [t_begin, t_end) with 64x64 tiles that stay resident in L1/L2, so both
// sides of every tile touch memory contiguously.
void transpose_blocked(const double *src, double *dst, std::int64_t n_streams,
                       std::int64_t trials, std::int64_t t_begin,
                       std::int64_t t_end) {
    constexpr std::int64_t kTile = 64;
    for (std::int64_t s0 = 0; s0 < n_streams; s0 += kTile) {
        const std::int64_t s1 = std::min(s0 + kTile, n_streams);
        for (std::int64_t t0 = t_begin; t0 < t_end; t0 += kTile) {
            const std::int64_t t1 = std::min(t0 + kTile, t_end);
            for (std::int64_t t = t0; t < t1; ++t) {
                const double *srow = src + t; // column t across streams
                double *drow = dst + t * n_streams;
                for (std::int64_t s = s0; s < s1; ++s) {
                    drow[s] = srow[s * trials];
                }
            }
        }
    }
}

// Launch fn(args...) on a new thread. If creation fails, join every thread
// started so far and return false so the caller can unwind cleanly instead
// of leaking running threads or letting an exception cross the C boundary.
template <typename Fn, typename... Args>
bool spawn_or_join(std::vector<std::thread> &threads, Fn &&fn,
                   Args &&...args) {
    try {
        threads.emplace_back(std::forward<Fn>(fn),
                             std::forward<Args>(args)...);
        return true;
    } catch (...) { // std::system_error etc.
        for (auto &thread : threads) {
            thread.join();
        }
        threads.clear();
        return false;
    }
}

} // namespace

// ------------------------------------------------------------------ C ABI
// Consumed by cpp/sjrng.py via ctypes. All four outputs are caller-allocated
// contiguous float64 arrays of shape (trials, n_streams), column i = stream i.
//
// This is a C ABI: no exception may escape (crossing ctypes is undefined).
// Status codes:
//   0  success
//   1  negative n_streams or trials
//   2  n_streams > 0 but eids/chs null
//   3  negative seed/stream or negative eid/ch (SeedSequence entropy >= 0)
//   4  trials > 0 but an output pointer is null
//   5  n_streams*trials cannot fit the addressable scratch (checked by
//      division before any multiplication)
//   6  out of memory
//   7  unexpected error inside a worker (never aborts the process)
//   8  thread creation failed (already-started threads joined first)
extern "C" {

const char *sjrng_build_id() {
    return "sjrng_core 1.0 (numpy Generator/PCG64/SeedSequence + ziggurat "
           "normal, bit-exact port)";
}

int sjrng_generate(long long seed, long long stream, long long n_streams,
                   const long long *eids, const long long *chs,
                   long long trials, double *istar, double *erandom,
                   double *wanders, double *accept_uniform, int n_threads) {
    if (n_streams < 0 || trials < 0) {
        return 1;
    }
    if (n_streams > 0 && (eids == nullptr || chs == nullptr)) {
        return 2;
    }
    if (seed < 0 || stream < 0) {
        return 3; // SeedSequence rejects negative entropy
    }
    for (long long i = 0; i < n_streams; ++i) {
        if (eids[i] < 0 || chs[i] < 0) {
            return 3;
        }
    }
    if (n_streams == 0 || trials == 0) {
        return 0;
    }
    if (istar == nullptr || erandom == nullptr || wanders == nullptr ||
        accept_uniform == nullptr) {
        return 4;
    }
    // Overflow guard, computed with division only: the four scratch arrays
    // need 4 * total doubles, so total must satisfy
    // 4 * total * sizeof(double) <= SIZE_MAX. Comparing via division avoids
    // the signed-overflow UB of multiplying first.
    const unsigned long long total_max =
        std::numeric_limits<std::size_t>::max() / 4 / sizeof(double);
    if (static_cast<unsigned long long>(n_streams) >
        total_max / static_cast<unsigned long long>(trials)) {
        return 5;
    }
    const std::size_t total = static_cast<std::size_t>(n_streams) *
                              static_cast<std::size_t>(trials);

    long long threads_wanted = n_threads > 0
                                   ? static_cast<long long>(n_threads)
                                   : (std::thread::hardware_concurrency()
                                          ? static_cast<long long>(
                                                std::thread::
                                                    hardware_concurrency())
                                          : 1LL);
    if (threads_wanted > n_streams) {
        threads_wanted = n_streams; // n_streams >= 1 here, fits in int range
    }

    // Scratch rows are (n_streams, trials); outputs are (trials, n_streams).
    std::vector<double> tmp;
    try {
        tmp.resize(static_cast<std::size_t>(4) * total);
    } catch (const std::bad_alloc &) {
        return 6; // caller falls back to the upstream numpy path
    } catch (...) {
        return 7;
    }
    double *tmp_istar = tmp.data();
    double *tmp_erandom = tmp_istar + total;
    double *tmp_wanders = tmp_erandom + total;
    double *tmp_accept = tmp_wanders + total;

    // Worker bodies must not let exceptions escape into std::thread (that
    // calls std::terminate); record the first failure and let the caller
    // report it after joining.
    std::atomic<int> worker_error{0};
    auto worker = [&](long long begin, long long end) {
        try {
            for (long long i = begin; i < end; ++i) {
                generate_stream(StreamArgs{seed, stream, eids[i], chs[i]},
                                trials, static_cast<std::size_t>(i),
                                tmp_istar, tmp_erandom, tmp_wanders,
                                tmp_accept);
            }
        } catch (...) {
            worker_error.store(7, std::memory_order_relaxed);
        }
    };

    if (threads_wanted == 1) {
        worker(0, n_streams);
    } else {
        std::vector<std::thread> threads;
        const long long chunk =
            (n_streams + threads_wanted - 1) / threads_wanted;
        bool ok = true;
        for (long long t = 0; t < threads_wanted && ok; ++t) {
            const long long begin = t * chunk;
            const long long end =
                begin + chunk > n_streams ? n_streams : begin + chunk;
            if (begin >= end) {
                break;
            }
            ok = spawn_or_join(threads, worker, begin, end);
        }
        if (ok) {
            for (auto &thread : threads) {
                thread.join();
            }
        } else {
            return 8; // threads already joined by spawn_or_join
        }
    }
    if (worker_error.load() != 0) {
        return worker_error.load();
    }

    // Blocked, bit-preserving transpose into the final column layout.
    double *outputs[4] = {istar, erandom, wanders, accept_uniform};
    const double *sources[4] = {tmp_istar, tmp_erandom, tmp_wanders,
                                tmp_accept};
    auto transpose_worker = [&](long long t_begin, long long t_end) {
        try {
            for (int k = 0; k < 4; ++k) {
                transpose_blocked(sources[k], outputs[k], n_streams, trials,
                                  t_begin, t_end);
            }
        } catch (...) {
            worker_error.store(7, std::memory_order_relaxed);
        }
    };
    if (threads_wanted == 1) {
        transpose_worker(0, trials);
    } else {
        std::vector<std::thread> threads;
        const long long tchunk = (trials + threads_wanted - 1) / threads_wanted;
        bool ok = true;
        for (long long t = 0; t < threads_wanted && ok; ++t) {
            const long long begin = t * tchunk;
            const long long end =
                begin + tchunk > trials ? trials : begin + tchunk;
            if (begin >= end) {
                break;
            }
            ok = spawn_or_join(threads, transpose_worker, begin, end);
        }
        if (ok) {
            for (auto &thread : threads) {
                thread.join();
            }
        } else {
            return 8;
        }
    }
    if (worker_error.load() != 0) {
        return worker_error.load();
    }
    return 0;
}

} // extern "C"
