// Build (GCC, on the benchmark machine):
// g++ -std=c++17 -O3 -march=native -flto -DNDEBUG -pthread main.cpp -o chacha20gf1024lfsr1024
// Run: ./chacha20gf1024lfsr1024 30000000 4 [full]
// Requires the unchanged ChaCha20GF1024LFSR1024.h and toy_kwise_1024_test.h.
// This adaptation has not been compiled or executed.
#include "ChaCha20GF1024LFSR1024.h"
#include "ChaCha20GF1024LFSR1024Parallel.h"
#include "ChaCha20GF1024LFSR1024Seed.h"
#include "toy_kwise_1024_test.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <immintrin.h>
#  include <intrin.h>
#  define DEMO_MSVC_X86 1
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#  include <immintrin.h>
#  define DEMO_GNU_X86 1
#  define DEMO_TARGET_RDRND __attribute__((target("rdrnd")))
#else
#  define DEMO_TARGET_RDRND
#endif

using RNG = ChaCha20GF1024LFSR1024;
using Position128 = RNG::Position;
using Field128 = RNG::Field;

static uint64_t sm64(uint64_t& s)
{
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void store64_le(uint8_t* p, uint64_t v)
{
    for (int i = 0; i < 8; ++i)
        p[i] = static_cast<uint8_t>(v >> (8 * i));
}

// This is reproducible benchmark data, NOT a claim of independent entropy.
// The expensive structured seed is prepared once and reused throughout.
static const RNG::PreparedSeed& reproducible_seed()
{
    static const RNG::PreparedSeed seed = [] {
        auto fill = [state = uint64_t{0x123456789ABCDEF0ULL}]
                    (uint8_t* dst, std::size_t count) mutable {
            while (count != 0) {
                const uint64_t v = sm64(state);
                const std::size_t n = std::min<std::size_t>(count, 8);
                for (std::size_t j = 0; j < n; ++j)
                    dst[j] = static_cast<uint8_t>(v >> (8 * j));
                dst += n;
                count -= n;
            }
        };
        return RNG::make_seed(fill);
    }();
    return seed;
}

static Position128 add_pos(Position128 p, uint64_t delta)
{
    if (p.add_u64(delta))
        throw std::overflow_error("test position overflow");
    return p;
}

// Regression vector for the new ChaCha20GF1024LFSR1024Counter128 layout:
// zero 256-bit key, stream_id = 0, block position = 0.
// The stream-id subkey is HChaCha20(key, stream_id || 0), then words 12..15
// are used as one 128-bit block index.
static bool test_chacha_zero_vector()
{
    static const uint64_t expected[8] = {
        0xd1013fbf182ad0bcULL, 0xacfda8a730de9292ULL,
        0xc72c00a6505eb6a4ULL, 0xd5c31ac9f7d2d62cULL,
        0xcfbfd2aae0838f72ULL, 0xddae8fb52d2dbd9aULL,
        0x139bc03fd85d0165ULL, 0x0f8e9e014310271eULL
    };

    uint8_t key[32] = {};
    ChaCha20GF1024LFSR1024Counter128 c(key, 0);
    for (int i = 0; i < 8; ++i)
        if (c.next_int() != expected[i]) return false;
    return true;
}

static bool test_embedded_polynomial_regression_vector()
{
    static const uint64_t expected[16] = {
        0xde9aff20072a8ee4ULL, 0x680b3c09a5cb1a1bULL,
        0xcf4adae3ecb51785ULL, 0xa0e824c25c835030ULL,
        0x12a3a8e92708f081ULL, 0x94b5d72863c6bc56ULL,
        0xf7362ba3c5e9b8c7ULL, 0xaa1bdb38b38755e3ULL,
        0xb148bb6c24e2eeddULL, 0x98a32053d2550c82ULL,
        0x1086db2d3e2075a1ULL, 0xe94c0e238000c700ULL,
        0xe0faed830bd798a1ULL, 0x43303620284f904cULL,
        0xfbd4a81b79200de2ULL, 0x04b4c74ded93e16bULL
    };

    RNG::Seed seed = reproducible_seed().bytes();
    // Keep the existing known-answer vector without inventing a new one:
    // embed its degree-511 polynomial into the 1024-coefficient family.
    // The first 32 + 8192 seed bytes are unchanged; higher coefficients are zero.
    // This checks the larger FFT, but not the added nonzero coefficients.
    constexpr std::size_t old_gf_bytes = 512 * RNG::GF_ELEMENT_BYTES;
    std::memset(seed.data() + RNG::CHACHA_SEED_BYTES + old_gf_bytes, 0,
                RNG::GF_SEED_BYTES - old_gf_bytes);
    // Zero numerator is a valid LFSR state and produces a zero mask. This
    // preserves the existing known-answer vector; it is not a new LFSR vector.
    std::memset(seed.data() + RNG::LFSR_NUMERATOR_OFFSET, 0, 128);
    ChaCha20GF1024LFSR1024 rng(seed);
    for (uint64_t v : expected)
        if (rng.next_int() != v) return false;
    return true;
}

// Composition reference: portable GF Horner + standalone ChaCha + LFSR
// debug seek. LFSR arithmetic is checked separately by a bit-serial reference.
// This is not a newly measured known-answer vector.
static bool test_hybrid_full_seed_reference()
{
    const auto& seed = reproducible_seed();
    constexpr uint64_t stream_id = 0x123456789ABCDEF0ULL;
    ChaCha20GF1024LFSR1024 hybrid(seed, stream_id);
    ChaCha20GF1024LFSR1024Counter128 chacha(seed.bytes().data(), stream_id);
    constexpr uint64_t block = RNG::OUTPUT_WORDS_PER_FFT;
    const Position128 positions[] = {
        {0, 0}, {1, 0}, {block - 2, 0}, {block - 1, 0},
        {block, 0}, {block + 1, 0}, {2 * block - 1, 0},
        {~uint64_t{0}, 0}, {0, 1}, {1, 1},
        {0x123456789ABCDEF0ULL, 0x0123456789ABCDEFULL},
        {~uint64_t{0}, ~uint64_t{0}}
    };
    for (const Position128 p : positions) {
        const Field128 g = hybrid.debug_gf_horner128(p.shr1());
        chacha.seek(p);
        const uint64_t expected = chacha.next_int() ^ ((p.lo & 1) ? g.hi : g.lo)
                                ^ hybrid.debug_lfsr_word(p);
        hybrid.seek(p);
        if (hybrid.next_int() != expected) return false;
    }
    return true;
}

// Independent slow recurrence: emit constant coefficient, cancel Q if it
// is one, then divide by z. No PCLMUL, inverse-series, or jump-table reuse.
struct BitSerialLfsrReference
{
    uint64_t q[16]{};
    uint64_t a[16]{};

    explicit BitSerialLfsrReference(const RNG::Seed& seed)
    {
        for (std::size_t i = 0; i < 16; ++i) {
            for (unsigned b = 0; b < 8; ++b) {
                q[i] |= uint64_t(seed[RNG::LFSR_POLYNOMIAL_OFFSET + 8 * i + b]) << (8 * b);
                a[i] |= uint64_t(seed[RNG::LFSR_NUMERATOR_OFFSET + 8 * i + b]) << (8 * b);
            }
        }
    }

    uint64_t next_word()
    {
        uint64_t word = 0;
        for (unsigned bit = 0; bit < 64; ++bit) {
            const uint64_t out = a[0] & 1;
            word |= out << bit;
            if (out != 0)
                for (unsigned j = 0; j < 16; ++j) a[j] ^= q[j];
            for (unsigned j = 0; j < 15; ++j)
                a[j] = (a[j] >> 1) | (a[j + 1] << 63);
            a[15] = (a[15] >> 1) | (out << 63);
        }
        return word;
    }
};

static bool test_lfsr_bit_reference()
{
    const auto& seed = reproducible_seed();
    RNG hybrid(seed);
    ChaCha20GF1024LFSR1024Counter128 chacha(seed.bytes().data());
    BitSerialLfsrReference reference(seed.bytes());
    const uint64_t seeks[] = {0, 1, 15, 16, 17, 1023, 1024, 2047, 2048, 4098};
    std::size_t next_seek = 0;
    for (uint64_t i = 0; i <= 4098; ++i) {
        const uint64_t mask = reference.next_word();
        const Field128 gf = hybrid.debug_gf_fft128(Position128(i / 2));
        const uint64_t expected = chacha.next_int() ^ ((i & 1) ? gf.hi : gf.lo) ^ mask;
        if (hybrid.next_int() != expected)
            return false;
        if (next_seek < sizeof(seeks) / sizeof(seeks[0]) && i == seeks[next_seek]) {
            if (hybrid.debug_lfsr_word(Position128(i)) != mask)
                return false;
            ++next_seek;
        }
    }
    return true;
}

static bool test_structured_seed()
{
    const auto& seed = reproducible_seed();
    RNG original(seed), restored(seed.bytes());
    for (unsigned i = 0; i < 64; ++i)
        if (original.next_int() != restored.next_int()) return false;

    RNG::Seed bad = seed.bytes();
    bad[RNG::LFSR_POLYNOMIAL_OFFSET] &= uint8_t{0xfe};
    bool even_rejected = false;
    try { (void)RNG::prepare_seed(bad); }
    catch (const std::invalid_argument&) { even_rejected = true; }

    // Q = z^1024+1 is reducible, despite having the required constant bit.
    std::memset(bad.data() + RNG::LFSR_POLYNOMIAL_OFFSET, 0, 128);
    bad[RNG::LFSR_POLYNOMIAL_OFFSET] = 1;
    bool reducible_rejected = false;
    try { (void)RNG::prepare_seed(bad); }
    catch (const std::invalid_argument&) { reducible_rejected = true; }
    return even_rejected && reducible_rejected;
}

static bool test_gf_arithmetic()
{
    uint64_t s = 0xA0761D6478BD642FULL;
    const Field128 one{1, 0};

    for (int i = 0; i < 100000; ++i) {
        const Field128 a{sm64(s), sm64(s)};
        const Field128 b{sm64(s), sm64(s)};
        const Field128 c{sm64(s), sm64(s)};

        const Field128 p = RNG::debug_gf_mul_portable(a, b);
        const Field128 f = RNG::debug_gf_mul_fast(a, b);
        if (p != f) return false;

        if (RNG::debug_gf_mul_fast(a, b ^ c) !=
            (RNG::debug_gf_mul_fast(a, b) ^
             RNG::debug_gf_mul_fast(a, c))) return false;

        if (RNG::debug_gf_mul_fast(a, one) != a) return false;
    }
    return true;
}

static bool test_fft_vs_horner()
{
    const Position128 fixed[] = {
        {0,0}, {1,0}, {2,0}, {7,0}, {8,0}, {255,0}, {256,0},
        {510,0}, {511,0}, {512,0}, {513,0},
        {1022,0}, {1023,0}, {1024,0}, {1025,0}, {2047,0}, {2048,0},
        {1234,0}, {65535,0}, {65536,0},
        {0x123456789ABC0000ULL,0},
        {0xFFFFFFFFFFFFFC00ULL,0},
        {0xFFFFFFFFFFFFFFFFULL,0},
        {0,1},
        {1,1},
        {0x123456789ABCDEF0ULL,0x0123456789ABCDEFULL},
        {0xFFFFFFFFFFFFFC00ULL,0xFEDCBA9876543210ULL}
    };

    for (int seed_no = 0; seed_no < 4; ++seed_no) {
        RNG::Seed seed = reproducible_seed().bytes();
        uint64_t ss = 0x123456789ABCDEF0ULL ^
                      (0x9E3779B97F4A7C15ULL * uint64_t(seed_no + 1));
        for (std::size_t i = 0; i < RNG::BASE_SEED_BYTES; i += 8)
            store64_le(seed.data() + i, sm64(ss));
        ChaCha20GF1024LFSR1024 g(seed);

        for (const Position128 p : fixed)
            if (g.debug_gf_fft128(p) != g.debug_gf_horner128(p)) return false;

        uint64_t r = 0xD1B54A32D192ED03ULL ^ uint64_t(seed_no);
        for (int i = 0; i < 500; ++i) {
            const Position128 p{sm64(r), sm64(r)};
            if (g.debug_gf_fft128(p) != g.debug_gf_horner128(p)) return false;
        }
    }
    return true;
}

static bool test_seek()
{
    const auto& seed = reproducible_seed();

    // Ordinary backward-compatible 64-bit seek check.
    {
        ChaCha20GF1024LFSR1024 seq(seed), rnd(seed);
        constexpr uint64_t POS = 12345;
        uint64_t expected = 0;
        for (uint64_t i = 0; i <= POS; ++i) expected = seq.next_int();
        rnd.seek(POS);
        if (expected != rnd.next_int()) return false;
    }

    // New 128-bit check: walk across the low-64-bit wrap and compare every
    // sequential output with an independent direct seek to the same position.
    {
        const Position128 start{std::numeric_limits<uint64_t>::max() - 12ULL,
                                0x0123456789ABCDEFULL};
        ChaCha20GF1024LFSR1024 seq(seed), direct(seed);
        seq.seek(start);
        for (uint64_t i = 0; i < 40; ++i) {
            const uint64_t a = seq.next_int();
            const Position128 p = add_pos(start, i);
            direct.seek(p);
            const uint64_t b = direct.next_int();
            if (a != b) return false;
        }
    }

    return true;
}

static bool test_bulk_identity()
{
    const auto& seed = reproducible_seed();

    struct Case { Position128 start; std::size_t count; };
    const Case cases[] = {
        {{0, 0}, 4099},
        {{1, 0}, 2051},
        {{RNG::OUTPUT_WORDS_PER_FFT - 1, 0}, 6149},
        {{RNG::OUTPUT_WORDS_PER_FFT, 0}, 4097},
        {{std::numeric_limits<uint64_t>::max() - 1500ULL, 0x123456789ABCDEF0ULL}, 4096},
        {{std::numeric_limits<uint64_t>::max() - 100ULL,
          std::numeric_limits<uint64_t>::max()}, 101}
    };

    for (const Case& tc : cases) {
        std::vector<uint64_t> scalar(tc.count), bulk(tc.count);
        ChaCha20GF1024LFSR1024 a(seed), b(seed);
        a.seek(tc.start);
        b.seek(tc.start);
        for (std::size_t i = 0; i < tc.count; ++i)
            scalar[i] = a.next_int();
        b.generate(bulk.data(), bulk.size());
        if (scalar != bulk) return false;
    }

    // A one-word request at the final representable position is valid, then
    // both APIs must report exhaustion rather than wrapping to zero.
    const Position128 last{std::numeric_limits<uint64_t>::max(),
                           std::numeric_limits<uint64_t>::max()};
    ChaCha20GF1024LFSR1024 a(seed), b(seed);
    a.seek(last);
    b.seek(last);
    uint64_t bv = 0;
    b.generate(&bv, 1);
    if (bv != a.next_int()) return false;

    bool a_threw = false, b_threw = false;
    try { (void)a.next_int(); } catch (const std::overflow_error&) { a_threw = true; }
    try { b.generate(&bv, 1); } catch (const std::overflow_error&) { b_threw = true; }
    return a_threw && b_threw;
}

static bool test_parallel_identity(unsigned threads)
{
    const auto& seed = reproducible_seed();

    constexpr std::size_t COUNT = 20000 + 37;
    std::vector<uint64_t> serial(COUNT), parallel(COUNT);

    // Deliberately cross the 64-bit low-half boundary. This verifies that the
    // parallel wrapper preserves the same logical 128-bit-position stream.
    const Position128 start{std::numeric_limits<uint64_t>::max() - 700ULL,
                            0x1111222233334444ULL};

    ChaCha20GF1024LFSR1024 s(seed);
    s.seek(start);
    for (std::size_t i = 0; i < COUNT; ++i)
        serial[i] = s.next_int();

    ChaCha20GF1024LFSR1024Parallel p(seed);
    p.fill_parallel_at(parallel.data(), parallel.size(), start, threads);
    if (serial != parallel) return false;

    // Two complete FFT output blocks ending at the last valid position.
    // This exercises the parallel exclusive-end overflow boundary.
    constexpr std::size_t tail_count = 2 * RNG::OUTPUT_WORDS_PER_FFT;
    const Position128 last{~uint64_t{0}, ~uint64_t{0}};
    const Position128 tail_start{last.lo - (tail_count - 1), last.hi};
    serial.resize(tail_count);
    parallel.resize(tail_count);
    s.seek(tail_start);
    s.generate(serial.data(), serial.size());
    p.seek(tail_start);
    p.fill_parallel(parallel.data(), parallel.size(), threads);
    if (serial != parallel || !p.exhausted() || p.position128() != last)
        return false;

    uint64_t extra = 0;
    try { p.fill_parallel(&extra, 1, threads); }
    catch (const std::overflow_error&) { return true; }
    return false;
}

static bool cpu_has_rdrand()
{
#if defined(DEMO_MSVC_X86)
    int regs[4] = {};
    __cpuid(regs, 1);
    return (static_cast<uint32_t>(regs[2]) & (1u << 30)) != 0;
#elif defined(DEMO_GNU_X86)
    __builtin_cpu_init();
    return __builtin_cpu_supports("rdrnd") != 0;
#else
    return false;
#endif
}

#if defined(DEMO_MSVC_X86)
static bool rdrand64(uint64_t& out)
{
#  if defined(_M_X64)
    return _rdrand64_step(reinterpret_cast<unsigned __int64*>(&out)) != 0;
#  else
    unsigned int lo = 0, hi = 0;
    if (!_rdrand32_step(&lo) || !_rdrand32_step(&hi)) return false;
    out = uint64_t(lo) | (uint64_t(hi) << 32);
    return true;
#  endif
}
#elif defined(DEMO_GNU_X86)
DEMO_TARGET_RDRND
static bool rdrand64(uint64_t& out)
{
#  if defined(__x86_64__)
    unsigned long long v = 0;
    const int ok = _rdrand64_step(&v);
    out = static_cast<uint64_t>(v);
    return ok != 0;
#  else
    unsigned int lo = 0, hi = 0;
    if (!_rdrand32_step(&lo) || !_rdrand32_step(&hi)) return false;
    out = uint64_t(lo) | (uint64_t(hi) << 32);
    return true;
#  endif
}
#else
static bool rdrand64(uint64_t&) { return false; }
#endif

// RDRAND is a hardware DRBG interface, not a proof of the entropy assumptions.
// It supplies a byte callback to the exact same structured seed sampler.
static void rdrand_demo_bytes(uint8_t* dst, std::size_t bytes)
{
    while (bytes != 0) {
        uint64_t v = 0;
        bool ok = false;
        for (int retry = 0; retry < 16 && !ok; ++retry)
            ok = rdrand64(v);
        if (!ok)
            throw std::runtime_error("RDRAND failed after 16 attempts");
        const std::size_t n = std::min<std::size_t>(bytes, 8);
        for (std::size_t j = 0; j < n; ++j)
            dst[j] = static_cast<uint8_t>(v >> (8 * j));
        dst += n;
        bytes -= n;
    }
}

static void demo_seeding()
{
    namespace seed_api = chacha20gf1024lfsr1024_seed;
    seed_api::SeedReport report;
    std::printf("Preparing structured seed (RDSEED first, OS fallback)...\n");
    std::fflush(stdout);
    try {
        const auto seed = seed_api::make_prepared_seed(&report);
        RNG rng(seed);
        std::printf("Hardware-first seed demo: OK  first=%016llx\n",
                    (unsigned long long)rng.next_int());
    }
    catch (const std::exception& e) {
        std::printf("Hardware-first seed demo: FAILED (%s)\n", e.what());
    }
    std::printf("RDSEED64 available:       %s\n", report.rdseed64_available ? "yes" : "no");
    std::printf("Seed source:              %s\n", seed_api::seed_source_name(report.source));
    std::printf("Fallback reason:          %s\n", seed_api::fallback_reason_name(report.fallback));
    std::printf("RDSEED bytes obtained:    %llu%s\n",
                (unsigned long long)report.rdseed_bytes,
                report.source == seed_api::SeedSource::RDSEED ? "" : " (not used in final seed)");
    std::printf("RDSEED failed attempts:   %llu\n", (unsigned long long)report.rdseed_failures);
    std::printf("OS bytes supplied:        %llu\n", (unsigned long long)report.os_bytes);

    if (!cpu_has_rdrand()) {
        std::printf("RDRAND seed demo:          unavailable\n");
        return;
    }
    std::printf("Preparing RDRAND structured seed...\n");
    std::fflush(stdout);
    try {
        const auto seed = RNG::make_seed(rdrand_demo_bytes);
        RNG rng(seed);
        std::printf("RDRAND seed demo:          OK  first=%016llx\n",
                    (unsigned long long)rng.next_int());
    }
    catch (const std::exception& e) {
        std::printf("RDRAND seed demo:          FAILED (%s)\n", e.what());
    }
}

static void bench_chacha(uint64_t count)
{
    uint8_t key[32] = {};
    uint64_t s = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < 4; ++i) store64_le(key + i * 8, sm64(s));
    ChaCha20GF1024LFSR1024Counter128 rng(key);
    volatile uint64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < count; ++i) sink ^= rng.next_int();
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double dcount = static_cast<double>(count);
    std::printf("%-32s %9.3f M uint64/s  %9.3f MiB/s  sink=%016llx\n",
        "ChaCha20GF1024LFSR1024Counter128", dcount / sec / 1e6, dcount * 8.0 / sec / (1024 * 1024),
        (unsigned long long)sink);
}

static void bench_single(uint64_t count)
{
    const auto& seed = reproducible_seed();
    ChaCha20GF1024LFSR1024 rng(seed);
    volatile uint64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < count; ++i) sink ^= rng.next_int();
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double dcount = static_cast<double>(count);
    std::printf("%-32s %9.3f M uint64/s  %9.3f MiB/s  sink=%016llx\n",
        "GF1024+LFSR1024 single", dcount / sec / 1e6, dcount * 8.0 / sec / (1024 * 1024),
        (unsigned long long)sink);
}

static void bench_bulk(uint64_t count)
{
    const auto& seed = reproducible_seed();
    ChaCha20GF1024LFSR1024 rng(seed);
    std::vector<uint64_t> out(static_cast<std::size_t>(count));

    const auto t0 = std::chrono::steady_clock::now();
    rng.generate(out.data(), out.size());
    const auto t1 = std::chrono::steady_clock::now();

    uint64_t sink = 0;
    for (uint64_t v : out) sink ^= v;
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double dcount = static_cast<double>(count);
    std::printf("%-32s %9.3f M uint64/s  %9.3f MiB/s  sink=%016llx\n",
        "GF1024+LFSR1024 bulk", dcount / sec / 1e6, dcount * 8.0 / sec / (1024 * 1024),
        (unsigned long long)sink);
}

static void bench_parallel(uint64_t count, unsigned threads)
{
    const auto& seed = reproducible_seed();
    ChaCha20GF1024LFSR1024Parallel rng(seed);
    std::vector<uint64_t> out(static_cast<std::size_t>(count));

    const auto t0 = std::chrono::steady_clock::now();
    rng.fill_parallel(out.data(), out.size(), threads);
    const auto t1 = std::chrono::steady_clock::now();

    uint64_t sink = 0;
    for (uint64_t v : out) sink ^= v; // outside timed region
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    char name[64];
    std::snprintf(name, sizeof(name), "GF1024+LFSR1024 parallel x%u", threads);
    const double dcount = static_cast<double>(count);
    std::printf("%-32s %9.3f M uint64/s  %9.3f MiB/s  sink=%016llx\n",
        name, dcount / sec / 1e6, dcount * 8.0 / sec / (1024 * 1024),
        (unsigned long long)sink);
}

static int run_main(int argc, char** argv)
{
    uint64_t count = 1000000ULL;
    unsigned threads = 4;
    if (argc > 1) count = std::strtoull(argv[1], nullptr, 10);
    if (argc > 2) threads = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(uint64_t))
        throw std::invalid_argument("Iterations must be positive and fit the output buffer");
    const bool full_toy = argc > 3 && std::strcmp(argv[3], "full") == 0;

    std::printf("PCLMUL runtime support:    %s\n",
        RNG::debug_cpu_has_pclmul() ? "yes" : "no");
    std::printf("sizeof(ChaCha20GF1024LFSR1024):  %zu bytes\n",
        sizeof(ChaCha20GF1024LFSR1024));
    std::printf("Full seed size:            %zu bytes\n",
        RNG::FULL_SEED_BYTES);
    std::printf("GF seed size:              %zu bytes\n",
        RNG::GF_SEED_BYTES);
    std::printf("LFSR seed size:            %zu bytes\n", RNG::LFSR_SEED_BYTES);
    std::printf("Shared LFSR tables:        %zu bytes\n", RNG::lfsr_shared_table_bytes());
    std::printf("Preparing reproducible structured seed (outside benchmarks)...\n");
    std::fflush(stdout);
    const auto seed_start = std::chrono::steady_clock::now();
    (void)reproducible_seed();
    const double seed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seed_start).count();
    std::printf("Seed preparation:         %.3f s\n", seed_seconds);
    bool all_ok = true;
    auto checked = [&all_ok](bool passed) {
        all_ok &= passed;
        return passed ? "OK" : "FAILED";
    };
    std::printf("ChaCha128 regression vec:  %s\n",
        checked(test_chacha_zero_vector()));
    std::printf("Embedded vec (zero LFSR):  %s\n",
        checked(test_embedded_polynomial_regression_vector()));
    std::printf("Hybrid composition ref:   %s\n",
        checked(test_hybrid_full_seed_reference()));
    std::printf("LFSR word/bit reference:   %s\n",
        checked(test_lfsr_bit_reference()));
    std::printf("Structured seed checks:   %s\n",
        checked(test_structured_seed()));
    std::printf("GF(2^128) fast/ref:        %s\n",
        checked(test_gf_arithmetic()));
    std::printf("GF128 FFT vs Horner:       %s\n",
        checked(test_fft_vs_horner()));
    std::printf("Hybrid 128-bit seek:       %s\n",
        checked(test_seek()));
    std::printf("Bulk vs next_int:          %s\n",
        checked(test_bulk_identity()));
    std::printf("Parallel identity x%u:      %s\n", threads,
        checked(test_parallel_identity(threads)));

    if (!all_ok) {
        std::fprintf(stderr, "Core checks failed; benchmarks skipped.\n");
        return 1;
    }
    demo_seeding();

    std::printf("Iterations: %llu\n", (unsigned long long)count);
    bench_chacha(count);
    bench_single(count);
    bench_bulk(count);
    bench_parallel(count, threads);

    // ------------------------------------------------------------------
    // Exhaustive toy-model verification of the exact k-wise claim.
    // Production uses 1024 coefficients over GF(2^128), with both 64-bit
    // halves exposed. The toy models use counting and exact rank to verify:
    // (A) exact k-wise independence,
    // (B) sharpness at k+1,
    // (C) invariance under a fixed XOR mask,
    // (D) a reducible-polynomial negative control,
    // (E) that projecting field values to fewer output bits preserves k-wise
    //     independence with exactly the expected multiplicities,
    // (F) split-half uniformity with and without a fixed XOR mask,
    // (G)/(H) k=8 rank checks and the gain over k=4 in the toy field.
    // These toy checks do not replace production GF/LFSR checks above and do
    // not establish the AGHP small-bias theorem; the toy header is unchanged.
    // ------------------------------------------------------------------
    std::printf("\nToy-model k-wise verification (%s mode, exact counting + binary rank,\n"
        "GF(2^4) k=4 and k=8%s):\n",
        full_toy ? "full" : "quick",
        full_toy ? " and GF(2^8) k=3" : "");
    std::fflush(stdout);
    const bool toy_ok = run_toy_kwise_1024_test(full_toy, /*gf8_sets=*/4, stdout);
    std::printf("Toy-model k-wise check:    %s\n", toy_ok ? "OK" : "FAILED");

    return all_ok && toy_ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    try {
        return run_main(argc, argv);
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 2;
    }
}
