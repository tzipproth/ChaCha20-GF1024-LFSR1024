#pragma once
// toy_kwise_1024_test.h
//
// Exhaustive counting and exact rank checks for the k-wise independence claim
// behind ChaCha20GF1024, scaled down to fields whose ENTIRE seed space can be
// enumerated (A-F), plus rank checks for a larger toy seed space (G-H).
// Field evaluations P(x) use a low-weight irreducible polynomial; split-half
// checks mirror the production mapping from one field value to two words.
//
//   Production: GF(2^128), 1024 coefficients, seed space 2^131072 (not enumerable)
//   Toy:        GF(2^4),  4 coefficients,   seed space 2^16
//               GF(2^8),  3 coefficients,   seed space 2^24
//   Rank toy:   GF(2^4),  8 coefficients,   seed space 2^32 (not enumerated)
//
// Checks (A)-(F) by exhaustive counting over each selected seed space:
//   (A) k-wise:    for k distinct positions every k-tuple occurs EXACTLY once
//   (B) sharpness: for k+1 positions only 2^(mk) of 2^(m(k+1)) tuples are reachable
//   (C) masking:   XOR with an arbitrary fixed position-dependent mask (the
//                  role of ChaCha20 in the hybrid) leaves (A) unchanged
//   (D) control:   with a REDUCIBLE polynomial (x^4+1) check (A) must FAIL,
//                  proving the test can detect a broken construction
//   (E) projection: project GF(2^4) outputs to 2 bits; every 4-tuple must
//                  occur exactly 2^(16-8)=256 times over the seed space
//   (F) split halves: use low/high 2-bit halves as adjacent output words,
//                  mirroring the production low64/high64 mapping
//
// Modes:
//   quick          GF(2^4): (A) over ALL 1820 position sets, (B)/(C) sampled,
//                  (D) negative control.  Suitable as a default self-test.
//   full          additionally (B)/(C) over ALL sets and GF(2^8) with
//                  `gf8_sets` sampled position sets.
//
// This does not test the 128-bit production field directly; it tests the theorem
// and the scaled construction, including split-half output. A passing toy test
// does NOT validate the production FFT, field implementation, seek or bulk API.
// (G) additionally checks k=8 over GF(2^4) by exact binary matrix rank,
//     avoiding enumeration of 2^32 seeds. All 12870 eight-point sets are covered.
// (H) checks split-half rank for k=8, structured 16-word uniformity, and a
//     selection that distinguishes k=4 from k=8.
// Entry point: run_toy_kwise_1024_test(full, gf8_sets, out).
// This header may coexist with the original toy_kwise_test.h.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace toy_kwise_1024_detail {

// ---------------------------------------------------------------------------
// Small binary field GF(2^M) with reduction polynomial x^M + LOW.
// ---------------------------------------------------------------------------
template <unsigned M, unsigned LOW>
struct SmallField
{
    static constexpr unsigned SIZE = 1u << M;
    static constexpr unsigned MASK = SIZE - 1;

    static unsigned mul_slow(unsigned a, unsigned b)
    {
        unsigned r = 0;
        for (unsigned i = 0; i < M; ++i) {
            if (b & 1u) r ^= a;
            b >>= 1;
            const unsigned carry = a & (1u << (M - 1));
            a = (a << 1) & MASK;
            if (carry) a ^= LOW;
        }
        return r;
    }

    // Full multiplication table (256x256 bytes for GF(2^8)).
    static const std::vector<uint8_t>& table()
    {
        static const std::vector<uint8_t> t = [] {
            std::vector<uint8_t> v(std::size_t(SIZE) * SIZE);
            for (unsigned a = 0; a < SIZE; ++a)
                for (unsigned b = 0; b < SIZE; ++b)
                    v[std::size_t(a) * SIZE + b] = uint8_t(mul_slow(a, b));
            return v;
        }();
        return t;
    }

    static inline unsigned mul(unsigned a, unsigned b)
    {
        return table()[std::size_t(a) * SIZE + b];
    }

    // Horner evaluation of P(x) = c[0] + c[1] x + ... + c[k-1] x^(k-1).
    static inline unsigned eval(const unsigned* c, unsigned k, unsigned x)
    {
        unsigned y = c[k - 1];
        for (unsigned i = k - 1; i-- > 0;)
            y = mul(y, x) ^ c[i];
        return y;
    }

    // Sanity: the field must actually be a field (multiplicative group of
    // order 2^M - 1).  Checks a^(2^M-1) == 1 for every nonzero a.
    static bool is_field()
    {
        for (unsigned a = 1; a < SIZE; ++a) {
            unsigned r = 1;
            for (unsigned e = 0; e < SIZE - 1; ++e) r = mul_slow(r, a);
            if (r != 1) return false;
        }
        return true;
    }
};

// Decode seed index -> k coefficients (M bits each).
template <unsigned M>
static inline void seed_to_coeffs(uint64_t seed, unsigned k, unsigned* c)
{
    for (unsigned i = 0; i < k; ++i)
        c[i] = unsigned((seed >> (M * i)) & ((1u << M) - 1));
}

// Arbitrary fixed mask standing in for the ChaCha20 stream: any function of
// the position works for the argument, so a cheap integer hash is enough.
static inline unsigned demo_mask(unsigned position, unsigned m_bits)
{
    uint64_t z = (uint64_t(position) + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return unsigned((z ^ (z >> 31)) & ((1u << m_bits) - 1));
}

// ---------------------------------------------------------------------------
// (A)/(C): every k-tuple exactly once over the full seed space.
// ---------------------------------------------------------------------------
template <class F, unsigned M>
static bool check_exactly_once(const unsigned* pos, unsigned k, bool masked)
{
    const uint64_t seeds = 1ull << (M * k);
    std::vector<uint8_t> count(std::size_t(seeds), 0); // tuple index -> count
    unsigned c[16];

    for (uint64_t s = 0; s < seeds; ++s) {
        seed_to_coeffs<M>(s, k, c);
        uint64_t tuple = 0;
        for (unsigned i = 0; i < k; ++i) {
            unsigned v = F::eval(c, k, pos[i]);
            if (masked) v ^= demo_mask(pos[i], M);
            tuple |= uint64_t(v) << (M * i);
        }
        if (++count[std::size_t(tuple)] != 1) return false; // seen twice
    }
    // seeds == number of tuples and every tuple hit at most once -> exactly once
    return true;
}

// ---------------------------------------------------------------------------
// (E): projection of each field value to fewer output bits preserves k-wise
// independence.  Production uses GF(2^128)->64 bits; the toy analogue uses
// GF(2^4)->2 bits and can exhaust the entire seed space.
// ---------------------------------------------------------------------------
template <class F, unsigned M>
static bool check_projected_uniform(const unsigned* pos, unsigned k, unsigned out_bits)
{
    const uint64_t seeds = 1ull << (M * k);
    const uint64_t tuples = 1ull << (out_bits * k);
    const uint64_t expected = seeds / tuples;
    std::vector<uint32_t> count(std::size_t(tuples), 0);
    unsigned c[16];
    const unsigned mask = (1u << out_bits) - 1u;

    for (uint64_t s = 0; s < seeds; ++s) {
        seed_to_coeffs<M>(s, k, c);
        uint64_t tuple = 0;
        for (unsigned i = 0; i < k; ++i) {
            const unsigned v = F::eval(c, k, pos[i]) & mask;
            tuple |= uint64_t(v) << (out_bits * i);
        }
        ++count[std::size_t(tuple)];
    }
    for (uint32_t n : count)
        if (n != expected) return false;
    return true;
}

// Production uses both 64-bit halves of each GF(2^128) evaluation for two
// adjacent output words.  This toy analogue splits a GF(2^4) value into two
// 2-bit words and checks arbitrary word-position selections.
template <class F, unsigned M>
static bool check_split_halves_uniform(const unsigned* word_pos, unsigned k, unsigned half_bits, bool masked = false)
{
    const uint64_t seeds = 1ull << (M * k);
    const uint64_t tuples = 1ull << (half_bits * k);
    const uint64_t expected = seeds / tuples;
    std::vector<uint32_t> count(std::size_t(tuples), 0);
    unsigned c[16];
    const unsigned mask = (1u << half_bits) - 1u;

    for (uint64_t seed = 0; seed < seeds; ++seed) {
        seed_to_coeffs<M>(seed, k, c);
        uint64_t tuple = 0;
        for (unsigned i = 0; i < k; ++i) {
            const unsigned eval_pos = word_pos[i] >> 1;
            unsigned v = F::eval(c, k, eval_pos);
            if (word_pos[i] & 1u)
                v >>= half_bits;
            v &= mask;
            if (masked) v ^= demo_mask(word_pos[i], half_bits);
            tuple |= uint64_t(v) << (half_bits * i);
        }
        ++count[std::size_t(tuple)];
    }

    for (uint32_t n : count)
        if (n != expected) return false;
    return true;
}

// ---------------------------------------------------------------------------
// (B): number of distinct reachable (k+1)-tuples.
// ---------------------------------------------------------------------------
template <class F, unsigned M>
static uint64_t count_reachable(const unsigned* pos, unsigned k1)
{
    const unsigned k = k1 - 1;
    const uint64_t seeds = 1ull << (M * k);
    const unsigned tuple_bits = M * k1;
    unsigned c[16];

    // A direct bitset is ideal for the small GF(2^4) checks, but the GF(2^8)
    // k+1 test has a 2^32 tuple universe and would reserve 512 MiB.  In that
    // case store only the 2^24 actually generated tuples, then sort/unique.
    if (tuple_bits <= 26) {
        const uint64_t tuples = 1ull << tuple_bits;
        std::vector<uint8_t> seen(std::size_t((tuples + 7) / 8), 0);
        uint64_t distinct = 0;
        for (uint64_t s = 0; s < seeds; ++s) {
            seed_to_coeffs<M>(s, k, c);
            uint64_t tuple = 0;
            for (unsigned i = 0; i < k1; ++i)
                tuple |= uint64_t(F::eval(c, k, pos[i])) << (M * i);
            uint8_t& byte = seen[std::size_t(tuple >> 3)];
            const uint8_t bit = uint8_t(1u << (tuple & 7));
            if (!(byte & bit)) { byte |= bit; ++distinct; }
        }
        return distinct;
    }

    if (tuple_bits <= 32) {
        std::vector<uint32_t> generated;
        generated.reserve(static_cast<std::size_t>(seeds));
        for (uint64_t s = 0; s < seeds; ++s) {
            seed_to_coeffs<M>(s, k, c);
            uint64_t tuple = 0;
            for (unsigned i = 0; i < k1; ++i)
                tuple |= uint64_t(F::eval(c, k, pos[i])) << (M * i);
            generated.push_back(static_cast<uint32_t>(tuple));
        }
        std::sort(generated.begin(), generated.end());
        return static_cast<uint64_t>(std::unique(generated.begin(), generated.end()) - generated.begin());
    }

    std::vector<uint64_t> generated;
    generated.reserve(static_cast<std::size_t>(seeds));
    for (uint64_t s = 0; s < seeds; ++s) {
        seed_to_coeffs<M>(s, k, c);
        uint64_t tuple = 0;
        for (unsigned i = 0; i < k1; ++i)
            tuple |= uint64_t(F::eval(c, k, pos[i])) << (M * i);
        generated.push_back(tuple);
    }
    std::sort(generated.begin(), generated.end());
    return static_cast<uint64_t>(std::unique(generated.begin(), generated.end()) - generated.begin());
}

// Enumerate all k-subsets of {0..n-1}.
static bool next_combination(unsigned* idx, unsigned k, unsigned n)
{
    int i = int(k) - 1;
    while (i >= 0 && idx[i] == n - k + unsigned(i)) --i;
    if (i < 0) return false;
    ++idx[i];
    for (unsigned j = unsigned(i) + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
    return true;
}

static double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}


// GF(2^4), k = 4
template <unsigned LOW>
static bool gf16_check_A(unsigned* pos_out_fail)
{
    using F = SmallField<4, LOW>;
    unsigned pos[4] = {0, 1, 2, 3};
    do {
        if (!check_exactly_once<F, 4>(pos, 4, false)) {
            if (pos_out_fail) std::copy(pos, pos + 4, pos_out_fail);
            return false;
        }
    } while (next_combination(pos, 4, F::SIZE));
    return true;
}


// The seed-to-output map is linear over GF(2). Full row rank is equivalent
// to exact uniformity over ALL seeds, without enumerating them. Row bit j
// records the contribution of seed bit j to one selected output bit.
static unsigned binary_rank(const uint64_t* rows, unsigned count)
{
    uint64_t pivots[64] = {};
    unsigned rank = 0;
    for (unsigned i = 0; i < count; ++i) {
        uint64_t row = rows[i];
        for (int bit = 63; bit >= 0; --bit) {
            if (!(row & (uint64_t{1} << bit))) continue;
            if (pivots[bit]) row ^= pivots[bit];
            else {
                pivots[bit] = row;
                ++rank;
                break;
            }
        }
    }
    return rank;
}

template <unsigned K, unsigned LOW = 0x3>
struct GF16ResponseRows
{
    static_assert(K > 0 && K <= 16, "Seed rows must fit in uint64_t");
    uint64_t rows[16][4] = {};

    GF16ResponseRows()
    {
        using F = SmallField<4, LOW>;
        for (unsigned bit = 0; bit < 4 * K; ++bit) {
            unsigned c[K] = {};
            c[bit / 4] = 1u << (bit % 4);
            for (unsigned x = 0; x < 16; ++x) {
                const unsigned y = F::eval(c, K, x);
                for (unsigned b = 0; b < 4; ++b)
                    if ((y >> b) & 1u) rows[x][b] |= uint64_t{1} << bit;
            }
        }
    }

    unsigned field_rank(const unsigned* positions, unsigned count) const
    {
        uint64_t selected[64] = {};
        if (count > 16) return 0;
        for (unsigned i = 0; i < count; ++i) {
            if (positions[i] >= 16) return 0;
            for (unsigned b = 0; b < 4; ++b)
                selected[4 * i + b] = rows[positions[i]][b];
        }
        return binary_rank(selected, 4 * count);
    }

    unsigned split_rank(const unsigned* positions, unsigned count) const
    {
        uint64_t selected[64] = {};
        if (count > 32) return 0;
        for (unsigned i = 0; i < count; ++i) {
            if (positions[i] >= 32) return 0;
            const unsigned x = positions[i] >> 1;
            const unsigned offset = (positions[i] & 1u) * 2;
            selected[2 * i] = rows[x][offset];
            selected[2 * i + 1] = rows[x][offset + 1];
        }
        return binary_rank(selected, 2 * count);
    }
};

template <class Say>
static bool check_doubled_order(Say& say)
{
    const GF16ResponseRows<8> doubled;
    unsigned pos[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    unsigned checked = 0;
    bool fields_ok = true;
    do {
        ++checked;
        if (doubled.field_rank(pos, 8) != 32) {
            fields_ok = false;
            say("    (G) rank FAILED at");
            for (unsigned x : pos) say(" %u", x);
            say("\n");
            break;
        }
    } while (next_combination(pos, 8, 16));
    say("    (G) GF(2^4), k=8: %u/12870 point sets, exact 32-bit rank  %s\n",
        checked, fields_ok ? "OK" : "FAILED");

    // k+1 full values remain dependent, despite distinct positions.
    const unsigned nine[9] = {0, 1, 2, 3, 4, 5, 6, 7, 15};
    const unsigned repeated[8] = {0, 1, 2, 3, 4, 5, 6, 6};
    const unsigned first8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const GF16ResponseRows<8, 0x1> reducible;
    const bool controls_ok = doubled.field_rank(nine, 9) == 32
        && doubled.field_rank(repeated, 8) == 28
        && reducible.field_rank(first8, 8) < 32;
    say("    (G) rank controls: k+1, duplicate point, reducible modulus  %s\n",
        controls_ok ? "OK" : "FAILED");

    const unsigned words[][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {0, 2, 4, 6, 8, 10, 12, 14},
        {1, 3, 5, 7, 9, 11, 13, 15},
        {0, 1, 4, 7, 16, 23, 28, 31},
        {0, 3, 8, 11, 16, 21, 26, 31},
        {16, 18, 20, 22, 24, 26, 28, 30}
    };
    bool split_ok = true;
    for (const auto& selection : words)
        split_ok &= doubled.split_rank(selection, 8) == 16;
    // Both halves of eight distinct points: 16 words, all 32 seed bits.
    const unsigned paired[16] = {
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 30, 31
    };
    split_ok &= doubled.split_rank(paired, 16) == 32;
    say("    (H) k=8 split halves and structured 16-word uniformity  %s\n",
        split_ok ? "OK" : "FAILED");

    // A concrete gain over k=4: low halves at field points 0..7.
    // Their XOR vanishes for degree <=3 over this binary subspace;
    // k=8 supplies eight independent complete field evaluations.
    const GF16ResponseRows<4> original;
    const unsigned low8[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    const unsigned old_rank = original.split_rank(low8, 8);
    const unsigned new_rank = doubled.split_rank(low8, 8);
    const bool gain_ok = old_rank < 16 && new_rank == 16;
    say("    (H) doubling k=4 -> k=8: selected-word rank %u -> %u of 16  %s\n",
        old_rank, new_rank, gain_ok ? "OK" : "FAILED");
    return fields_ok && controls_ok && split_ok && gain_ok;
}

// Returns true if all checks pass.  `out` may be nullptr for silent operation.
static bool run_tests(bool full, unsigned gf8_sets, std::FILE* out)
{
    auto say = [out](const char* fmt, auto... args) {
        if (!out) return;
        if constexpr (sizeof...(args) == 0) std::fputs(fmt, out);
        else std::fprintf(out, fmt, args...);
    };
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;

    {
        using F = SmallField<4, 0x3>;
        constexpr unsigned M = 4, K = 4;
        say("  GF(2^4) k=%u seed space 2^%u%s\n", K, M * K, F::is_field() ? "" : "  FIELD CHECK FAILED");
        ok &= F::is_field();

        unsigned failpos[4];
        if (gf16_check_A<0x3>(failpos))
            say("    (A) k-wise:    all 1820 position sets, every 4-tuple exactly once  OK\n");
        else { say("    (A) FAILED at %u %u %u %u\n", failpos[0], failpos[1], failpos[2], failpos[3]); ok = false; }

        // (B) sharpness and (C) masking: all sets in full mode, every 40th / 20th otherwise
        unsigned pos5[K + 1] = {0, 1, 2, 3, 4};
        unsigned n5 = 0, idx = 0;
        bool sharpness_ok = true;
        do {
            if (full || idx++ % 40 == 0) {
                ++n5;
                if (count_reachable<F, M>(pos5, K + 1) != (1ull << (M * K))) {
                    say("    (B) FAILED at %u %u %u %u %u\n", pos5[0], pos5[1], pos5[2], pos5[3], pos5[4]);
                    sharpness_ok = false; ok = false; break;
                }
            }
        } while (next_combination(pos5, K + 1, F::SIZE));
        say("    (B) sharpness: %u/4368 sets of 5 positions reach 2^%u of 2^%u tuples  %s\n", n5, M * K, M * (K + 1), sharpness_ok ? "OK" : "FAILED");

        unsigned posm[K] = {0, 1, 2, 3};
        unsigned nm = 0; idx = 0;
        bool masking_ok = true;
        do {
            if (full || idx++ % 20 == 0) {
                ++nm;
                if (!check_exactly_once<F, M>(posm, K, true)) {
                    say("    (C) FAILED (masked) at %u %u %u %u\n", posm[0], posm[1], posm[2], posm[3]);
                    masking_ok = false; ok = false; break;
                }
            }
        } while (next_combination(posm, K, F::SIZE));
        say("    (C) masking:   XOR with fixed mask preserves (A) on %u/1820 sets  %s\n", nm, masking_ok ? "OK" : "FAILED");

        // (D) negative control: reducible x^4+1 must break (A)
        const bool control_fails = !gf16_check_A<0x1>(nullptr) && !SmallField<4, 0x1>::is_field();
        say("    (D) control:   reducible x^4+1 breaks (A)  %s\n", control_fails ? "OK" : "FAILED (test is not discriminating!)");
        ok &= control_fails;

        // (E) Selecting one half of the production field value is a
        // GF(2^128)->64-bit projection. Check its GF(2^4)->2-bit analogue.
        const unsigned posp[K] = {0, 1, 2, 3};
        const bool projection_ok = check_projected_uniform<F, M>(posp, K, 2);
        say("    (E) projection: GF(2^4)->2 bits, every 4-tuple occurs 256 times  %s\n",
            projection_ok ? "OK" : "FAILED");
        ok &= projection_ok;

        const unsigned split_sets[][K] = {
            {0, 1, 2, 3},   // both halves of two evaluation points
            {0, 2, 4, 6},   // low halves of four distinct points
            {1, 3, 5, 7},   // high halves of four distinct points
            {0, 1, 4, 7}    // mixed halves and mixed multiplicities
        };
        bool split_ok = true;
        for (const auto& ws : split_sets) {
            split_ok &= check_split_halves_uniform<F, M>(ws, K, 2);
            split_ok &= check_split_halves_uniform<F, M>(ws, K, 2, true);
        }
        say("    (F) split halves: paired GF value halves preserve 4-wise output uniformity  %s\n",
            split_ok ? "OK" : "FAILED");
        ok &= split_ok;
    }

    ok &= check_doubled_order(say);

    if (full) {
        using F = SmallField<8, 0x1B>;
        constexpr unsigned M = 8, K = 3;
        say("  GF(2^8) k=%u seed space 2^%u%s\n", K, M * K, F::is_field() ? "" : "  FIELD CHECK FAILED");
        ok &= F::is_field();

        uint64_t rng = 0xD1B54A32D192ED03ULL;
        auto next = [&rng]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
        std::vector<std::vector<unsigned>> sets = {{0, 1, 2}, {0, 1, 255}, {0, 128, 255}, {1, 2, 4}};
        while (sets.size() < gf8_sets) {
            unsigned p[K];
            do { for (unsigned& v : p) v = unsigned(next() & 0xFF); }
            while (p[0] == p[1] || p[0] == p[2] || p[1] == p[2]);
            sets.push_back({p[0], p[1], p[2]});
        }
        if (sets.size() > gf8_sets) sets.resize(gf8_sets ? gf8_sets : 1);

        bool gf8_ok = true;
        for (const auto& s : sets)
            if (!check_exactly_once<F, M>(s.data(), K, false)) {
                say("    (A) FAILED at %u %u %u\n", s[0], s[1], s[2]); gf8_ok = false; ok = false; break;
            }
        say("    (A) k-wise:    %zu position sets, every 3-tuple exactly once  %s\n", sets.size(), gf8_ok ? "OK" : "FAILED");

        const unsigned pos4[K + 1] = {0, 1, 2, 3};
        if (count_reachable<F, M>(pos4, K + 1) != (1ull << (M * K))) { say("    (B) FAILED\n"); ok = false; }
        else say("    (B) sharpness: 4 positions reach 2^%u of 2^%u tuples  OK\n", M * K, M * (K + 1));

        if (!check_exactly_once<F, M>(sets[0].data(), K, true)) { say("    (C) FAILED (masked)\n"); ok = false; }
        else say("    (C) masking:   XOR with fixed mask preserves (A)  OK\n");
    }

    say("  toy test time: %.1f s\n", seconds_since(t0));
    return ok;
}

} // namespace toy_kwise_1024_detail

// Same arguments as the original runner; distinct name permits side-by-side use.
// gf8_sets==0 retains the original convention of checking one GF(2^8) set.
static bool run_toy_kwise_1024_test(bool full, unsigned gf8_sets, std::FILE* out)
{
    return toy_kwise_1024_detail::run_tests(full, gf8_sets, out);
}
