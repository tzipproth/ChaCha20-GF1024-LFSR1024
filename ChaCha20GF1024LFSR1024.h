#ifndef CHACHA20GF1024LFSR1024_SINGLE_FILE_INCLUDED
#define CHACHA20GF1024LFSR1024_SINGLE_FILE_INCLUDED

// ChaCha20GF1024LFSR1024.cpp -- self-contained C++17, include-style implementation.
// Include this file from your main.cpp. No main() is supplied. Classes are inline;
// this file does not depend on ChaCha20GF1024FFT.h or the old seed/parallel headers.
// Public generator: ChaCha20GF1024LFSR1024.
// Define CHACHA20GF1024LFSR1024_FORCE_PORTABLE consistently in all translation
// units to disable PCLMUL and the four-block ChaCha SIMD path.
//
// Output: original GF1024/ChaCha stream XOR three independent AGHP LFSR masks.
// Each LFSR mask is the coefficient sequence of A(z)/Q(z), where Q is uniformly
// chosen among monic irreducible degree-1024 binary polynomials and A is an
// independent uniform polynomial of degree < 1024. All three (Q,A) pairs are
// independent of each other and of the GF/ChaCha seed. Coefficients are output
// LSB-first in each uint64_t. This is the rational-series form of AGHP's LFSR;
// multiplication by Q maps the first 1024 sequence bits bijectively to A.
//
// With independent ideal seeds, for each fixed public stream_id:
//  * Any <=1024 distinct output words remain exactly jointly uniform.
//  * Every fixed nonempty bit parity over the 2^128-word domain has bias
//    <= ((2^134-1)/(2^1024-2^512))^3 < 2^-2667.
//  * Any 5000 distinct output bits have total variation distance < 2^-168
//    from ideal independent bits (TV uses half the L1 distance).
//  * Every fixed GF(2) polynomial p of degree <=3 over the full bit domain:
//    |E[(-1)^p(R)] - E[(-1)^p(U)]| < 2^-218.25 for ideal uniform U.
//    The probability error is half this bound, not a bias relative to 1/2.
//    Viola, Theorem 2: https://eccc.weizmann.ac.il/report/2007/132/download
//    Independent GF/ChaCha XOR preserves it: p(x XOR M) has the same degree bound.
// These are ensemble guarantees, not tests of a particular initialized stream.
// They do not cover adaptive seed-dependent parity choices, joint use of
// different stream IDs with one seed, or independence from a disclosed seed.
// The cryptographic claim still depends on the custom ChaCha/HChaCha component.
// Reference: Alon, Goldreich, Hastad, Peralta, "Simple Constructions of Almost
// k-wise Independent Random Variables", section 3, construction 1:
// https://web.math.princeton.edu/~nalon/PDFS/aghp4.pdf
//
// Serialized seed (17184 bytes, little-endian; incompatible with old seeds):
//   [0,32)       ChaCha key
//   [32,16416)   1024 GF(2^128) monomial coefficients (original layout)
//   [16416,16672) Q0 then A0 (128 bytes each)
//   [16672,16928) Q1 then A1 (128 bytes each)
//   [16928,17184) Q2 then A2 (128 bytes each)
// Each Q has implicit z^1024 and Q(0)=1. Q/A bits encode degrees 0..1023.
// All Q parts are STRUCTURED: arbitrary bytes are not a valid full seed.
// prepare_seed checks every Q for irreducibility; it cannot establish sampling
// uniformity or independence. make_seed performs separate rejection sampling
// when its callback supplies fresh independent uniform bytes.
// Zero numerator A is accepted, as required by this distribution.
// Never derive this extra seed from the ChaCha key when claiming the theorems.
//
// Usage (fill_random_bytes must return void and throw on entropy-source failure):
//   using RNG = ChaCha20GF1024LFSR1024;
//   auto seed = RNG::make_seed(fill_random_bytes);
//   RNG rng(seed, 0);                      // shared immutable jump tables
//   rng.generate(dst, count);
//   rng.generate_parallel_at(dst, count, RNG::Position(0), 4);
//   auto bytes = seed.bytes();             // persist the exact sampled seed
//   RNG restored(bytes, 0);                // validates Q and rebuilds tables
//
// uint64_t/default constructors use deterministic SplitMix64 convenience
// seeding and provide NONE of the ideal-seed distribution theorems. Initialization can be
// expensive; prepare once and reuse PreparedSeed or copy a generator.
// next_bit has a separate word reservoir, as in the previous implementation.
// For one contiguous bit stream do not interleave next_bit and word APIs.

#include <array>
#include <memory>
#include <thread>
#include <vector>
#include <exception>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64) || defined(_M_IX86))
#  include <intrin.h>
#  include <wmmintrin.h>
#  define CHACHA20GF1024LFSR1024FFT_MSVC_X86 1
#  define CHACHA20GF1024LFSR1024FFT_FORCE_INLINE __forceinline
#  if (defined(_M_X64) || defined(_M_AMD64)) && \
      !defined(CHACHA20GF1024LFSR1024_FORCE_PORTABLE)
#    define CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4 1
#  endif
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#  include <wmmintrin.h>
#  define CHACHA20GF1024LFSR1024FFT_GNU_X86 1
#  define CHACHA20GF1024LFSR1024FFT_FORCE_INLINE inline __attribute__((always_inline))
#  define CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL __attribute__((target("pclmul,sse2")))
#  if defined(__x86_64__) && !defined(CHACHA20GF1024LFSR1024_FORCE_PORTABLE)
#    define CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4 1
#  endif
#else
#  define CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#endif


namespace chacha20gf1024lfsr1024_detail {

// =============================================================================
// 128-bit position / GF(2^128) value types
// =============================================================================

struct ChaCha20GF1024Position128
{
    uint64_t lo = 0;
    uint64_t hi = 0;

    constexpr ChaCha20GF1024Position128() = default;
    constexpr ChaCha20GF1024Position128(uint64_t low, uint64_t high = 0)
        : lo(low), hi(high) {}

    constexpr bool operator==(const ChaCha20GF1024Position128& o) const
    {
        return lo == o.lo && hi == o.hi;
    }
    constexpr bool operator!=(const ChaCha20GF1024Position128& o) const
    {
        return !(*this == o);
    }
    constexpr bool operator<(const ChaCha20GF1024Position128& o) const
    {
        return hi < o.hi || (hi == o.hi && lo < o.lo);
    }

    constexpr bool is_max() const
    {
        return lo == ~uint64_t{0} && hi == ~uint64_t{0};
    }

    // Adds a 64-bit amount. Returns true on overflow past 2^128-1.
    bool add_u64(uint64_t v)
    {
        const uint64_t old = lo;
        lo += v;
        if (lo < old) {
            ++hi;
            if (hi == 0)
                return true;
        }
        return false;
    }

    bool increment()
    {
        return add_u64(1);
    }

    constexpr ChaCha20GF1024Position128 shr1() const
    {
        return ChaCha20GF1024Position128((lo >> 1) | (hi << 63), hi >> 1);
    }

    constexpr ChaCha20GF1024Position128 shr3() const
    {
        return ChaCha20GF1024Position128((lo >> 3) | (hi << 61), hi >> 3);
    }

    constexpr ChaCha20GF1024Position128 aligned_down_1024() const
    {
        return ChaCha20GF1024Position128(lo & ~uint64_t{1023}, hi);
    }

    constexpr std::size_t low10() const
    {
        return static_cast<std::size_t>(lo & uint64_t{1023});
    }
};

struct ChaCha20GF1024Field128
{
    uint64_t lo = 0;
    uint64_t hi = 0;

    constexpr ChaCha20GF1024Field128() = default;
    constexpr ChaCha20GF1024Field128(uint64_t low, uint64_t high = 0)
        : lo(low), hi(high) {}

    constexpr bool operator==(const ChaCha20GF1024Field128& o) const
    {
        return lo == o.lo && hi == o.hi;
    }
    constexpr bool operator!=(const ChaCha20GF1024Field128& o) const
    {
        return !(*this == o);
    }
    constexpr bool is_zero() const
    {
        return lo == 0 && hi == 0;
    }
};

inline ChaCha20GF1024Field128 operator^(ChaCha20GF1024Field128 a,
                                       ChaCha20GF1024Field128 b)
{
    return ChaCha20GF1024Field128(a.lo ^ b.lo, a.hi ^ b.hi);
}
inline ChaCha20GF1024Field128& operator^=(ChaCha20GF1024Field128& a,
                                        ChaCha20GF1024Field128 b)
{
    a.lo ^= b.lo;
    a.hi ^= b.hi;
    return a;
}

// =============================================================================
// ChaCha20GF1024Counter128
// =============================================================================
//
// A position-indexed ChaCha20 component.  A 64-bit stream_id is first mapped
// to a subkey with HChaCha20 (128-bit HChaCha input = stream_id || 0).  The
// resulting key then uses words 12..15 as one 128-bit block index.  This keeps
// the entire 128-bit position space available while providing computational
// stream separation through subkey derivation.
//
// HChaCha20 output words are 0,1,2,3,12,13,14,15 after 20 rounds, without the
// ChaCha feed-forward addition.
class ChaCha20GF1024Counter128
{
public:
    using Position = ChaCha20GF1024Position128;

    static constexpr std::size_t KEY_BYTES = 32;
    static constexpr std::size_t BLOCK_BYTES = 64;
    static constexpr std::size_t WORDS_PER_BLOCK = 8;
#if defined(CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4)
    static constexpr std::size_t BUFFERED_BLOCKS = 4;
#else
    static constexpr std::size_t BUFFERED_BLOCKS = 1;
#endif
    static constexpr std::size_t BUFFER_WORDS = BUFFERED_BLOCKS * WORDS_PER_BLOCK;

    ChaCha20GF1024Counter128()
    {
        uint8_t key[KEY_BYTES] = {};
        init(key, 0);
    }

    explicit ChaCha20GF1024Counter128(const uint8_t* key32, uint64_t stream_id = 0)
    {
        init(key32, stream_id);
    }

    uint64_t next_int()
    {
        if (buffer_index_ >= BUFFER_WORDS)
            refill();

        return buffer_[buffer_index_++];
    }

    void generate(uint64_t* dst, std::size_t count)
    {
        // Consume an already-buffered partial block first.
        while (count != 0 && buffer_index_ < BUFFER_WORDS) {
            *dst++ = buffer_[buffer_index_++];
            --count;
        }

        // Four independent blocks map naturally to the four 32-bit lanes of
        // SSE2.  x86-64 guarantees SSE2, so no runtime dispatch is needed.
#if defined(CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4)
        while (count >= 4 * WORDS_PER_BLOCK) {
            generate_current_4_blocks(dst);
            increment_counter_by_4();
            dst += 4 * WORDS_PER_BLOCK;
            count -= 4 * WORDS_PER_BLOCK;
            buffer_index_ = BUFFER_WORDS;
        }
#endif

        // Remaining full blocks can be written directly to the caller's buffer.
        while (count >= WORDS_PER_BLOCK) {
            generate_current_block(dst);
            increment_counter();
            dst += WORDS_PER_BLOCK;
            count -= WORDS_PER_BLOCK;
            buffer_index_ = BUFFER_WORDS;
        }

        if (count != 0) {
            refill();
            while (count-- != 0)
                *dst++ = buffer_[buffer_index_++];
        }
    }

    int next_bit()
    {
        // Bit calls consume a word into a separate LSB-first bit reservoir.
        // Interleaving this API with next_int()/generate() therefore does not
        // define one shared call-ordered bit stream.
        if (bit_index_ == 64) {
            current_bits_ = next_int();
            bit_index_ = 0;
        }
        const int bit = static_cast<int>((current_bits_ >> bit_index_) & 1ULL);
        ++bit_index_;
        return bit;
    }

    void seek(Position word_position)
    {
        const Position block_counter = word_position.shr3();
        const std::size_t word_in_block = static_cast<std::size_t>(word_position.lo & 7ULL);

        set_counter(block_counter);
#if defined(CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4)
        generate_current_4_blocks(buffer_);
        increment_counter_by_4();
#else
        generate_current_block(buffer_);
        increment_counter();
#endif
        buffer_index_ = word_in_block;
        current_bits_ = 0;
        bit_index_ = 64;
    }

    void seek(uint64_t word_position)
    {
        seek(Position(word_position));
    }

private:
    uint32_t state_[16]{};
    uint64_t buffer_[BUFFER_WORDS]{};
    std::size_t buffer_index_ = BUFFER_WORDS;
    uint64_t current_bits_ = 0;
    int bit_index_ = 64;

    static uint32_t load32_le(const uint8_t* p)
    {
        return  static_cast<uint32_t>(p[0])
            | (static_cast<uint32_t>(p[1]) << 8)
            | (static_cast<uint32_t>(p[2]) << 16)
            | (static_cast<uint32_t>(p[3]) << 24);
    }

    static uint32_t rotl32(uint32_t x, int n)
    {
        return (x << n) | (x >> (32 - n));
    }

    static void quarterround(uint32_t& a, uint32_t& b,
                             uint32_t& c, uint32_t& d)
    {
        a += b; d ^= a; d = rotl32(d, 16);
        c += d; b ^= c; b = rotl32(b, 12);
        a += b; d ^= a; d = rotl32(d, 8);
        c += d; b ^= c; b = rotl32(b, 7);
    }

    static void rounds20(uint32_t x[16])
    {
        for (int i = 0; i < 10; ++i) {
            quarterround(x[0], x[4], x[8],  x[12]);
            quarterround(x[1], x[5], x[9],  x[13]);
            quarterround(x[2], x[6], x[10], x[14]);
            quarterround(x[3], x[7], x[11], x[15]);

            quarterround(x[0], x[5], x[10], x[15]);
            quarterround(x[1], x[6], x[11], x[12]);
            quarterround(x[2], x[7], x[8],  x[13]);
            quarterround(x[3], x[4], x[9],  x[14]);
        }
    }

#if defined(CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4)
    static __m128i rotl32x4(__m128i x, int n)
    {
        return _mm_or_si128(_mm_slli_epi32(x, n),
                            _mm_srli_epi32(x, 32 - n));
    }

    static void quarterround4(__m128i& a, __m128i& b,
                              __m128i& c, __m128i& d)
    {
        a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a); d = rotl32x4(d, 16);
        c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c); b = rotl32x4(b, 12);
        a = _mm_add_epi32(a, b); d = _mm_xor_si128(d, a); d = rotl32x4(d, 8);
        c = _mm_add_epi32(c, d); b = _mm_xor_si128(b, c); b = rotl32x4(b, 7);
    }

    static void rounds20x4(__m128i x[16])
    {
        for (int i = 0; i < 10; ++i) {
            quarterround4(x[0], x[4], x[8],  x[12]);
            quarterround4(x[1], x[5], x[9],  x[13]);
            quarterround4(x[2], x[6], x[10], x[14]);
            quarterround4(x[3], x[7], x[11], x[15]);

            quarterround4(x[0], x[5], x[10], x[15]);
            quarterround4(x[1], x[6], x[11], x[12]);
            quarterround4(x[2], x[7], x[8],  x[13]);
            quarterround4(x[3], x[4], x[9],  x[14]);
        }
    }
#endif

    static void hchacha20_subkey(const uint8_t* key32, uint64_t stream_id,
                                 uint32_t out_key[8])
    {
        uint32_t x[16] = {
            0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
            0,0,0,0,0,0,0,0,
            static_cast<uint32_t>(stream_id),
            static_cast<uint32_t>(stream_id >> 32),
            0, 0
        };
        for (int i = 0; i < 8; ++i)
            x[4 + i] = load32_le(key32 + 4 * i);

        rounds20(x);
        out_key[0] = x[0];
        out_key[1] = x[1];
        out_key[2] = x[2];
        out_key[3] = x[3];
        out_key[4] = x[12];
        out_key[5] = x[13];
        out_key[6] = x[14];
        out_key[7] = x[15];
    }

    void init(const uint8_t* key32, uint64_t stream_id)
    {
        state_[0] = 0x61707865u;
        state_[1] = 0x3320646eu;
        state_[2] = 0x79622d32u;
        state_[3] = 0x6b206574u;

        uint32_t subkey[8];
        hchacha20_subkey(key32, stream_id, subkey);
        for (int i = 0; i < 8; ++i)
            state_[4 + i] = subkey[i];

        state_[12] = state_[13] = state_[14] = state_[15] = 0;
        buffer_index_ = BUFFER_WORDS;
        current_bits_ = 0;
        bit_index_ = 64;
    }

    void set_counter(Position counter)
    {
        state_[12] = static_cast<uint32_t>(counter.lo);
        state_[13] = static_cast<uint32_t>(counter.lo >> 32);
        state_[14] = static_cast<uint32_t>(counter.hi);
        state_[15] = static_cast<uint32_t>(counter.hi >> 32);
        buffer_index_ = BUFFER_WORDS;
    }

    void increment_counter()
    {
        ++state_[12];
        if (state_[12] == 0) {
            ++state_[13];
            if (state_[13] == 0) {
                ++state_[14];
                if (state_[14] == 0)
                    ++state_[15];
            }
        }
    }

#if defined(CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4)
    void increment_counter_by_4()
    {
        const uint32_t old = state_[12];
        state_[12] += 4;
        if (state_[12] < old) {
            ++state_[13];
            if (state_[13] == 0) {
                ++state_[14];
                if (state_[14] == 0)
                    ++state_[15];
            }
        }
    }

    void generate_current_4_blocks(uint64_t out[4 * WORDS_PER_BLOCK]) const
    {
        uint32_t counters[4][4];
        for (uint32_t lane = 0; lane < 4; ++lane) {
            counters[lane][0] = state_[12] + lane;
            const uint32_t carry0 = counters[lane][0] < state_[12] ? 1u : 0u;
            counters[lane][1] = state_[13] + carry0;
            const uint32_t carry1 = carry0 != 0 && counters[lane][1] == 0 ? 1u : 0u;
            counters[lane][2] = state_[14] + carry1;
            const uint32_t carry2 = carry1 != 0 && counters[lane][2] == 0 ? 1u : 0u;
            counters[lane][3] = state_[15] + carry2;
        }

        __m128i x[16];
        __m128i initial[16];
        for (int i = 0; i < 12; ++i)
            x[i] = _mm_set1_epi32(static_cast<int>(state_[i]));
        for (int i = 12; i < 16; ++i) {
            const int c = i - 12;
            x[i] = _mm_setr_epi32(static_cast<int>(counters[0][c]),
                                  static_cast<int>(counters[1][c]),
                                  static_cast<int>(counters[2][c]),
                                  static_cast<int>(counters[3][c]));
        }
        for (int i = 0; i < 16; ++i)
            initial[i] = x[i];

        rounds20x4(x);
        for (int i = 0; i < 16; ++i)
            x[i] = _mm_add_epi32(x[i], initial[i]);

        alignas(16) uint32_t lo[4];
        alignas(16) uint32_t hi[4];
        for (int word = 0; word < 8; ++word) {
            _mm_store_si128(reinterpret_cast<__m128i*>(lo), x[2 * word]);
            _mm_store_si128(reinterpret_cast<__m128i*>(hi), x[2 * word + 1]);
            for (int lane = 0; lane < 4; ++lane) {
                out[lane * WORDS_PER_BLOCK + word] =
                    static_cast<uint64_t>(lo[lane]) |
                    (static_cast<uint64_t>(hi[lane]) << 32);
            }
        }
    }
#endif

    void generate_current_block(uint64_t out[WORDS_PER_BLOCK]) const
    {
        uint32_t x[16];
        std::memcpy(x, state_, sizeof(x));
        rounds20(x);
        for (int i = 0; i < 16; ++i)
            x[i] += state_[i];
        for (std::size_t i = 0; i < WORDS_PER_BLOCK; ++i)
            out[i] = static_cast<uint64_t>(x[2 * i])
                   | (static_cast<uint64_t>(x[2 * i + 1]) << 32);
    }

    void refill()
    {
#if defined(CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4)
        generate_current_4_blocks(buffer_);
        increment_counter_by_4();
#else
        generate_current_block(buffer_);
        increment_counter();
#endif
        buffer_index_ = 0;
    }
};

// =============================================================================
// ChaCha20GF1024FFT
// =============================================================================
//
//   Let j = floor(i/2).  Then
//   R(2j)   = ChaCha20(subkey(stream_id), 2j)   XOR low64(P(j))
//   R(2j+1) = ChaCha20(subkey(stream_id), 2j+1) XOR high64(P(j))
//
// P is a degree-1023 polynomial over GF(2^128), defined by the irreducible
// polynomial x^128 + x^7 + x^2 + x + 1.  The 1024 coefficients require an
// 16384-byte perfectly uniform GF seed.  Any 1024 distinct field positions give
// 1024 independent uniform GF(2^128) values.  Exposing the low/high 64-bit
// halves as adjacent output words preserves exact 1024-wise independence of
// arbitrary uint64_t output positions, for a fixed ChaCha stream or a
// ChaCha seed independent of the GF seed. This is a single-stream theorem.
//
// The 1024 coefficients are converted once from monomial basis to a normalized
// subspace/novel basis.  Each aligned block of 1024 consecutive GF evaluation
// points is evaluated by the 10-stage additive FFT and supplies 2048 uint64_t
// output words by using both 64-bit halves of every field value.
class ChaCha20GF1024FFT
{
public:
    using Position = ChaCha20GF1024Position128;
    using Field = ChaCha20GF1024Field128;

    static constexpr std::size_t CHACHA_SEED_BYTES = 32;
    static constexpr std::size_t GF_COEFFICIENTS = 1024;
    static constexpr std::size_t GF_ELEMENT_BYTES = 16;
    static constexpr std::size_t GF_SEED_BYTES = GF_COEFFICIENTS * GF_ELEMENT_BYTES;
    static constexpr std::size_t FULL_SEED_BYTES = CHACHA_SEED_BYTES + GF_SEED_BYTES;
    static constexpr std::size_t FFT_LOGN = 10;
    static constexpr std::size_t FFT_SIZE = std::size_t{1} << FFT_LOGN;
    static constexpr std::size_t OUTPUT_WORDS_PER_FFT = FFT_SIZE * 2;
    static_assert(GF_COEFFICIENTS == FFT_SIZE, "FFT and coefficient counts must agree");

    ChaCha20GF1024FFT(const uint8_t* chacha_seed32,
                    const uint8_t* gf_seed16384,
                    uint64_t stream_id = 0)
        : chacha_(chacha_seed32, stream_id)
    {
        init_gf_from_bytes(gf_seed16384);
        finish_initialization();
    }

    explicit ChaCha20GF1024FFT(const uint8_t* full_seed16416,
                             uint64_t stream_id = 0)
        : chacha_(full_seed16416, stream_id)
    {
        init_gf_from_bytes(full_seed16416 + CHACHA_SEED_BYTES);
        finish_initialization();
    }

    // Convenience constructor only. It creates only 2^64 possible complete
    // initial states and therefore DOES NOT provide the full-seed-space exact
    // 1024-wise guarantee or 256 bits of ChaCha key entropy.
    explicit ChaCha20GF1024FFT(uint64_t seed, uint64_t stream_id = 0)
        : chacha_()
    {
        uint8_t chacha_seed[CHACHA_SEED_BYTES];
        uint8_t gf_seed[GF_SEED_BYTES];
        uint64_t sm = seed;
        fill_from_splitmix64(chacha_seed, sizeof(chacha_seed), sm);
        fill_from_splitmix64(gf_seed, sizeof(gf_seed), sm);
        chacha_ = ChaCha20GF1024Counter128(chacha_seed, stream_id);
        init_gf_from_bytes(gf_seed);
        finish_initialization();
    }

    ChaCha20GF1024FFT()
        : ChaCha20GF1024FFT(uint64_t{0})
    {
    }

    uint64_t next_int()
    {
        if (exhausted_)
            throw std::overflow_error("ChaCha20GF1024FFT: 128-bit word-position space exhausted");

        const Position eval_position = word_position_.shr1();
        ensure_gf_eval_block(eval_position);
        const Field g = gf_block_[eval_position.low10()];
        const uint64_t c = chacha_.next_int();
        const uint64_t out = c ^ ((word_position_.lo & 1ULL) ? g.hi : g.lo);

        if (word_position_.is_max())
            exhausted_ = true;
        else
            word_position_.increment();

        return out;
    }

    void generate(uint64_t* dst, std::size_t count)
    {
        if (count == 0)
            return;
        if (exhausted_)
            throw std::overflow_error("ChaCha20GF1024FFT: 128-bit word-position space exhausted");

        // Validate the last requested word before writing anything.
        Position last = word_position_;
        if (last.add_u64(static_cast<uint64_t>(count - 1)))
            throw std::overflow_error("ChaCha20GF1024FFT: requested range exceeds 128-bit position space");

        while (count != 0) {
            const Position eval_position = word_position_.shr1();
            ensure_gf_eval_block(eval_position);
            std::size_t eval_offset = eval_position.low10();
            const std::size_t first_half = static_cast<std::size_t>(word_position_.lo & 1ULL);
            const std::size_t available = (FFT_SIZE - eval_offset) * 2 - first_half;
            const std::size_t chunk = std::min<std::size_t>(count, available);

            chacha_.generate(dst, chunk);

            std::size_t i = 0;
            if (first_half != 0 && i < chunk) {
                dst[i++] ^= gf_block_[eval_offset++].hi;
            }
            while (i + 1 < chunk) {
                const Field g = gf_block_[eval_offset++];
                dst[i++] ^= g.lo;
                dst[i++] ^= g.hi;
            }
            if (i < chunk)
                dst[i] ^= gf_block_[eval_offset].lo;

            Position chunk_last = word_position_;
            const bool overflow = chunk_last.add_u64(static_cast<uint64_t>(chunk - 1));
            (void)overflow; // already ruled out by the full-range validation above
            if (chunk_last.is_max()) {
                exhausted_ = true;
                word_position_ = chunk_last;
            }
            else {
                chunk_last.increment();
                word_position_ = chunk_last;
            }

            dst += chunk;
            count -= chunk;
        }
    }

    int next_bit()
    {
        // Bit calls consume a word into a separate LSB-first bit reservoir.
        // Interleaving this API with next_int()/generate() therefore does not
        // define one shared call-ordered bit stream.
        if (bit_index_ == 64) {
            current_bits_ = next_int();
            bit_index_ = 0;
        }
        const int bit = static_cast<int>((current_bits_ >> bit_index_) & 1ULL);
        ++bit_index_;
        return bit;
    }

    void seek(Position word_position)
    {
        chacha_.seek(word_position);
        word_position_ = word_position;
        exhausted_ = false;
        // Keep a cached FFT block when seeking within the same aligned block.
        // ensure_gf_eval_block() is keyed by gf_block_base_ and will replace it if needed.
        current_bits_ = 0;
        bit_index_ = 64;
    }

    void seek(uint64_t word_position)
    {
        seek(Position(word_position));
    }

    Position position128() const
    {
        return word_position_;
    }

    // position128() is the final representable position both immediately
    // before and immediately after its output; this accessor distinguishes
    // those two states for checkpoint/restart users.
    bool exhausted() const noexcept
    {
        return exhausted_;
    }

    static constexpr std::size_t gf_seed_state_bytes() { return GF_SEED_BYTES; }
    static constexpr std::size_t gf_cached_block_bytes() { return FFT_SIZE * sizeof(Field); }

    // Slow direct reference evaluation of the full GF(2^128) value. Test use only.
    Field debug_gf_horner128(Position position) const
    {
        return horner_reference(position_to_field(position));
    }

    // FFT result for an arbitrary position, full GF(2^128) value. Test use only.
    Field debug_gf_fft128(Position position)
    {
        ensure_gf_eval_block(position);
        return gf_block_[position.low10()];
    }

    // Backward-friendly low-64 projection helpers for ordinary 64-bit positions.
    uint64_t debug_gf_horner(uint64_t position) const
    {
        return debug_gf_horner128(Position(position)).lo;
    }
    uint64_t debug_gf_fft(uint64_t position)
    {
        return debug_gf_fft128(Position(position)).lo;
    }

    static Field debug_gf_mul_portable(Field a, Field b)
    {
        return gf_mul_portable(a, b);
    }

    static Field debug_gf_mul_fast(Field a, Field b)
    {
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (cpu_has_pclmul())
            return gf_mul_pclmul(a, b);
#endif
        return gf_mul_portable(a, b);
    }

    static bool debug_cpu_has_pclmul()
    {
        return cpu_has_pclmul();
    }

private:
    // x^128 + x^7 + x^2 + x + 1 -> x^128 == x^7+x^2+x+1.
    static constexpr uint64_t GF_REDUCTION = 0x87ULL;

    struct FFTTables
    {
        Field subspace[FFT_LOGN][FFT_LOGN]{};
        Field gamma[FFT_LOGN]{};
        Field gamma_inv[FFT_LOGN]{};
        // Prefix XORs of normalized subspace values. Incrementing a binary
        // node index toggles bits 0..ctz(node), so one XOR advances its factor.
        Field psi_delta[FFT_LOGN][FFT_LOGN]{};
    };

    ChaCha20GF1024Counter128 chacha_;

    // Keep the original coefficients for the independent Horner reference.
    // The three main arrays each occupy 16384 bytes; tables are shared static.
    Field gf_monomial_[GF_COEFFICIENTS]{};
    Field gf_novel_[GF_COEFFICIENTS]{};
    Field gf_block_[FFT_SIZE]{};
    Position gf_block_base_{};
    bool gf_block_valid_ = false;

    Position word_position_{};
    uint64_t current_bits_ = 0;
    int bit_index_ = 64;
    bool use_pclmul_ = false;
    bool exhausted_ = false;

    static uint64_t load64_le(const uint8_t* p)
    {
        return  static_cast<uint64_t>(p[0])
            | (static_cast<uint64_t>(p[1]) << 8)
            | (static_cast<uint64_t>(p[2]) << 16)
            | (static_cast<uint64_t>(p[3]) << 24)
            | (static_cast<uint64_t>(p[4]) << 32)
            | (static_cast<uint64_t>(p[5]) << 40)
            | (static_cast<uint64_t>(p[6]) << 48)
            | (static_cast<uint64_t>(p[7]) << 56);
    }

    static void store64_le(uint8_t* p, uint64_t v)
    {
        for (int i = 0; i < 8; ++i)
            p[i] = static_cast<uint8_t>(v >> (8 * i));
    }

    static uint64_t splitmix64_step(uint64_t& state)
    {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    static void fill_from_splitmix64(uint8_t* dst, std::size_t bytes,
                                     uint64_t& sm_state)
    {
        while (bytes >= 8) {
            const uint64_t v = splitmix64_step(sm_state);
            store64_le(dst, v);
            dst += 8;
            bytes -= 8;
        }
        if (bytes != 0) {
            uint8_t tmp[8];
            store64_le(tmp, splitmix64_step(sm_state));
            std::memcpy(dst, tmp, bytes);
        }
    }

    void init_gf_from_bytes(const uint8_t* seed16384)
    {
        for (std::size_t i = 0; i < GF_COEFFICIENTS; ++i) {
            const Field v(load64_le(seed16384 + 16 * i),
                          load64_le(seed16384 + 16 * i + 8));
            gf_novel_[i] = v;
            gf_monomial_[i] = v;
        }
    }

    static Field position_to_field(Position p)
    {
        return Field(p.lo, p.hi);
    }

    static Field gf_mul_portable(Field a, Field b)
    {
        Field r;
        for (int i = 0; i < 128; ++i) {
            if (b.lo & 1ULL)
                r ^= a;

            b.lo = (b.lo >> 1) | (b.hi << 63);
            b.hi >>= 1;

            const uint64_t carry = a.hi >> 63;
            a.hi = (a.hi << 1) | (a.lo >> 63);
            a.lo <<= 1;
            if (carry)
                a.lo ^= GF_REDUCTION;
        }
        return r;
    }

#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    static CHACHA20GF1024LFSR1024FFT_FORCE_INLINE Field gf_mul_pclmul_prepared(__m128i va, __m128i va01, Field b)
    {
        const __m128i vb = _mm_set_epi64x(static_cast<long long>(b.hi),
                                        static_cast<long long>(b.lo));
        const __m128i p00 = _mm_clmulepi64_si128(va, vb, 0x00);
        const __m128i p11 = _mm_clmulepi64_si128(va, vb, 0x11);
        const __m128i vb01 = _mm_xor_si128(vb, _mm_shuffle_epi32(vb, 0x4e));
        __m128i px = _mm_clmulepi64_si128(va01, vb01, 0x00);
        px = _mm_xor_si128(px, _mm_xor_si128(p00, p11));
        return reduce_cl_product(p00, px, p11);
    }

#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    static Field gf_mul_pclmul(Field a, Field b)
    {
        const __m128i va = _mm_set_epi64x(static_cast<long long>(a.hi),
                                        static_cast<long long>(a.lo));
        const __m128i va01 = _mm_xor_si128(va, _mm_shuffle_epi32(va, 0x4e));
        return gf_mul_pclmul_prepared(va, va01, b);
    }
#endif

#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    static CHACHA20GF1024LFSR1024FFT_FORCE_INLINE Field reduce_cl_product(__m128i p00, __m128i cross, __m128i p11)
    {
        const uint64_t p00lo = static_cast<uint64_t>(_mm_cvtsi128_si64(p00));
        const uint64_t p00hi = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(p00, 8)));
        const uint64_t xlo   = static_cast<uint64_t>(_mm_cvtsi128_si64(cross));
        const uint64_t xhi   = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(cross, 8)));
        const uint64_t p11lo = static_cast<uint64_t>(_mm_cvtsi128_si64(p11));
        const uint64_t p11hi = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(p11, 8)));

        uint64_t c0 = p00lo;
        uint64_t c1 = p00hi ^ xlo;
        const uint64_t c2 = p11lo ^ xhi;
        const uint64_t c3 = p11hi;

        // Fold H*x^128 with x^128 = 1+x+x^2+x^7.  H=(c2,c3).
        uint64_t t0 = 0, t1 = 0, t2 = 0;
        t0 ^= c2; t1 ^= c3; // shift 0

        t0 ^= c2 << 1;
        t1 ^= (c3 << 1) | (c2 >> 63);
        t2 ^= c3 >> 63;

        t0 ^= c2 << 2;
        t1 ^= (c3 << 2) | (c2 >> 62);
        t2 ^= c3 >> 62;

        t0 ^= c2 << 7;
        t1 ^= (c3 << 7) | (c2 >> 57);
        t2 ^= c3 >> 57;

        c0 ^= t0;
        c1 ^= t1;

        // t2 has at most 7 significant bits and represents terms x^(128+j).
        // Fold those once more; the result is below degree 14.
        const uint64_t f = t2 ^ (t2 << 1) ^ (t2 << 2) ^ (t2 << 7);
        c0 ^= f;

        return Field(c0, c1);
    }
#endif

    static bool cpu_has_pclmul()
    {
#if defined(CHACHA20GF1024LFSR1024_FORCE_PORTABLE)
        return false;
#else
        // CPUID capability is process-invariant; do not query it in hot test loops.
        static const bool supported = []() {
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86)
            int regs[4] = {};
            __cpuid(regs, 1);
            return (static_cast<uint32_t>(regs[2]) & (1u << 1)) != 0;
#elif defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
            __builtin_cpu_init();
            return __builtin_cpu_supports("pclmul") != 0;
#else
            return false;
#endif
        }();
        return supported;
#endif
    }

    Field gf_mul(Field a, Field b) const
    {
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (use_pclmul_)
            return gf_mul_pclmul(a, b);
#endif
        return gf_mul_portable(a, b);
    }

    // a^(2^128-2).  Initialization-only; simplicity matters more than speed.
    static Field gf_inv_portable(Field a)
    {
        Field r(1, 0);
        Field p = a;
        for (unsigned bit = 0; bit < 128; ++bit) {
            if (bit != 0) // exponent 2^128-2 has bits 1..127 set, bit 0 clear
                r = gf_mul_portable(r, p);
            p = gf_mul_portable(p, p);
        }
        return r;
    }

    static Field eval_subspace_portable(const FFTTables& tables,
                                        std::size_t level, Field x)
    {
        Field r;
        Field xp = x;
        for (std::size_t j = 0; j <= level; ++j) {
            r ^= gf_mul_portable(tables.subspace[level][j], xp);
            if (j != level)
                xp = gf_mul_portable(xp, xp);
        }
        return r;
    }

    static FFTTables make_fft_tables()
    {
        FFTTables tables;
        tables.subspace[0][0] = Field(1, 0); // s_0(X)=X

        for (std::size_t i = 0; i < FFT_LOGN; ++i) {
            const Field beta(uint64_t{1} << i, 0);
            tables.gamma[i] = eval_subspace_portable(tables, i, beta);
            tables.gamma_inv[i] = gf_inv_portable(tables.gamma[i]);

            Field delta;
            for (std::size_t j = i + 1; j < FFT_LOGN; ++j) {
                const Field sj = eval_subspace_portable(
                    tables, i, Field(uint64_t{1} << j, 0));
                delta ^= gf_mul_portable(sj, tables.gamma_inv[i]);
                tables.psi_delta[i][j - i - 1] = delta;
            }

            if (i + 1 == FFT_LOGN)
                break;

            Field next[FFT_LOGN] = {};
            next[0] = gf_mul_portable(tables.gamma[i], tables.subspace[i][0]);
            for (std::size_t j = 1; j <= i; ++j) {
                next[j] = gf_mul_portable(tables.subspace[i][j - 1],
                                          tables.subspace[i][j - 1])
                        ^ gf_mul_portable(tables.gamma[i], tables.subspace[i][j]);
            }
            next[i + 1] = gf_mul_portable(tables.subspace[i][i],
                                          tables.subspace[i][i]);
            for (std::size_t j = 0; j <= i + 1; ++j)
                tables.subspace[i + 1][j] = next[j];
        }
        return tables;
    }

    static const FFTTables& fft_tables()
    {
        // These values depend only on the fixed field and FFT basis, not on
        // the seed.  C++ guarantees thread-safe one-time initialization.
        static const FFTTables tables = make_fft_tables();
        return tables;
    }

    void monomial_to_novel(Field* data, std::size_t logn,
                           const FFTTables& tables)
    {
        if (logn == 0)
            return;

        const std::size_t level = logn - 1;
        const std::size_t half = std::size_t{1} << level;
        const std::size_t n = half * 2;

        for (std::size_t idx = n; idx-- > half;) {
            const Field q = data[idx];
            const std::size_t qi = idx - half;
            for (std::size_t j = 0; j < level; ++j)
                data[qi + (std::size_t{1} << j)] ^=
                    gf_mul(q, tables.subspace[level][j]);
            data[idx] = gf_mul(tables.gamma[level], q);
        }

        monomial_to_novel(data, logn - 1, tables);
        monomial_to_novel(data + half, logn - 1, tables);
    }

    void finish_initialization()
    {
        use_pclmul_ = cpu_has_pclmul();
        const FFTTables& tables = fft_tables();
        monomial_to_novel(gf_novel_, FFT_LOGN, tables);
        word_position_ = Position();
        gf_block_base_ = Position();
        gf_block_valid_ = false;
        current_bits_ = 0;
        bit_index_ = 64;
        exhausted_ = false;
    }

    Field horner_reference(Field x) const
    {
        Field y = gf_monomial_[GF_COEFFICIENTS - 1];
        for (std::size_t i = GF_COEFFICIENTS - 1; i-- > 0;)
            y = gf_mul_portable(y, x) ^ gf_monomial_[i];
        return y;
    }

    void ensure_gf_eval_block(Position position)
    {
        const Position base = position.aligned_down_1024();
        if (gf_block_valid_ && gf_block_base_ == base)
            return;

        std::memcpy(gf_block_, gf_novel_, sizeof(gf_block_));
        fft_block(base);
        gf_block_base_ = base;
        gf_block_valid_ = true;
    }

    void prepare_base_t(Position base, Field base_t[FFT_LOGN],
                        const FFTTables& tables) const
    {
        Field s = position_to_field(base); // s_0(x) = x
        for (std::size_t i = 0; i < FFT_LOGN; ++i) {
            base_t[i] = gf_mul(s, tables.gamma_inv[i]);
            if (i + 1 != FFT_LOGN) {
                // s_{i+1}(x) = s_i(x)^2 + gamma_i*s_i(x).
                s = gf_mul(s, s) ^ gf_mul(tables.gamma[i], s);
            }
        }
    }

    static std::size_t trailing_zero_index(std::size_t value)
    {
        // Only called for nonzero node indices below FFT_SIZE.
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86)
        unsigned long index;
        _BitScanForward(&index, static_cast<unsigned long>(value));
        return static_cast<std::size_t>(index);
#elif defined(__GNUC__) || defined(__clang__)
        return static_cast<std::size_t>(__builtin_ctz(static_cast<unsigned>(value)));
#else
        std::size_t index = 0;
        while ((value & 1) == 0) {
            value >>= 1;
            ++index;
        }
        return index;
#endif
    }

    void fft_iterative_portable(Field* data,
                                const Field base_t[FFT_LOGN],
                                const FFTTables& tables) const
    {
        for (std::size_t level = FFT_LOGN; level-- > 0;) {
            const std::size_t half = std::size_t{1} << level;
            const std::size_t node_size = half * 2;
            const std::size_t nodes = FFT_SIZE / node_size;
            Field t = base_t[level];

            for (std::size_t node = 0; node < nodes; ++node) {
                if (node != 0) {
                    const std::size_t bit = trailing_zero_index(node);
                    t ^= tables.psi_delta[level][bit];
                }
                Field* const block = data + node * node_size;
                for (std::size_t i = 0; i < half; ++i) {
                    const Field upper = block[i + half];
                    const Field left = block[i] ^ gf_mul_portable(t, upper);
                    block[i] = left;
                    block[i + half] = left ^ upper;
                }
            }
        }
    }

#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    void fft_iterative_pclmul(Field* data,
                              const Field base_t[FFT_LOGN],
                              const FFTTables& tables) const
    {
        for (std::size_t level = FFT_LOGN; level-- > 0;) {
            const std::size_t half = std::size_t{1} << level;
            const std::size_t node_size = half * 2;
            const std::size_t nodes = FFT_SIZE / node_size;
            Field t = base_t[level];

            for (std::size_t node = 0; node < nodes; ++node) {
                if (node != 0) {
                    const std::size_t bit = trailing_zero_index(node);
                    t ^= tables.psi_delta[level][bit];
                }
                Field* const block = data + node * node_size;
                // All butterflies in this node share t. Prepare its SIMD
                // representation and Karatsuba half-XOR once per node.
                const __m128i vt = _mm_set_epi64x(static_cast<long long>(t.hi),
                                                static_cast<long long>(t.lo));
                const __m128i vt01 = _mm_xor_si128(vt, _mm_shuffle_epi32(vt, 0x4e));
                for (std::size_t i = 0; i < half; ++i) {
                    const Field upper = block[i + half];
                    const Field left = block[i] ^ gf_mul_pclmul_prepared(vt, vt01, upper);
                    block[i] = left;
                    block[i + half] = left ^ upper;
                }
            }
        }
    }
#endif

    void fft_block(Position base)
    {
        const FFTTables& tables = fft_tables();
        Field base_t[FFT_LOGN];
        prepare_base_t(base, base_t, tables);
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (use_pclmul_) {
            fft_iterative_pclmul(gf_block_, base_t, tables);
            return;
        }
#endif
        fft_iterative_portable(gf_block_, base_t, tables);
    }
};

// =============================================================================
// AGHP LFSR1024. A/Q formal series; exact Rabin test; 128-bit WORD jumps.
// =============================================================================
class LFSR1024
{
public:
    static constexpr std::size_t LIMBS = 16;
    static constexpr std::size_t POLY_BYTES = 128;
    static constexpr std::size_t SEED_BYTES = 256;
    using Position = ChaCha20GF1024Position128;
    using Value = std::array<uint64_t, LIMBS>;
    using Wide = std::array<uint64_t, 2 * LIMBS>;
    using Raw = std::array<uint64_t, LIMBS + 1>;

    struct Parameters
    {
        Value q{};                         // implicit monic z^1024
        uint64_t inverse_low = 0;           // 1/Q modulo z^64
        uint64_t inverse_reversed = 0;      // reversed-Q inverse modulo z^64
        std::array<Value, 128> word_jump{}; // z^(-64*2^i) modulo Q
        bool accelerated = false;
    };
    using SharedParameters = std::shared_ptr<const Parameters>;

    // Read exactly 128 bytes in the public little-endian coefficient order.
    static Value decode(const uint8_t* bytes)
    {
        Value v{};
        for (std::size_t j = 0; j < LIMBS; ++j)
            for (unsigned b = 0; b < 8; ++b)
                v[j] |= uint64_t(bytes[8 * j + b]) << (8 * b);
        return v;
    }

    static SharedParameters prepare(const uint8_t* polynomial)
    {
        auto p = std::make_shared<Parameters>();
        init_arithmetic(*p, decode(polynomial));
        if (!is_irreducible(*p))
            throw std::invalid_argument("LFSR1024: Q must be monic irreducible of degree 1024");
        make_jumps(*p);
        return p;
    }

    // Callback must fill the complete requested buffer or throw. Each trial
    // is fresh: forcing Q(0)=1 preserves uniformity among odd monic candidates.
    // Exact rejection sampling yields uniform irreducible Q. A trial limit
    // causes an explicit failure, never a fallback polynomial.
    template<class FillRandom>
    static SharedParameters sample(uint8_t* polynomial, FillRandom& fill,
                                   std::size_t max_attempts)
    {
        auto p = std::make_shared<Parameters>();
        for (std::size_t attempt = 0; attempt < max_attempts; ++attempt) {
            fill(polynomial, POLY_BYTES);
            polynomial[0] |= uint8_t{1};
            const Value q = decode(polynomial);
            if (has_small_factor(q))
                continue;
            init_arithmetic(*p, q);
            if (!is_irreducible(*p))
                continue;
            make_jumps(*p);
            return p;
        }
        throw std::runtime_error("LFSR1024: irreducible-polynomial sampling attempt limit reached");
    }

    LFSR1024(SharedParameters p, const uint8_t* numerator)
        : params_(std::move(p)), initial_(decode(numerator)), state_(initial_)
    {
        if (!params_)
            throw std::invalid_argument("LFSR1024: missing prepared parameters");
    }

    void seek(Position word_position)
    {
        // Work modulo Q: discarding j words maps A to A*z^(-64*j).
        // Keeping the exponent in WORD units avoids truncating a 134-bit bit
        // index to 128 bits. Maximum supported index is still 2^128-1 words.
        Value v = initial_;
        uint64_t index = word_position.lo;
        for (unsigned bit = 0; index != 0; ++bit, index >>= 1)
            if (index & 1)
                v = multiply_mod(v, params_->word_jump[bit], *params_);
        index = word_position.hi;
        for (unsigned bit = 64; index != 0; ++bit, index >>= 1)
            if (index & 1)
                v = multiply_mod(v, params_->word_jump[bit], *params_);
        state_ = v;
    }

    uint64_t next_word() noexcept
    {
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (params_->accelerated)
            return next_word_fast();
#endif
        return next_word_portable();
    }

    void xor_words(uint64_t* dst, std::size_t count) noexcept
    {
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (params_->accelerated) {
            xor_words_fast(dst, count);
            return;
        }
#endif
        for (std::size_t i = 0; i < count; ++i)
            dst[i] ^= next_word_portable();
    }

private:
    struct Product { uint64_t lo, hi; };
    SharedParameters params_;
    Value initial_{};
    Value state_{};

    static Product clmul_portable(uint64_t a, uint64_t b) noexcept
    {
        Product out{0, 0};
        uint64_t lo = a, hi = 0;
        for (unsigned i = 0; i < 64; ++i) {
            const uint64_t mask = uint64_t{0} - (b & 1);
            out.lo ^= lo & mask;
            out.hi ^= hi & mask;
            hi = (hi << 1) | (lo >> 63);
            lo <<= 1;
            b >>= 1;
        }
        return out;
    }

#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    static inline Product clmul_fast(uint64_t a, uint64_t b) noexcept
    {
        const __m128i aa = _mm_set_epi64x(0, static_cast<long long>(a));
        const __m128i bb = _mm_set_epi64x(0, static_cast<long long>(b));
        const __m128i v = _mm_clmulepi64_si128(aa, bb, 0x00);
        return Product{static_cast<uint64_t>(_mm_cvtsi128_si64(v)),
                       static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(v, 8)))};
    }
#endif

    static uint64_t reverse_bits(uint64_t v) noexcept
    {
        v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
        v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
        v = ((v >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((v & 0x0f0f0f0f0f0f0f0fULL) << 4);
        v = ((v >> 8) & 0x00ff00ff00ff00ffULL) | ((v & 0x00ff00ff00ff00ffULL) << 8);
        v = ((v >> 16) & 0x0000ffff0000ffffULL) | ((v & 0x0000ffff0000ffffULL) << 16);
        return (v >> 32) | (v << 32);
    }

    static uint64_t inverse_series64(uint64_t odd) noexcept
    {
        // odd^64 == 1 mod z^64 in characteristic two. Therefore odd^63
        // is its inverse; compute the product of odd^(1,2,4,8,16,32).
        uint64_t inverse = 1;
        for (unsigned i = 0; i < 6; ++i) {
            inverse = clmul_portable(inverse, odd).lo;
            odd = clmul_portable(odd, odd).lo;
        }
        return inverse;
    }

    static void init_arithmetic(Parameters& p, const Value& q)
    {
        p.q = q;
        p.accelerated = ChaCha20GF1024FFT::debug_cpu_has_pclmul();
        if ((q[0] & 1) == 0)
            throw std::invalid_argument("LFSR1024: Q constant coefficient must be one");
        p.inverse_low = inverse_series64(q[0]);
        const uint64_t reversed_low = uint64_t{1} | (reverse_bits(q[LIMBS - 1]) << 1);
        p.inverse_reversed = inverse_series64(reversed_low);
    }

    // Dense polynomial division 64 quotient bits at a time. For each leading
    // word H, reverse(H)*inverse(reverse(Q)) determines its quotient word.
    // This works with arbitrary dense Q; no sparse-polynomial assumption.
    static Value reduce_portable(Wide a, const Parameters& p) noexcept
    {
        for (std::size_t top = 2 * LIMBS; top-- > LIMBS;) {
            const uint64_t quotient = reverse_bits(
                clmul_portable(reverse_bits(a[top]), p.inverse_reversed).lo);
            const std::size_t offset = top - LIMBS;
            for (std::size_t j = 0; j < LIMBS; ++j) {
                const Product v = clmul_portable(quotient, p.q[j]);
                a[offset + j] ^= v.lo;
                a[offset + j + 1] ^= v.hi;
            }
            a[top] ^= quotient; // implicit z^1024 coefficient
        }
        Value out{};
        std::copy_n(a.begin(), LIMBS, out.begin());
        return out;
    }

#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    static Value reduce_fast(Wide a, const Parameters& p) noexcept
    {
        for (std::size_t top = 2 * LIMBS; top-- > LIMBS;) {
            const uint64_t quotient = reverse_bits(
                clmul_fast(reverse_bits(a[top]), p.inverse_reversed).lo);
            const std::size_t offset = top - LIMBS;
            for (std::size_t j = 0; j < LIMBS; ++j) {
                const Product v = clmul_fast(quotient, p.q[j]);
                a[offset + j] ^= v.lo;
                a[offset + j + 1] ^= v.hi;
            }
            a[top] ^= quotient;
        }
        Value out{};
        std::copy_n(a.begin(), LIMBS, out.begin());
        return out;
    }

#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    static Value multiply_mod_fast(const Value& a, const Value& b,
                                   const Parameters& p) noexcept
    {
        Wide product{};
        for (std::size_t i = 0; i < LIMBS; ++i)
            for (std::size_t j = 0; j < LIMBS; ++j) {
                const Product v = clmul_fast(a[i], b[j]);
                product[i + j] ^= v.lo;
                product[i + j + 1] ^= v.hi;
            }
        return reduce_fast(product, p);
    }
#endif

    static Value multiply_mod(const Value& a, const Value& b,
                              const Parameters& p) noexcept
    {
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (p.accelerated)
            return multiply_mod_fast(a, b, p);
#endif
        Wide product{};
        for (std::size_t i = 0; i < LIMBS; ++i)
            for (std::size_t j = 0; j < LIMBS; ++j) {
                const Product v = clmul_portable(a[i], b[j]);
                product[i + j] ^= v.lo;
                product[i + j + 1] ^= v.hi;
            }
        return reduce_portable(product, p);
    }

    static uint64_t spread32(uint32_t bits) noexcept
    {
        uint64_t v = bits;
        v = (v | (v << 16)) & 0x0000ffff0000ffffULL;
        v = (v | (v << 8)) & 0x00ff00ff00ff00ffULL;
        v = (v | (v << 4)) & 0x0f0f0f0f0f0f0f0fULL;
        v = (v | (v << 2)) & 0x3333333333333333ULL;
        v = (v | (v << 1)) & 0x5555555555555555ULL;
        return v;
    }

    static Value square_mod(const Value& a, const Parameters& p) noexcept
    {
        Wide squared{};
        // Squaring over GF(2) merely inserts zero bits; no cross products.
        for (std::size_t i = 0; i < LIMBS; ++i) {
            squared[2 * i] = spread32(static_cast<uint32_t>(a[i]));
            squared[2 * i + 1] = spread32(static_cast<uint32_t>(a[i] >> 32));
        }
#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
        if (p.accelerated)
            return reduce_fast(squared, p);
#endif
        return reduce_portable(squared, p);
    }

    static int degree(const Raw& a) noexcept
    {
        for (std::size_t j = a.size(); j-- > 0;) {
            uint64_t v = a[j];
            if (v == 0)
                continue;
            int bit = 0;
            if (v >> 32) { v >>= 32; bit += 32; }
            if (v >> 16) { v >>= 16; bit += 16; }
            if (v >> 8)  { v >>= 8;  bit += 8; }
            if (v >> 4)  { v >>= 4;  bit += 4; }
            if (v >> 2)  { v >>= 2;  bit += 2; }
            if (v >> 1)  { bit += 1; }
            return static_cast<int>(64 * j) + bit;
        }
        return -1;
    }

    static bool coprime(const Value& a, const Parameters& p) noexcept
    {
        Raw left{}, right{};
        std::copy(p.q.begin(), p.q.end(), left.begin());
        left[LIMBS] = 1;
        std::copy(a.begin(), a.end(), right.begin());
        int dr = degree(right);
        while (dr >= 0) {
            int dl = degree(left);
            while (dl >= dr) {
                const unsigned distance = static_cast<unsigned>(dl - dr);
                const std::size_t word = distance / 64;
                const unsigned bits = distance % 64;
                for (std::size_t j = 0; j + word < left.size(); ++j) {
                    left[j + word] ^= right[j] << bits;
                    if (bits != 0 && j + word + 1 < left.size())
                        left[j + word + 1] ^= right[j] >> (64 - bits);
                }
                dl = degree(left);
            }
            std::swap(left, right);
            dr = degree(right);
        }
        return degree(left) == 0; // only nonzero constant over GF(2) is 1
    }

    static bool has_small_factor(const Value& q) noexcept
    {
        if ((q[0] & 1) == 0)
            return true;
        uint64_t parity = 1; // includes the implicit monic coefficient
        for (uint64_t word : q)
            parity ^= word;
        parity ^= parity >> 32; parity ^= parity >> 16;
        parity ^= parity >> 8;  parity ^= parity >> 4;
        parity ^= parity >> 2;  parity ^= parity >> 1;
        if ((parity & 1) == 0) // Q(1)=0 -> factor z+1
            return true;
        static constexpr uint32_t small[] = {
            0x7, 0xb, 0xd, 0x13, 0x19, 0x1f,
            0x25, 0x29, 0x2f, 0x37, 0x3b, 0x3d
        };
        for (uint32_t divisor : small) {
            unsigned d = 0;
            for (uint32_t t = divisor; t >>= 1;)
                ++d;
            uint32_t rem = 1; // leading coefficient at degree 1024
            for (std::size_t bit = 1024; bit-- > 0;) {
                rem = (rem << 1) | uint32_t((q[bit / 64] >> (bit % 64)) & 1);
                if (rem & (uint32_t{1} << d))
                    rem ^= divisor;
            }
            if (rem == 0)
                return true;
        }
        return false;
    }

    static bool is_irreducible(const Parameters& p) noexcept
    {
        if (has_small_factor(p.q))
            return false;
        // Exact Rabin criterion. 1024 has just one distinct prime divisor, 2:
        // gcd(z^(2^512)-z,Q)=1 and z^(2^1024)=z modulo Q.
        Value x{};
        x[0] = 2;
        Value h = x;
        for (unsigned i = 1; i <= 1024; ++i) {
            h = square_mod(h, p);
            if (i < 1024 && h == x)
                return false;
            if (i == 512) {
                Value g = h;
                g[0] ^= 2;
                if (!coprime(g, p))
                    return false;
            }
        }
        return h == x;
    }

    static Value divide_x(Value a, const Parameters& p) noexcept
    {
        const uint64_t bit = a[0] & 1;
        const uint64_t mask = uint64_t{0} - bit;
        for (std::size_t i = 0; i < LIMBS; ++i)
            a[i] ^= p.q[i] & mask;
        for (std::size_t i = 0; i + 1 < LIMBS; ++i)
            a[i] = (a[i] >> 1) | (a[i + 1] << 63);
        a[LIMBS - 1] = (a[LIMBS - 1] >> 1) | (bit << 63);
        return a;
    }

    static void make_jumps(Parameters& p) noexcept
    {
        Value inverse_word{};
        inverse_word[0] = 1;
        for (unsigned i = 0; i < 64; ++i)
            inverse_word = divide_x(inverse_word, p);
        p.word_jump[0] = inverse_word;
        for (std::size_t i = 1; i < p.word_jump.size(); ++i)
            p.word_jump[i] = square_mod(p.word_jump[i - 1], p);
    }

    uint64_t next_word_portable() noexcept
    {
        // B = (A/Q) mod z^64, then A' = (A + B*Q)/z^64.
        const uint64_t out = clmul_portable(state_[0], params_->inverse_low).lo;
        uint64_t carry = clmul_portable(out, params_->q[0]).hi;
        for (std::size_t j = 1; j < LIMBS; ++j) {
            const Product p = clmul_portable(out, params_->q[j]);
            state_[j - 1] = state_[j] ^ carry ^ p.lo;
            carry = p.hi;
        }
        state_[LIMBS - 1] = carry ^ out; // implicit monic coefficient
        return out;
    }

#if defined(CHACHA20GF1024LFSR1024FFT_MSVC_X86) || defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    uint64_t next_word_fast() noexcept
    {
        const uint64_t out = clmul_fast(state_[0], params_->inverse_low).lo;
        uint64_t carry = clmul_fast(out, params_->q[0]).hi;
        for (std::size_t j = 1; j < LIMBS; ++j) {
            const Product p = clmul_fast(out, params_->q[j]);
            state_[j - 1] = state_[j] ^ carry ^ p.lo;
            carry = p.hi;
        }
        state_[LIMBS - 1] = carry ^ out;
        return out;
    }

#  if defined(CHACHA20GF1024LFSR1024FFT_GNU_X86)
    CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#  endif
    void xor_words_fast(uint64_t* dst, std::size_t count) noexcept
    {
        for (std::size_t i = 0; i < count; ++i)
            dst[i] ^= next_word_fast();
    }
#endif
};

} // namespace chacha20gf1024lfsr1024_detail

using ChaCha20GF1024LFSR1024Position128 =
    chacha20gf1024lfsr1024_detail::ChaCha20GF1024Position128;
using ChaCha20GF1024LFSR1024Field128 =
    chacha20gf1024lfsr1024_detail::ChaCha20GF1024Field128;
using ChaCha20GF1024LFSR1024Counter128 =
    chacha20gf1024lfsr1024_detail::ChaCha20GF1024Counter128;

class ChaCha20GF1024LFSR1024
{
    using Base = chacha20gf1024lfsr1024_detail::ChaCha20GF1024FFT;
    using Mask = chacha20gf1024lfsr1024_detail::LFSR1024;
public:
    using Position = ChaCha20GF1024LFSR1024Position128;
    using Field = ChaCha20GF1024LFSR1024Field128;
    static constexpr std::size_t CHACHA_SEED_BYTES = Base::CHACHA_SEED_BYTES;
    static constexpr std::size_t GF_COEFFICIENTS = Base::GF_COEFFICIENTS;
    static constexpr std::size_t GF_ELEMENT_BYTES = Base::GF_ELEMENT_BYTES;
    static constexpr std::size_t GF_SEED_BYTES = Base::GF_SEED_BYTES;
    static constexpr std::size_t BASE_SEED_BYTES = Base::FULL_SEED_BYTES;
    // Exactly three AGHP masks; this is not a configurable generator parameter.
    static constexpr std::size_t LFSR_COMPONENTS = 3;
    static constexpr std::size_t LFSR_COMPONENT_SEED_BYTES = Mask::SEED_BYTES;
    static constexpr std::size_t LFSR_SEED_BYTES = LFSR_COMPONENTS * LFSR_COMPONENT_SEED_BYTES;
    // Legacy names identify component zero. Use the indexed helpers for all three.
    static constexpr std::size_t LFSR_POLYNOMIAL_OFFSET = BASE_SEED_BYTES;
    static constexpr std::size_t LFSR_NUMERATOR_OFFSET = BASE_SEED_BYTES + Mask::POLY_BYTES;
    static constexpr std::size_t FULL_SEED_BYTES = BASE_SEED_BYTES + LFSR_SEED_BYTES;
    static constexpr std::size_t FFT_LOGN = Base::FFT_LOGN;
    static constexpr std::size_t FFT_SIZE = Base::FFT_SIZE;
    static constexpr std::size_t OUTPUT_WORDS_PER_FFT = Base::OUTPUT_WORDS_PER_FFT;
    using Seed = std::array<uint8_t, FULL_SEED_BYTES>;

    // component must be in [0, LFSR_COMPONENTS).
    static constexpr std::size_t lfsr_polynomial_offset(std::size_t component)
    {
        return BASE_SEED_BYTES + component * LFSR_COMPONENT_SEED_BYTES;
    }
    static constexpr std::size_t lfsr_numerator_offset(std::size_t component)
    {
        return lfsr_polynomial_offset(component) + Mask::POLY_BYTES;
    }

private:
    using ParameterSet = std::array<Mask::SharedParameters, LFSR_COMPONENTS>;
public:

    // Immutable sampled/validated seed plus expensive precomputed tables.
    // Construct through make_seed or prepare_seed, then share among instances.
    class PreparedSeed
    {
    public:
        PreparedSeed(const PreparedSeed&) = default;
        PreparedSeed(PreparedSeed&&) noexcept = default;
        PreparedSeed& operator=(const PreparedSeed&) = default;
        PreparedSeed& operator=(PreparedSeed&&) noexcept = default;
        const Seed& bytes() const noexcept { return bytes_; }
    private:
        friend class ChaCha20GF1024LFSR1024;
        Seed bytes_;
        ParameterSet parameters_;
        PreparedSeed(Seed bytes, ParameterSet parameters)
            : bytes_(std::move(bytes)), parameters_(std::move(parameters)) {}
    };

    template<class FillRandom>
    static PreparedSeed make_seed(FillRandom&& fill,
                                  std::size_t max_polynomial_attempts = 1000000)
    {
        static_assert(std::is_same<decltype(fill(static_cast<uint8_t*>(nullptr),
                                                 std::size_t{})), void>::value,
                      "Random-byte callback must return void and throw on failure");
        if (max_polynomial_attempts == 0)
            throw std::invalid_argument("ChaCha20GF1024LFSR1024: attempt limit is zero");
        Seed bytes{};
        fill(bytes.data(), BASE_SEED_BYTES);
        ParameterSet parameters;
        for (std::size_t j = 0; j < LFSR_COMPONENTS; ++j) {
            // Fresh bytes for every Q and A. The candidate limit is per Q.
            // Independently sampled equal polynomials are valid; do not reject them.
            parameters[j] = Mask::sample(bytes.data() + lfsr_polynomial_offset(j),
                                         fill, max_polynomial_attempts);
            fill(bytes.data() + lfsr_numerator_offset(j), Mask::POLY_BYTES);
        }
        return PreparedSeed(std::move(bytes), std::move(parameters));
    }

    static PreparedSeed prepare_seed(const Seed& bytes)
    {
        ParameterSet parameters;
        for (std::size_t j = 0; j < LFSR_COMPONENTS; ++j)
            parameters[j] = Mask::prepare(bytes.data() + lfsr_polynomial_offset(j));
        return PreparedSeed(bytes, std::move(parameters));
    }

    // A checked byte-count overload prevents accidentally importing a former
    // 16416-byte GF seed or 16672-byte one-mask seed as the new three-mask seed.
    static PreparedSeed prepare_seed(const uint8_t* bytes, std::size_t size)
    {
        if (bytes == nullptr || size != FULL_SEED_BYTES)
            throw std::invalid_argument("ChaCha20GF1024LFSR1024: expected 17184 seed bytes (three AGHP masks)");
        Seed copy{};
        std::memcpy(copy.data(), bytes, copy.size());
        return prepare_seed(copy);
    }

    explicit ChaCha20GF1024LFSR1024(const PreparedSeed& seed, uint64_t stream_id = 0)
        : base_(seed.bytes_.data(), stream_id),
          masks_{{Mask(seed.parameters_[0], seed.bytes_.data() + lfsr_numerator_offset(0)),
                  Mask(seed.parameters_[1], seed.bytes_.data() + lfsr_numerator_offset(1)),
                  Mask(seed.parameters_[2], seed.bytes_.data() + lfsr_numerator_offset(2))}}
    {
    }

    explicit ChaCha20GF1024LFSR1024(const Seed& seed, uint64_t stream_id = 0)
        : ChaCha20GF1024LFSR1024(prepare_seed(seed), stream_id)
    {
    }

    ChaCha20GF1024LFSR1024(const uint8_t* seed, std::size_t bytes,
                         uint64_t stream_id = 0)
        : ChaCha20GF1024LFSR1024(prepare_seed(seed, bytes), stream_id)
    {
    }

    // Deterministic convenience only. The ideal independent-seed assumptions
    // are not satisfied. No theorem follows from merely passing the
    // irreducibility check. Prefer a reusable PreparedSeed in performance work.
    explicit ChaCha20GF1024LFSR1024(uint64_t seed, uint64_t stream_id = 0)
        : ChaCha20GF1024LFSR1024(make_deterministic_seed(seed), stream_id)
    {
    }

    ChaCha20GF1024LFSR1024() : ChaCha20GF1024LFSR1024(uint64_t{0}) {}

    uint64_t next_int()
    {
        // Base checks exhaustion before any mask advances.
        uint64_t word = base_.next_int();
        for (auto& mask : masks_)
            word ^= mask.next_word();
        return word;
    }

    void generate(uint64_t* dst, std::size_t count)
    {
        if (count == 0)
            return;
        if (dst == nullptr)
            throw std::invalid_argument("ChaCha20GF1024LFSR1024: null output buffer");
        if (base_.exhausted())
            throw std::overflow_error("ChaCha20GF1024LFSR1024: position space exhausted");
        validate_range(base_.position128(), count);
        // Apply all three masks while each GF-sized output chunk is cache-hot.
        while (count != 0) {
            const std::size_t chunk = std::min(count, OUTPUT_WORDS_PER_FFT);
            base_.generate(dst, chunk);
            for (auto& mask : masks_)
                mask.xor_words(dst, chunk);
            dst += chunk;
            count -= chunk;
        }
    }

    int next_bit()
    {
        if (bit_index_ == 64) {
            current_bits_ = next_int();
            bit_index_ = 0;
        }
        return static_cast<int>((current_bits_ >> bit_index_++) & 1);
    }

    void seek(Position position)
    {
        // No mutable global/lazy tables: concurrent seeks in independent
        // copies only read the immutable shared parameters.
        if (position != base_.position128() || base_.exhausted()) {
            for (auto& mask : masks_)
                mask.seek(position);
            base_.seek(position);
        }
        current_bits_ = 0;
        bit_index_ = 64;
    }

    void seek(uint64_t position) { seek(Position(position)); }
    Position position128() const noexcept { return base_.position128(); }
    bool exhausted() const noexcept { return base_.exhausted(); }

    // Does not change this instance's position, mask state or bit reservoir.
    // Copies inherit the same GF coefficients and all three LFSR seeds. No reseeding and
    // no altered stream IDs. Start positions may be full 128-bit word indices.
    // Do not mutate this instance concurrently with this call.
    void generate_parallel_at(uint64_t* dst, std::size_t count, Position start,
                              unsigned thread_count = 0) const
    {
        if (count == 0)
            return;
        if (dst == nullptr)
            throw std::invalid_argument("ChaCha20GF1024LFSR1024: null output buffer");
        validate_range(start, count);
        if (thread_count == 0)
            thread_count = std::max(1u, std::thread::hardware_concurrency());
        const std::size_t blocks = (count - 1) / OUTPUT_WORDS_PER_FFT + 1;
        const unsigned workers = static_cast<unsigned>(
            std::min<std::size_t>(thread_count, blocks));
        if (workers <= 1) {
            auto local = *this;
            local.seek(start);
            local.generate(dst, count);
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(workers);
        std::vector<std::exception_ptr> errors(workers);
        const std::size_t per_worker = blocks / workers;
        const std::size_t extra = blocks % workers;
        std::size_t first_block = 0;
        try {
            for (unsigned t = 0; t < workers; ++t) {
                const std::size_t assigned = per_worker + (t < extra ? 1 : 0);
                const std::size_t offset = first_block * OUTPUT_WORDS_PER_FFT;
                // Only the final worker can receive a partial block. Avoid
                // multiplying a rounded-up final block count past SIZE_MAX.
                const std::size_t words = t + 1 == workers
                    ? count - offset : assigned * OUTPUT_WORDS_PER_FFT;
                Position at = start;
                at.add_u64(static_cast<uint64_t>(offset)); // range checked above
                threads.emplace_back([this, dst, offset, words, at, t, &errors]() {
                    try {
                        auto local = *this;
                        local.seek(at);
                        local.generate(dst + offset, words);
                    }
                    catch (...) {
                        errors[t] = std::current_exception();
                    }
                });
                first_block += assigned;
            }
        }
        catch (...) {
            for (auto& thread : threads)
                if (thread.joinable()) thread.join();
            throw;
        }
        for (auto& thread : threads)
            thread.join();
        for (const auto& error : errors)
            if (error) std::rethrow_exception(error);
    }

    void generate_parallel_at(uint64_t* dst, std::size_t count, uint64_t start,
                              unsigned thread_count = 0) const
    {
        generate_parallel_at(dst, count, Position(start), thread_count);
    }

    // Consuming parallel API. On worker failure the caller's position stays
    // unchanged, but parts of dst may already have been written.
    void generate_parallel(uint64_t* dst, std::size_t count, unsigned thread_count = 0)
    {
        if (count == 0)
            return;
        if (exhausted())
            throw std::overflow_error("ChaCha20GF1024LFSR1024: position space exhausted");
        const Position start = position128();
        const Position last = validate_range(start, count);
        generate_parallel_at(dst, count, start, thread_count);
        // Preserve the old next_bit reservoir, matching ordinary generate().
        const uint64_t reservoir = current_bits_;
        const int reservoir_index = bit_index_;
        if (last.is_max()) {
            seek(last);
            (void)next_int(); // advance all components and mark exhaustion
        }
        else {
            Position next = last;
            next.increment();
            seek(next);
        }
        current_bits_ = reservoir;
        bit_index_ = reservoir_index;
    }

    static constexpr std::size_t gf_seed_state_bytes() { return GF_SEED_BYTES; }
    static constexpr std::size_t gf_cached_block_bytes() { return Base::gf_cached_block_bytes(); }
    static constexpr std::size_t lfsr_shared_table_bytes() { return LFSR_COMPONENTS * sizeof(Mask::Parameters); }
    static bool debug_cpu_has_pclmul() { return Base::debug_cpu_has_pclmul(); }
    static Field debug_gf_mul_portable(Field a, Field b) { return Base::debug_gf_mul_portable(a, b); }
    static Field debug_gf_mul_fast(Field a, Field b) { return Base::debug_gf_mul_fast(a, b); }
    Field debug_gf_horner128(Position p) const { return base_.debug_gf_horner128(p); }
    Field debug_gf_fft128(Position p) { return base_.debug_gf_fft128(p); }
    uint64_t debug_gf_horner(uint64_t p) const { return base_.debug_gf_horner(p); }
    uint64_t debug_gf_fft(uint64_t p) { return base_.debug_gf_fft(p); }
    uint64_t debug_lfsr_word(Position p) const
    {
        uint64_t word = 0;
        for (std::size_t j = 0; j < LFSR_COMPONENTS; ++j)
            word ^= debug_lfsr_component_word(j, p);
        return word;
    }
    uint64_t debug_lfsr_component_word(std::size_t component, Position p) const
    {
        auto copy = masks_.at(component);
        copy.seek(p);
        return copy.next_word();
    }

private:
    Base base_;
    std::array<Mask, LFSR_COMPONENTS> masks_;
    uint64_t current_bits_ = 0;
    int bit_index_ = 64;

    static Position validate_range(Position start, std::size_t count)
    {
        static_assert(sizeof(std::size_t) <= sizeof(uint64_t), "size_t wider than supported");
        if (count != 0 && start.add_u64(static_cast<uint64_t>(count - 1)))
            throw std::overflow_error("ChaCha20GF1024LFSR1024: requested range exceeds 2^128 words");
        return start;
    }

    static PreparedSeed make_deterministic_seed(uint64_t seed)
    {
        // Fill callbacks here always request multiples of eight bytes.
        auto fill = [state = seed](uint8_t* dst, std::size_t count) mutable {
            while (count != 0) {
                uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                z ^= z >> 31;
                const std::size_t bytes = std::min<std::size_t>(count, 8);
                for (std::size_t i = 0; i < bytes; ++i)
                    dst[i] = static_cast<uint8_t>(z >> (8 * i));
                dst += bytes;
                count -= bytes;
            }
        };
        return make_seed(fill);
    }
};

#undef CHACHA20GF1024LFSR1024FFT_FORCE_INLINE
#undef CHACHA20GF1024LFSR1024FFT_CHACHA_SIMD4
#undef CHACHA20GF1024LFSR1024FFT_TARGET_PCLMUL
#undef CHACHA20GF1024LFSR1024FFT_GNU_X86
#undef CHACHA20GF1024LFSR1024FFT_MSVC_X86

#endif // CHACHA20GF1024LFSR1024_SINGLE_FILE_INCLUDED
