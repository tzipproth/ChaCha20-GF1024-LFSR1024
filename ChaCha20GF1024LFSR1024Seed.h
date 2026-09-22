#pragma once

#include "ChaCha20GF1024LFSR1024.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#  if defined(_MSC_VER)
#    pragma comment(lib, "bcrypt.lib")
#  endif
#elif defined(__linux__)
#  include <cerrno>
#  include <sys/random.h>
#else
#  error "ChaCha20GF1024LFSR1024Seed.h currently supports Windows and Linux."
#endif

#if defined(_MSC_VER) && defined(_M_X64)
#  include <intrin.h>
#  include <immintrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#  include <cpuid.h>
#  include <immintrin.h>
#endif

// RDSEED-first or explicit OS-only initialization for the structured seed.
// The LFSR polynomial is sampled with the core's exact rejection procedure;
// filling the complete seed array directly with random bytes is NOT valid.
// Neither RDSEED success nor OS CSPRNG output establishes the ideal entropy assumed
// by the information-theoretic theorems. This helper provides practical seeding.
// Initialization may be expensive. Reuse PreparedSeed across worker instances.
namespace chacha20gf1024lfsr1024_seed
{
    inline bool os_random_bytes(void* dst, std::size_t bytes)
    {
        auto* p = static_cast<std::uint8_t*>(dst);

#if defined(_WIN32)
        // BCryptGenRandom takes ULONG lengths. Chunk for completeness.
        while (bytes != 0) {
            const ULONG chunk = static_cast<ULONG>(
                bytes > static_cast<std::size_t>(0xFFFFFFFFu)
                    ? 0xFFFFFFFFu : bytes);
            const NTSTATUS st = BCryptGenRandom(
                nullptr, reinterpret_cast<PUCHAR>(p), chunk,
                BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            if (!BCRYPT_SUCCESS(st))
                return false;
            p += chunk;
            bytes -= chunk;
        }
        return true;

#elif defined(__linux__)
        while (bytes != 0) {
            const ssize_t n = ::getrandom(p, bytes, 0);
            if (n > 0) {
                p += static_cast<std::size_t>(n);
                bytes -= static_cast<std::size_t>(n);
                continue;
            }
            if (n < 0 && errno == EINTR)
                continue;
            return false;
        }
        return true;
#endif
    }

    using FullSeed = ChaCha20GF1024LFSR1024::Seed;
    using PreparedSeed = ChaCha20GF1024LFSR1024::PreparedSeed;

    inline PreparedSeed make_os_prepared_seed()
    {
        auto fill = [](std::uint8_t* dst, std::size_t bytes) {
            if (!os_random_bytes(dst, bytes))
                throw std::runtime_error("OS random generator failed");
        };
        return ChaCha20GF1024LFSR1024::make_seed(fill);
    }

    // Serialization convenience. Re-importing these bytes checks the
    // polynomial again; use make_os_prepared_seed() for immediate construction.
    inline FullSeed make_os_full_seed()
    {
        const PreparedSeed seed = make_os_prepared_seed();
        return seed.bytes();
    }

    inline ChaCha20GF1024LFSR1024 make_os_seeded_rng(std::uint64_t stream_id = 0)
    {
        const PreparedSeed seed = make_os_prepared_seed();
        return ChaCha20GF1024LFSR1024(seed, stream_id);
    }

    enum class SeedSource { None, RDSEED, OS };
    enum class FallbackReason { None, Unavailable, RetryBudgetExceeded };

    struct SeedOptions
    {
        // Limits count instruction attempts/failures, not elapsed wall time.
        unsigned rdseed_attempts_per_word = 1024;
        std::uint64_t rdseed_max_failures = 65536;
    };

    struct SeedReport
    {
        bool rdseed64_available = false;
        SeedSource source = SeedSource::None;
        FallbackReason fallback = FallbackReason::None;
        // Includes random bytes used for rejected polynomial candidates.
        // If source == OS, all rdseed_bytes were discarded on restart.
        std::uint64_t rdseed_bytes = 0;
        std::uint64_t rdseed_failures = 0;
        std::uint64_t os_bytes = 0;
    };

    inline const char* os_source_name() noexcept
    {
#if defined(_WIN32)
        return "OS / BCryptGenRandom";
#else
        return "OS / getrandom";
#endif
    }

    inline const char* seed_source_name(SeedSource source) noexcept
    {
        switch (source) {
        case SeedSource::RDSEED: return "RDSEED (64-bit)";
        case SeedSource::OS: return os_source_name();
        default: return "none (no completed seed)";
        }
    }

    inline const char* fallback_reason_name(FallbackReason reason) noexcept
    {
        switch (reason) {
        case FallbackReason::Unavailable:
            return "RDSEED64 unavailable on this CPU or build";
        case FallbackReason::RetryBudgetExceeded:
            return "RDSEED retry budget exceeded";
        default: return "none";
        }
    }

    // Only the 64-bit instruction is used. Other architectures and 32-bit
    // builds use the OS source. Do not infer physical quality from CPUID.
    inline bool rdseed64_available() noexcept
    {
#if defined(_MSC_VER) && defined(_M_X64)
        int regs[4]{};
        __cpuid(regs, 0);
        if (regs[0] < 7) return false;
        __cpuidex(regs, 7, 0);
        return (static_cast<unsigned>(regs[1]) & (1u << 18)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        if (__get_cpuid_max(0, nullptr) < 7) return false;
        unsigned a = 0, b = 0, c = 0, d = 0;
        __cpuid_count(7, 0, a, b, c, d);
        return (b & (1u << 18)) != 0;
#else
        return false;
#endif
    }

    namespace detail
    {
        struct RdseedExhausted {};

#if defined(_MSC_VER) && defined(_M_X64)
        inline bool rdseed_word(std::uint64_t& out) noexcept
        {
            unsigned __int64 value = 0;
            if (!_rdseed64_step(&value)) return false;
            out = static_cast<std::uint64_t>(value);
            return true;
        }
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        // No global -mrdseed is needed; never call before CPUID succeeds.
        __attribute__((target("rdseed")))
        inline bool rdseed_word(std::uint64_t& out) noexcept
        {
            unsigned long long value = 0;
            if (!_rdseed64_step(&value)) return false;
            out = static_cast<std::uint64_t>(value);
            return true;
        }
#else
        inline bool rdseed_word(std::uint64_t&) noexcept { return false; }
#endif

        inline void retry_pause() noexcept
        {
#if (defined(_MSC_VER) && defined(_M_X64)) || \
    ((defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__))
            _mm_pause();
#endif
        }

        inline void rdseed_bytes(std::uint8_t* dst, std::size_t bytes,
                                 const SeedOptions& options, SeedReport& report)
        {
            while (bytes != 0) {
                std::uint64_t value = 0;
                bool ok = false;
                for (unsigned attempt = 0; attempt < options.rdseed_attempts_per_word; ++attempt) {
                    if (rdseed_word(value)) { ok = true; break; }
                    ++report.rdseed_failures;
                    if (report.rdseed_failures >= options.rdseed_max_failures)
                        throw RdseedExhausted{};
                    retry_pause();
                }
                if (!ok) throw RdseedExhausted{};
                // Success flag, not value, decides validity. Zero is valid.
                const std::size_t n = bytes < 8 ? bytes : 8;
                for (std::size_t j = 0; j < n; ++j)
                    dst[j] = static_cast<std::uint8_t>(value >> (8 * j));
                report.rdseed_bytes += n;
                dst += n;
                bytes -= n;
            }
        }
    }

    // RDSEED-first initialization, with an all-or-nothing OS restart.
    // Reports instruction/API usage, NOT a proof of entropy or independence.
    // Only a bounded RDSEED acquisition failure triggers the restart; errors
    // from allocation, polynomial sampling, or the OS source propagate.
    inline PreparedSeed make_prepared_seed(SeedReport* report = nullptr,
                                          const SeedOptions& options = SeedOptions{})
    {
        SeedReport local;
        SeedReport& info = report ? *report : local;
        info = SeedReport{};
        if (options.rdseed_attempts_per_word == 0 || options.rdseed_max_failures == 0)
            throw std::invalid_argument("RDSEED retry budgets must be nonzero");
        info.rdseed64_available = rdseed64_available();
        if (info.rdseed64_available) {
            try {
                auto fill = [&](std::uint8_t* dst, std::size_t bytes) {
                    detail::rdseed_bytes(dst, bytes, options, info);
                };
                auto seed = ChaCha20GF1024LFSR1024::make_seed(fill);
                info.source = SeedSource::RDSEED;
                return seed;
            }
            catch (const detail::RdseedExhausted&) {
                info.fallback = FallbackReason::RetryBudgetExceeded;
            }
        }
        else {
            info.fallback = FallbackReason::Unavailable;
        }

        // Start make_seed AGAIN: new ChaCha key, GF coefficients, denominator
        // candidates and numerator. No RDSEED prefix survives into this seed.
        auto fill = [&](std::uint8_t* dst, std::size_t bytes) {
            if (!os_random_bytes(dst, bytes))
                throw std::runtime_error("OS random generator failed during seed fallback");
            info.os_bytes += bytes;
        };
        auto seed = ChaCha20GF1024LFSR1024::make_seed(fill);
        info.source = SeedSource::OS;
        return seed;
    }

    inline FullSeed make_full_seed(SeedReport* report = nullptr,
                                  const SeedOptions& options = SeedOptions{})
    {
        const PreparedSeed seed = make_prepared_seed(report, options);
        return seed.bytes();
    }

    inline ChaCha20GF1024LFSR1024 make_seeded_rng(std::uint64_t stream_id = 0,
                                                SeedReport* report = nullptr,
                                                const SeedOptions& options = SeedOptions{})
    {
        const PreparedSeed seed = make_prepared_seed(report, options);
        return ChaCha20GF1024LFSR1024(seed, stream_id);
    }
}
