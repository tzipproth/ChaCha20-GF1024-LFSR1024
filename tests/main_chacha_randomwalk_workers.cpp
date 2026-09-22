// Build (GCC / Linux):
// g++ -std=c++17 -O3 -march=native -flto -DNDEBUG -pthread main_chacha_randomwalk_workers.cpp -o chacha_randomwalk
// Run: ./chacha_randomwalk [threads=4] [report_seconds=10] [walk_power=16]
// Requires the current core and RDSEED-first Seed.h. No Parallel.h is needed.
// Source reviewed only; this revision has not been compiled or run.
// Statistical reference: independent fair bits (not an assertion that bounded
// independence proves the null model for the complete accumulated sample).
#include <cerrno>
#include <csignal>
#include <exception>
#include <limits>
#include <mutex>

#include "ChaCha20GF1024LFSR1024Seed.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

volatile std::sig_atomic_t interrupted = 0;
extern "C" void HandleInterrupt(int) { interrupted = 1; }

// Joins also during stack unwinding after partial thread creation or reporting
// failure. Declaration order in main keeps everything workers use alive.
struct WorkerJoiner {
    std::atomic<bool>& stop;
    std::vector<std::thread>& workers;
    ~WorkerJoiner() {
        stop.store(true, std::memory_order_release);
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
    }
};

static unsigned Popcount64(std::uint64_t v) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_popcountll(v));
#else
    // Portable SWAR; no unchecked POPCNT CPU requirement on MSVC builds.
    v -= (v >> 1) & 0x5555555555555555ULL;
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return static_cast<unsigned>((v * 0x0101010101010101ULL) >> 56);
#endif
}

// 1,048,576 uint64_t = 8 MiB per worker.
// For walk powers up to 22 this always contains an integer number of walks.
constexpr std::size_t BATCH_WORDS = 1u << 20;
constexpr unsigned DEFAULT_WALK_POWER = 16; // 2^16 = 65536 bits
constexpr unsigned MIN_WALK_POWER = 6;      // at least one uint64_t
constexpr unsigned MAX_WALK_POWER = 22;     // keeps per-worker statistics modest

// We only inspect the walk after every TWO bits. In scaled coordinates
// q = S/2, each 2-bit pair changes q by -1, 0, or +1.
//
// For one byte (= four 2-bit pairs) precompute:
//   * total scaled displacement after the byte
//   * for each possible starting q in [-4,+4], which of the four even
//     positions inside the byte return to zero.
//
// If |q| > 4 at byte entry, a zero return within the next four pair-steps
// is impossible, so the hot loop only needs one range test and one add.
struct ByteWalkLUT {
    std::array<std::int8_t, 256> delta{};
    std::array<std::array<std::int8_t, 256>, 4> prefixQ{};
    std::array<std::array<std::uint8_t, 256>, 9> hitMask{}; // start q=-4..+4

    ByteWalkLUT() {
        for (unsigned b = 0; b < 256; ++b) {
            int q = 0;
            int prefix[4]{};

            for (unsigned j = 0; j < 4; ++j) {
                const unsigned pair = (b >> (2 * j)) & 3u;
                const unsigned ones = (pair & 1u) + ((pair >> 1) & 1u);
                q += static_cast<int>(ones) - 1; // 00:-1, 01/10:0, 11:+1
                prefix[j] = q;
                prefixQ[j][b] = static_cast<std::int8_t>(q);
            }

            delta[b] = static_cast<std::int8_t>(q);

            for (int start = -4; start <= 4; ++start) {
                std::uint8_t mask = 0;
                for (unsigned j = 0; j < 4; ++j) {
                    if (start + prefix[j] == 0)
                        mask |= static_cast<std::uint8_t>(1u << j);
                }
                hitMask[static_cast<std::size_t>(start + 4)][b] = mask;
            }
        }
    }
};

static const ByteWalkLUT& GetByteWalkLUT()
{
    static const ByteWalkLUT lut;
    return lut;
}

struct WalkStatistics {
    // returnToZero[i] counts S_x == 0 at x = 2*(i+1).
    std::vector<std::uint64_t> returnToZero;

    // Path statistics are only needed at the power-of-two checkpoints x=2^p.
    // neverReturnedAtPower[p-1] counts walks that have not revisited zero at
    // any positive time <= 2^p.
    std::vector<std::uint64_t> neverReturnedAtPower;

    // Sum, over all walks, of the number of positive-time visits to zero by
    // checkpoint x=2^p. Dividing by completedWalks gives the observed mean
    // local time at zero (excluding the starting point at x=0).
    std::vector<std::uint64_t> zeroVisitSumAtPower;

    // maxAbsAtPower[p-1] = largest observed |S_x| at x=2^p.
    std::vector<std::uint32_t> maxAbsAtPower;

    // At checkpoint x: E[S_x]=0, E[S_x^2]=x,
    // E[S_x^4]=3*x*x-2*x, hence Var(S_x^2)=2*x*(x-1).
    std::vector<long double> positionSumAtPower;
    std::vector<long double> squareSumAtPower;
    std::uint64_t bitChanges = 0;
    std::uint64_t adjacentPairs = 0;

    std::uint64_t completedWalks = 0;
    std::uint64_t wordsProcessed = 0;

    WalkStatistics() = default;
    WalkStatistics(std::size_t returnPoints, unsigned walkPower)
        : returnToZero(returnPoints, 0),
          neverReturnedAtPower(walkPower, 0),
          zeroVisitSumAtPower(walkPower, 0),
          maxAbsAtPower(walkPower, 0),
          positionSumAtPower(walkPower, 0),
          squareSumAtPower(walkPower, 0) {}
};

// One writer (worker), one reader (main). The worker first writes all plain
// snapshot fields and then publishes ackEpoch with release semantics.
// Main requests the next epoch only after reading every snapshot, so workers
// cannot overwrite snapshots while main aggregates them.
struct WorkerSnapshot {
    WalkStatistics stats;
    std::atomic<std::uint64_t> ackEpoch{0};
};

static void AddSnapshot(WalkStatistics& total, const WorkerSnapshot& s)
{
    for (std::size_t i = 0; i < total.returnToZero.size(); ++i)
        total.returnToZero[i] += s.stats.returnToZero[i];

    for (std::size_t i = 0; i < total.maxAbsAtPower.size(); ++i) {
        total.positionSumAtPower[i] += s.stats.positionSumAtPower[i];
        total.squareSumAtPower[i] += s.stats.squareSumAtPower[i];
        total.neverReturnedAtPower[i] += s.stats.neverReturnedAtPower[i];
        total.zeroVisitSumAtPower[i] += s.stats.zeroVisitSumAtPower[i];
        total.maxAbsAtPower[i] =
            std::max(total.maxAbsAtPower[i], s.stats.maxAbsAtPower[i]);
    }

    total.bitChanges += s.stats.bitChanges;
    total.adjacentPairs += s.stats.adjacentPairs;
    total.completedWalks += s.stats.completedWalks;
    total.wordsProcessed += s.stats.wordsProcessed;
}

static std::vector<long double> BuildReturnProbabilities(std::size_t returnPoints)
{
    std::vector<long double> probabilities(returnPoints);

    // P(S_0=0)=1 and
    // P(S_x=0) = P(S_{x-2}=0) * (x-1)/x for even x.
    long double p = 1.0L;
    for (std::size_t i = 0; i < returnPoints; ++i) {
        const unsigned long long x = 2ULL * static_cast<unsigned long long>(i + 1);
        p *= static_cast<long double>(x - 1ULL) /
             static_cast<long double>(x);
        probabilities[i] = p;
    }

    return probabilities;
}

// Numerically evaluated binomial tail P(|S_n| >= absS), not a normal approximation.
// The observed absS always has the correct parity for n.
static long double TwoSidedPositionTail(std::uint64_t n, std::uint32_t absS)
{
    if (absS == 0)
        return 1.0L;
    if (absS > n || ((n - absS) & 1ULL) != 0)
        return 0.0L;

    const std::uint64_t k = (n + absS) / 2ULL;
    const long double logP =
        std::lgamma(static_cast<long double>(n) + 1.0L) -
        std::lgamma(static_cast<long double>(k) + 1.0L) -
        std::lgamma(static_cast<long double>(n - k) + 1.0L) -
        static_cast<long double>(n) * std::log(2.0L);

    long double term = std::exp(logP);
    long double oneSide = term;
    std::uint64_t sPos = absS;

    while (sPos + 2ULL <= n) {
        term *= static_cast<long double>(n - sPos) /
                static_cast<long double>(n + sPos + 2ULL);
        oneSide += term;
        sPos += 2ULL;
        if (term < oneSide * 1.0e-24L)
            break;
    }

    const long double twoSided = 2.0L * oneSide;
    return twoSided > 1.0L ? 1.0L : twoSided;
}

// Exact first two moments of the number L_x of positive-time visits to zero
// by an even time x.  If p_x=P(S_x=0) and a=(x+1)p_x, then
//   E[L_x]   = a - 1
//   Var[L_x] = x + 2 - a - a^2.
// The variance identity follows by expanding L_x as a sum of return
// indicators and using the Markov property for joint return probabilities.
static void ZeroVisitMoments(std::uint64_t x,
                             long double returnProbability,
                             long double& mean,
                             long double& variance)
{
    const long double a =
        (static_cast<long double>(x) + 1.0L) * returnProbability;
    mean = a - 1.0L;
    variance = static_cast<long double>(x) + 2.0L - a - a * a;
    if (variance < 0.0L && variance > -1.0e-18L)
        variance = 0.0L;
}

static void PrintWalkStatistics(const WalkStatistics& s,
                                const std::vector<long double>& probabilities,
                                std::uint64_t walkBits,
                                unsigned walkPower,
                                unsigned threads,
                                std::uint64_t reportNumber,
                                double totalSeconds,
                                double intervalSeconds,
                                std::uint64_t previousWords,
                                std::uint64_t previousWalks)
{
    const std::uint64_t deltaWords = s.wordsProcessed - previousWords;
    const std::uint64_t deltaWalks = s.completedWalks - previousWalks;

    const long double bits =
        static_cast<long double>(s.wordsProcessed) * 64.0L;

    const double intervalMWords = intervalSeconds > 0.0
        ? static_cast<double>(deltaWords) / intervalSeconds / 1.0e6
        : 0.0;
    const double totalMWords = totalSeconds > 0.0
        ? static_cast<double>(s.wordsProcessed) / totalSeconds / 1.0e6
        : 0.0;

    const double intervalGiBs = intervalSeconds > 0.0
        ? static_cast<double>(deltaWords) * 8.0 /
          intervalSeconds / (1024.0 * 1024.0 * 1024.0)
        : 0.0;
    const double totalGiBs = totalSeconds > 0.0
        ? static_cast<double>(s.wordsProcessed) * 8.0 /
          totalSeconds / (1024.0 * 1024.0 * 1024.0)
        : 0.0;

    const double intervalMWalks = intervalSeconds > 0.0
        ? static_cast<double>(deltaWalks) / intervalSeconds / 1.0e6
        : 0.0;
    const double totalMWalks = totalSeconds > 0.0
        ? static_cast<double>(s.completedWalks) / totalSeconds / 1.0e6
        : 0.0;

    // Largest single-coordinate |z| as a diagnostic only. The z-values over
    // different x are correlated because they come from the same walks.
    long double maxAbsZ = 0.0L;
    long double maxZ = 0.0L;
    std::uint64_t maxZX = 0;

    for (std::size_t i = 0; i < s.returnToZero.size(); ++i) {
        const long double p = probabilities[i];
        const long double expected =
            static_cast<long double>(s.completedWalks) * p;
        const long double variance =
            static_cast<long double>(s.completedWalks) * p * (1.0L - p);

        if (variance > 0.0L) {
            const long double z =
                (static_cast<long double>(s.returnToZero[i]) - expected) /
                std::sqrt(variance);
            const long double az = std::fabs(z);
            if (az > maxAbsZ) {
                maxAbsZ = az;
                maxZ = z;
                maxZX = 2ULL * static_cast<std::uint64_t>(i + 1);
            }
        }
    }

    std::cout
        << "\n======================================================================\n"
        << "Report " << reportNumber << "   threads=" << threads << '\n'
        << "Time: total=" << std::fixed << std::setprecision(3)
        << totalSeconds << " s"
        << "   interval=" << intervalSeconds << " s\n"
        << "Work: uint64=" << s.wordsProcessed
        << "   bits=" << std::fixed << std::setprecision(0) << bits
        << "   completed " << walkBits << "-bit walks=" << s.completedWalks << '\n'
        << "Data: " << std::fixed << std::setprecision(3)
        << (static_cast<long double>(s.wordsProcessed) * 8.0L /
            (1024.0L * 1024.0L * 1024.0L))
        << " GiB generated+analysed\n"
        << "Rate interval: " << std::setprecision(3)
        << intervalMWords << " M uint64/s   "
        << intervalGiBs << " GiB/s   "
        << intervalMWalks << " M walks/s\n"
        << "Rate total:    "
        << totalMWords << " M uint64/s   "
        << totalGiBs << " GiB/s   "
        << totalMWalks << " M walks/s\n"
        << "Diagnostic max |z| over x=2.." << walkBits << ": "
        << std::setprecision(4) << static_cast<double>(maxAbsZ)
        << " at x=" << maxZX
        << " (z=" << std::showpos << static_cast<double>(maxZ)
        << std::noshowpos << ")\n\n";

    const long double walks = static_cast<long double>(s.completedWalks);
    const long double pairs = static_cast<long double>(s.adjacentPairs);
    const long double changeZ = pairs > 0
        ? (2.0L * s.bitChanges - pairs) / std::sqrt(pairs) : 0;
    const long double finalDriftZ = walks > 0
        ? s.positionSumAtPower.back() / std::sqrt(walks * walkBits) : 0;
    const long double onesPercent = bits > 0
        ? 50.0L + 50.0L * s.positionSumAtPower.back() / bits : 0;

    std::cout << "Bit balance: " << std::fixed << std::setprecision(6)
        << onesPercent << "% ones (expected 50%); z=" << std::showpos
        << std::setprecision(3) << finalDriftZ << std::noshowpos << '\n'
        << "Adjacent bits: " << std::setprecision(6)
        << (pairs > 0 ? 100.0L * s.bitChanges / pairs : 0)
        << "% changes (expected 50%); z=" << std::showpos
        << std::setprecision(3) << changeZ << std::noshowpos << '\n'
        << "  Positive change-z: excess alternation; negative: excess repetition.\n"
        << "  Counts cross word/walk/batch boundaries, but not worker lanes.\n\n";

    std::cout << "Power-of-two checkpoints (ratios: observed / expected)\n"
        << std::right
        << std::setw(9) << "steps"
        << std::setw(11) << "at0/exp"
        << std::setw(10) << "z0"
        << std::setw(11) << "never/exp"
        << std::setw(10) << "zNever"
        << std::setw(11) << "visits/exp"
        << std::setw(10) << "zVisits"
        << std::setw(10) << "zDrift"
        << std::setw(10) << "zWidth"
        << std::setw(10) << "max|S|" << '\n'
        << std::string(102, '-') << '\n';

    for (unsigned power = 1; power <= walkPower; ++power) {
        const std::uint64_t x = std::uint64_t{1} << power;
        const std::size_t pi = power - 1;
        const long double p = probabilities[static_cast<std::size_t>(x / 2 - 1)];
        const long double expected = walks * p;
        const long double sd = std::sqrt(walks * p * (1 - p));
        long double meanVisits = 0, varVisits = 0;
        ZeroVisitMoments(x, p, meanVisits, varVisits);
        const long double visitSD = std::sqrt(walks * varVisits);
        const long double driftSD = std::sqrt(walks * x);
        const long double widthSD = std::sqrt(2.0L * walks * x * (x - 1));
        const auto at0 = s.returnToZero[static_cast<std::size_t>(x / 2 - 1)];
        const auto never = s.neverReturnedAtPower[pi];
        const auto visits = s.zeroVisitSumAtPower[pi];
        auto ratio = [](long double observed, long double target) {
            return target > 0 ? observed / target : 0;
        };
        auto z = [](long double difference, long double sigma) {
            return sigma > 0 ? difference / sigma : 0;
        };
        auto printRatio = [](long double v) {
            std::cout << std::setw(11) << std::fixed << std::setprecision(5) << v;
        };
        auto printZ = [](long double v) {
            // Scientific format also remains readable for very large defects.
            std::cout << std::setw(10) << std::showpos;
            if (std::fabs(v) >= 10000)
                std::cout << std::scientific << std::setprecision(2);
            else
                std::cout << std::fixed << std::setprecision(3);
            std::cout << v << std::noshowpos;
        };
        std::cout << std::setw(9) << x;
        printRatio(ratio(at0, expected));
        printZ(z(static_cast<long double>(at0) - expected, sd));
        printRatio(ratio(never, expected));
        printZ(z(static_cast<long double>(never) - expected, sd));
        printRatio(ratio(visits, walks * meanVisits));
        printZ(z(static_cast<long double>(visits) - walks * meanVisits, visitSD));
        printZ(z(s.positionSumAtPower[pi], driftSD));
        printZ(z(s.squareSumAtPower[pi] - walks * x, widthSD));
        std::cout << std::setw(10) << s.maxAbsAtPower[pi] << '\n';
    }

    const long double meanSquareRatio = walks > 0
        ? s.squareSumAtPower.back() / (walks * walkBits) : 0;
    const std::uint32_t maxAbs = s.maxAbsAtPower.back();
    const long double expectedAtLeastMax = walks * TwoSidedPositionTail(walkBits, maxAbs);
    std::cout << std::fixed << std::setprecision(6)
        << "\nFinal mean S^2 / steps: " << meanSquareRatio << " (expected 1).\n"
        << "Largest final |S|: " << maxAbs << " = " << std::setprecision(3)
        << static_cast<long double>(maxAbs) / std::sqrt(static_cast<long double>(walkBits))
        << " * sqrt(steps); expected walks reaching this endpoint magnitude or greater: "
        << std::scientific << expectedAtLeastMax << std::fixed << '\n'
        << "  This is an endpoint statistic, not the maximum along a path or a p-value.\n"
        << "Legend: at0 = endpoint is zero; never = no return since start;\n"
        << "        visits = mean number of returns, excluding the starting point.\n"
        << "        zDrift: positive means excess 1s; negative means excess 0s.\n"
        << "        zWidth: positive means wider endpoints; negative means narrower.\n"
        << "        Width uses S^2 about zero, so bias can also increase it.\n"
        << "Ratios should fluctuate around 1, z-values around 0 under independent fair bits.\n"
        << "A z-value is the signed deviation in theoretical standard-deviation units.\n"
        << "Rows, metrics and cumulative reports are correlated. Individual large z-values\n"
        << "and max|z| are diagnostics, not an overall pass/fail or global p-value.\n"
        << "Small expected counts need discrete analysis; z is not an exact normal p-value.\n"
        << std::flush;
}

static void Worker(unsigned workerId,
                   const ChaCha20GF1024LFSR1024::PreparedSeed& seed,
                   std::size_t wordsPerWalk,
                   std::size_t returnPoints,
                   unsigned walkPower,
                   std::uint64_t maxWorkerWords,
                   std::atomic<bool>& stop,
                   std::atomic<std::uint64_t>& requestedEpoch,
                   WorkerSnapshot& published)
{
    // Every worker uses the same prepared seed and the same public stream ID,
    // but starts in a disjoint 2^64-word lane of the 128-bit position space.
    // This keeps all generated words in one logical RNG stream instead of
    // assuming independence between different stream IDs. The counter limit
    // below stops the run well before a worker can leave its lane.
    ChaCha20GF1024LFSR1024 rng(seed);
    rng.seek(ChaCha20GF1024LFSR1024::Position{0,
        static_cast<std::uint64_t>(workerId)});

    std::vector<std::uint64_t> buffer(BATCH_WORDS);
    WalkStatistics stats(returnPoints, walkPower);
    std::uint64_t seenEpoch = 0;
    bool havePreviousBit = false;
    unsigned previousBit = 0;

    const ByteWalkLUT& lut = GetByteWalkLUT();

    while (!stop.load(std::memory_order_acquire)) {
        if (stats.wordsProcessed > maxWorkerWords - BATCH_WORDS)
            throw std::overflow_error("statistics counter limit reached; start a fresh run");
        // Single-threaded RNG inside each worker; all parallelism is at the
        // worker-pipeline level (generation + analysis together).
        rng.generate(buffer.data(), buffer.size());

        for (std::size_t walkBase = 0;
             walkBase < buffer.size();
             walkBase += wordsPerWalk) {

            int q = 0;                    // scaled position q = S/2
            std::size_t returnIndex = 0;   // x=2,4,...
            std::size_t nextCheckpointWord = 1; // x=64,128,256,...
            unsigned checkpointPower = 6;
            std::uint32_t zeroVisits = 0;  // positive-time visits to S=0 so far

            auto recordCheckpoint = [&](unsigned power,
                                        int checkpointQ,
                                        std::uint32_t visits) {
                const std::size_t pi = static_cast<std::size_t>(power - 1);
                if (visits == 0)
                    ++stats.neverReturnedAtPower[pi];
                stats.zeroVisitSumAtPower[pi] += visits;

                const long double position = 2.0L * checkpointQ;
                stats.positionSumAtPower[pi] += position;
                stats.squareSumAtPower[pi] += position * position;
                const std::uint32_t v = static_cast<std::uint32_t>(
                    2 * std::abs(checkpointQ));
                stats.maxAbsAtPower[pi] =
                    std::max(stats.maxAbsAtPower[pi], v);
            };

            for (std::size_t wi = 0; wi < wordsPerWalk; ++wi) {
                std::uint64_t w = buffer[walkBase + wi];
                // Under independent fair bits, adjacent XORs are independent
                // fair bits (the map from first bit + all XORs is bijective).
                stats.bitChanges += Popcount64((w ^ (w >> 1)) & 0x7fffffffffffffffULL);
                stats.adjacentPairs += 63;
                if (havePreviousBit) {
                    stats.bitChanges += previousBit ^ static_cast<unsigned>(w & 1u);
                    ++stats.adjacentPairs;
                }
                previousBit = static_cast<unsigned>(w >> 63);
                havePreviousBit = true;

                // Consume LSB -> MSB. One byte contains four 2-bit pair-steps.
                for (unsigned byteIndex = 0; byteIndex < 8; ++byteIndex) {
                    const unsigned b = static_cast<unsigned>(w & 0xffu);
                    w >>= 8;

                    std::uint8_t mask = 0;
                    if (static_cast<unsigned>(q + 4) <= 8u) {
                        mask = lut.hitMask[static_cast<std::size_t>(q + 4)][b];

                        if (mask & 0x1u) ++stats.returnToZero[returnIndex + 0];
                        if (mask & 0x2u) ++stats.returnToZero[returnIndex + 1];
                        if (mask & 0x4u) ++stats.returnToZero[returnIndex + 2];
                        if (mask & 0x8u) ++stats.returnToZero[returnIndex + 3];
                    }

                    // x=2 and x=4 lie inside the first byte.  Record the path
                    // statistics at the exact sub-byte checkpoint, before the
                    // rest of this byte can add later zero visits.
                    if (wi == 0 && byteIndex == 0) {
                        const std::uint32_t visits2 =
                            static_cast<std::uint32_t>(mask & 0x1u ? 1u : 0u);
                        const std::uint32_t visits4 = visits2 +
                            static_cast<std::uint32_t>(mask & 0x2u ? 1u : 0u);
                        recordCheckpoint(1,
                                         static_cast<int>(lut.prefixQ[0][b]),
                                         visits2);
                        recordCheckpoint(2,
                                         static_cast<int>(lut.prefixQ[1][b]),
                                         visits4);
                    }

                    zeroVisits +=
                        static_cast<std::uint32_t>((mask & 0x1u) != 0) +
                        static_cast<std::uint32_t>((mask & 0x2u) != 0) +
                        static_cast<std::uint32_t>((mask & 0x4u) != 0) +
                        static_cast<std::uint32_t>((mask & 0x8u) != 0);

                    q += lut.delta[b];
                    returnIndex += 4;

                    // Remaining checkpoints inside the first word.
                    if (wi == 0) {
                        if (byteIndex == 0)
                            recordCheckpoint(3, q, zeroVisits); // x=8
                        else if (byteIndex == 1)
                            recordCheckpoint(4, q, zeroVisits); // x=16
                        else if (byteIndex == 3)
                            recordCheckpoint(5, q, zeroVisits); // x=32
                    }
                }

                // From x=64 onward each power-of-two checkpoint is a word boundary.
                if (checkpointPower <= walkPower && wi + 1 == nextCheckpointWord) {
                    recordCheckpoint(checkpointPower, q, zeroVisits);
                    nextCheckpointWord <<= 1;
                    ++checkpointPower;
                }
            }

            ++stats.completedWalks;
        }

        stats.wordsProcessed += static_cast<std::uint64_t>(buffer.size());

        // One atomic read per 8 MiB processed by this worker.
        const std::uint64_t requested =
            requestedEpoch.load(std::memory_order_acquire);

        if (requested != seenEpoch) {
            // Pre-sized snapshot: no allocation in the snapshot path.
            std::copy(stats.returnToZero.begin(),
                      stats.returnToZero.end(),
                      published.stats.returnToZero.begin());
            std::copy(stats.neverReturnedAtPower.begin(),
                      stats.neverReturnedAtPower.end(),
                      published.stats.neverReturnedAtPower.begin());
            std::copy(stats.zeroVisitSumAtPower.begin(),
                      stats.zeroVisitSumAtPower.end(),
                      published.stats.zeroVisitSumAtPower.begin());
            std::copy(stats.maxAbsAtPower.begin(),
                      stats.maxAbsAtPower.end(),
                      published.stats.maxAbsAtPower.begin());
            std::copy(stats.positionSumAtPower.begin(), stats.positionSumAtPower.end(),
                      published.stats.positionSumAtPower.begin());
            std::copy(stats.squareSumAtPower.begin(), stats.squareSumAtPower.end(),
                      published.stats.squareSumAtPower.begin());
            published.stats.bitChanges = stats.bitChanges;
            published.stats.adjacentPairs = stats.adjacentPairs;
            published.stats.completedWalks = stats.completedWalks;
            published.stats.wordsProcessed = stats.wordsProcessed;

            published.ackEpoch.store(requested, std::memory_order_release);
            seenEpoch = requested;
        }
    }
}

static unsigned ParseUnsigned(const char* s, const char* name)
{
    // Reject signs, whitespace, overflow and narrowing before conversion.
    if (*s < '0' || *s > '9')
        throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno == ERANGE || *end != '\0' || v > std::numeric_limits<unsigned>::max())
        throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    return static_cast<unsigned>(v);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        // Syntax:
        //   program [threads] [report_seconds] [walk_power]
        //
        // Examples:
        //   ./chacha_randomwalk                 -> 4 threads, 10 s, 2^16 bits
        //   ./chacha_randomwalk 4 10 13         -> 8192-bit walks
        //   ./chacha_randomwalk 4 10 16         -> 65536-bit walks
        //   ./chacha_randomwalk 8 5 18          -> 262144-bit walks
        unsigned threads = 4;
        unsigned reportSeconds = 10;
        unsigned walkPower = DEFAULT_WALK_POWER;

        if (argc > 1)
            threads = ParseUnsigned(argv[1], "thread count");
        if (argc > 2)
            reportSeconds = ParseUnsigned(argv[2], "report seconds");
        if (argc > 3)
            walkPower = ParseUnsigned(argv[3], "walk power");
        if (argc > 4)
            throw std::runtime_error(
                "too many arguments; expected [threads] [report_seconds] [walk_power]");

        if (threads == 0)
            threads = std::max(1u, std::thread::hardware_concurrency());
        if (reportSeconds == 0)
            throw std::runtime_error("report seconds must be >= 1");
        if (walkPower < MIN_WALK_POWER || walkPower > MAX_WALK_POWER) {
            throw std::runtime_error(
                "walk power must be between " + std::to_string(MIN_WALK_POWER) +
                " and " + std::to_string(MAX_WALK_POWER));
        }

        const std::uint64_t walkBits = std::uint64_t{1} << walkPower;
        const std::size_t wordsPerWalk =
            static_cast<std::size_t>(walkBits / 64ULL);
        const std::size_t returnPoints =
            static_cast<std::size_t>(walkBits / 2ULL);

        if (wordsPerWalk == 0 || BATCH_WORDS % wordsPerWalk != 0) {
            throw std::runtime_error(
                "selected walk length must divide the 8 MiB worker batch exactly");
        }

        // Build all immutable structures before starting worker timing. Seed
        // preparation includes sampling and validating the degree-1024 LFSR
        // polynomial and is intentionally performed only once.
        (void)GetByteWalkLUT();
        const std::vector<long double> probabilities =
            BuildReturnProbabilities(returnPoints);
        namespace seed_api = chacha20gf1024lfsr1024_seed;
        seed_api::SeedReport seedReport;
        std::cout << "Preparing structured seed (RDSEED first, OS fallback)...\n" << std::flush;
        const auto seed = seed_api::make_prepared_seed(&seedReport);
        std::cout << "RDSEED64 available: " << (seedReport.rdseed64_available ? "yes" : "no")
            << "\nSeed source: " << seed_api::seed_source_name(seedReport.source)
            << "\nFallback: " << seed_api::fallback_reason_name(seedReport.fallback)
            << "\nRDSEED bytes: " << seedReport.rdseed_bytes
            << (seedReport.source == seed_api::SeedSource::RDSEED ? "" : " (discarded)")
            << "; failed attempts: " << seedReport.rdseed_failures
            << "; OS bytes: " << seedReport.os_bytes << "\n";
        const std::uint64_t maxWorkerWords =
            std::numeric_limits<std::uint64_t>::max() / 64 / threads;
        if (maxWorkerWords < BATCH_WORDS)
            throw std::runtime_error("too many workers for the counter budget");

        const long double counterKiB =
            static_cast<long double>(returnPoints) * sizeof(std::uint64_t) / 1024.0L;

        std::cout
            << "ChaCha20GF1024LFSR1024 fully parallel 1D random-walk return test\n"
            << "mode:              infinite (Ctrl+C to stop)\n"
            << "workers:           " << threads << '\n'
            << "report interval:   ~" << reportSeconds << " s\n"
            << "walk power:        " << walkPower << '\n'
            << "walk length:       2^" << walkPower << " = " << walkBits << " bits\n"
            << "observations:      S_x==0 at every even x=2.." << walkBits << '\n'
            << "console table:     x=2,4,8,...," << walkBits << '\n'
            << "path statistics:   returns, survival, visits, drift, width, extremes + bit changes\n"
            << "return counters:   " << returnPoints << " per worker ("
            << std::fixed << std::setprecision(1) << static_cast<double>(counterKiB)
            << " KiB)\n"
            << "rounds:            20 (ChaCha20)\n"
            << "seed:              structured "
            << ChaCha20GF1024LFSR1024::FULL_SEED_BYTES
            << " bytes; source reported above\n"
            << "worker positions:  one logical stream, high64 = 0.."
            << (threads - 1) << '\n'
            << "bit order:         LSB -> MSB within each uint64_t; 0=-1, 1=+1\n"
            << "parallelism:       each worker generates + analyses locally\n"
            << "snapshot polling:  once per 8 MiB/worker/batch, plus a stop check\n"
            << "H0:                P(S_x=0)=C(x,x/2)/2^x for even x\n"
            << "H0 scope:          independent fair bits across all sampled positions\n"
            << "                   This is a statistical reference, not implied for the\n"
            << "                   whole sample by the RNG's finite-order guarantees.\n"
            << "Rate scope:        generation + analysis + startup/reporting overhead\n"
            << "Stop:              Ctrl+C; joins workers after their current batch\n"
            << "syntax:            [threads] [report_seconds] [walk_power]\n"
            << std::flush;

        std::atomic<std::uint64_t> requestedEpoch{0};
        std::unique_ptr<WorkerSnapshot[]> snapshots(new WorkerSnapshot[threads]);
        for (unsigned t = 0; t < threads; ++t) {
            snapshots[t].stats.returnToZero.assign(returnPoints, 0);
            snapshots[t].stats.neverReturnedAtPower.assign(walkPower, 0);
            snapshots[t].stats.zeroVisitSumAtPower.assign(walkPower, 0);
            snapshots[t].stats.maxAbsAtPower.assign(walkPower, 0);
            snapshots[t].stats.positionSumAtPower.assign(walkPower, 0);
            snapshots[t].stats.squareSumAtPower.assign(walkPower, 0);
        }

        std::atomic<bool> stop{false};
        std::mutex errorMutex;
        std::exception_ptr workerError;
        std::vector<std::thread> workers;
        workers.reserve(threads);
        WorkerJoiner joiner{stop, workers};
        std::signal(SIGINT, HandleInterrupt);
        std::signal(SIGTERM, HandleInterrupt);

        using clock = std::chrono::steady_clock;
        const auto start = clock::now(); // includes worker setup; seed is excluded
        for (unsigned t = 0; t < threads; ++t) {
            workers.emplace_back([&, t] {
                try {
                    Worker(t, seed, wordsPerWalk, returnPoints, walkPower,
                           maxWorkerWords, stop, requestedEpoch, snapshots[t]);
                }
                catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(errorMutex);
                        if (!workerError) workerError = std::current_exception();
                    }
                    stop.store(true, std::memory_order_release);
                }
            });
        }

        auto previousTime = start;
        auto nextReport = start + std::chrono::seconds(reportSeconds);

        std::uint64_t epoch = 0;
        std::uint64_t previousWords = 0;
        std::uint64_t previousWalks = 0;
        std::uint64_t reportNumber = 0;

        while (!interrupted && !stop.load(std::memory_order_acquire)) {
            while (clock::now() < nextReport && !interrupted &&
                   !stop.load(std::memory_order_acquire))
                std::this_thread::sleep_until(std::min(nextReport,
                    clock::now() + std::chrono::milliseconds(100)));
            if (interrupted || stop.load(std::memory_order_acquire)) break;

            ++epoch;
            requestedEpoch.store(epoch, std::memory_order_release);

            // Wait until each worker publishes one coherent cumulative snapshot.
            for (unsigned t = 0; t < threads; ++t) {
                while (snapshots[t].ackEpoch.load(std::memory_order_acquire) != epoch &&
                       !interrupted && !stop.load(std::memory_order_acquire))
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (interrupted || stop.load(std::memory_order_acquire)) break;
            }

            if (interrupted || stop.load(std::memory_order_acquire)) break;
            const auto now = clock::now();

            WalkStatistics total(returnPoints, walkPower);
            for (unsigned t = 0; t < threads; ++t)
                AddSnapshot(total, snapshots[t]);

            const double totalSeconds =
                std::chrono::duration<double>(now - start).count();
            const double intervalSeconds =
                std::chrono::duration<double>(now - previousTime).count();

            ++reportNumber;
            PrintWalkStatistics(total,
                                probabilities,
                                walkBits,
                                walkPower,
                                threads,
                                reportNumber,
                                totalSeconds,
                                intervalSeconds,
                                previousWords,
                                previousWalks);

            previousWords = total.wordsProcessed;
            previousWalks = total.completedWalks;
            previousTime = now;

            nextReport += std::chrono::seconds(reportSeconds);
            const auto afterReport = clock::now();
            if (nextReport <= afterReport)
                nextReport = afterReport + std::chrono::seconds(reportSeconds);
        }
        stop.store(true, std::memory_order_release);
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
        if (workerError) std::rethrow_exception(workerError);
        std::cout << "\nStopped. Last printed report is the last complete snapshot;\n"
                     "work since that snapshot is not included.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
