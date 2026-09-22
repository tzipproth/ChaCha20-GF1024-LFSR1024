# ChaCha20GF1024LFSR1024
## Exact bounded independence, small-bias guarantees and computational pseudorandomness

> ChaCha20 is already an excellent computational pseudorandom generator. What exact mathematical distribution properties can be added?

ChaCha20GF1024LFSR1024 combines three independently seeded components:

- a position-indexed ChaCha20/HChaCha20 stream;
- a polynomial with 1024 coefficients over `GF(2^128)`, evaluated using an additive FFT;
- a 1024-bit LFSR with a randomly selected irreducible feedback polynomial, implementing an AGHP small-bias construction.

The output is their wordwise XOR and has **exact 1024-wise joint independence of arbitrary 64-bit output words** under the seed assumptions below. The LFSR adds bounds on arbitrary fixed bit parities and on the joint distribution of selected bit positions beyond that exact guarantee.

Under the ideal, independent seed distributions specified below, for each fixed public stream identifier:

| Selection / test | Guarantee | Source |
|---|---|---|
| Any at most 1024 distinct 64-bit word positions | Exactly jointly uniform; all bits in those words are jointly independent | GF component |
| Any at most 1024 distinct bit positions anywhere | Exactly jointly uniform | GF component |
| Both halves of each of 1024 distinct GF evaluations: 2048 words | Exactly jointly uniform for this structured selection | GF component |
| Any 1600 distinct bit positions anywhere | Joint distribution has total variation distance less than `2^-90` from 1600 independent fair bits | LFSR small-bias guarantee |
| XOR of any fixed nonempty selection of bits in the entire `2^134`-bit domain | Bias less than `2^-889`; probability of either parity outcome differs from 1/2 by less than `2^-890` | LFSR small-bias guarantee |

These are distribution theorems over random initialization, not statistical test results or promises about the appearance of every fixed-seed stream. They do not assert exact independence of arbitrary 1600 bits or joint uniformity of the whole stream. The ChaCha component supplies a separate, conditional computational security claim.

**GF1024** refers to the coefficient count and guaranteed word-independence order, not the field size. **LFSR1024** refers to the degree/state width of the separate binary recurrence. The polynomial-evaluation field remains `GF(2^128)`.

## 1. Overview

| Property | Value |
|---|---:|
| Field | GF(2^128) |
| Polynomial degree, at most | 1023 |
| Independent field coefficients | 1024 |
| Guaranteed arbitrary-word independence | 1024-wise |
| GF seed | 16384 bytes |
| ChaCha seed | 32 bytes |
| LFSR polynomial + numerator encoding | 256 bytes |
| Full structured seed encoding | 16672 bytes |
| LFSR state width | 1024 bits |
| Conservative global parity-bias bound | less than 2^-889 |
| Field evaluations per FFT | 1024 |
| 64-bit output words per FFT | 2048 |
| FFT stages | 10 |
| Supported word positions per stream | 2^128 |

The supplied `main.cpp` includes the generator, parallel wrapper, seed helper and toy checks.

## 2. Construction

For output-word position `i`, define:

```text
j = floor(i / 2)
P(x) = a0 + a1*x + ... + a1023*x^1023

G(2j)   = low64(P(j))
G(2j+1) = high64(P(j))
R(i)    = C(i) XOR G(i) XOR B_word(i)
```

Each coefficient is a 128-bit field element. The GF seed is decoded as 1024 monomial-basis coefficients in increasing degree order. Each coefficient uses a little-endian low 64-bit limb followed by a little-endian high limb.

### Field arithmetic

The field is defined by the irreducible polynomial:

```text
x^128 + x^7 + x^2 + x + 1
```

This is the polynomial used by GHASH/GCM. The implementation uses a low-bit polynomial representation, where reduction of an overflowed term uses `0x87`. Sharing the field polynomial does not imply identical byte/bit conventions to every GHASH implementation.

Addition is XOR. Multiplication is carry-less polynomial multiplication followed by reduction. The x86 fast path uses PCLMULQDQ with Karatsuba multiplication; the reference path uses portable shift/XOR arithmetic.

### ChaCha component

The standalone class is `ChaCha20GF1024LFSR1024Counter128`. It:

1. Derives a subkey with HChaCha20 from the 256-bit key and the input `stream_id || 0`, where the two components are 64 bits each.
2. Uses the 20-round ChaCha core with feed-forward.
3. Stores a 128-bit block index in state words 12 through 15.

The word position divided by eight selects the block; its low three bits select the 64-bit word within that block. HChaCha20 uses the standard output-word selection `0,1,2,3,12,13,14,15`, without feed-forward.

This is a **project-specific counter layout**, not the RFC 8439 IETF layout and not a standardized XChaCha mode. The computational claim assumes this composition behaves pseudorandomly within the contemplated query bounds. The large address space is not itself a security bound.

### LFSR small-bias component

Work over binary polynomials, with addition and multiplication modulo two. Select:

- `Q(z)` uniformly among all monic irreducible degree-1024 binary polynomials;
- `A(z)` independently and uniformly among all polynomials of degree below 1024, **including zero**.

The mask bits are the coefficients of the formal power series:

```text
B(z) = A(z) / Q(z) = b0 + b1*z + b2*z^2 + ...
B_word(i) = sum over r = 0..63 of b_(64*i+r) * 2^r
```

Thus bits are packed LSB-first in each output word. Since `Q(0)=1`, division as a formal power series is well-defined. Given `Q`, the first 1024 bits correspond bijectively to `A`.

This is a rational-series representation of the random-feedback LFSR construction in [Alon–Goldreich–Håstad–Peralta (AGHP), construction 1](https://web.math.princeton.edu/~nalon/PDFS/aghp4.pdf). Its characteristic polynomial is the reciprocal of `Q`; reciprocal polynomials preserve irreducibility and uniform sampling. **Random selection of the polynomial matters:** a single hard-coded feedback polynomial with only a random initial state does not provide the global bound claimed here.

## 3. Exact independence theorem

Let `k = 1024` and let the coefficients of `P` be independent uniform elements of `GF(2^128)`.

### Complete field values

For any `r <= k` distinct field points, the vector

```text
(P(x1), ..., P(xr))
```

is exactly uniform over `GF(2^128)^r`. The evaluation map has rank `r`: the relevant Vandermonde matrix has distinct evaluation points and therefore full row rank.

At `r = k`, interpolation is a bijection between coefficient vectors and evaluation vectors. At smaller `r`, every evaluation vector has exactly the same number of coefficient preimages.

### Split-half 64-bit words

Any `t <= 1024` distinct word positions refer to at most `t` distinct field points. Each selected field point contributes either its low half, high half, or both.

The complete field values are jointly uniform. Selecting coordinates of a uniform bit vector preserves uniformity of the selected coordinates. Consequently, the selected `t` words are jointly uniform over all `2^(64*t)` possibilities.

Condition on any fixed ChaCha stream and any fixed LFSR seed. Their XOR is a fixed mask, so XOR with it preserves this exact distribution. Independently randomizing either mask preserves the result as well.

### Exact counting

The GF seed contains `1024 * 128 = 131072` bits. For every fixed ChaCha stream, fixed LFSR seed and every selection of `t <= 1024` words:

```text
GF-seed preimages per output tuple = 2^(131072 - 64*t)
```

For 1024 words, every 65536-bit tuple therefore occurs exactly `2^65536` times over the GF seed space.

There is a stronger structured statement: choose 1024 distinct field points and expose **both halves** of each. Those 2048 words are jointly uniform, with exactly one GF-seed preimage per tuple for a fixed ChaCha stream and fixed LFSR seed. Positions `0..2047` are one example.

This does **not** imply arbitrary 2048-wise word independence: other selections may spread over more than 1024 field points.

### Meaning of “identical to ideal random bits”

For an observer restricted to the selected words, without seed information or additional outputs, their distribution is identical to independent ideal random bits. This is an exact equality, not a computational approximation.

A fixed-seed instance is nevertheless deterministic. The theorem describes a family of streams under random initialization; it does not create fresh physical entropy at each call.

### Bit granularity

Any collection of bits contained within at most 1024 words inherits exact joint uniformity. In particular, any 1024 distinct bit positions are covered.

This is not a guarantee for arbitrary 65536 bit positions spread across the entire stream. For example, one bit from each of more than 1024 words can lie outside the stated theorem.

## 4. What the LFSR adds: bit parities and almost independence

### Global parity guarantee

Let `n = 64 * 2^128 = 2^134`, the number of addressed bits. For any **fixed nonempty** subset `S` of these bit positions, define:

```text
parity_S = XOR of the output bits at positions in S
bias_S   = |Pr[parity_S = 0] - Pr[parity_S = 1]|
```

For this implementation's parameters:

```text
bias_S <= (n - 1) / (2^1024 - 2^512) < 2^-889.
```

The probability of either parity outcome therefore differs from `1/2` by less than `2^-890`. A parity can involve thousands, millions, or more bits, scattered anywhere within the supported domain. This does **not** mean those bits are all jointly independent: this particular guarantee concerns their XOR.

Here is the parameter calculation for the implemented recurrence. There are exactly `(2^1024 - 2^512) / 1024` monic irreducible degree-1024 binary polynomials. Represent the chosen parity by the nonzero polynomial `H(x) = sum_(t in S) x^t`, of degree at most `n-1`. For fixed feedback polynomial, the parity is a linear functional of the uniform initial state. It is either balanced or identically zero. The latter case occurs only if that feedback polynomial divides `H`. There can be at most `floor((n-1)/1024)` such distinct factors. Dividing this count by the number of possible feedback polynomials gives the stated bound.

For independent masks, parity expectations multiply:

```text
E[(-1)^(parity(X XOR Y))]
    = E[(-1)^(parity(X))] * E[(-1)^(parity(Y))].
```

Consequently, XOR with independent GF and ChaCha outputs cannot increase the LFSR's parity-bias bound. No unproven property of ChaCha is needed for this statement; its stream may be fixed.

The subset is fixed independently of the random seed and observed outputs. This theorem does not give the same bound to an observer who learns the feedback polynomial and then chooses a parity tailored to its recurrence.

### From parity bounds to joint distributions

For a fixed selection of `t` distinct bit positions, small bias bounds every nonconstant binary Fourier coefficient of its distribution. Parseval's identity and Cauchy–Schwarz give:

```text
TV(distribution of selected t bits, uniform t bits)
    <= min(1, sqrt(2^t - 1) * epsilon / 2),
where epsilon < 2^-889.
```

Here `TV` is total variation distance, **half** the sum of absolute probability differences. In particular:

```text
t = 1600  =>  TV < 2^-90.
```

This controls the entire joint distribution of those 1600 bits, not just pairs or XORs. For any event depending only on that fixed selection, its probability differs from the ideal probability by less than `2^-90`, regardless of computational power. It does not cover conditioning on a disclosed seed or arbitrary additional outputs.

The example `1600` is a useful operating point of the bound, not a new exact independence order or a hard cutoff. Pairwise independence follows exactly from the GF theorem. The LFSR supplies a quantitative guarantee for scattered selections outside that theorem, plus parity control even for much larger selections. It is incorrect to summarize the result as “epsilon-close for all larger selections” without specifying their size and the distance used.

### Concrete improvement over the GF mask alone

For the existing GF position mapping, the polynomial degree bound implies:

```text
XOR over j = 0..2047 of P(j) = 0
G(0) XOR G(2) XOR ... XOR G(4094) = 0.
```

To see this, the points `0..2047` form an 11-dimensional binary subspace. Every monomial of degree below 1024 has binary-coordinate degree at most 10, so its sum over that subspace vanishes.

With only the GF mask and a fixed ChaCha stream, the corresponding output XOR is a fixed constant. **With the independently randomized LFSR, each fixed bit parity of that XOR instead obeys the small-bias bound.** This is a specific algebraic defect for which an unconditional improvement can be stated.

The relation gives an upper bound of 2047 on arbitrary-word independence when both other masks are fixed. It does not give that upper bound when the LFSR parameters are randomized. The exact maximum independence order of the complete hybrid is not established here.

## 5. Computational security and multiple streams

If the ChaCha/HChaCha component is computationally indistinguishable from uniform under the stated usage assumptions, XOR with independently generated GF and LFSR components preserves that computational property. Conversely, fixing the ChaCha stream leaves both the GF and small-bias guarantees intact under their respective seed assumptions.

A failure of the ChaCha assumption would not invalidate bounded GF independence, but the hybrid would not thereby remain a secure cryptographic generator: GF plus LFSR alone has the stated bounded distribution guarantees, but no cryptographic security theorem is claimed for that combination. Algebraic structure outside the covered test classes remains relevant.

The exact theorem is a **single-stream theorem**. Changing `stream_id` changes the ChaCha subkey but does not change the GF or LFSR masks when the same full seed is reused. At equal positions in two such streams:

```text
R_s(i) XOR R_t(i) = C_s(i) XOR C_t(i)
```

Both shared masks cancel. HChaCha-based stream separation is a computational property, not proof of joint information-theoretic independence across stream identifiers.

Parallel range generation is different: it evaluates disjoint positions of the same stream and retains that stream's exact theorem.

## 6. Seed requirements and limits

### Structured seed format

| Byte range (end exclusive) | Contents | Bytes |
|---|---|---:|
| `[0, 32)` | ChaCha key | 32 |
| `[32, 16416)` | 1024 GF(2^128) monomial coefficients | 16384 |
| `[16416, 16544)` | Coefficients of `Q`, degrees 0 through 1023; leading coefficient implicit | 128 |
| `[16544, 16672)` | Coefficients of `A`, degrees 0 through 1023 | 128 |
| Total | Serialized structured seed | 16672 |

Binary polynomial coefficients use increasing bit order within little-endian limbs. The `Q` section is constrained: **an arbitrary uniformly filled 16672-byte buffer is not a valid structured seed**. The serialized length is not a claim of that many independent entropy bits.

For both information-theoretic guarantees together, the GF coefficients must be jointly uniform, `Q` must be uniform over the stated irreducible polynomials, and `A` must be uniform including zero. These parts must be independent of each other and of any randomized ChaCha key. A fixed ChaCha key is permitted for the distribution theorems.

`make_seed(fill)` samples candidate polynomials and rejects reducible ones, using an exact irreducibility test. With fresh independent uniform input bytes, successful rejection sampling produces the required distribution. The default limit is 1,000,000 candidates; exhaustion throws instead of substituting a fixed polynomial. `prepare_seed` validates a serialized seed and rebuilds its tables; validation cannot establish how the seed was sampled.

A zero GF coefficient vector and a zero numerator `A` are mathematically valid members of the ideal distributions. Zero `A` produces a zero LFSR mask. The denominator `Q` must satisfy its structural requirements. Rejecting other seeds merely because their output looks unusual would change the sampling distribution.

### Hardware-first initialization with OS fallback

`ChaCha20GF1024LFSR1024Seed.h` offers `make_prepared_seed()`, `make_full_seed()` and `make_seeded_rng()`. These helpers first try direct **64-bit RDSEED**, then restart the complete structured seed generation using the OS source if RDSEED is unavailable or its retry budget is exhausted.

The helper checks CPUID before executing RDSEED and checks the instruction's success flag on every call. A successfully returned zero is accepted. It uses the 64-bit instruction only; unsupported CPUs, other architectures and 32-bit builds take the OS path. The implementation supports MSVC and GCC/Clang x86-64 builds without requiring a global `-mrdseed` option.

`SeedOptions` defaults to at most **1024 attempts per 64-bit value** and **65536 failed attempts in total per seed-generation attempt**, using `PAUSE` between retries. The limits count attempts, not elapsed time; they do not bound polynomial-validation time or OS blocking. Both settings must be nonzero and can be changed by the caller.

On a retry failure, all partially generated RDSEED seed material is abandoned. The core sampler is called again with OS bytes, starting from a new ChaCha key and GF coefficients and resampling the LFSR denominator and numerator. There is no silent mixture of RDSEED and OS bytes in the completed seed. Allocation errors, the core's polynomial-candidate limit, and OS failures propagate as exceptions; they are not treated as RDSEED availability failures.

The OS fallback uses:

- Windows: `BCryptGenRandom`;
- Linux: `getrandom`, handling partial reads and interrupted calls.

The functions `make_os_prepared_seed()`, `make_os_full_seed()` and `make_os_seeded_rng()` are **explicitly OS-only** and do not use RDSEED. Prefer the prepared form of either policy when constructing several generator instances.

### Seed-source reporting

An optional `SeedReport*` records the policy outcome; the helper itself does not print. For example:

```cpp
namespace seed_api = chacha20gf1024lfsr1024_seed;
seed_api::SeedReport report;
auto seed = seed_api::make_prepared_seed(&report);
ChaCha20GF1024LFSR1024 rng(seed);
// seed_api::seed_source_name(report.source)
// seed_api::fallback_reason_name(report.fallback)
```

| Report field | Meaning |
|---|---|
| `rdseed64_available` | This CPU/build permits the implemented 64-bit RDSEED path |
| `source` | `RDSEED`, `OS`, or `None` if no seed was completed |
| `fallback` | None, unavailable, or retry budget exceeded |
| `rdseed_bytes` | Successfully obtained bytes, including rejected polynomial candidates; discarded if fallback occurred |
| `rdseed_failures` | Instruction calls that reported no value available |
| `os_bytes` | Bytes supplied by successful OS fill requests during fallback |

The report is reset on entry. After an exception, `source` remains `None` unless seed preparation had already completed. Counters are acquisition diagnostics, not measurements of entropy, and may exceed the 16672-byte serialized seed size because polynomial sampling can reject candidates. The report is separate from the serialized seed; store it separately if provenance must be retained across save/restore.

`main.cpp` displays the actual source and fallback reason for the hardware-first seed demonstration. Direct RDRAND remains a separate, explicitly labeled demonstration. Correctness checks and throughput benchmarks still use reproducible deterministic seed material; the hardware-first demonstration does not change their seed policy or checksums. Seed initialization does not alter the subsequent generator algorithm or its per-word cost.

### What hardware provenance does and does not establish

RDSEED is intended to supply conditioned physical entropy for seeding, whereas RDRAND provides hardware DRBG output. The instruction can temporarily fail to supply a value, so availability detection alone is insufficient. See the [Intel DRNG implementation guide](https://cdrdv2-public.intel.com/864722/drng-software-implementation-guide.pdf).

A source report establishes which instruction/API was successfully used, not perfect entropy or a proved statistical distance from ideal initialization. In a virtual machine, it also depends on the virtual CPU. OS APIs do not report the hardware contribution to each returned seed. The mathematical assumptions in the preceding section remain conditional on the actual source and its correct operation.

The 64-bit-only implementation also avoids the 16/32-bit RDSEED forms affected by the specific Zen-5 issue described in [AMD-SB-7055](https://www.amd.com/en/resources/product-security/bulletin/amd-sb-7055.html). This is not a general certification against all hardware or firmware defects.

### Convenience initialization

The `uint64_t` constructor expands a 64-bit seed using SplitMix64; the default constructor uses zero. These modes are useful for reproducibility but reach at most `2^64` initialized states. They do not satisfy either ideal-seed distribution theorem or provide 256 bits of ChaCha key entropy.

The reproducible seed used by the demo tests and benchmarks is also deterministically expanded. It is not evidence of an ideal seed source.

### Seed-size optimality

Exact independence of 1024 selected 64-bit words requires at least 65536 random seed bits by elementary counting. This is a **necessary lower bound**, not a proof that this seed size is achievable for the complete supported position domain. Additional domain-wide constraints can impose stronger bounds.

The implementation uses 131072 GF seed bits. No general minimal-seed claim is made. For the stronger structured guarantee of 2048 words formed from 1024 complete evaluations, however, all 131072 GF seed bits are used bijectively.

More generally, a deterministic generator with `s` seed bits produces at most `2^s` distinct streams. It cannot produce a jointly uniform `n`-bit output for `n > s`. Additional deterministic mixing does not remove this limit.

## 7. Position domain, random access and exhaustion

The hybrid supports:

```text
0 <= word_position < 2^128
```

This corresponds to `2^131` bytes, approximately `2.72e39` bytes. It uses `2^127` distinct GF evaluation points and `2^125` ChaCha blocks.

The original ChaCha layout with a fixed 64-bit nonce and a 64-bit block counter supports `2^67` 64-bit words. The hybrid's addressable word domain is larger by a factor of `2^61`. This comparison concerns addressing, not a cryptographic security guarantee for the entire domain.

Sequential generation does not wrap. The final word can be returned, after which further nonempty requests throw `std::overflow_error`. Both the hybrid and parallel wrapper expose `exhausted()`; after exhaustion, `position128()` reports the final position. An explicit `seek()` resets exhaustion and deliberately permits rereading a position.

This is not a proof of a minimal output period. Equal values at different positions are expected, as with ideal random data.

The standalone ChaCha counter class is a lower-level component; the hybrid and parallel wrapper implement the documented finite-domain exhaustion contract.

## 8. FFT, LFSR costs and initialization

The 1024 monomial coefficients are converted once to a normalized subspace/novel basis. Each aligned block of 1024 consecutive field points is an affine binary subspace, so a ten-stage additive FFT evaluates the polynomial at all those points.

Both halves of every field value are used, yielding 2048 words per block.

The current source performs the following numbers of field-multiplication calls per FFT block, counting squaring as multiplication:

| Work | Count |
|---|---:|
| Butterfly multiplications | 10 * 512 = 5120 |
| Affine-base preparation | 10 + 2 * 9 = 28 |
| Total | 5148 |
| Output words | 2048 |
| Multiplications per word | 2.5137 |

These are field-operation counts, not total runtime estimates: ChaCha, memory traffic, instruction scheduling and API overhead also contribute.

The implementation uses the following optimizations:

- Prefix-XOR factor updates remove the temporary array of node factors.
- Nonzero node indices use an intrinsic trailing-zero count where available.
- The PCLMUL multiplier and its Karatsuba half-XOR are prepared once per node.
- Multiplication/reduction helpers are forced inline inside the targeted fast path.
- ChaCha uses four-block SSE2 generation on x86-64 and direct bulk output.

The FFT computes the same polynomial as Horner evaluation. No coefficients are dropped and no independence guarantee is traded for speed.

Neither universal FFT optimality nor fastest-possible implementation is claimed. Larger blocks can benefit continuous generation while increasing the cost of isolated seeks that miss the cache.

### LFSR generation and seeking

The LFSR processes 64 bits at a time. For its current numerator state:

```text
W      = (A / Q) mod z^64
A_next = (A + W*Q) / z^64
```

The fast path uses **17 carry-less 64-by-64 multiplications per output word**: one to recover `W`, and 16 for dense feedback, plus XORs and state updates. These PCLMUL operations are not the same cost unit as the complete GF(2^128) multiplication counts above. Their ratio does not directly predict the slowdown. A portable shift/XOR path is available.

Random access computes `A_i = A_initial * z^(-64*i) mod Q`. The implementation precomputes 128 binary powers of the word jump, so seeking uses the full 128-bit word position without truncating a 134-bit bit index. Workers can jump directly to disjoint ranges without processing the preceding output.

Sampling and validating a degree-1024 irreducible polynomial and preparing jump tables can be expensive. For degree 1024 the exact Rabin test checks `gcd(z^(2^512)-z, Q)=1` and `z^(2^1024)=z mod Q`, with preliminary small-factor rejection. Prepare once and reuse `PreparedSeed` or copy an initialized generator. Seed preparation is outside the throughput benchmarks and is reported separately. No measured slowdown factor is claimed here.

## 9. Serial, bit and parallel APIs

The main class provides `next_int()`, `generate(dst, count)`, `next_bit()`, 64-bit and 128-bit `seek()`, `position128()`, and `exhausted()`.

Seeking inside the currently cached GF block preserves the FFT result. Bulk generation combines ChaCha output, cached GF halves and the LFSR word mask.

**Bit-call semantics:** `next_bit()` consumes a word into a separate LSB-first reservoir. Interleaving it with `next_int()` or `generate()` does not produce a single shared call-ordered bit stream. `seek()` discards the bit reservoir. The position/exhaustion accessors describe word allocation; buffered bits can remain after the final word has been allocated.

The parallel wrapper splits the middle of the requested range at **2048-word** boundaries. Unaligned prefix/suffix ranges are generated serially. Each worker copies the prototype and seeks to its range; no mutable generator state is shared between workers. Immutable LFSR polynomial parameters and jump tables are shared; workers do not repeat polynomial validation.

- `fill_parallel()` advances the wrapper's position.
- `fill_parallel_at()` evaluates an explicit range without advancing that position.
- A thread count of zero uses the reported hardware concurrency, with a minimum of one.
- Worker exceptions are rethrown in the calling thread.
- If thread creation fails after earlier workers have started, those workers are joined before unwinding.
- A valid range ending at `2^128-1` does not require an unrepresentable exclusive-end position.

## 10. Measured performance — Ryzen 9 5900X

Results are from the supplied `benchmark-results.tar.gz`: 18 regular runs (three each with 1, 2, 4, 8, 12 and 24 threads) and a separate full-test run. Every run generated 30,000,000 64-bit words per benchmark. The full-test run is excluded from the performance aggregates below.

| Environment | Recorded value |
|---|---|
| Processor | AMD Ryzen 9 5900X, 12 cores / 24 logical CPUs exposed to the guest |
| Execution environment | WSL2, Microsoft hypervisor |
| Kernel | `6.18.33.1-microsoft-standard-WSL2`, x86-64 |
| Compiler | `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0` |
| Runtime PCLMUL support | Yes in every log |
| Workload | 30,000,000 uint64_t values; 240,000,000 output bytes |

The compilation command supplied for this measurement series is shown in section 13. The logs record compiler and system information but do not independently record the compiler invocation. These are measurements of this WSL2 environment, not bare-metal Linux results.

### Serial throughput

Medians and ranges below use all 18 regular runs, because each invocation measures the same serial paths regardless of its parallel-thread argument. `M words/s` means millions of 64-bit words per second; `MiB/s` uses 2^20 bytes per MiB.

| Mode | Median M words/s | Median MiB/s | Min–max M words/s |
|---|---:|---:|---:|
| ChaCha component alone | 181.236 | 1382.716 | 179.690–183.643 |
| Hybrid, next_int | 33.669 | 256.875 | 33.325–33.855 |
| Hybrid, bulk | 37.352 | 284.971 | 36.889–37.535 |

### Parallel throughput

Each row uses three runs at the stated thread count. Speedup is relative to the serial bulk median above. The one-thread row measures the parallel wrapper's single-thread path.

| Threads | Median M words/s | Median MiB/s | Min–max M words/s | Speedup vs. serial bulk |
|---:|---:|---:|---:|---:|
| 1 | 37.280 | 284.427 | 37.177–37.392 | 1.00x |
| 2 | 73.695 | 562.247 | 73.579–73.765 | 1.97x |
| 4 | 140.667 | 1073.205 | 140.606–141.980 | 3.77x |
| 8 | 270.894 | 2066.759 | 200.444–273.263 | 7.25x |
| 12 | 261.269 | 1993.322 | 254.086–266.569 | 6.99x |
| 24 | 411.476 | 3139.316 | 410.630–413.517 | 11.02x |

The highest measured median was **411.476 M words/s (3139.316 MiB/s)** at 24 threads. Scaling was not monotonic: the 8-thread runs ranged from 200.444 to 273.263 M words/s, while the 12-thread median was below the 8-thread median. The logs do not establish the cause; the logs do not record scheduling, host load or CPU frequency during the measurements. Three runs describe this sample rather than proving a universally optimal thread count.

### Timing boundaries and checksums

The generator is initialized before throughput timing. Bulk output buffers are allocated before timing, and their checksums are computed afterward. Parallel timing includes worker creation, seeking and joining. The scalar loop includes a volatile checksum update, so scalar-versus-bulk differences are not exclusively API costs. A bulk output buffer occupies 240,000,000 bytes (about 229 MiB).

Every regular run and the full-test run reported the same benchmark checksums:

| Component | Checksum |
|---|---|
| ChaCha component alone | `083d93371401937a` |
| Hybrid, scalar / bulk / parallel | `713e68911fc782f0` |

XOR checksums are consistency signals, not collision-free proofs. The production checks also compare complete output sequences.

The reproducible structured seed preparation took **0.018–0.019 seconds**, with a median of **0.018 seconds** over the regular runs. This measures preparation from the deterministic benchmark byte source; it does not measure RDSEED acquisition or fresh random-polynomial initialization latency. Hardware-first and RDRAND seed demonstrations run separately from the timed throughput loops.

## 11. Memory footprint

GF1024 stores three major per-instance arrays:

| Array | Bytes |
|---|---:|
| Original monomial coefficients, retained for Horner reference | 16384 |
| Novel-basis coefficients | 16384 |
| Cached FFT results | 16384 |
| Total for these arrays | 49152 |

The FFT tables and normalization constants are **shared static data**, not part of each instance. ChaCha state, buffering and position bookkeeping add per-instance storage.

Every supplied run reported `sizeof(ChaCha20GF1024LFSR1024)` = **49840 bytes**, and **16536 bytes** for the shared LFSR parameters and tables. Heap-allocated shared parameters are not included in `sizeof`.

The LFSR adds 128-byte initial and current numerator states per instance, plus bookkeeping. Its immutable jump-power array occupies **16384 bytes**, with additional polynomial coefficients and metadata. These heap-allocated parameters are shared among copies or instances constructed from the same `PreparedSeed`. They are additional to `sizeof(rng)`, not included in it. A retained `PreparedSeed` also stores the 16672-byte serialization. The current demo prints object size and shared-table size separately.

Object size depends on compiler, architecture and configuration and is not part of the theorem. Each parallel worker holds its own generator copy in addition to the wrapper's prototype.

## 12. Validation and reported status

All **18 quick-mode runs** and the separate **full-mode run** report successful production checks and a successful toy suite. Parallel identity checks passed for every measured thread count: 1, 2, 4, 8, 12 and 24.

`full-test.txt` reports:

```text
ChaCha128 regression vec:  OK
Embedded vec (zero LFSR):  OK
Hybrid composition ref:   OK
LFSR word/bit reference:   OK
Structured seed checks:   OK
GF(2^128) fast/ref:        OK
GF128 FFT vs Horner:       OK
Hybrid 128-bit seek:       OK
Bulk vs next_int:          OK
Parallel identity x4:      OK
Toy-model k-wise check:    OK
```

The hardware-first and RDRAND demonstrations succeeded in all 19 logs. Each hardware-first run reports `RDSEED (64-bit)`, no fallback, zero failed RDSEED attempts and zero OS bytes. This records the instruction/API outcome in the WSL2 environment; it does not establish physical entropy quality. The OS fallback path was not exercised by these runs.

### Current production checks

- **ChaCha known-answer vector:** checks the zero-key regression vector.
- **Embedded polynomial vector:** checks a fixed known-answer vector for a lower-degree polynomial embedded in the 1024-coefficient array, with the upper half of the coefficients set to zero. The LFSR numerator is zero for this check. It checks the embedded GF/ChaCha vector, not a nonzero LFSR known-answer vector.
- **Full-seed hybrid reference:** populates all coefficients and compares output against standalone ChaCha XOR portable Horner evaluation XOR the LFSR debug mask, including boundary and high-position cases. This composition check alone is not an independent LFSR reference.
- **LFSR word/bit reference:** compares the optimized word recurrence with a separate bit-at-a-time rational-series recurrence over 4099 words and selected seeks.
- **Structured seeds:** checks export/restore and rejection of invalid or reducible denominator polynomials.
- **Field arithmetic:** compares portable and PCLMUL multiplication and checks distributivity and multiplication by one.
- **FFT versus Horner:** compares fixed and pseudorandom field positions, including the 1024-point boundaries and nonzero upper position limbs.
- **Seek and bulk:** compares direct seeking and sequential generation, and bulk versus scalar output, including carry and exhaustion cases.
- **Parallel:** compares complete output sequences across the low-64-bit carry and checks a two-block range ending at the final valid position, followed by exhaustion.

The current `main.cpp` returns a failure status if a boolean production check or the toy suite fails. The seeding demonstrations print their own status and do not contribute to that combined boolean result.

### Toy checks

`toy_kwise_1024_test.h` supplies:

| Check | Scope |
|---|---|
| A | GF(2^4), k=4: enumerate all seeds for all 1820 four-point sets |
| B | Full-field dependence at k+1 points |
| C | Preservation under a fixed XOR mask |
| D | Reducible-polynomial negative control |
| E | Uniformity after projection to fewer output bits |
| F | Split-half word uniformity, with and without a fixed mask |
| G | GF(2^4), k=8: exact binary rank for all 12870 eight-point sets, plus controls |
| H | Selected split-half sets, structured 16-word uniformity, and a k=4 versus k=8 comparison |

The rank checks analyze the complete linear seed-to-output map without enumerating all `2^32` k=8 seeds. In the reported comparison, the selected-word rank rose from **12 to 16 out of 16**, demonstrating full uniformity of that toy selection after doubling the order.

Quick mode samples 110 of 4368 position sets for B and 91 of 1820 for C. Full mode checks all those GF(2^4) sets and adds GF(2^8), k=3 checks with exhaustive seed enumeration for selected position sets.

**Both quick and full modes passed in the supplied logs.** The full run took **15.1 seconds for the toy suite**, checking all 4368 five-point sets for B and all 1820 four-point sets for C in GF(2^4). The k=8 rank check covered all 12870 eight-point sets. The GF(2^8), k=3 extension exhaustively enumerated all `2^24` seeds for **four selected three-point sets**, with dependence and masking checks also passing. It does not enumerate every possible GF(2^8) point set.

The toy suite validates scaled GF constructions, not the production LFSR arithmetic or the small-bias theorem. Its results complement the production reference checks and do not replace the general proofs.

## 13. Build and run

The code uses C++17. Only `main.cpp` is compiled explicitly because implementations reside in the included headers.

### Linux / GCC

From the directory containing the five source files listed below:

```bash
g++ -std=c++17 -O3 -march=native -flto -DNDEBUG -pthread main.cpp -o chacha20gf1024lfsr1024
./chacha20gf1024lfsr1024 30000000 4
```

The arguments are word count, thread count, and optionally `full` for the full toy suite:

```bash
./chacha20gf1024lfsr1024 30000000 4 full
```

The program runs correctness checks, seed demonstrations, benchmarks and the selected toy suite. It is not a benchmark-only command.

### CPU portability

`-march=native` targets the build machine and can make the executable incompatible with older CPUs. A less CPU-specific build is:

```bash
g++ -std=c++17 -O3 -flto -pthread main.cpp -o chacha20gf1024lfsr1024
```

GCC/Clang use a function-specific PCLMUL target and runtime detection; a global `-mpclmul` flag is not required.

To select the portable field implementation and disable the explicit four-block ChaCha SIMD path:

```bash
g++ -std=c++17 -O3 -pthread -DCHACHA20GF1024LFSR1024_FORCE_PORTABLE main.cpp -o chacha20gf1024lfsr1024-portable
```

Define `CHACHA20GF1024LFSR1024_FORCE_PORTABLE` consistently across translation units. “Portable” selects source paths; it does not forbid compiler-generated SIMD or change the mathematical output.

### Windows and other compilers

For MinGW-w64, append `-lbcrypt`:

```bash
g++ -std=c++17 -O3 -march=native -flto -DNDEBUG -pthread main.cpp -o chacha20gf1024lfsr1024.exe -lbcrypt
```

MSVC links BCrypt via the seed header's pragma. The project is intended for MSVC, GCC and Clang. The supplied measurements and successful tests cover GCC on Ubuntu under WSL2; cross-compiler validation is not claimed.

The seed helper currently supports Windows and Linux. The core generator itself does not depend on OS random APIs.

## 14. API examples

### Practical initialization and bulk output

```cpp
#include "ChaCha20GF1024LFSR1024Seed.h"
#include <vector>

auto seed = chacha20gf1024lfsr1024_seed::make_prepared_seed();
ChaCha20GF1024LFSR1024 rng(seed);

uint64_t first = rng.next_int();
std::vector<uint64_t> values(4096);
rng.generate(values.data(), values.size());
```

Use the prepared seed to avoid repeating irreducibility checks and jump-table preparation:

```cpp
ChaCha20GF1024LFSR1024 another(seed); // independent mutable state, same stream
const auto saved = seed.bytes();    // serialized Seed value
ChaCha20GF1024LFSR1024 restored(saved); // validates Q and rebuilds tables
```

For a raw serialized buffer, pass its explicit size:

```cpp
ChaCha20GF1024LFSR1024 restored2(saved.data(), saved.size());
```

The serialized seed must contain exactly `FULL_SEED_BYTES` (16672 bytes), including a valid irreducible LFSR denominator. The core header is self-contained.

### Deterministic convenience mode

```cpp
#include "ChaCha20GF1024LFSR1024.h"

ChaCha20GF1024LFSR1024 rng(uint64_t{12345}); // Reproducible, not full-entropy mode.
```

### Random access

Using an initialized generator:

```cpp
ChaCha20GF1024LFSR1024::Position position{
    0x0123456789abcdefULL,
    0x0000000000000001ULL
};
rng.seek(position);
uint64_t value = rng.next_int();

rng.seek(1'000'000ULL);
```

### Parallel evaluation

```cpp
#include "ChaCha20GF1024LFSR1024Parallel.h"
#include "ChaCha20GF1024LFSR1024Seed.h"
#include <vector>

auto seed = chacha20gf1024lfsr1024_seed::make_prepared_seed();
ChaCha20GF1024LFSR1024Parallel rng(seed);

std::vector<uint64_t> values(1'000'000);
rng.fill_parallel(values.data(), values.size(), 4);
```

### Toy runner

```cpp
#include "toy_kwise_1024_test.h"

bool ok = run_toy_kwise_1024_test(false, 4, stdout);
```

Use `true` for full mode. The `gf8_sets` argument is used in full mode; zero retains the convention of checking one GF(2^8) set. Passing a null output pointer suppresses logging.

## 15. Files

| Current file | Role |
|---|---|
| ChaCha20GF1024LFSR1024.h | Self-contained core: ChaCha, GF FFT, LFSR, structured sampler and jump tables |
| ChaCha20GF1024LFSR1024Parallel.h | Parallel generation of disjoint ranges of one stream |
| ChaCha20GF1024LFSR1024Seed.h | RDSEED-first seeding, complete OS fallback, source reporting, and explicit OS-only helpers |
| toy_kwise_1024_test.h | Exhaustive counting and binary-rank toy checks |
| main.cpp | Production checks, seed demonstrations and benchmarks |
| README.md | Construction, guarantees, usage and measured results |

Use `FULL_SEED_BYTES`, `GF_SEED_BYTES` and `LFSR_SEED_BYTES` for seed buffer sizes, and `OUTPUT_WORDS_PER_FFT` for word-range alignment. The standalone ChaCha class is `ChaCha20GF1024LFSR1024Counter128`; the toy runner is `run_toy_kwise_1024_test`.

## 16. Scope and future work

The implementation is an experimental prototype with the measured performance and successful production, quick-mode and full-mode checks documented above. It does not claim:

- a universal ranking as “more random” than ChaCha20;
- unconditional cryptographic security or a cryptographic audit;
- ideal initialization from an expanded short seed;
- arbitrary exact independence beyond the proven order, or whole-stream statistical closeness to ideal bits;
- exact cross-stream independence merely from different stream identifiers;
- a proven minimal period of 2^128 words;
- backtracking resistance after state compromise;
- minimum seed size or universally optimal FFT performance;
- fresh entropy generation by deterministic expansion.

Possible future work includes wider SIMD/VPCLMUL, ARM PMULL, further fixed-factor arithmetic optimization, additional platform measurements, and determining the exact split-half independence order above the established lower bound.

The current implementation includes one AGHP random-feedback small-bias component. It does not implement a sum of multiple small-bias generators to fool higher-degree GF(2) polynomial tests (a different construction with separate parameters). Larger GF coefficient counts and such additional constructions remain possible future work.

## 17. References

- Noga Alon, Oded Goldreich, Johan Håstad and René Peralta, [Simple Constructions of Almost k-wise Independent Random Variables](https://web.math.princeton.edu/~nalon/PDFS/aghp4.pdf), especially the random-feedback LFSR construction.
- Joseph Naor and Moni Naor, [Small-Bias Probability Spaces: Efficient Constructions and Applications](https://www.wisdom.weizmann.ac.il/~naor/PAPERS/bias.pdf).

- Daniel J. Bernstein, [ChaCha, a variant of Salsa20](https://cr.yp.to/chacha/chacha-20080120.pdf).
- Y. Nir and A. Langley, [RFC 8439: ChaCha20 and Poly1305 for IETF Protocols](https://www.rfc-editor.org/rfc/rfc8439.html).
- S. Arciszewski, [XChaCha Internet-Draft](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha), including HChaCha20.
- James Aspnes, [k-wise Independence](https://www.cs.yale.edu/homes/aspnes/pinewiki/KwiseIndependence.html).
- NIST, [SP 800-38D: GCM and GMAC](https://csrc.nist.gov/pubs/sp/800/38/d/final), for the binary field polynomial.
- Shuhong Gao and Todd Mateer, [Additive Fast Fourier Transforms over Finite Fields](https://www.math.clemson.edu/~sgao/papers/GM10.pdf).
- Sian-Jheng Lin, Wei-Ho Chung and Yunghsiang S. Han, [Novel Polynomial Basis and Its Application to Reed-Solomon Erasure Codes](https://arxiv.org/abs/1404.3458).
- Intel, [Enabling High-Performance Galois-Counter-Mode on Intel Architecture Processors](https://www.intel.com/content/dam/www/public/us/en/documents/software-support/enabling-high-performance-gcm.pdf).
- Linux, [getrandom(2)](https://man7.org/linux/man-pages/man2/getrandom.2.html).
- Microsoft, [BCryptGenRandom](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom).
- GCC, [Optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) and [x86 options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html).

These references describe the constituent methods. They do not constitute an audit or endorsement of this combined implementation.
