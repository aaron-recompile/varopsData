# M2 MacBook Pro ARM64 Benchmark

## Hardware & Environment

- **CPU:** Apple M2  
- **Architecture:** ARM64  
- **OS:** macOS  
- **Compiler:** Clang 16.0.0  
- **SHA256 implementation:** `arm_shani` (1-way / 2-way), hardware-accelerated  

## Results Summary

- **Total tests:** 875  
- **Suggested varops/weight:** 3853 (current value: 5200)  

This benchmark runs the full 875-case varops test matrix on an Apple M2 machine and compares the effective CPU cost of different operations under the current Taproot varops model.

## Key Finding: RIPEMD160 / SHA1 are Underpriced on ARM

| Operation                    | Time (s) | Schnorr-equivalent ops | Varops used |
|-----------------------------|----------|-------------------------|-------------|
| Schnorr validation          | 1.86     | 80,000                  | baseline    |
| RIPEMD160_DROP_DUP_520Bx2   | 1.63     | 69,900                  | 3.33%       |
| SHA1_DROP_DUP_520Bx2        | 1.06     | 45,706                  | 3.33%       |
| SHA256_DROP_DUP_520Bx2      | 0.49     | 21,061                  | 36.67%      |

RIPEMD160 consumes ~87% of Schnorr's CPU time but only 3.33% of the varops budget — roughly a **30× gap** between actual cost and budgeted cost.  

This is consistent with the fact that Apple Silicon has hardware acceleration for SHA256 but not for RIPEMD160 or SHA1.

## Additional Notes

- Stack-heavy scripts (e.g. `NIP`, `TUCK` on large data) appear correctly priced: they reach 100% varops at the point where CPU usage is saturated.
- `MUL` on large numbers also appears correctly priced under the current model.
- These results may be useful for discussions on whether RIPEMD160 / SHA1 cost multipliers should be increased, or whether these opcodes should be treated differently in future TapScript designs. See the related PR discussion for context:  
  https://github.com/rustyrussell/bips/pull/1

## Contributor

- **GitHub:** [@aaron-recompile](https://github.com/aaron-recompile)  
- **Date:** 2025-12


## Rerun (2025-12-22): Worst-case varops stress tests

This rerun was performed after the addition of new worst-case benchmarks
(e.g. 3DUP + repeated hash opcodes) in commit `f6137273a3` on the `gsr` branch.

Key observations:

- The slowest 100% varops operation observed was:
  - `TUCK_DROP_100KBx2`: 2.588s (vs Schnorr 1.945s)
- Based on this, the implied upper bound for varops would be:
  - ~3900 varops / weight unit (down from current 5200)

Notably, some RIPEMD160-heavy sequences (e.g. repeated 3DUP + RIPEMD160)
remain significantly slower than Schnorr validation while consuming
a small fraction of the varops budget, reinforcing the baseline
underpricing observation on ARM64.

Detailed results are provided in:
- `varops_m2_macbook_pro_arm64_aaron_3dup_ripemd160.csv`