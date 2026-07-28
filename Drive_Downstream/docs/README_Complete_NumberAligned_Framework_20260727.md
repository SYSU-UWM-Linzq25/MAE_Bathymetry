# Downstream MAE complete number-aligned framework

This package reorganizes the complete `Drive_Downstream_Backup.zip`, not only the current model comparison.

## Fixed meaning of prefixes

- **A**: split generation, data audit, source-chain inspection
- **B**: HPC/command drivers for A-stage preparation and QA
- **C**: smoke/minitest training
- **D**: model training and staged fine-tuning
- **E**: checkpoint tile-level evaluation
- **F**: full-river prediction, GT/error calculation, workflow submission
- **G**: dashboard and cross-model analysis

## Number families

- **001**: original LCC stage3 pipeline
- **002**: river-holdout stage4 final-mask pipeline
- **003**: NoData-safe stage4 experiment
- **004**: zero-NoData/core-loss fix
- **005**: canonical all-river NoData/core-loss model
- **010 / 010g / 010q**: LOORO cross-validation, grouped holdout, and suspicious-tile source QA
- **020**: v2 dual-mask **NormOnly**
- **025**: **NormThenMeter**, initialized from family 020
- **030**: **MeterOnly**
- **034**: **MeterThenNorm**, initialized from family 030
- **090**: cross-model comparisons

## Main current comparison

```text
A020 split
  ├─ D021 NormOnly ──> D026 NormThenMeter ──> E026 / F026 -> F028
  └─ D031 MeterOnly ────────────────────────> E031 / F031 -> F033

F028 + F033 -> G092/G093 local best-middle-worst reach analysis
```

## Compatibility policy

Existing scientific logic and default data/result roots were preserved. Some aligned code files still read or write legacy product names such as `F060_summary.json`, `F062_summary.json`, or output roots containing historical labels. This is intentional so existing completed results remain usable. The framework CSV identifies these cases.

## Recommended staged run order

1. Use the 001–005 families only for historical reproduction or regression checks.
2. Use 010/010g for legacy LOORO cross-validation and 010q for suspicious-tile diagnosis.
3. Generate v2 splits with A020/A020s.
4. Train NormOnly using D021 or D024.
5. Train NormThenMeter using D026/D027; D025 consumes the NormOnly checkpoint.
6. Train MeterOnly using D031/D032.
7. Run E026/F026/F028 for NormThenMeter and E031/F031/F033 for MeterOnly.
8. Run G092/G093 for the detailed global and local-reach comparison.
9. Family 034 currently contains training code only; missing E/F stages are listed as planned rows rather than silently fabricated.
