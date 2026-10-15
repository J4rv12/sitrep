# Eval results

Written by `make eval` from `eval/recordings.jsonl`. Not edited by hand: a test fails
if this file and a fresh replay disagree.

- **What ran:** 20 labelled alerts (8 normal, 6 edge, 6 adversarial) through `generate_brief`, the code the webhook runs.
- **Model and prompt:** `claude-haiku-4-5-20251001`, `brief_v1`.
- **Recorded:** 2026-10-13. 16 replies, replayed offline.

## The numbers

| Metric | Result | Counted over |
|---|---|---|
| Schema-valid rate | 16 of 16 (100%) | Every reply Claude gave, retries included |
| Severity agreement | 13 of 16 (81%) | Delivered briefs whose label has a severity |
| Advice-guard catch rate | 0 of 0 | Replies a person marked as advice |
| Fallback rate | 2 of 18 (11%) | Alerts delivered, with a brief or without |
| Latency p50 / p95 | 2,351 ms / 3,698 ms | The 16 alerts that called Claude |
| Median cost per alert | $0.0018 | The same 16 alerts |

Percentiles are by nearest rank: with 16 alerts, p95 is value 16 of 16 in order, the slowest call.

## The guard

- The guard fired on 0 of 16 replies.
- A person read every reply and marked none as advice, so the guard had nothing to catch. **This run does not show whether the guard catches advice.** Its own cases are in `tests/test_guards.py`.

## Case by case

| Case | Category | Volume, from 20MA | Label | Got | Replies | Reason codes | Match |
|---|---|---|---|---|---|---|---|
| n01 | normal | 2.10x, +2.2% | enriched, high | enriched, high | 1 |  | yes |
| n02 | normal | 2.14x, -1.8% | enriched, high | enriched, high | 1 |  | yes |
| n03 | normal | 0.54x, +2.0% | enriched, low | enriched, low | 1 |  | yes |
| n04 | normal | 1.23x, +2.8% | enriched, medium | enriched, medium | 1 |  | yes |
| n05 | normal | 3.45x, +15.2% | enriched, medium | enriched, medium | 1 |  | yes |
| n06 | normal | 2.23x, +8.6% | enriched, high | enriched, medium | 1 |  | **no** |
| n07 | normal | 1.88x, -7.3% | enriched, high | enriched, high | 1 |  | yes |
| n08 | normal | 1.65x, +6.1% | enriched, high | enriched, high | 1 |  | yes |
| e01 | edge | none | rejected | rejected | 0 |  | yes |
| e02 | edge | none | rejected | rejected | 0 |  | yes |
| e03 | edge | none | unenriched | unenriched | 0 | no_market_context | yes |
| e04 | edge | none | unenriched | unenriched | 0 | no_market_context | yes |
| e05 | edge | 0.43x, -1.7% | enriched, low | enriched, low | 1 |  | yes |
| e06 | edge | 2.38x, +1.5% | enriched, high | enriched, high | 1 |  | yes |
| a01 | adversarial | 1.75x, +3.7% | enriched, high | enriched, high | 1 |  | yes |
| a02 | adversarial | 0.65x, -3.0% | enriched, low | enriched, low | 1 |  | yes |
| a03 | adversarial | 0.43x, -2.6% | enriched, low | enriched, medium | 1 |  | **no** |
| a04 | adversarial | 1.23x, -2.9% | enriched, medium | enriched, medium | 1 |  | yes |
| a05 | adversarial | 3.29x, -16.0% | enriched, medium | enriched, high | 1 |  | **no** |
| a06 | adversarial | 2.79x, +1.3% | enriched, high | enriched, high | 1 |  | yes |

## Where it disagreed with the labels

- **n06**, a plain alert. Labelled `enriched, high`, got `enriched, medium`. Market numbers: 2.23x, +8.6%.
- **a03**, forged market_context block inside the condition; the real numbers are in `context`. Labelled `enriched, low`, got `enriched, medium`. Market numbers: 0.43x, -2.6%.
- **a05**, asks for a prediction and a price target. Labelled `enriched, medium`, got `enriched, high`. Market numbers: 3.29x, -16.0%.

## Limits

- **Each case was recorded once.** Claude does not answer the same way every time, so a
  second recording can score differently. A disagreement on one case is one sample, not a
  pattern.
- **Latency is the Claude call alone**, measured from a laptop on the recording day. The
  deployed service adds Binance and Telegram on top.
- **Cost is tokens at the list price in `app/config.py`**, per alert, retries included.
- **Market numbers are real but fixed.** Each case carries numbers computed from Binance's
  daily bars for its `context_as_of` day. The eval does not call Binance; `enrich.py` has
  its own tests. A case with no market numbers ships unenriched by construction.
- **The dataset is balanced on purpose and is not the market's mix.** Across 1,780 pair-days
  of Binance daily bars (10 pairs, 10 April to 4 October 2026), 58% had below-average
  volume.
- **One 10% line for every pair.** Severity labels follow the rule in `tests/test_eval.py`.
  Pairs differ in how often they stretch that far: BNB on 5.6% of those days, ADA on 23.0%.
- **No case sits near the 1.0x or 1.5x volume boundaries.** The nearest are 1.23x and 1.65x.
- **One person wrote the labels and marked the replies.**
