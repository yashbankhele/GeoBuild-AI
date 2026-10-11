# Architecture and evaluation notes

## Current data paths

- **Demo path:** synthetic sample image pair -> OpenCV grayscale difference -> mask/overlay/metrics.
- **Satellite path:** STAC search -> Sentinel-2 L2A scene -> B02/B03/B04/B08/SCL loading -> true-colour preview + NDVI + first-pass validity mask.
- **Evaluation path:** uploaded image pair + labelled binary mask -> precision/recall/F1/IoU/accuracy.
- **Supervised baseline:** RGB pair pixel features + mask -> Random Forest -> pixel-level smoke-test metrics.

## Current gaps

- The satellite loader does not clip to the exact requested AOI.
- It does not co-register two dates or compute a multi-date satellite change map.
- The optional Random Forest evaluation uses a random pixel split; spatial leakage may inflate scores.
- No trained model or model score is shipped in this repository.
- All built-in time-series records are synthetic.

## Recommended research protocol

1. Define a clear change taxonomy and annotation guide.
2. Collect multiple aligned image pairs from different sites and dates.
3. Have a human create binary or semantic reference masks.
4. Split by site/date before fitting any model.
5. Compare a fixed classical baseline and a supervised model on the same held-out pairs.
6. Report precision, recall, F1, IoU, class prevalence, failure examples and data limitations.
