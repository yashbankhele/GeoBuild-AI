# Data provenance

- `data/sample_images/before.png` and `after.png` are generated locally by `generate_sample_images.py`. They are synthetic aerial-style illustrations, not photographs or satellite scenes.
- `data/demo/sites.csv` and `observations.csv` are synthetic example records created for dashboard demonstrations.
- Live scene search in `pages/2_Sentinel2_Explorer.py` queries the public Microsoft Planetary Computer STAC API and loads assets from the Sentinel-2 L2A collection.
- Any labelled mask uploaded to the evaluation page must be provided/annotated by the user and aligned to the image pair. No labelled benchmark dataset is bundled.
