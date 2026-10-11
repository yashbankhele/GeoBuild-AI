# GeoBuild AI — Satellite Change Monitoring Portfolio

**Version 3.0 | Python · Streamlit · OpenCV · Rasterio · Sentinel-2 STAC · scikit-learn**

GeoBuild AI is a portfolio prototype exploring image-difference workflows, Sentinel-2 scene discovery, basic raster processing, NDVI visualization, and labelled change-mask evaluation.

> **Transparency:** built-in images and site time-series records are synthetic. Sentinel-2 Explorer connects to the public Microsoft Planetary Computer STAC catalogue when internet is available. The image-difference method is classical computer vision, not a pretrained deep-learning model. Real-data results require careful inspection and validation.

## Features

- Multi-page Streamlit app and demo overview
- Before/after image comparison using OpenCV
- Threshold, smoothing and connected-region controls
- Change mask, red overlay, PNG/CSV/Markdown exports
- Sentinel-2 L2A STAC search by AOI, date range and cloud cover
- Scene preview using B02/B03/B04 and NDVI from B08/B04
- SCL-based first-pass cloud/shadow/snow exclusion mask
- Labelled-mask evaluation: precision, recall, F1, IoU, accuracy and confusion matrix
- Optional supervised Random Forest pixel-feature baseline
- Multi-site synthetic analytics, data-quality checks and tests

## Windows setup (PowerShell)

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_sample_images.py
python -m pytest -q
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Pages

1. **Overview** — demo site KPIs and observation history
2. **Image Comparison** — upload images and inspect a classical change mask
3. **Sentinel-2 Explorer** — query scenes from Microsoft Planetary Computer and preview RGB/NDVI
4. **Labelled Evaluation** — compare predictions with a human-labelled mask; optional Random Forest baseline
5. **Site Analytics** — synthetic observation trends
6. **Data Quality & Project Guide** — data checks, architecture and limitations

## Sentinel-2 data source

The explorer uses the public STAC endpoint:
`https://planetarycomputer.microsoft.com/api/stac/v1`

It searches the `sentinel-2-l2a` collection. Use a small WGS84 bounding box and date range first. Public catalogue/data availability, network speed, and scene assets may vary. The current module is a first-pass integration: it does not yet clip to the exact AOI or co-register separate scenes for change detection.

## Evaluation protocol

Upload the exact earlier image, later image, and a human-labelled binary mask of identical dimensions. White/non-zero pixels mean change. The app reports precision, recall, F1, IoU and accuracy.

For a credible research result, create labelled masks for multiple independent site/date pairs, keep a separate untouched test set, and split by site or date rather than randomly splitting nearby pixels. The included Random Forest uses a random pixel split only as a development smoke test and must not be presented as a robust generalization benchmark.

## Important limitations

- Synthetic demo imagery and time series are illustrative, not real monitoring results.
- RGB image resizing is not geographic registration.
- The satellite loader is not yet exact-AOI clipped or multi-date registered.
- Clouds, shadows, seasonality, illumination and sensor differences can cause false changes.
- NDVI is a vegetation index, not a construction-progress metric.
- Flagged pixel share is not construction completion percentage or real-world area.
- No validated model accuracy claim is included.

## Roadmap

- AOI clipping, projected area calculations and consistent multi-date alignment
- Improved cloud masking and quality screening
- Labelled masks across multiple independent sites
- Site-held-out model training and evaluation
- Comparison with suitable change-detection/segmentation networks
- Deployment, screenshots, demo video and reproducibility checks

## License

Add an appropriate license before redistributing or accepting external contributions.
