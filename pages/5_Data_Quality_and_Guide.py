import streamlit as st
import pandas as pd
from src.demo_data import load_sites, load_observations

st.title("📚 Data Quality & Project Guide")
sites, obs = load_sites(), load_observations()
st.subheader("Demo data checks")
checks = [
    ("Unique site IDs", sites.site_id.is_unique),
    ("No missing observation cells", not obs.isna().any().any()),
    ("Change percentages in range", obs.changed_pixel_pct.between(0,100).all()),
    ("Quality scores in range", obs.image_quality.between(0,100).all()),
]
for label, passed in checks:
    (st.success if passed else st.error)(f"{'PASS' if passed else 'FAIL'} · {label}")
st.write(f"Sites: {len(sites)} · Observations: {len(obs)} · Missing cells: {int(obs.isna().sum().sum())}")
st.dataframe(sites, use_container_width=True, hide_index=True)
st.subheader("Architecture")
st.code("""Sentinel-2 STAC search (optional internet module)
              |
              v
    Load RGB + NIR + SCL bands
              |
              v
     RGB preview + NDVI + cloud mask

Uploaded image pair -> OpenCV baseline -> change mask
                                      |
                                      v
                           labelled mask evaluation
                                      |
                                      v
                       precision / recall / F1 / IoU
""")
st.subheader("Current limitations")
st.markdown("""
- Built-in images and site observations are synthetic.
- Sentinel-2 scene search and loading are real-data integration features, but availability and results depend on STAC coverage/network.
- Current scene loader does not yet clip to exact AOI or align two dates for change analysis.
- Pixel difference is not a trained deep-learning model.
- Evaluation requires a human-labelled, aligned ground-truth mask.
- Random pixel splits can leak spatial context; use held-out sites or dates for credible generalization estimates.
""")
st.subheader("Roadmap")
st.markdown("""
1. AOI-aware clipping and consistent reprojection across dates.
2. Better cloud/shadow screening and scene selection.
3. Build labelled masks for several independent sites.
4. Train and evaluate on site-level splits; report precision, recall, F1, IoU and uncertainty.
5. Compare classical baseline with segmentation/change-detection networks.
6. Deploy with cached STAC results, error handling, screenshots and demo video.
""")
