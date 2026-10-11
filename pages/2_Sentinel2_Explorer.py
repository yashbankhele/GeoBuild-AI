import io
import numpy as np
import streamlit as st
from PIL import Image
from src.satellite import search_sentinel2_items, load_sentinel_item

st.title("🌍 Sentinel-2 Explorer")
st.write("Search Sentinel-2 L2A scenes from Microsoft Planetary Computer and inspect a true-colour preview plus NDVI.")
st.info("Requires internet. Satellite search/download may take time. Start with a small bounding box and a short date range.")

with st.form("stac_search"):
    c1, c2 = st.columns(2)
    with c1:
        min_lon = st.number_input("Min longitude", value=73.80, format="%.5f")
        min_lat = st.number_input("Min latitude", value=18.45, format="%.5f")
    with c2:
        max_lon = st.number_input("Max longitude", value=73.90, format="%.5f")
        max_lat = st.number_input("Max latitude", value=18.55, format="%.5f")
    c3, c4, c5 = st.columns(3)
    with c3:
        start_date = st.date_input("Start date", value=__import__("datetime").date(2025, 1, 1))
    with c4:
        end_date = st.date_input("End date", value=__import__("datetime").date(2025, 3, 31))
    with c5:
        max_cloud = st.slider("Max scene cloud cover (%)", 0, 100, 20)
    submitted = st.form_submit_button("Search Sentinel-2 scenes", type="primary")

if submitted:
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
    bbox = [min_lon, min_lat, max_lon, max_lat]
    try:
        with st.spinner("Searching public STAC catalogue..."):
            items = search_sentinel2_items(bbox, start_date.isoformat(), end_date.isoformat(), max_cloud)
        if not items:
            st.warning("No scenes matched. Try a wider date range, larger cloud limit, or a different AOI.")
            st.stop()
        st.session_state["sentinel_items"] = items
        st.success(f"Found {len(items)} matching scene(s).")
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "Item ID": i.id,
            "Date": i.datetime.isoformat() if i.datetime else "unknown",
            "Cloud cover (%)": i.properties.get("eo:cloud_cover"),
            "Platform": i.properties.get("platform", "unknown"),
        } for i in items]), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Satellite search failed: {exc}")
        st.caption("Check internet access, dates, bounding box, and dependency installation.")
        st.stop()

items = st.session_state.get("sentinel_items", [])
if items:
    labels = [f"{i.datetime.date() if i.datetime else 'unknown'} | cloud {i.properties.get('eo:cloud_cover', '?')}% | {i.id}" for i in items]
    chosen_idx = st.selectbox("Choose scene to load", range(len(items)), format_func=lambda x: labels[x])
    if st.button("Load selected scene", type="primary"):
        try:
            with st.spinner("Downloading/reprojecting scene bands. This may take a few minutes..."):
                data = load_sentinel_item(items[chosen_idx])
            st.session_state["loaded_scene"] = data
            st.success("Scene loaded. Review the cloud/valid-data mask before interpreting.")
        except Exception as exc:
            st.error(f"Could not load scene assets: {exc}")
            st.caption("Some scenes may have missing assets or network-access issues. Try another scene.")

data = st.session_state.get("loaded_scene")
if data:
    m = data["meta"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Scene cloud cover", f"{m.get('cloud_cover', 0):.1f}%")
    c2.metric("Valid pixels", f"{m['valid_pixel_pct']:.1f}%")
    c3.metric("Raster dimensions", f"{m['width']} × {m['height']}")
    st.write(f"**Scene:** `{m['item_id']}` · **Date:** {m['date']} · **CRS:** `{m['crs']}`")
    a, b = st.columns(2)
    a.image(data["rgb"], caption="True-colour preview (percentile-stretched)", use_container_width=True)
    ndvi = data["ndvi"]
    ndvi_display = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
    b.image(ndvi_display, caption="NDVI preview (grayscale stretch; not a classified map)", clamp=True, use_container_width=True)
    valid_preview = (data["valid_mask"].astype(np.uint8) * 255)
    st.image(valid_preview, caption="SCL-based valid-pixel mask (white = included)", clamp=True, use_container_width=True)
    st.download_button("Download NDVI array as CSV (preview)", 
                       __import__("pandas").DataFrame(ndvi).to_csv(index=False).encode(),
                       "sentinel2_ndvi_preview.csv", "text/csv")
    st.warning("Scene loading is a first-pass implementation. Inspect the AOI footprint and cloud mask. The current page does not yet clip exactly to the requested bounding box or register multiple scenes for change detection.")
