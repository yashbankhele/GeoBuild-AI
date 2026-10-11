import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from src.change_detection import compare_images
from src.reporting import make_markdown_report

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "sample_images"
st.title("🛰️ Image Comparison")
st.write("Classical computer-vision baseline for a pair of images.")

def load_rgb(src):
    return np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)

def png_bytes(array, mode="RGB"):
    buf = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buf, format="PNG")
    return buf.getvalue()

with st.sidebar:
    use_samples = st.toggle("Use synthetic sample images", value=True)
    threshold = st.slider("Difference threshold", 5, 100, 25)
    blur = st.select_slider("Smoothing kernel", [1, 3, 5, 7], value=3)
    min_region = st.slider("Minimum connected region (pixels)", 0, 500, 0, step=25)

if use_samples:
    if not (SAMPLES / "before.png").exists():
        st.error("Sample images missing. Run python generate_sample_images.py")
        st.stop()
    before, after = load_rgb(SAMPLES / "before.png"), load_rgb(SAMPLES / "after.png")
    source_note = "Synthetic sample images"
else:
    a, b = st.columns(2)
    with a:
        before_file = st.file_uploader("Earlier image", type=["png", "jpg", "jpeg"], key="before")
    with b:
        after_file = st.file_uploader("Later image", type=["png", "jpg", "jpeg"], key="after")
    if before_file is None or after_file is None:
        st.info("Upload both images.")
        st.stop()
    before, after = load_rgb(before_file), load_rgb(after_file)
    source_note = "User-uploaded images"

a, b = st.columns(2)
a.image(before, caption="Earlier image", use_container_width=True)
b.image(after, caption="Later image", use_container_width=True)

if st.button("Run image comparison", type="primary", use_container_width=True):
    r = compare_images(before, after, threshold, blur, min_region)
    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged pixels", f"{r.changed_pixels:,}")
    m2.metric("Total pixels", f"{r.total_pixels:,}")
    m3.metric("Flagged pixel share", f"{r.changed_percentage:.2f}%")
    t1, t2, t3 = st.tabs(["Overlay", "Difference", "Mask"])
    t1.image(r.overlay_rgb, use_container_width=True)
    t2.image(r.difference_gray, clamp=True, use_container_width=True)
    t3.image(r.change_mask, clamp=True, use_container_width=True)
    summary = pd.DataFrame([
        ("Input", source_note), ("Threshold", threshold), ("Smoothing kernel", blur),
        ("Minimum region size", min_region), ("Flagged pixels", r.changed_pixels),
        ("Total pixels", r.total_pixels), ("Flagged pixel share (%)", round(r.changed_percentage, 3))
    ], columns=["Metric", "Value"])
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button("Download summary CSV", summary.to_csv(index=False).encode(),
                       "change_summary.csv", "text/csv")
    st.download_button("Download mask PNG", png_bytes(r.change_mask, "L"),
                       "change_mask.png", "image/png")
    st.download_button("Download overlay PNG", png_bytes(r.overlay_rgb),
                       "change_overlay.png", "image/png")
    report = make_markdown_report("Image comparison", threshold, r.changed_pixels,
                                  r.total_pixels, r.changed_percentage, source_note)
    st.download_button("Download Markdown report", report.encode(),
                       "geobuild_report.md", "text/markdown")
    st.warning("Pixel share is not construction completion or square metres. This method is not a trained ML model.")
