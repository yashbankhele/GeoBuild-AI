import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from src.change_detection import compare_images
from src.evaluation import evaluate_binary_mask
from src.ml_model import train_pixel_classifier, predict_pixel_mask

st.title("🧪 Labelled Model Evaluation")
st.write("Evaluate a predicted change mask against a human-labelled ground-truth mask. Optionally train a small supervised pixel classifier.")

st.warning("Metrics are meaningful only when the ground-truth mask is carefully labelled and aligned to the exact image pair. Do not evaluate on the same pixels used to fit a model and call it a final benchmark.")

def read_rgb(file):
    return np.asarray(Image.open(file).convert("RGB"), dtype=np.uint8)

def read_mask(file):
    return np.asarray(Image.open(file).convert("L"), dtype=np.uint8)

a, b, c = st.columns(3)
with a:
    before_f = st.file_uploader("Earlier RGB image", type=["png", "jpg", "jpeg"], key="eval_before")
with b:
    after_f = st.file_uploader("Later RGB image", type=["png", "jpg", "jpeg"], key="eval_after")
with c:
    truth_f = st.file_uploader("Ground-truth mask (white = changed)", type=["png", "jpg", "jpeg"], key="eval_truth")

if before_f and after_f and truth_f:
    try:
        before, after, truth = read_rgb(before_f), read_rgb(after_f), read_mask(truth_f)
        if before.shape[:2] != after.shape[:2]:
            st.error("For evaluation, the two images must already be aligned and have identical dimensions.")
            st.stop()
        if before.shape[:2] != truth.shape:
            st.error("Ground-truth mask dimensions must exactly match the images.")
            st.stop()
        threshold = st.slider("Baseline threshold", 5, 100, 25)
        if st.button("Evaluate classical baseline", type="primary"):
            result = compare_images(before, after, threshold=threshold)
            metrics = evaluate_binary_mask(result.change_mask, truth)
            st.subheader("Classical image-difference baseline")
            cols = st.columns(5)
            for col, name, label in zip(cols, ["precision","recall","f1","iou","accuracy"],
                                        ["Precision","Recall","F1-score","IoU","Accuracy"]):
                col.metric(label, f"{metrics[name]:.3f}")
            st.write("Confusion matrix — rows are true labels [unchanged, changed], columns are predictions [unchanged, changed].")
            st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                         index=["True unchanged", "True changed"],
                         columns=["Pred unchanged", "Pred changed"]), use_container_width=True)
            st.image(result.overlay_rgb, caption="Baseline prediction overlay", use_container_width=True)
            st.download_button("Download metrics JSON",
                __import__("json").dumps({k:v for k,v in metrics.items()}, indent=2).encode(),
                "baseline_evaluation_metrics.json", "application/json")

        st.subheader("Optional supervised Random Forest baseline")
        st.write("This model learns pixel features from the labelled image pair. Its random pixel split is a development smoke test, not a reliable spatial holdout benchmark.")
        if st.button("Train and evaluate pixel classifier"):
            try:
                model, report, n_train, n_test = train_pixel_classifier(before, after, truth)
                st.success(f"Trained on {n_train:,} sampled pixels; held out {n_test:,} pixels.")
                st.json(report)
                pred = predict_pixel_mask(model, before, after)
                st.image(pred, caption="Random Forest predicted change mask", clamp=True, use_container_width=True)
                rf_metrics = evaluate_binary_mask(pred, truth)
                st.write("Full-image metrics (optimistic if same image was used for training):")
                st.json({k:rf_metrics[k] for k in ["precision","recall","f1","iou","accuracy"]})
                st.caption("For an honest model benchmark, split by different sites/date pairs and keep a separate untouched test set.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    except Exception as exc:
        st.error(f"Could not process evaluation inputs: {exc}")
else:
    st.info("Upload an earlier image, later image, and a matching labelled mask to run evaluation.")

with st.expander("How to create a labelled mask"):
    st.markdown("""
    - Create a black-and-white mask with an image editor or annotation tool.
    - White pixels (value > 0) mean true change; black pixels mean unchanged.
    - The mask must be exactly the same dimensions and spatial alignment as both images.
    - Save the provenance, labelling rules, and annotator decisions.
    - Evaluate on image pairs and locations not used during model fitting.
    """)
