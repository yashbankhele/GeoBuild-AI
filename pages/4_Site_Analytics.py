import streamlit as st
import pandas as pd
import plotly.express as px
from src.demo_data import load_sites, load_observations

st.title("📈 Site Analytics")
st.caption("Illustrative synthetic time series — separate from live Sentinel-2 data.")
sites, obs = load_sites(), load_observations()
chosen = st.multiselect("Sites", sites.site_name.tolist(), default=sites.site_name.tolist())
metric = st.selectbox("Metric", ["changed_pixel_pct", "image_quality"],
    format_func=lambda x: {"changed_pixel_pct":"Flagged pixel share (%)","image_quality":"Illustrative image quality score"}[x])
filtered = obs[obs.site_id.isin(sites[sites.site_name.isin(chosen)].site_id)]
if filtered.empty:
    st.info("Select at least one site.")
    st.stop()
label = "Flagged pixel share (%)" if metric == "changed_pixel_pct" else "Image quality score"
fig = px.line(filtered, x="date", y=metric, color="site_name", markers=True, title=f"{label} over time")
fig.update_layout(height=430)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(filtered.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Export filtered observations", filtered.to_csv(index=False).encode(), "site_observations.csv", "text/csv")
