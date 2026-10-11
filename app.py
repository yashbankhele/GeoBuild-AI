import streamlit as st
import plotly.express as px
from src.demo_data import load_sites, load_observations

st.set_page_config(page_title="GeoBuild AI", page_icon="🏗️", layout="wide")
sites = load_sites()
obs = load_observations()

st.title("🏗️ GeoBuild AI")
st.subheader("Satellite imagery · geospatial analysis · change monitoring")
st.caption("Portfolio Edition v3.0 | Demo data is labelled clearly; satellite access is available on a separate page.")

with st.sidebar:
    st.header("Project navigation")
    st.markdown("""
    - Overview
    - Image comparison
    - Sentinel-2 Explorer
    - Labelled model evaluation
    - Site analytics
    - Data quality & guide
    """)
    st.info("Satellite search requires internet access. Demo pages work offline.")

site_options = sites["site_name"].tolist()
selected_site = st.selectbox("Select demonstration site", site_options)
site = sites.loc[sites.site_name == selected_site].iloc[0]
site_obs = obs.loc[obs.site_id == site.site_id].sort_values("date")

st.markdown("### Portfolio overview")
a, b, c, d = st.columns(4)
a.metric("Demo sites", len(sites))
b.metric("Observations", len(obs))
c.metric("Latest flagged-pixel share", f"{site_obs.iloc[-1]['changed_pixel_pct']:.1f}%")
d.metric("Real satellite module", "Available", help="Connects to Microsoft Planetary Computer STAC when internet and data coverage are available.")

left, right = st.columns([1.3, 1])
with left:
    fig = px.line(site_obs, x="date", y="changed_pixel_pct", markers=True,
                  title="Illustrative change indicator by observation date",
                  labels={"date": "Observation date", "changed_pixel_pct": "Flagged pixels (%)"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("#### Selected demo site")
    st.write(f"**ID:** `{site['site_id']}`")
    st.write(f"**Label:** {site['area_label']}")
    st.write(f"**Illustrative coordinates:** {site['latitude']:.4f}, {site['longitude']:.4f}")
    st.write(f"**Data note:** {site['data_note']}")

st.subheader("Recent observations")
st.dataframe(site_obs.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download observations CSV", site_obs.to_csv(index=False).encode(),
                   f"{site['site_id']}_observations.csv", "text/csv")

st.warning("The site time-series values are synthetic. They are not outputs from the satellite module and must not be presented as real monitoring results.")
