import streamlit as st
from visualizations import rq5_remote_evolution
from preprocess import render_country_filter

df_full = st.session_state.get("df")
if df_full is None:
    st.warning("No data loaded. Return to the Home page first.")
    st.stop()

st.title("RQ5 – Remote Work Trend Post-COVID")
st.markdown(
    "Tracks the year-by-year shift between on-site, hybrid, and remote arrangements "
    "from 2020 to 2025, revealing whether remote work became a lasting structural norm."
)

country_filter = render_country_filter(df_full)

df = df_full.copy()
if country_filter is not None:
    df = df[df["res_iso2"].isin(country_filter)]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

valid = df.dropna(subset=["remote_category"])
col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(valid):,}")
col2.metric("Remote share (latest year)", (
    f"{valid[valid.work_year == valid.work_year.max()]['remote_category'].eq('Remote').mean():.0%}"
))
col3.metric("Years covered", f"{int(valid.work_year.min())}–{int(valid.work_year.max())}")

st.plotly_chart(rq5_remote_evolution(df), use_container_width=True)
st.caption(
    "Each bar shows the proportion of job listings per work arrangement for that year. "
    "Records without remote_ratio data are excluded."
)
