import streamlit as st
from visualizations import rq1_salary_evolution
from preprocess import render_country_filter

df_full = st.session_state.get("df")
if df_full is None:
    st.warning("No data loaded. Return to the Home page first.")
    st.stop()

st.title("RQ1 – Salary Evolution Over the Years")
st.markdown(
    "Tracks whether Data Science salaries show continuous appreciation from 2020 to 2025, "
    "both for the overall market and for key roles."
)

year_min = int(df_full.work_year.min())
year_max = int(df_full.work_year.max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

country_filter = render_country_filter(df_full)

df = df_full[df_full.work_year.between(*year_range)]
if country_filter is not None:
    df = df[df["res_iso2"].isin(country_filter)]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(df):,}")
col2.metric("Median salary (USD)", f"${df.salary_in_usd.median():,.0f}")
col3.metric("Years", f"{year_range[0]}–{year_range[1]}")

st.plotly_chart(rq1_salary_evolution(df), use_container_width=True)
st.caption(
    "Dashed line = overall market median. Solid lines = individual job titles. "
    "Hover for exact values."
)
