import streamlit as st
from visualizations import rq2_same_vs_different_country
from preprocess import render_country_filter

df_full = st.session_state.get("df")
if df_full is None:
    st.warning("No data loaded. Return to the Home page first.")
    st.stop()

st.title("RQ2 – Local vs. Foreign Employment")
st.markdown(
    "Compares median salaries for professionals employed by a company in their own country "
    "versus those working for a company abroad, broken down by experience level."
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

valid = df.dropna(subset=["same_country_label"])
col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(valid):,}")
col2.metric("Same-country jobs", f"{(valid.same_country_label == 'Same country').sum():,}")
col3.metric("Foreign-company jobs", f"{(valid.same_country_label == 'Different country').sum():,}")

st.plotly_chart(rq2_same_vs_different_country(df), use_container_width=True)
st.caption("Rows without employee_residence data are excluded from this chart.")
