import streamlit as st
from preprocess import SENIORITY_LABELS, SENIORITY_ORDER
from visualizations import bubble_map

df_full = st.session_state.get("df")
if df_full is None:
    st.warning("No data loaded. Return to the Home page first.")
    st.stop()

st.title("RQ3 – Large Companies: Geographic Salary by Experience")
st.markdown(
    "Median salary by country, filtered by company size. "
    "Bubble size represents job count; color represents median annual salary in USD."
)

st.sidebar.header("Filters")

selected_levels = st.sidebar.segmented_control(
    "Seniority",
    options=SENIORITY_ORDER,
    format_func=lambda k: SENIORITY_LABELS[k],
    selection_mode="multi",
    default=["SE"],
)

selected_sizes = st.sidebar.segmented_control(
    "Company size",
    options=["S", "M", "L"],
    format_func=lambda k: {"S": "Small (S)", "M": "Medium (M)", "L": "Large (L)"}[k],
    selection_mode="multi",
    default=["L"],
)

year_min = int(df_full.work_year.min())
year_max = int(df_full.work_year.max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

if not selected_levels:
    st.warning("Select at least one seniority level.")
    st.stop()

if not selected_sizes:
    st.warning("Select at least one company size.")
    st.stop()

df = df_full[
    df_full.company_size.isin(selected_sizes)
    & df_full.experience_level.isin(selected_levels)
    & df_full.work_year.between(*year_range)
    & df_full.iso3.notna()
]

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

bubble_map(df, selected_levels, year_range)
