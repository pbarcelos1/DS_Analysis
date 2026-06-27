import streamlit as st
from preprocess import get_data

if "df" not in st.session_state:
    df = get_data()
    if df is not None:
        st.session_state["df"] = df

if "df" not in st.session_state:
    st.info("Upload a CSV file above to get started.")
    st.stop()

df = st.session_state["df"]

st.title("Data Science Salary Dashboard")
st.markdown(
    f"Dataset: **{len(df):,} records** · "
    f"{int(df.work_year.min())}–{int(df.work_year.max())} · "
    f"{df.company_location.nunique()} countries"
)
st.markdown("---")
st.subheader("Select a Research Question")

rqs = [
    (
        "RQ1", "Salary Evolution", "📈",
        "How have Data Science salaries changed from 2020 to 2025?",
        "pages/rq1.py",
    ),
    (
        "RQ2", "Local vs. Foreign Employment", "🌐",
        "Do professionals at foreign companies earn more than those at local ones?",
        "pages/rq2.py",
    ),
    (
        "RQ3", "Large Companies by Country", "🏢",
        "Which countries pay senior professionals most in large companies?",
        "pages/rq3.py",
    ),
    (
        "RQ4", "Remote Work & Salary", "💻",
        "Does remote or hybrid work affect salary across experience levels?",
        "pages/rq4.py",
    ),
    (
        "RQ5", "Remote Work Trend", "📊",
        "How have remote work arrangements evolved post-COVID (2020–2025)?",
        "pages/rq5.py",
    ),
]

col1, col2, col3 = st.columns(3)
col4, col5, _ = st.columns(3)
containers = [col1, col2, col3, col4, col5]

for (rq_num, title, icon, question, path), col in zip(rqs, containers):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon} {rq_num}")
            st.markdown(f"**{title}**")
            st.caption(question)
            if st.button("Explore →", key=rq_num, use_container_width=True):
                st.switch_page(path)
