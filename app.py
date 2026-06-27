import streamlit as st

st.set_page_config(
    page_title="DS Salary Dashboard",
    page_icon="📊",
    layout="wide",
)

pg = st.navigation(
    {
        "": [
            st.Page("pages/home.py", title="Home", icon="🏠", default=True),
        ],
        "Research Questions": [
            st.Page("pages/rq1.py", title="RQ1 – Salary Evolution", icon="📈"),
            st.Page("pages/rq2.py", title="RQ2 – Local vs. Foreign", icon="🌐"),
            st.Page("pages/rq3.py", title="RQ3 – Large Companies", icon="🏢"),
            st.Page("pages/rq4.py", title="RQ4 – Remote Work & Salary", icon="💻"),
            st.Page("pages/rq5.py", title="RQ5 – Remote Work Trend", icon="📊"),
        ],
    }
)
pg.run()
