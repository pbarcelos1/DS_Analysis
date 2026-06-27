import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from preprocess import (
    SENIORITY_LABELS,
    SENIORITY_ORDER,
    EXP_ORDER,
    REMOTE_ORDER,
    COLORS_COUNTRY,
    COLORS_REMOTE,
    COLORS_EXP_LEVEL,
)

_TARGET_JOB_TITLES = [
    "Data Scientist",
    "Data Engineer",
    "Data Analyst",
    "Machine Learning Engineer",
]

_FONT = "Arial, sans-serif"
_BASE_LAYOUT = dict(
    font=dict(family=_FONT, size=13),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
    margin=dict(l=80, r=40, t=80, b=60),
)
_GRID = dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc")


def bubble_map(df: pd.DataFrame, selected_levels: list[str], year_range: tuple[int, int]) -> None:
    """Bubble map: median salary by country, filtered to countries with ≥10 jobs."""
    agg = (
        df.groupby("iso3")
        .agg(
            median_salary=("salary_in_usd", "median"),
            job_count=("salary_in_usd", "count"),
        )
        .reset_index()
    )
    agg = agg[agg.job_count > 2]
    agg["log_count"] = agg["job_count"] ** 0.7
    df = df[df.iso3.isin(agg.iso3)]

    if df.empty or agg.empty:
        st.warning(
            "No data matches the current filters. "
            "Try broadening the year range or selecting additional seniority levels."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total jobs", f"{len(df):,}")
    col2.metric("Median salary (USD)", f"${df.salary_in_usd.median():,.0f}")
    col3.metric("Countries represented", f"{agg.iso3.nunique()}")

    levels_str = ", ".join(
        SENIORITY_LABELS[k] for k in SENIORITY_ORDER if k in selected_levels
    )
    years_str = (
        f"{year_range[0]}–{year_range[1]}"
        if year_range[0] != year_range[1]
        else str(year_range[0])
    )
    chart_title = f"Median Data-Science Salary by Country  ·  {levels_str}  ·  {years_str}"

    fig = px.scatter_geo(
        agg,
        locations="iso3",
        locationmode="ISO-3",
        size="log_count",
        color="median_salary",
        size_max=50,
        color_continuous_scale="Plasma",
        projection="natural earth",
        title=chart_title,
        hover_name="iso3",
        hover_data={
            "median_salary": ":$,.0f",
            "job_count": ":,",
            "iso3": False,
            "log_count": False,
        },
        labels={
            "median_salary": "Median salary (USD)",
            "job_count": "Job count",
        },
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Median salary (USD)"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bubble size represents job count; color represents median annual salary in USD.")


# ── RQ chart functions (return go.Figure) ────────────────────────────────────

def rq1_salary_evolution(df: pd.DataFrame) -> go.Figure:
    """Line chart: median salary per year for overall market and key job titles."""

    def _yearly_median(data: pd.DataFrame, label: str) -> pd.DataFrame:
        return (
            data.groupby("work_year")
            .agg(median_salary=("salary_in_usd", "median"), n=("salary_in_usd", "count"))
            .reset_index()
            .assign(series=label)
        )

    frames = [_yearly_median(df, "All Data Science")]
    for title in _TARGET_JOB_TITLES:
        sub = df[df["job_title"] == title]
        if not sub.empty:
            frames.append(_yearly_median(sub, title))

    combined = pd.concat(frames, ignore_index=True)

    fig = px.line(
        combined,
        x="work_year",
        y="median_salary",
        color="series",
        markers=True,
        labels={
            "work_year": "Year",
            "median_salary": "Median Salary (USD)",
            "series": "Job Title",
        },
        category_orders={"work_year": sorted(df["work_year"].unique())},
        custom_data=["n", "series"],
        title="RQ1 – Median Salary Evolution in Data Science Professions (2020–2025)",
    )
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Year: %{x}<br>"
            "Median Salary: $%{y:,.0f}<br>"
            "Records: %{customdata[0]:,}<extra></extra>"
        ),
    )
    for trace in fig.data:
        if trace.name == "All Data Science":
            trace.line.width = 3.5
            trace.line.dash = "dot"
            trace.marker.size = 10

    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(dtick=1, **_GRID),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)", **_GRID),
        hovermode="x unified",
        legend_title="Job Title",
        height=540,
    )
    return fig


def rq2_same_vs_different_country(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: median salary by work-location type at each experience level."""
    df2 = df.dropna(subset=["employee_residence", "same_country_label"])
    agg = (
        df2.groupby(["exp_label", "same_country_label"])
        .agg(median_salary=("salary_in_usd", "median"), n=("salary_in_usd", "count"))
        .reset_index()
    )

    fig = px.bar(
        agg,
        x="exp_label",
        y="median_salary",
        color="same_country_label",
        barmode="group",
        labels={
            "exp_label": "Experience Level",
            "median_salary": "Median Salary (USD)",
            "same_country_label": "Work Location",
        },
        category_orders={
            "exp_label": EXP_ORDER,
            "same_country_label": ["Same country", "Different country"],
        },
        color_discrete_map=COLORS_COUNTRY,
        custom_data=["n", "same_country_label"],
        title="RQ2 – Median Salary: Same-Country vs. Foreign-Company Employment by Experience",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Experience: %{x}<br>"
            "Median Salary: $%{y:,.0f}<br>"
            "Records: %{customdata[0]:,}<extra></extra>"
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Experience Level", **_GRID),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)", **_GRID),
        legend_title="Work Location",
        height=500,
    )
    return fig


def rq3_large_companies(df: pd.DataFrame) -> go.Figure:
    """Horizontal grouped bar: top 15 countries in large companies by experience level."""
    large = df[df["company_size"] == "L"]
    eligible = (
        large.groupby("company_location")["salary_in_usd"]
        .count()
        .loc[lambda s: s >= 10]
        .index
    )
    large = large[large["company_location"].isin(eligible)]

    agg = (
        large.groupby(["company_location", "exp_label"])
        .agg(median_salary=("salary_in_usd", "median"), n=("salary_in_usd", "count"))
        .reset_index()
    )

    se_ex_median = (
        large[large["experience_level"].isin(["SE", "EX"])]
        .groupby("company_location")["salary_in_usd"]
        .median()
        .reset_index(name="se_ex_median")
    )
    top15 = se_ex_median.nlargest(min(15, len(se_ex_median)), "se_ex_median")["company_location"].tolist()
    sort_order = (
        se_ex_median[se_ex_median["company_location"].isin(top15)]
        .sort_values("se_ex_median", ascending=True)["company_location"]
        .tolist()
    )

    combined = agg[agg["company_location"].isin(top15)]

    fig = px.bar(
        combined,
        y="company_location",
        x="median_salary",
        color="exp_label",
        barmode="group",
        orientation="h",
        labels={
            "company_location": "Country (ISO code)",
            "median_salary": "Median Salary (USD)",
            "exp_label": "Experience Level",
        },
        category_orders={"company_location": sort_order, "exp_label": EXP_ORDER},
        color_discrete_map=COLORS_EXP_LEVEL,
        custom_data=["n", "exp_label"],
        title="RQ3 – Top 15 Countries in Large Companies: Median Salary by Experience Level",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Country: %{y}<br>"
            "Median Salary: $%{x:,.0f}<br>"
            "Records: %{customdata[0]:,}<extra></extra>"
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)", **_GRID),
        yaxis=dict(title="Country (ISO code)", **_GRID),
        legend_title="Experience Level",
        height=680,
    )
    return fig


def rq4_remote_salary(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: median salary by remote work category at each experience level."""
    df4 = df.dropna(subset=["remote_category"])
    agg = (
        df4.groupby(["exp_label", "remote_category"])
        .agg(median_salary=("salary_in_usd", "median"), n=("salary_in_usd", "count"))
        .reset_index()
    )

    fig = px.bar(
        agg,
        x="exp_label",
        y="median_salary",
        color="remote_category",
        barmode="group",
        labels={
            "exp_label": "Experience Level",
            "median_salary": "Median Salary (USD)",
            "remote_category": "Work Arrangement",
        },
        category_orders={"exp_label": EXP_ORDER, "remote_category": REMOTE_ORDER},
        color_discrete_map=COLORS_REMOTE,
        custom_data=["n", "remote_category"],
        title="RQ4 – Median Salary by Remote Work Category and Experience Level",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Experience: %{x}<br>"
            "Median Salary: $%{y:,.0f}<br>"
            "Records: %{customdata[0]:,}<extra></extra>"
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(title="Experience Level", **_GRID),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)", **_GRID),
        legend_title="Work Arrangement",
        height=500,
    )
    return fig


def rq5_remote_evolution(df: pd.DataFrame) -> go.Figure:
    """Pie chart per year: proportion of on-site / hybrid / remote."""
    df5 = df.dropna(subset=["remote_category"])
    years = sorted(df5["work_year"].unique())
    n_years = len(years)

    ncols = min(3, n_years)
    nrows = math.ceil(n_years / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "domain"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=[str(int(y)) for y in years],
    )

    for idx, year in enumerate(years):
        row = idx // ncols + 1
        col = idx % ncols + 1
        subset = df5[df5["work_year"] == year]
        counts = (
            subset.groupby("remote_category", sort=False)
            .size()
            .reindex(REMOTE_ORDER)
            .fillna(0)
            .reset_index()
        )
        counts.columns = ["remote_category", "count"]
        total = counts["count"].sum()

        fig.add_trace(
            go.Pie(
                labels=counts["remote_category"],
                values=counts["count"],
                marker_colors=[COLORS_REMOTE[cat] for cat in counts["remote_category"]],
                legendgroup="remote",
                showlegend=(idx == 0),
                name="",
                customdata=[[total]] * len(counts),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Proportion: %{percent}<br>"
                    f"Records: %{{value:,}} of {int(total):,}<extra></extra>"
                ),
                textinfo="percent",
                hole=0.3,
                rotation=90,
                direction="clockwise",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        title="RQ5 – Evolution of Remote Work Arrangements in Data Science (2020–2025)",
        legend_title="Work Arrangement",
        height=300 * nrows,
    )
    return fig
