#!/usr/bin/env python3
"""
Data Science Salary Visualization Pipeline
Research Questions RQ1–RQ5
Dataset: fused_salaries.csv (74,799 records, 2020–2025)

Run with:
    python src/create_rq_visualizations.py
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "data/fused_salaries.csv"

OUTPUT_PATHS = {
    "html": "outputs/figures/html",
    "png": "outputs/figures/png",
    "processed": "outputs/processed",
}

# Experience level codes → readable labels, ordered entry → executive
EXP_LABELS = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
EXP_ORDER = ["Entry-level", "Mid-level", "Senior", "Executive"]

# Remote ratio values → readable labels
REMOTE_LABELS = {0: "On-site", 50: "Hybrid", 100: "Remote"}
REMOTE_ORDER = ["On-site", "Hybrid", "Remote"]

# Consistent color maps used across charts
COLORS_COUNTRY = {
    "Same country": "#3A86FF",
    "Different country": "#FF6B6B",
}

COLORS_REMOTE = {
    "On-site": "#3A86FF",
    "Hybrid": "#FF9F1C",
    "Remote": "#6BCB77",
}

COLORS_EXP_LEVEL = {
    "Entry-level": "#4E9AF1",
    "Mid-level":   "#F1A54E",
    "Senior":      "#5AB55E",
    "Executive":   "#E05C5C",
}

# Job titles to highlight individually in RQ1
TARGET_JOB_TITLES = [
    "Data Scientist",
    "Data Engineer",
    "Data Analyst",
    "Machine Learning Engineer",
]

FIGURE_WIDTH = 1100
FIGURE_HEIGHT = 620
FONT_FAMILY = "Arial, sans-serif"
FONT_SIZE = 13


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def create_output_folders():
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    print("Output folders ready.")


def save_figure(fig: go.Figure, name: str) -> None:
    """Save a Plotly figure as both interactive HTML and static PNG."""
    html_path = os.path.join(OUTPUT_PATHS["html"], f"{name}.html")
    png_path = os.path.join(OUTPUT_PATHS["png"], f"{name}.png")

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"  HTML → {html_path}")

    try:
        fig.write_image(png_path, width=FIGURE_WIDTH, height=FIGURE_HEIGHT, scale=2)
        print(f"  PNG  → {png_path}")
    except Exception as exc:
        print(f"  PNG export failed: {exc}")
        print("  Fix: pip install -U kaleido  (requires plotly >= 6.1.1)")


def apply_base_layout(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    """Apply consistent layout defaults to every figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=FIGURE_WIDTH,
        height=kwargs.get("height", FIGURE_HEIGHT),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
        margin=dict(l=80, r=40, t=80, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Load raw dataset, validate columns, drop unusable rows, derive helper
    columns, and persist a clean copy for downstream reproducibility.
    """
    required_cols = [
        "work_year", "experience_level", "employment_type", "job_title",
        "salary_in_usd", "employee_residence", "remote_ratio",
        "company_location", "company_size",
    ]

    df = pd.read_csv(path)
    print(f"  Raw records: {len(df):,}")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Drop rows where core analysis variables are absent
    core_cols = ["salary_in_usd", "work_year", "experience_level", "job_title"]
    before = len(df)
    df = df.dropna(subset=core_cols)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with nulls in: {core_cols}")

    # ── Derived columns ──────────────────────────────────────────────────────

    # same_country: True only when employee_residence is known and matches employer country.
    # Rows with null employee_residence are marked NA so they don't pollute RQ2 comparisons.
    # Using pandas nullable Boolean dtype to avoid FutureWarning on mixed-type assignment.
    df["same_country"] = df["employee_residence"].eq(df["company_location"]).astype("boolean")
    df.loc[df["employee_residence"].isna(), "same_country"] = pd.NA
    df["same_country_label"] = df["same_country"].map(
        {True: "Same country", False: "Different country"}
    )

    # remote_category: readable label for the three remote_ratio values.
    # Rows with null remote_ratio (dataset records that lacked this field) stay NaN.
    df["remote_category"] = df["remote_ratio"].map(REMOTE_LABELS)

    # Human-readable experience label ordered from entry to executive
    df["exp_label"] = df["experience_level"].map(EXP_LABELS)

    out_path = os.path.join(OUTPUT_PATHS["processed"], "ds_salaries_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"  Processed data saved → {out_path}")
    print(f"  Final records: {len(df):,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RQ1 – Salary evolution over time
# ─────────────────────────────────────────────────────────────────────────────

def rq1_salary_evolution(df: pd.DataFrame) -> None:
    """
    Line chart of median salary by year.
    One overall market line + one line per key job title.
    Reveals whether the field shows continuous appreciation.
    """
    print("\n[RQ1] Salary evolution over time …")

    def yearly_median(data: pd.DataFrame, label: str) -> pd.DataFrame:
        agg = (
            data.groupby("work_year")
            .agg(median_salary=("salary_in_usd", "median"), n=("salary_in_usd", "count"))
            .reset_index()
        )
        agg["series"] = label
        return agg

    frames = [yearly_median(df, "All Data Science")]

    for title in TARGET_JOB_TITLES:
        sub = df[df["job_title"] == title]
        if not sub.empty:
            frames.append(yearly_median(sub, title))

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
    )

    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Year: %{x}<br>"
            "Median Salary: $%{y:,.0f}<br>"
            "Records: %{customdata[0]:,}"
            "<extra></extra>"
        ),
    )

    # Make the "All Data Science" overall line thicker and dashed for emphasis
    for trace in fig.data:
        if trace.name == "All Data Science":
            trace.line.width = 3.5
            trace.line.dash = "dot"
            trace.marker.size = 10

    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)"),
        hovermode="x unified",
        legend_title="Job Title",
    )

    apply_base_layout(
        fig,
        "RQ1 – Median Salary Evolution in Data Science Professions (2020–2025)",
    )

    save_figure(fig, "rq1_salary_evolution")


# ─────────────────────────────────────────────────────────────────────────────
# RQ2 – Same country vs. different country by experience level
# ─────────────────────────────────────────────────────────────────────────────

def rq2_same_vs_different_country(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: median salary by work-location type (same vs. foreign
    company) at each experience level.
    Quantifies whether international employment commands a salary premium.
    """
    print("\n[RQ2] Same vs. different country salary by experience level …")

    # Drop records without a known employee_residence (same_country undefined)
    df2 = df.dropna(subset=["employee_residence", "same_country_label"]).copy()

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
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Experience: %{x}<br>"
            "Median Salary: $%{y:,.0f}<br>"
            "Records: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)"),
        xaxis_title="Experience Level",
        legend_title="Work Location",
    )

    apply_base_layout(
        fig,
        "RQ2 – Median Salary: Same-Country vs. Foreign-Company Employment by Experience",
    )

    save_figure(fig, "rq2_same_vs_different_country")


# ─────────────────────────────────────────────────────────────────────────────
# RQ3 – Large companies: top countries by salary, all four experience levels
# ─────────────────────────────────────────────────────────────────────────────

def rq3_large_companies_country_experience(df: pd.DataFrame) -> None:
    """
    Horizontal grouped bar chart restricted to large companies (company_size == 'L').
    Shows median salary for all four experience levels (EN / MI / SE / EX) per country.
    Countries are ranked by SE+EX combined median salary (top 15) so the axis
    ordering reflects which geographies most reward seniority — while the four
    individual bars reveal the full salary gradient across career stages.
    """
    print("\n[RQ3] Large companies – top countries by experience level …")

    large = df[df["company_size"] == "L"].copy()

    # Keep only countries with at least 10 total records in large companies
    # to avoid median estimates based on very small, unrepresentative samples
    MIN_RECORDS = 10
    eligible_countries = (
        large.groupby("company_location")["salary_in_usd"]
        .count()
        .loc[lambda s: s >= MIN_RECORDS]
        .index
    )
    large = large[large["company_location"].isin(eligible_countries)]
    print(f"  Countries with ≥{MIN_RECORDS} records in large companies: {len(eligible_countries)}")

    # Median per country per individual experience level
    agg = (
        large.groupby(["company_location", "exp_label"])
        .agg(median_salary=("salary_in_usd", "median"), n=("salary_in_usd", "count"))
        .reset_index()
    )

    # Rank countries by SE+EX combined median to preserve the "most value
    # experienced professionals" framing from the research question
    se_ex_median = (
        large[large["experience_level"].isin(["SE", "EX"])]
        .groupby("company_location")["salary_in_usd"]
        .median()
        .reset_index(name="se_ex_median")
    )
    top15_countries = (
        se_ex_median.nlargest(min(15, len(se_ex_median)), "se_ex_median")["company_location"].tolist()
    )

    combined = agg[agg["company_location"].isin(top15_countries)].copy()

    # Ascending sort so the highest-paying country appears at the top of the chart
    sort_order = (
        se_ex_median[se_ex_median["company_location"].isin(top15_countries)]
        .sort_values("se_ex_median", ascending=True)["company_location"]
        .tolist()
    )

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
        category_orders={
            "company_location": sort_order,
            "exp_label": EXP_ORDER,
        },
        color_discrete_map=COLORS_EXP_LEVEL,
        custom_data=["n", "exp_label"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Country: %{y}<br>"
            "Median Salary: $%{x:,.0f}<br>"
            "Records: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)"),
        yaxis_title="Country (ISO code)",
        legend_title="Experience Level",
    )

    apply_base_layout(
        fig,
        "RQ3 – Top 15 Countries in Large Companies: Median Salary by Experience Level",
        height=720,
    )

    save_figure(fig, "rq3_large_companies_country_experience")


# ─────────────────────────────────────────────────────────────────────────────
# RQ4 – Remote work impact on salary by experience level
# ─────────────────────────────────────────────────────────────────────────────

def rq4_remote_salary_by_experience(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: median salary by remote work category at each
    experience level.
    Answers whether the salary effect of remote/hybrid work is uniform
    across seniority or concentrated at specific levels.
    """
    print("\n[RQ4] Remote work impact on salary by experience level …")

    # Exclude records with unknown remote_ratio
    df4 = df.dropna(subset=["remote_category"]).copy()

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
        category_orders={
            "exp_label": EXP_ORDER,
            "remote_category": REMOTE_ORDER,
        },
        color_discrete_map=COLORS_REMOTE,
        custom_data=["n", "remote_category"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Experience: %{x}<br>"
            "Median Salary: $%{y:,.0f}<br>"
            "Records: %{customdata[0]:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Median Salary (USD)"),
        xaxis_title="Experience Level",
        legend_title="Work Arrangement",
    )

    apply_base_layout(
        fig,
        "RQ4 – Median Salary by Remote Work Category and Experience Level",
    )

    save_figure(fig, "rq4_remote_salary_by_experience")


# ─────────────────────────────────────────────────────────────────────────────
# RQ5 – Remote work arrangement evolution post-COVID
# ─────────────────────────────────────────────────────────────────────────────

def rq5_remote_work_evolution(df: pd.DataFrame) -> None:
    """
    100 % stacked bar chart: yearly proportion of on-site / hybrid / remote.
    Captures the post-pandemic trajectory and whether remote work became a
    lasting norm or reverted toward pre-pandemic on-site dominance.
    """
    print("\n[RQ5] Remote work arrangement evolution (2020–2025) …")

    df5 = df.dropna(subset=["remote_category"]).copy()

    counts = (
        df5.groupby(["work_year", "remote_category"])
        .size()
        .reset_index(name="count")
    )

    totals = counts.groupby("work_year")["count"].sum().reset_index(name="total")
    counts = counts.merge(totals, on="work_year")
    counts["pct"] = (counts["count"] / counts["total"] * 100).round(1)

    fig = px.bar(
        counts,
        x="work_year",
        y="pct",
        color="remote_category",
        barmode="stack",
        labels={
            "work_year": "Year",
            "pct": "Proportion (%)",
            "remote_category": "Work Arrangement",
        },
        category_orders={
            "work_year": sorted(df5["work_year"].unique()),
            "remote_category": REMOTE_ORDER,
        },
        color_discrete_map=COLORS_REMOTE,
        custom_data=["count", "total", "remote_category"],
        text_auto=False,
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>"
            "Year: %{x}<br>"
            "Proportion: %{y:.1f}%<br>"
            "Records: %{customdata[0]:,} of %{customdata[1]:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        yaxis=dict(ticksuffix="%", range=[0, 100], title="Proportion (%)"),
        xaxis=dict(dtick=1, title="Year"),
        legend_title="Work Arrangement",
    )

    apply_base_layout(
        fig,
        "RQ5 – Evolution of Remote Work Arrangements in Data Science (2020–2025)",
    )

    save_figure(fig, "rq5_remote_work_evolution")


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """\
# Visualization Summary – Data Science Salary Analysis

**Dataset:** `data/fused_salaries.csv` · 74,799 records · 2020–2025
**Main salary variable:** `salary_in_usd` (median used for all comparisons)
**Generated by:** `src/create_rq_visualizations.py`

---

## RQ1 – Salary Adjustment Over the Years

**Objective:** Determine whether Data Science salaries show a continuous pattern
of appreciation from 2020 to 2025, both for the market overall and for key roles.

**Visualization:** Line chart – `outputs/figures/html/rq1_salary_evolution.html`
**Variables:** `work_year`, `salary_in_usd`, `job_title`
**Series:** Overall market + Data Scientist, Data Engineer, Data Analyst, Machine Learning Engineer

### Interpretation to be completed after visual inspection

---

## RQ2 – Same-Country vs. Foreign-Company Employment

**Objective:** Assess whether professionals employed by companies located in a
different country than their residence earn systematically higher or lower salaries,
and whether this effect varies with experience level.

**Visualization:** Grouped bar chart – `outputs/figures/html/rq2_same_vs_different_country.html`
**Variables:** `employee_residence`, `company_location` → `same_country_label`; `experience_level` → `exp_label`; `salary_in_usd`
**Groups:** Same country / Different country × Entry-level / Mid-level / Senior / Executive

### Interpretation to be completed after visual inspection

---

## RQ3 – Large Companies: Geographic Salary Differences by Experience

**Objective:** Identify which countries hosting large companies (company_size == 'L')
most reward experienced professionals, and compare this pattern against entry-level
salaries in the same locations.

**Visualization:** Horizontal grouped bar chart – `outputs/figures/html/rq3_large_companies_country_experience.html`
**Variables:** `company_size` (filtered to 'L'), `company_location`, `experience_level`, `salary_in_usd`
**Groups:** Top 15 countries by experienced median salary; Experienced (SE/EX) vs. Entry-level (EN)

### Interpretation to be completed after visual inspection

---

## RQ4 – Impact of Remote Work on Salary by Experience Level

**Objective:** Investigate whether remote or hybrid arrangements are associated
with salary differences, and whether this effect is uniform across experience levels
or concentrated in specific seniority groups.

**Visualization:** Grouped bar chart – `outputs/figures/html/rq4_remote_salary_by_experience.html`
**Variables:** `remote_ratio` → `remote_category` (On-site / Hybrid / Remote); `experience_level` → `exp_label`; `salary_in_usd`

### Interpretation to be completed after visual inspection

---

## RQ5 – Remote Work Evolution After COVID-19

**Objective:** Track the year-by-year shift in work arrangements (on-site, hybrid,
remote) from 2020 to 2025 to determine whether remote work became a lasting
structural change or reverted to pre-pandemic norms.

**Visualization:** 100% stacked bar chart – `outputs/figures/html/rq5_remote_work_evolution.html`
**Variables:** `work_year`, `remote_ratio` → `remote_category`
**Metric:** Proportion (%) of records per work arrangement category per year

### Interpretation to be completed after visual inspection
"""


def write_summary_report() -> None:
    out_path = "outputs/visualization_summary.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(REPORT_TEMPLATE)
    print(f"\n  Markdown report → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    create_output_folders()

    print(f"\nLoading dataset: {DATA_PATH}")
    df = load_and_preprocess(DATA_PATH)
    print(f"\nDataset ready: {len(df):,} records, {df['work_year'].nunique()} years.\n")

    rq1_salary_evolution(df)
    rq2_same_vs_different_country(df)
    rq3_large_companies_country_experience(df)
    rq4_remote_salary_by_experience(df)
    rq5_remote_work_evolution(df)
    write_summary_report()

    print("\nDone. All figures saved to outputs/figures/.")


if __name__ == "__main__":
    main()
