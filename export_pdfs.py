"""Export all RQ visualizations as PDF files to outputs/figures/."""

import sys
import os

# Stub out streamlit so preprocess/visualizations can be imported without a server
import types

_captured_fig = None

_st_stub = types.ModuleType("streamlit")
_st_stub.cache_data = lambda fn=None, **kw: (fn if fn else lambda f: f)
_st_stub.warning = lambda *a, **kw: None
_st_stub.error = lambda *a, **kw: None
_st_stub.caption = lambda *a, **kw: None
_st_stub.columns = lambda *a, **kw: [types.SimpleNamespace(metric=lambda *a, **kw: None)] * 3
_st_stub.file_uploader = lambda *a, **kw: None

def _capture_plotly_chart(fig, **kw):
    global _captured_fig
    _captured_fig = fig

_st_stub.plotly_chart = _capture_plotly_chart
sys.modules["streamlit"] = _st_stub

from preprocess import get_data, _ISO2_TO_CONTINENT, CONTINENT_ORDER
from visualizations import (
    rq1_salary_evolution,
    rq2_same_vs_different_country,
    bubble_map,
    rq4_remote_salary,
    rq5_remote_evolution,
)

OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("Loading data…")
    df = get_data()
    print(f"  {len(df):,} records loaded.")

    year_min = int(df.work_year.min())
    year_max = int(df.work_year.max())
    df_rq3 = df[
        df.company_size.isin(["L"])
        & df.experience_level.isin(["SE"])
        & df.work_year.between(year_min, year_max)
        & df.iso3.notna()
    ]
    bubble_map(df_rq3, ["SE"], (year_min, year_max))
    rq3_fig_senior = _captured_fig

    df_rq3_en = df[
        df.company_size.isin(["L"])
        & df.experience_level.isin(["EN"])
        & df.work_year.between(year_min, year_max)
        & df.iso3.notna()
    ]
    bubble_map(df_rq3_en, ["EN"], (year_min, year_max))
    rq3_fig_entry = _captured_fig

    rq2_charts = []
    for continent in CONTINENT_ORDER:
        codes = {c for c, cont in _ISO2_TO_CONTINENT.items() if cont == continent}
        df_cont = df[df["res_iso2"].isin(codes)]
        if df_cont.empty:
            continue
        slug = continent.lower().replace(" ", "_")
        fig = rq2_same_vs_different_country(df_cont)
        fig.update_layout(title=f"RQ2 – {continent}: Median Salary by Work Location & Experience")
        rq2_charts.append((f"rq2_{slug}", fig))

    charts = (
        [("rq1_salary_evolution", rq1_salary_evolution(df))]
        + rq2_charts
        + [
            ("rq3_bubble_map_senior", rq3_fig_senior),
            ("rq3_bubble_map_entry", rq3_fig_entry),
            ("rq4_remote_salary", rq4_remote_salary(df)),
            ("rq5_remote_evolution", rq5_remote_evolution(df)),
        ]
    )

    for name, fig in charts:
        path = os.path.join(OUT_DIR, f"{name}.pdf")
        fig.write_image(path, format="pdf")
        print(f"  Saved {path}")

    print("Done.")


if __name__ == "__main__":
    main()
