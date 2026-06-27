#!/usr/bin/env python3
"""
Random Forest salary predictor with SHAP feature-importance analysis.

Uses the same ISO-2/ISO-3 mappings defined in preprocess.py (inlined here to
avoid the streamlit dependency that preprocess.py carries at import time).

Output: outputs/figures/shap_salary_prediction.png
"""

import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── ISO mappings (same data as preprocess.py) ─────────────────────────────────

_TO_ISO3 = {
    "AD": "AND", "AE": "ARE", "AL": "ALB", "AM": "ARM", "AR": "ARG", "AS": "ASM",
    "AT": "AUT", "AU": "AUS", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "BO": "BOL", "BR": "BRA", "BS": "BHS", "CA": "CAN", "CD": "COD",
    "CF": "CAF", "CH": "CHE", "CL": "CHL", "CN": "CHN", "CO": "COL",
    "CR": "CRI", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "DO": "DOM", "DZ": "DZA", "EC": "ECU", "EE": "EST", "EG": "EGY",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR", "GH": "GHA",
    "GI": "GIB", "GR": "GRC", "GT": "GTM", "HK": "HKG", "HN": "HND",
    "HR": "HRV", "HU": "HUN", "ID": "IDN", "IE": "IRL", "IL": "ISR",
    "IN": "IND", "IQ": "IRQ", "IR": "IRN", "IT": "ITA", "JM": "JAM",
    "JO": "JOR", "JP": "JPN", "KE": "KEN", "KR": "KOR", "LB": "LBN",
    "LS": "LSO", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MA": "MAR",
    "MD": "MDA", "MK": "MKD", "ML": "MLI", "MT": "MLT", "MU": "MUS",
    "MX": "MEX", "MY": "MYS", "NG": "NGA", "NL": "NLD", "NO": "NOR",
    "NZ": "NZL", "OM": "OMN", "PA": "PAN", "PE": "PER", "PH": "PHL",
    "PK": "PAK", "PL": "POL", "PR": "PRI", "PT": "PRT", "QA": "QAT",
    "RO": "ROU", "RS": "SRB", "RU": "RUS", "SA": "SAU", "SE": "SWE",
    "SG": "SGP", "SI": "SVN", "SK": "SVK", "SV": "SLV", "TH": "THA",
    "TR": "TUR", "TW": "TWN", "UA": "UKR", "US": "USA", "VE": "VEN",
    "VN": "VNM", "XK": "XKX", "ZA": "ZAF", "ZM": "ZMB",
}

_ISO2_TO_CONTINENT = {
    "DZ": "Africa",  "CD": "Africa",  "CF": "Africa",  "EG": "Africa",
    "GH": "Africa",  "KE": "Africa",  "LS": "Africa",  "MA": "Africa",
    "ML": "Africa",  "MU": "Africa",  "NG": "Africa",  "ZA": "Africa", "ZM": "Africa",
    "AE": "Asia",    "AM": "Asia",    "CN": "Asia",    "HK": "Asia",   "ID": "Asia",
    "IL": "Asia",    "IN": "Asia",    "IQ": "Asia",    "IR": "Asia",   "JP": "Asia",
    "JO": "Asia",    "KR": "Asia",    "LB": "Asia",    "MY": "Asia",   "OM": "Asia",
    "PH": "Asia",    "PK": "Asia",    "QA": "Asia",    "SA": "Asia",   "SG": "Asia",
    "TH": "Asia",    "TR": "Asia",    "TW": "Asia",    "VN": "Asia",
    "AD": "Europe",  "AL": "Europe",  "AT": "Europe",  "BA": "Europe", "BE": "Europe",
    "BG": "Europe",  "CH": "Europe",  "CY": "Europe",  "CZ": "Europe", "DE": "Europe",
    "DK": "Europe",  "EE": "Europe",  "ES": "Europe",  "FI": "Europe", "FR": "Europe",
    "GB": "Europe",  "GI": "Europe",  "GR": "Europe",  "HR": "Europe", "HU": "Europe",
    "IE": "Europe",  "IT": "Europe",  "LT": "Europe",  "LU": "Europe", "LV": "Europe",
    "MD": "Europe",  "MK": "Europe",  "MT": "Europe",  "NL": "Europe", "NO": "Europe",
    "PL": "Europe",  "PT": "Europe",  "RO": "Europe",  "RS": "Europe", "RU": "Europe",
    "SE": "Europe",  "SI": "Europe",  "SK": "Europe",  "UA": "Europe", "XK": "Europe",
    "BS": "North America", "CA": "North America", "CR": "North America",
    "DO": "North America", "GT": "North America", "HN": "North America",
    "JM": "North America", "MX": "North America", "PA": "North America",
    "PR": "North America", "SV": "North America", "US": "North America",
    "AR": "South America", "BO": "South America", "BR": "South America",
    "CL": "South America", "CO": "South America", "EC": "South America",
    "PE": "South America", "VE": "South America",
    "AS": "Oceania", "AU": "Oceania", "NZ": "Oceania",
}

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_PATH  = "data/fused_salaries.csv"
OUT_PATH   = "outputs/figures/shap_salary_prediction.png"

EXP_ORDER  = ["EN", "MI", "SE", "EX"]
SIZE_ORDER = ["S", "M", "L"]
TOP_JOBS   = 30
TOP_COUNTRIES = 20

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)

# ISO-3 — same as preprocess.load_csv
df["iso3_company"]   = df["company_location"].str.upper().map(_TO_ISO3)
df["iso3_residence"] = df["employee_residence"].str.upper().map(_TO_ISO3)

# Continent — derived via same _ISO2_TO_CONTINENT mapping
df["res_continent"]  = df["employee_residence"].map(_ISO2_TO_CONTINENT).fillna("Other")
df["comp_continent"] = df["company_location"].map(_ISO2_TO_CONTINENT).fillna("Other")

# Same-country flag (mirrors add_derived_columns logic)
df["same_country"] = (df["employee_residence"] == df["company_location"]).astype(int)

# ── Encode features ────────────────────────────────────────────────────────────

# Ordinal: experience_level (EN=0 < MI=1 < SE=2 < EX=3)
df["experience_level"] = df["experience_level"].map(
    {v: i for i, v in enumerate(EXP_ORDER)}
)

# Ordinal: company_size (S=0 < M=1 < L=2)
df["company_size"] = df["company_size"].map(
    {v: i for i, v in enumerate(SIZE_ORDER)}
)

# Nominal: employment_type
le_emp = LabelEncoder()
df["employment_type"] = le_emp.fit_transform(df["employment_type"].astype(str))

# Nominal: job_title — keep top-30, group the rest as "Other"
top_jobs = df["job_title"].value_counts().nlargest(TOP_JOBS).index
df["job_title"] = df["job_title"].where(df["job_title"].isin(top_jobs), "Other")
le_job = LabelEncoder()
df["job_title"] = le_job.fit_transform(df["job_title"].astype(str))

# Nominal: employee_residence — top-20 ISO-2 codes + "Other"
# Preserves the US vs. rest-of-world signal that continent grouping loses
top_res = df["employee_residence"].value_counts().nlargest(TOP_COUNTRIES).index
df["employee_residence"] = df["employee_residence"].where(
    df["employee_residence"].isin(top_res), "Other"
)
le_res = LabelEncoder()
df["employee_residence"] = le_res.fit_transform(df["employee_residence"].astype(str))

# Nominal: company_location — top-20 ISO-2 codes + "Other"
top_comp = df["company_location"].value_counts().nlargest(TOP_COUNTRIES).index
df["company_location"] = df["company_location"].where(
    df["company_location"].isin(top_comp), "Other"
)
le_comp = LabelEncoder()
df["company_location"] = le_comp.fit_transform(df["company_location"].astype(str))

# ── Train/test split ───────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "work_year":          "Work Year",
    "experience_level":   "Experience Level",
    "employment_type":    "Employment Type",
    "job_title":          "Job Title",
    "employee_residence": "Employee Country",
    "remote_ratio":       "Remote Ratio",
    "company_location":   "Company Country",
    "company_size":       "Company Size",
    "same_country":       "Same Country",
}

X = df[list(FEATURE_LABELS)].rename(columns=FEATURE_LABELS)
y = df["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model ──────────────────────────────────────────────────────────────────────

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE : ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R²  : {r2_score(y_test, y_pred):.3f}")

# ── SHAP ───────────────────────────────────────────────────────────────────────

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

def _shap_to_buffer(plot_type: str | None = None) -> io.BytesIO:
    """Render a shap.summary_plot into a BytesIO PNG buffer."""
    kwargs = dict(show=False)
    if plot_type:
        kwargs["plot_type"] = plot_type
    shap.summary_plot(shap_values, X_test, **kwargs)
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close("all")
    return buf

buf_beeswarm = _shap_to_buffer()
buf_bar      = _shap_to_buffer("bar")

# Combine the two plots side by side
img_bs  = mpimg.imread(buf_beeswarm)
img_bar = mpimg.imread(buf_bar)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle(
    "SHAP Feature Importance — Data Science Salary Prediction (USD)",
    fontsize=15, fontweight="bold", y=1.01,
)
ax1.imshow(img_bs);  ax1.axis("off"); ax1.set_title("Beeswarm — directional impact per observation", fontsize=11)
ax2.imshow(img_bar); ax2.axis("off"); ax2.set_title("Mean |SHAP| — overall feature importance", fontsize=11)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"SHAP plot saved → {OUT_PATH}")
