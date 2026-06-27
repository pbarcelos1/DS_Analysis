import sys
import pandas as pd
import streamlit as st

_TO_ISO3 = {
    # ISO-2 → ISO-3
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
    # Full country names present in this dataset → ISO-3
    "ALGERIA": "DZA", "ANDORRA": "AND", "ARMENIA": "ARM",
    "BAHAMAS": "BHS", "BOSNIA AND HERZEGOVINA": "BIH",
    "CENTRAL AFRICAN REPUBLIC": "CAF", "CROATIA": "HRV",
    "ECUADOR": "ECU", "ESTONIA": "EST", "GHANA": "GHA",
    "GIBRALTAR": "GIB", "HONDURAS": "HND", "HONG KONG": "HKG",
    "IRAN": "IRN", "IRAQ": "IRQ", "LATVIA": "LVA", "LEBANON": "LBN",
    "LITHUANIA": "LTU", "LUXEMBOURG": "LUX", "MALTA": "MLT",
    "MAURITIUS": "MUS", "MOLDOVA": "MDA", "PUERTO RICO": "PRI",
    "QATAR": "QAT", "SLOVENIA": "SVN",
}

# ISO-2 → continent
_ISO2_TO_CONTINENT: dict[str, str] = {
    "DZ": "Africa", "CD": "Africa", "CF": "Africa", "EG": "Africa",
    "GH": "Africa", "KE": "Africa", "LS": "Africa", "MA": "Africa",
    "ML": "Africa", "MU": "Africa", "NG": "Africa", "ZA": "Africa", "ZM": "Africa",
    "AE": "Asia", "AM": "Asia", "CN": "Asia", "HK": "Asia", "ID": "Asia",
    "IL": "Asia", "IN": "Asia", "IQ": "Asia", "IR": "Asia", "JP": "Asia",
    "JO": "Asia", "KR": "Asia", "LB": "Asia", "MY": "Asia", "OM": "Asia",
    "PH": "Asia", "PK": "Asia", "QA": "Asia", "SA": "Asia", "SG": "Asia",
    "TH": "Asia", "TR": "Asia", "TW": "Asia", "VN": "Asia",
    "AD": "Europe", "AL": "Europe", "AT": "Europe", "BA": "Europe",
    "BE": "Europe", "BG": "Europe", "CH": "Europe", "CY": "Europe",
    "CZ": "Europe", "DE": "Europe", "DK": "Europe", "EE": "Europe",
    "ES": "Europe", "FI": "Europe", "FR": "Europe", "GB": "Europe",
    "GI": "Europe", "GR": "Europe", "HR": "Europe", "HU": "Europe",
    "IE": "Europe", "IT": "Europe", "LT": "Europe", "LU": "Europe",
    "LV": "Europe", "MD": "Europe", "MK": "Europe", "MT": "Europe",
    "NL": "Europe", "NO": "Europe", "PL": "Europe", "PT": "Europe",
    "RO": "Europe", "RS": "Europe", "RU": "Europe", "SE": "Europe",
    "SI": "Europe", "SK": "Europe", "UA": "Europe", "XK": "Europe",
    "BS": "North America", "CA": "North America", "CR": "North America",
    "DO": "North America", "GT": "North America", "HN": "North America",
    "JM": "North America", "MX": "North America", "PA": "North America",
    "PR": "North America", "SV": "North America", "US": "North America",
    "AR": "South America", "BO": "South America", "BR": "South America",
    "CL": "South America", "CO": "South America", "EC": "South America",
    "PE": "South America", "VE": "South America",
    "AS": "Oceania", "AU": "Oceania", "NZ": "Oceania",
}

# ISO-2 → readable country name
_ISO2_TO_NAME: dict[str, str] = {
    "AD": "Andorra", "AE": "UAE", "AL": "Albania", "AM": "Armenia",
    "AR": "Argentina", "AS": "American Samoa", "AT": "Austria",
    "AU": "Australia", "BA": "Bosnia & Herz.", "BE": "Belgium",
    "BG": "Bulgaria", "BO": "Bolivia", "BR": "Brazil", "BS": "Bahamas",
    "CA": "Canada", "CD": "DR Congo", "CF": "Central African Rep.",
    "CH": "Switzerland", "CL": "Chile", "CN": "China", "CO": "Colombia",
    "CR": "Costa Rica", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "DO": "Dominican Rep.",
    "DZ": "Algeria", "EC": "Ecuador", "EE": "Estonia", "EG": "Egypt",
    "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "GH": "Ghana", "GI": "Gibraltar", "GR": "Greece", "GT": "Guatemala",
    "HK": "Hong Kong", "HN": "Honduras", "HR": "Croatia", "HU": "Hungary",
    "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IN": "India",
    "IQ": "Iraq", "IR": "Iran", "IT": "Italy", "JM": "Jamaica",
    "JO": "Jordan", "JP": "Japan", "KE": "Kenya", "KR": "South Korea",
    "LB": "Lebanon", "LS": "Lesotho", "LT": "Lithuania", "LU": "Luxembourg",
    "LV": "Latvia", "MA": "Morocco", "MD": "Moldova", "MK": "N. Macedonia",
    "ML": "Mali", "MT": "Malta", "MU": "Mauritius", "MX": "Mexico",
    "MY": "Malaysia", "NG": "Nigeria", "NL": "Netherlands", "NO": "Norway",
    "NZ": "New Zealand", "OM": "Oman", "PA": "Panama", "PE": "Peru",
    "PH": "Philippines", "PK": "Pakistan", "PL": "Poland", "PR": "Puerto Rico",
    "PT": "Portugal", "QA": "Qatar", "RO": "Romania", "RS": "Serbia",
    "RU": "Russia", "SA": "Saudi Arabia", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SV": "El Salvador", "TH": "Thailand",
    "TR": "Turkey", "TW": "Taiwan", "UA": "Ukraine", "US": "United States",
    "VE": "Venezuela", "VN": "Vietnam", "XK": "Kosovo", "ZA": "South Africa",
    "ZM": "Zambia",
}

CONTINENT_ORDER = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]

# Full country name (uppercase) → ISO-2, used for employee_residence normalization
_FULLNAME_TO_ISO2: dict[str, str] = {
    name.upper(): iso2 for iso2, name in _ISO2_TO_NAME.items()
}
_FULLNAME_TO_ISO2.update({
    "BOSNIA AND HERZEGOVINA": "BA",
    "CENTRAL AFRICAN REPUBLIC": "CF",
})

# Used for the sidebar filter and the bubble map
SENIORITY_LABELS = {
    "EN": "Entry (EN)",
    "MI": "Mid (MI)",
    "SE": "Senior (SE)",
    "EX": "Executive (EX)",
}
SENIORITY_ORDER = ["EN", "MI", "SE", "EX"]

# Used as derived column values in the RQ charts
EXP_LABELS = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
EXP_ORDER = ["Entry-level", "Mid-level", "Senior", "Executive"]

REMOTE_LABELS = {0: "On-site", 50: "Hybrid", 100: "Remote"}
REMOTE_ORDER = ["On-site", "Hybrid", "Remote"]

COLORS_COUNTRY = {"Same country": "#3A86FF", "Different country": "#FF6B6B"}
COLORS_REMOTE = {"On-site": "#3A86FF", "Hybrid": "#FF9F1C", "Remote": "#6BCB77"}
COLORS_EXP_LEVEL = {
    "Entry-level": "#4E9AF1",
    "Mid-level": "#F1A54E",
    "Senior": "#5AB55E",
    "Executive": "#E05C5C",
}


def _to_res_iso2(val) -> str | None:
    """Normalize an employee_residence value (ISO-2 or full name) to ISO-2."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if s in _FULLNAME_TO_ISO2:
        return _FULLNAME_TO_ISO2[s]
    if s in _ISO2_TO_CONTINENT:
        return s
    return None


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["iso3"] = df["company_location"].str.upper().map(_TO_ISO3)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["same_country"] = df["employee_residence"].eq(df["company_location"]).astype("boolean")
    df.loc[df["employee_residence"].isna(), "same_country"] = pd.NA
    df["same_country_label"] = df["same_country"].map(
        {True: "Same country", False: "Different country"}
    )
    df["remote_category"] = df["remote_ratio"].map(REMOTE_LABELS)
    df["exp_label"] = df["experience_level"].map(EXP_LABELS)
    df["res_iso2"] = df["employee_residence"].map(_to_res_iso2)
    return df


def render_country_filter(df_full: pd.DataFrame, key_prefix: str = "cf") -> set[str] | None:
    """
    Renders continent/country checkboxes in the sidebar.

    Each continent has a "select all" checkbox. When unchecked an expander
    appears with individual country checkboxes.

    Returns the set of selected ISO-2 employee-residence codes, or None when
    every available country is selected (caller can skip the filter entirely).
    """
    available = sorted(
        c for c in df_full["res_iso2"].dropna().unique()
        if c in _ISO2_TO_CONTINENT
    )
    if not available:
        return None

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter by Country**")

    col_sel, col_desel = st.sidebar.columns(2)
    if col_sel.button("Select all", key=f"{key_prefix}_btn_sel"):
        for continent in CONTINENT_ORDER:
            st.session_state[f"{key_prefix}_cont_{continent}"] = True
        for code in available:
            if f"{key_prefix}_{code}" in st.session_state:
                st.session_state[f"{key_prefix}_{code}"] = True
    if col_desel.button("Deselect all", key=f"{key_prefix}_btn_desel"):
        for continent in CONTINENT_ORDER:
            st.session_state[f"{key_prefix}_cont_{continent}"] = False
        for code in available:
            if f"{key_prefix}_{code}" in st.session_state:
                st.session_state[f"{key_prefix}_{code}"] = False

    selected: set[str] = set()
    fully_selected = True

    for continent in CONTINENT_ORDER:
        codes = sorted(
            [c for c in available if _ISO2_TO_CONTINENT[c] == continent],
            key=lambda c: _ISO2_TO_NAME.get(c, c),
        )
        if not codes:
            continue

        cont_on = st.sidebar.checkbox(
            f"{continent} ({len(codes)})",
            value=True,
            key=f"{key_prefix}_cont_{continent}",
        )

        if cont_on:
            selected.update(codes)
        else:
            fully_selected = False
            with st.sidebar.expander("Countries", expanded=True):
                for code in codes:
                    if st.checkbox(
                        _ISO2_TO_NAME.get(code, code),
                        value=False,
                        key=f"{key_prefix}_{code}",
                    ):
                        selected.add(code)

    return None if fully_selected else selected


_DEFAULT_PATHS = ["data/fused_salaries.csv", "data.csv"]


def get_data() -> pd.DataFrame | None:
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    if cli_path:
        try:
            return add_derived_columns(load_csv(cli_path))
        except FileNotFoundError:
            st.error(f"File not found: {cli_path}")
            return None
    for p in _DEFAULT_PATHS:
        try:
            return add_derived_columns(load_csv(p))
        except FileNotFoundError:
            continue
    uploaded = st.file_uploader("Upload salaries CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df["iso3"] = df["company_location"].str.upper().map(_TO_ISO3)
        return add_derived_columns(df)
    return None
