import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NPS Problems – Robust v5", layout="wide")
st.title("Monthly NPS Problems – Regional-wise (Robust v5)")

# ---- Reference headers you expect in the files (we'll match them robustly) ----
JUNE_COLS_REF = {
    "Created At": "Created At",
    "Feedback": "3_Elevate കോഴ്സുകൾ കൂടുതൽ മെച്ചപ്പടുത്താൻ നിങ്ങളുടെ വിലയേറിയ അഭിപ്രായങ്ങൾ നൽകുക",
    "Rating": "4_നിങ്ങളുടെ Elevate കോഴ്സിനെ നിങ്ങളുടെ സുഹൃത്തുക്കൾക്ക്  നിർദേശിക്കാൻ നിങ്ങൾ എത്രമാത്രം തയ്യാറാണെന്ന്  1 മുതൽ 10 വരെ ഉള്ള റേറ്റിംഗ് നൽകി ഞങ്ങളെ അറിയിക്കുക .\n(10 = തീർച്ചയായും നിർദേശിക്കും, 1 = നിർദേശിക്കില്ല)",
    "Vertical": "Vertical",
    "Courses": "Courses",
    "Region": "Region",
    "Status": "Status",
    "Is promoter": "Is promoter",
    "Is Detractor": "Is Detractor",
    "No: of Responses": "No: of Responses",
}
JULY_COLS_REF = {
    "Created At": "Created At",
    "Feedback": "4_మీరు ఇచ్చే ఈ feedback మా courses ని improve చేసుకోడానికి help అవుతుంది.",
    "Rating": "3_Elevate courses ని మీ friends కి recommend చేస్తారా? \nOut of 10, elevate కి మీరు ఎంత rating ఇస్తారు. \n(10 = most likely; 1= least likely)",
    "Vertical": "Vertical",
    "Courses": "Courses",
    "Region": "Region",
    "Status": "Status",
    "Is promoter": "Is promoter",
    "Is Detractor": "Is Detractor",
    "No: of Responses": "No: of Responses",
}
NEEDED_ORDER = ["Created At","Feedback","Rating","Vertical","Courses","Region","Status","Is promoter","Is Detractor","No: of Responses"]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def _build_norm_map(cols):
    # map normalized -> original name
    m = {}
    for c in cols:
        m[_norm(c)] = c
    return m

@st.cache_data(show_spinner=False)
def _read_csv_bytes(bytes_data: bytes):
    # Try pyarrow first (fast), fallback to default
    try:
        return pd.read_csv(io.BytesIO(bytes_data), engine="pyarrow")
    except Exception:
        return pd.read_csv(io.BytesIO(bytes_data))

def _resolve_columns(df: pd.DataFrame, refs: dict) -> dict:
    """
    Return a dict {target_name: actual_column_in_df}
    Uses normalization; if exact normalized not found, tries substring match.
    Raises a clear error if any essential column cannot be mapped.
    """
    norm_map = _build_norm_map(df.columns)
    resolved = {}
    missing = []
    for target, ref in refs.items():
        ref_norm = _norm(ref)
        # 1) exact normalized match
        if ref_norm in norm_map:
            resolved[target] = norm_map[ref_norm]
            continue
        # 2) try relaxed substring search among normalized headers
        candidates = [orig for nrm, orig in norm_map.items() if ref_norm in nrm or nrm in ref_norm]
        if candidates:
            # pick the longest overlap candidate (heuristic)
            resolved[target] = candidates[0]
            continue
        missing.append((target, ref))
    if missing:
        # Build a helpful error message with available columns
        raise ValueError(
            "Could not match expected columns.\n"
            f"Missing: {missing}\n\n"
            f"Incoming columns: {list(df.columns)}"
        )
    return resolved

def _load_and_standardize(file_bytes: bytes, refs: dict, file_month_label: str) -> pd.DataFrame:
    raw = _read_csv_bytes(file_bytes)
    try:
        mapping = _resolve_columns(raw, refs)
    except Exception as e:
        st.error("Column mapping error.\n\n" + str(e))
        st.stop()

    df = raw.rename(columns=mapping)
    # Keep only rows that look like raw responses
    df = df[df["Vertical"].notna() & df["Courses"].notna() & df["Region"].notna()].copy()

    # Types
    df["Rating"] = pd.to_numeric(df.get("Rating"), errors="coerce")
    df["Is promoter"] = pd.to_numeric(df.get("Is promoter"), errors="coerce").fillna(0).astype("int8")
    df["Is Detractor"] = pd.to_numeric(df.get("Is Detractor"), errors="coerce").fillna(0).astype("int8")
    df["No: of Responses"] = pd.to_numeric(df.get("No: of Responses"), errors="coerce").fillna(1).astype("int16")
    df["Created At"] = pd.to_datetime(df.get("Created At"), errors="coerce")

    for c in ["Vertical","Courses","Region","Status"]:
        if c in df:
            df[c] = df[c].astype("category")

    # Friendly Month for legacy sections
    df["Month"] = "June 2025" if file_month_label == "2025-06" else "July 2025"
    # Fallback month from file label (for timestamp-derived section)
    df["FileMonth"] = file_month_label
    # Timestamp-derived Month (YYYY-MM)
    month_ts = df["Created At"].dt.to_period("M").astype(str).replace("NaT", pd.NA)
    df["Month_TS"] = month_ts.fillna(df["FileMonth"])

    # Ensure output column order
    for col in NEEDED_ORDER:
        if col not in df.columns:
            df[col] = np.nan

    return df[NEEDED_ORDER + ["Month","FileMonth","Month_TS"]]

def _nps_value(group):
    r = group["Rating"]
    total = r.notna().sum()
    if total == 0: return np.nan
    prom = ((r >= 9) & (r <= 10)).sum()
    det  = ((r >= 0) & (r <= 6)).sum()
    return ((prom/total) - (det/total)) * 100.0

def _nps_stats(group):
    r = group["Rating"].dropna()
    total = len(r)
    prom = ((r >= 9) & (r <= 10)).sum()
    det  = ((r >= 0) & (r <= 6)).sum()
    pas  = ((r >= 7) & (r <= 8)).sum()
    nps = ((prom/total) - (det/total)) * 100.0 if total else np.nan
    return pd.Series({"Responses": total, "Promoters": prom, "Passives": pas, "Detractors": det, "NPS": nps})

@st.cache_data(show_spinner=False)
def compute_all(june_bytes: bytes, july_bytes: bytes):
    june = _load_and_standardize(june_bytes, JUNE_COLS_REF, "2025-06")
    july = _load_and_standardize(july_bytes, JULY_COLS_REF, "2025-07")
    combined = pd.concat([june, july], ignore_index=True)

    # Problems = Detractors (precompute)
    problems = combined[combined["Is Detractor"] == 1].copy()
    problems_full = (
        problems.groupby(["Region","Courses","Status","Month"], dropna=False)
        .agg(
            Detractor_Count=("Is Detractor","sum"),
            Avg_Detractor_Rating=("Rating","mean"),
            Example_Comments=("Feedback", lambda x: "; ".join(x.dropna().astype(str).head(3)))
        ).reset_index()
    )
    compare_full = (
        problems_full.pivot(index=["Region","Courses","Status"], columns="Month", values="Detractor_Count")
        .fillna(0).reset_index()
    )
    if "June 2025" not in compare_full.columns: compare_full["June 2025"] = 0
    if "July 2025" not in compare_full.columns: compare_full["July 2025"] = 0
    compare_full["Δ Detractors (Jul - Jun)"] = compare_full["July 2025"] - compare_full["June 2025"]

    nps_full = (
        combined.groupby(["Region","Courses","Status","Month"], dropna=False)
        .apply(_nps_value).reset_index(name="NPS")
    )

    # Consecutive months via timestamp (YYYY-MM)
    overall_consec = (combined.groupby("Month_TS").apply(_nps_stats).reset_index().rename(columns={"Month_TS":"Month"}).sort_values("Month"))
    region_consec  = (combined.groupby(["Region","Month_TS"]).apply(_nps_stats).reset_index().rename(columns={"Month_TS":"Month"}).sort_values(["Region","Month"]))
    rcs_consec     = (combined.groupby(["Region","Courses","Status","Month_TS"]).apply(_nps_stats).reset_index().rename(columns={"Month_TS":"Month"}).sort_values(["Region","Courses","Status","Month"]))

    regions = ["All"] + sorted(combined["Region"].dropna().astype(str).unique())
    courses = ["All"] + sorted(combined["Courses"].dropna().astype(str).unique())
    statuses = ["All"] + sorted(combined["Status"].dropna().astype(str).unique())

    return problems_full, compare_full, nps_full, overall_consec, region_consec, rcs_consec, regions, courses, statuses

# ---------- Upload ----------
c1, c2 = st.columns(2)
with c1:
    june_file = st.file_uploader("Upload **June** CSV", type=["csv"])
with c2:
    july_file = st.file_uploader("Upload **July** CSV", type=["csv"])

perf_lite = st.toggle("⚡ Lite mode (render top rows; charts on click)", value=True)

if not june_file or not july_file:
    st.info("Upload both files to continue.")
    st.stop()

try:
    (problems_full, compare_full, nps_full,
     overall_consec, region_consec, rcs_consec,
     regions, courses, statuses) = compute_all(june_file.getvalue(), july_file.getvalue())
except Exception as e:
    st.error("Processing error:\n\n" + str(e))
    st.stop()

# ---------- Filters ----------
st.subheader("Filters")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    f_region = st.selectbox("Region", regions)
with fc2:
    f_course = st.selectbox("Course", courses)
with fc3:
    f_status = st.selectbox("Batch / Status", statuses)

def _mask(df):
    m = pd.Series(True, index=df.index)
    if "Region" in df.columns and f_region != "All": m &= (df["Region"].astype(str) == f_region)
    if "Courses" in df.columns and f_course != "All": m &= (df["Courses"].astype(str) == f_course)
    if "Status"  in df.columns and f_status  != "All": m &= (df["Status"].astype(str)  == f_status)
    return m

problems_view = problems_full[_mask(problems_full)]
compare_view  = compare_full[_mask(compare_full)]
nps_view      = nps_full[_mask(nps_full)]

# ---------- Tables ----------
st.header("Regional-wise Monthly NPS Problems (Detractors)")
st.dataframe(problems_view.head(50) if perf_lite else problems_view, use_container_width=True)

st.header("June vs July – Detractor Comparison")
st.dataframe(compare_view.head(50) if perf_lite else compare_view, use_container_width=True)

st.header("NPS by Month (Region/Course/Batch)")
st.dataframe(nps_view.head(50) if perf_lite else nps_view, use_container_width=True)

# ---------- Consecutive months NPS ----------
st.header("Consecutive Months NPS (from timestamps)")
st.subheader("Overall by Month")
st.dataframe(overall_consec.head(50) if perf_lite else overall_consec, use_container_width=True)
if not perf_lite or st.button("Draw overall NPS chart"):
    if not overall_consec.empty:
        st.line_chart(overall_consec.set_index("Month")[["NPS"]], use_container_width=True)

st.subheader("Region-wise by Month")
reg_view = overall_consec if f_region == "All" else region_consec[region_consec["Region"].astype(str) == f_region]
st.dataframe(reg_view.head(50) if perf_lite else reg_view, use_container_width=True)

st.subheader("Region–Course–Status by Month")
rcs_view = rcs_consec[_mask(rcs_consec)]
st.dataframe(rcs_view.head(50) if perf_lite else rcs_view, use_container_width=True)

# ---------- Downloads ----------
def _download(df, label):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    st.download_button(f"⬇️ Download {label}", buf.getvalue(),
                       file_name=f"{label.replace(' ','_').lower()}.csv",
                       mime="text/csv", use_container_width=True)

c3, c4, c5 = st.columns(3)
with c3: _download(problems_view, "regional_monthly_nps_problems")
with c4: _download(compare_view,  "june_vs_july_detractor_comparison")
with c5: _download(nps_view,      "nps_by_month_region_course_batch")

c6, c7, c8 = st.columns(3)
with c6: _download(overall_consec, "consecutive_months_overall_nps")
with c7: _download(region_consec,  "consecutive_months_region_nps")
with c8: _download(rcs_consec,     "consecutive_months_region_course_status_nps")
