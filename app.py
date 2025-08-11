import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NPS Problems – FAST v3", layout="wide")
st.title("Monthly NPS Problems – Regional-wise (FAST v3)")

# ---------- Fixed column maps (your exact headers) ----------
JUNE_COLS = {
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
JULY_COLS = {
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
NEEDED = ["Created At","Feedback","Rating","Vertical","Courses","Region","Status","Is promoter","Is Detractor","No: of Responses"]

# ---------- Cached loaders (hash on file bytes so uploads are read once) ----------
@st.cache_data(show_spinner=False)
def _read_csv_bytes(file_bytes: bytes, usecols):
    return pd.read_csv(io.BytesIO(file_bytes), usecols=usecols, engine="pyarrow")

@st.cache_data(show_spinner=False)
def load_and_standardize_from_bytes(file_bytes: bytes, mapping: dict, month_label: str) -> pd.DataFrame:
    df = _read_csv_bytes(file_bytes, [mapping[k] for k in mapping])
    df = df.rename(columns={v: k for k, v in mapping.items()})
    df = df[df["Vertical"].notna() & df["Courses"].notna() & df["Region"].notna()].copy()

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Is promoter"] = pd.to_numeric(df["Is promoter"], errors="coerce").fillna(0).astype("int8")
    df["Is Detractor"] = pd.to_numeric(df["Is Detractor"], errors="coerce").fillna(0).astype("int8")
    df["No: of Responses"] = pd.to_numeric(df["No: of Responses"], errors="coerce").fillna(1).astype("int16")
    df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")

    for c in ["Vertical","Courses","Region","Status"]:
        df[c] = df[c].astype("category")

    for col in NEEDED:
        if col not in df.columns:
            df[col] = np.nan

    df["Month"] = month_label
    return df[NEEDED + ["Month"]]

# ---------- Compute all summaries ONCE (cached) ----------
def _nps_calc(g):
    r = g["Rating"]
    total = r.notna().sum()
    if total == 0: return np.nan
    prom = ((r >= 9) & (r <= 10)).sum()
    det  = ((r >= 0) & (r <= 6)).sum()
    return ((prom/total) - (det/total)) * 100.0

@st.cache_data(show_spinner=False)
def compute_all(june_bytes: bytes, july_bytes: bytes):
    june = load_and_standardize_from_bytes(june_bytes, JUNE_COLS, "June 2025")
    july = load_and_standardize_from_bytes(july_bytes, JULY_COLS, "July 2025")
    combined = pd.concat([june, july], ignore_index=True)

    # Problems = Detractors (precompute ONCE)
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
        .apply(_nps_calc).reset_index(name="NPS")
    )

    # Unique values for filters
    regions = ["All"] + sorted(combined["Region"].dropna().astype(str).unique())
    courses = ["All"] + sorted(combined["Courses"].dropna().astype(str).unique())
    statuses = ["All"] + sorted(combined["Status"].dropna().astype(str).unique())

    return problems_full, compare_full, nps_full, regions, courses, statuses

# ---------- UI: upload & precompute (once) ----------
c1, c2 = st.columns(2)
with c1:
    june_file = st.file_uploader("Upload **June** CSV", type=["csv"])
with c2:
    july_file = st.file_uploader("Upload **July** CSV", type=["csv"])

perf_lite = st.toggle("⚡ Lite mode (renders top rows, charts only on click)", value=True)

if not june_file or not july_file:
    st.info("Upload both files to continue.")
    st.stop()

problems_full, compare_full, nps_full, regions, courses, statuses = compute_all(june_file.getvalue(), july_file.getvalue())

# ---------- Filters (ONLY filter precomputed tables – instant) ----------
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
    if f_region != "All": m &= (df["Region"].astype(str) == f_region)
    if f_course != "All": m &= (df["Courses"].astype(str) == f_course)
    if f_status != "All": m &= (df["Status"].astype(str) == f_status)
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

# ---------- Charts (on demand) ----------
st.header("Charts")
if not perf_lite or st.button("Draw charts now"):
    topj = (problems_view[problems_view["Month"] == "July 2025"]
            .sort_values("Detractor_Count", ascending=False).head(10))
    if topj.empty:
        st.info("No July detractors for current filter.")
    else:
        fig = plt.figure()
        labels = (topj["Region"] + " | " + topj["Courses"]).astype(str).str.slice(0, 70)
        plt.barh(labels, topj["Detractor_Count"].values)
        plt.xlabel("Detractor Count"); plt.ylabel("Region | Course")
        plt.gca().invert_yaxis()
        st.pyplot(fig, use_container_width=True)

    # Simple trend (uses filtered, precomputed data)
    pv = (problems_view
          .pivot(index=["Region","Courses","Status"], columns="Month", values="Detractor_Count")
          .fillna(0).reset_index())
    if not pv.empty:
        june_val = pv.get("June 2025", pd.Series([0])).iloc[0] if "June 2025" in pv.columns else 0
        july_val = pv.get("July 2025", pd.Series([0])).iloc[0] if "July 2025" in pv.columns else 0
        fig2 = plt.figure(); plt.bar(["June 2025","July 2025"], [june_val, july_val])
        plt.ylabel("Detractor Count"); st.pyplot(fig2, use_container_width=True)
else:
    st.info("Charts skipped in ⚡ Lite mode.")
