import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NPS Problems – Fast Mode", layout="wide")
st.title("Monthly NPS Problems – Regional-wise (FAST)")

# -----------------------------
# Exact column names in your CSVs (no auto-detect for speed)
# -----------------------------
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

NEEDED = [
    "Created At", "Feedback", "Rating", "Vertical", "Courses", "Region",
    "Status", "Is promoter", "Is Detractor", "No: of Responses"
]

# -----------------------------
# Cached helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_standardize(file, mapping: dict, month_label: str) -> pd.DataFrame:
    usecols = [mapping[k] for k in mapping]
    df = pd.read_csv(file, usecols=usecols)
    df = df.rename(columns={v: k for k, v in mapping.items()})

    # Keep only raw response rows
    df = df[df["Vertical"].notna() & df["Courses"].notna() & df["Region"].notna()].copy()

    # Types
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Is promoter"] = pd.to_numeric(df["Is promoter"], errors="coerce").fillna(0).astype("int8")
    df["Is Detractor"] = pd.to_numeric(df["Is Detractor"], errors="coerce").fillna(0).astype("int8")
    df["No: of Responses"] = pd.to_numeric(df["No: of Responses"], errors="coerce").fillna(1).astype("int16")
    df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")

    # Memory-friendly categories
    for c in ["Vertical", "Courses", "Region", "Status"]:
        df[c] = df[c].astype("category")

    for col in NEEDED:
        if col not in df.columns:
            df[col] = np.nan

    df["Month"] = month_label
    return df[NEEDED + ["Month"]]

@st.cache_data(show_spinner=False)
def build_summaries(combined: pd.DataFrame):
    # Problems = Detractors
    problems = combined[combined["Is Detractor"] == 1].copy()

    problems_summary = (
        problems.groupby(["Region", "Courses", "Status", "Month"], dropna=False)
        .agg(
            Detractor_Count=("Is Detractor", "sum"),
            Avg_Detractor_Rating=("Rating", "mean"),
            Example_Comments=("Feedback", lambda x: "; ".join(x.dropna().astype(str).head(3)))
        )
        .reset_index()
        .sort_values(["Region", "Courses", "Status", "Month"])
    )

    # June vs July comparison
    compare = (
        problems_summary
        .pivot(index=["Region", "Courses", "Status"], columns="Month", values="Detractor_Count")
        .fillna(0)
        .reset_index()
    )
    if "June 2025" not in compare.columns:
        compare["June 2025"] = 0
    if "July 2025" not in compare.columns:
        compare["July 2025"] = 0
    compare["Δ Detractors (Jul - Jun)"] = compare["July 2025"] - compare["June 2025"]

    # NPS calc
    def nps_calc(g):
        r = g["Rating"]
        total = r.notna().sum()
        if total == 0:
            return np.nan
        prom = ((r >= 9) & (r <= 10)).sum()
        det = ((r >= 0) & (r <= 6)).sum()
        return ((prom / total) - (det / total)) * 100.0

    nps_table = (
        combined.groupby(["Region", "Courses", "Status", "Month"], dropna=False)
        .apply(nps_calc)
        .reset_index(name="NPS")
        .sort_values(["Region", "Courses", "Status", "Month"])
    )

    return problems_summary, compare, nps_table

def download_csv(df: pd.DataFrame, label: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        f"⬇️ Download {label}",
        buf.getvalue(),
        file_name=f"{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )

# -----------------------------
# Inputs
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    june_file = st.file_uploader("Upload **June** CSV", type=["csv"])
with c2:
    july_file = st.file_uploader("Upload **July** CSV", type=["csv"])

perf_lite = st.toggle("⚡ Lite mode (render fewer rows & draw charts on click)", value=True)

if not june_file or not july_file:
    st.info("Upload both files to continue.")
    st.stop()

june = load_and_standardize(june_file, JUNE_COLS, "June 2025")
july = load_and_standardize(july_file, JULY_COLS, "July 2025")
combined = pd.concat([june, july], ignore_index=True)

# -----------------------------
# Filters
# -----------------------------
st.subheader("Filters")
fc1, fc2, fc3 = st.columns(3)
regions = ["All"] + sorted(combined["Region"].dropna().astype(str).unique())
courses = ["All"] + sorted(combined["Courses"].dropna().astype(str).unique())
statuses = ["All"] + sorted(combined["Status"].dropna().astype(str).unique())

with fc1:
    f_region = st.selectbox("Region", regions)
with fc2:
    f_course = st.selectbox("Course", courses)
with fc3:
    f_status = st.selectbox("Batch / Status", statuses)

f = combined
if f_region != "All":
    f = f[f["Region"].astype(str) == f_region]
if f_course != "All":
    f = f[f["Courses"].astype(str) == f_course]
if f_status != "All":
    f = f[f["Status"].astype(str) == f_status]

problems_summary, compare_table, nps_table = build_summaries(f)

# -----------------------------
# Tables
# -----------------------------
st.header("Regional-wise Monthly NPS Problems (Detractors)")
st.dataframe(problems_summary.head(100) if perf_lite else problems_summary, use_container_width=True)
download_csv(problems_summary, "regional_monthly_nps_problems")

st.header("June vs July – Detractor Comparison")
st.dataframe(compare_table.head(100) if perf_lite else compare_table, use_container_width=True)
download_csv(compare_table, "june_vs_july_detractor_comparison")

st.header("NPS by Month (Region/Course/Batch)")
st.dataframe(nps_table.head(100) if perf_lite else nps_table, use_container_width=True)
download_csv(nps_table, "nps_by_month_region_course_batch")

# -----------------------------
# Charts (on demand to keep it fast)
# -----------------------------
st.header("Charts")
if not perf_lite or st.button("Draw charts now"):
    # Top July problem courses
    topj = (
        problems_summary[problems_summary["Month"] == "July 2025"]
        .sort_values("Detractor_Count", ascending=False)
        .head(10)
    )
    if topj.empty:
        st.info("No July detractors for the current filter.")
    else:
        fig = plt.figure()
        labels = (topj["Region"] + " | " + topj["Courses"]).astype(str).str.slice(0, 70)
        plt.barh(labels, topj["Detractor_Count"].values)
        plt.xlabel("Detractor Count")
        plt.ylabel("Region | Course")
        plt.gca().invert_yaxis()
        st.pyplot(fig, use_container_width=True)

    # Trend for a selected triple
    st.subheader("Trend (June vs July) for a Selected Group")
    g1, g2, g3 = st.columns(3)
    with g1:
        t_region = st.selectbox("Trend Region", regions, key="t_r")
    with g2:
        t_course = st.selectbox("Trend Course", courses, key="t_c")
    with g3:
        t_status = st.selectbox("Trend Status", statuses, key="t_s")

    tf = problems_summary.copy()
    if t_region != "All":
        tf = tf[tf["Region"].astype(str) == t_region]
    if t_course != "All":
        tf = tf[tf["Courses"].astype(str) == t_course]
    if t_status != "All":
        tf = tf[tf["Status"].astype(str) == t_status]

    if not tf.empty:
        pv = (
            tf.pivot(index=["Region", "Courses", "Status"], columns="Month", values="Detractor_Count")
            .fillna(0)
            .reset_index()
        )
        if not pv.empty:
            june_val = pv.iloc[0]["June 2025"] if "June 2025" in pv.columns else 0
            july_val = pv.iloc[0]["July 2025"] if "July 2025" in pv.columns else 0
            fig2 = plt.figure()
            plt.bar(["June 2025", "July 2025"], [june_val, july_val])
            plt.ylabel("Detractor Count")
            st.pyplot(fig2, use_container_width=True)
    else:
        st.info("No detractor data for the selected group.")
else:
    st.info("Charts skipped in ⚡ Lite mode. Click the button above to draw them.")
