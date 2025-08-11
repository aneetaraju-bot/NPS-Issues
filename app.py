# app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monthly NPS Problems (Regional-wise)", layout="wide")

st.title("Monthly NPS Problems – Regional-wise (Courses, Batches, Timestamps)")
st.caption("Upload your June & July NPS CSVs. The app auto-detects feedback/rating columns even if headers are in Malayalam/Telugu.")

# -----------------------------
# Helpers
# -----------------------------
KNOWN_COLS = {
    "created_at": ["created at", "created_at"],
    "vertical": ["vertical"],
    "courses": ["courses", "course"],
    "region": ["region"],
    "status": ["status", "batch", "batch type"],
    "is_promoter": ["is promoter", "promoter"],
    "is_detractor": ["is detractor", "detractor"],
    "num_responses": ["no: of responses", "no of responses", "responses", "count", "response count"],
}

def _best_match(col, targets):
    c = col.strip().lower()
    return any(t in c for t in targets)

def detect_created_at(df):
    for c in df.columns:
        if _best_match(c, KNOWN_COLS["created_at"]):
            return c
    # fallback: exact "Created At" if present with whitespace/newlines
    for c in df.columns:
        if c.replace("\n"," ").strip().lower() == "created at":
            return c
    return None

def detect_feedback_col(df):
    """
    Try to find the open-text feedback column.
    Heuristics:
      1) Column name includes 'feedback' (any language: detect 'feedback')
      2) Otherwise choose object/text column with longest average string length,
         excluding known structural columns.
    """
    # 1) direct 'feedback' match
    for c in df.columns:
        if "feedback" in c.lower():
            return c

    # 2) fallback: pick a text-like column with longest avg length
    structural_like = set()
    for key in KNOWN_COLS:
        structural_like |= {c for c in df.columns if _best_match(c, KNOWN_COLS[key])}
    structural_like |= {detect_created_at(df)} if detect_created_at(df) else set()

    candidates = []
    for c in df.columns:
        if c in structural_like:
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(200)
            if sample.empty: 
                continue
            avg_len = sample.str.len().mean()
            candidates.append((avg_len, c))
    if candidates:
        candidates.sort(reverse=True)  # longest first
        return candidates[0][1]
    return None

def detect_rating_col(df):
    """
    Try to find NPS rating (0-10) column even if header is in Malayalam/Telugu.
    Heuristics:
      1) Column name includes 'rating' (case-insensitive)
      2) Otherwise pick a column that becomes numeric and mostly within 0..10
    """
    # 1) header contains 'rating'
    for c in df.columns:
        if "rating" in c.lower():
            return c

    # 2) numeric within 0..10
    best = None
    best_score = -1
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        in_range = ((s >= 0) & (s <= 10)).sum()
        score = in_range
        if score > best_score and in_range >= max(5, int(0.2*len(s))):
            best_score = score
            best = c
    return best

def detect_by_known(df, key):
    for c in df.columns:
        if _best_match(c, KNOWN_COLS[key]):
            return c
    return None

def standardize_month(df, month_label):
    """
    Return a standardized dataframe with columns:
    ['Created At','Feedback','Rating','Vertical','Courses','Region','Status','Is promoter','Is Detractor','No: of Responses','Month']
    """
    df = df.copy()

    c_created = detect_created_at(df)
    c_feedback = detect_feedback_col(df)
    c_rating  = detect_rating_col(df)
    c_vertical = detect_by_known(df, "vertical")
    c_courses  = detect_by_known(df, "courses")
    c_region   = detect_by_known(df, "region")
    c_status   = detect_by_known(df, "status")
    c_prom     = detect_by_known(df, "is_promoter")
    c_det      = detect_by_known(df, "is_detractor")
    c_nresp    = detect_by_known(df, "num_responses")

    # Filter to rows that look like raw responses (i.e., must have Vertical/Courses/Region)
    mask = pd.Series(True, index=df.index)
    for needed in [c_vertical, c_courses, c_region]:
        if needed:
            mask &= df[needed].notna()

    sdf = df.loc[mask, [c for c in [c_created, c_feedback, c_rating, c_vertical, c_courses, c_region, c_status, c_prom, c_det, c_nresp] if c is not None]].copy()

    rename_map = {}
    if c_created: rename_map[c_created] = "Created At"
    if c_feedback: rename_map[c_feedback] = "Feedback"
    if c_rating: rename_map[c_rating] = "Rating"
    if c_vertical: rename_map[c_vertical] = "Vertical"
    if c_courses: rename_map[c_courses] = "Courses"
    if c_region: rename_map[c_region] = "Region"
    if c_status: rename_map[c_status] = "Status"
    if c_prom: rename_map[c_prom] = "Is promoter"
    if c_det: rename_map[c_det] = "Is Detractor"
    if c_nresp: rename_map[c_nresp] = "No: of Responses"

    sdf.rename(columns=rename_map, inplace=True)

    # Ensure all expected columns exist
    for col in ["Created At","Feedback","Rating","Vertical","Courses","Region","Status","Is promoter","Is Detractor","No: of Responses"]:
        if col not in sdf.columns:
            sdf[col] = np.nan

    # Types
    sdf["Rating"] = pd.to_numeric(sdf["Rating"], errors="coerce")
    sdf["Is promoter"] = pd.to_numeric(sdf["Is promoter"], errors="coerce").fillna(0).astype(int)
    sdf["Is Detractor"] = pd.to_numeric(sdf["Is Detractor"], errors="coerce").fillna(0).astype(int)
    sdf["No: of Responses"] = pd.to_numeric(sdf["No: of Responses"], errors="coerce").fillna(1).astype(int)

    # Timestamps
    with pd.option_context("mode.chained_assignment", None):
        sdf["Created At"] = pd.to_datetime(sdf["Created At"], errors="coerce")

    sdf["Month"] = month_label
    return sdf

def nps_calc(group):
    # NPS = %Promoters (9-10) - %Detractors (0-6)
    r = group["Rating"]
    total = r.notna().sum()
    if total == 0:
        return np.nan
    promoters = ((r >= 9) & (r <= 10)).sum()
    detractors = ((r >= 0) & (r <= 6)).sum()
    return ((promoters / total) - (detractors / total)) * 100.0

def make_download(df, label):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=f"⬇️ Download {label} (CSV)",
        data=buf.getvalue(),
        file_name=f"{label.replace(' ','_').lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )

# -----------------------------
# File inputs
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    june_file = st.file_uploader("Upload **June** CSV", type=["csv"])
with c2:
    july_file = st.file_uploader("Upload **July** CSV", type=["csv"])

if not june_file or not july_file:
    st.info("Upload both **June** and **July** CSVs to begin.")
    st.stop()

raw_june = pd.read_csv(june_file)
raw_july = pd.read_csv(july_file)

june = standardize_month(raw_june, "June 2025")
july = standardize_month(raw_july, "July 2025")
combined = pd.concat([june, july], ignore_index=True)

# -----------------------------
# Filters
# -----------------------------
st.subheader("Filters")
fc1, fc2, fc3 = st.columns(3)

regions = ["All"] + sorted([x for x in combined["Region"].dropna().astype(str).unique()])
courses = ["All"] + sorted([x for x in combined["Courses"].dropna().astype(str).unique()])
statuses = ["All"] + sorted([x for x in combined["Status"].dropna().astype(str).unique()])

with fc1:
    f_region = st.selectbox("Region", regions)
with fc2:
    f_course = st.selectbox("Course", courses)
with fc3:
    f_status = st.selectbox("Batch / Status", statuses)

f = combined.copy()
if f_region != "All": f = f[f["Region"].astype(str) == f_region]
if f_course != "All": f = f[f["Courses"].astype(str) == f_course]
if f_status != "All": f = f[f["Status"].astype(str) == f_status]

# -----------------------------
# Problem list (Detractors)
# -----------------------------
st.header("Regional-wise Monthly NPS Problems")
problems = f[f["Is Detractor"] == 1].copy()

problems_summary = (
    problems.groupby(["Region","Courses","Status","Month"], dropna=False)
    .agg(
        Detractor_Count=("Is Detractor","sum"),
        Avg_Detractor_Rating=("Rating","mean"),
        Example_Comments=("Feedback", lambda x: "; ".join(x.dropna().astype(str).head(3)))
    )
    .reset_index()
    .sort_values(["Region","Courses","Status","Month"])
)

st.dataframe(problems_summary, use_container_width=True)
make_download(problems_summary, "regional_monthly_nps_problems")

# -----------------------------
# June vs July comparison
# -----------------------------
st.header("June vs July – Comparison (per Region/Course/Batch)")

pivot = (
    problems_summary
    .pivot(index=["Region","Courses","Status"], columns="Month", values="Detractor_Count")
    .fillna(0)
    .reset_index()
)
if "June 2025" not in pivot.columns: pivot["June 2025"] = 0
if "July 2025" not in pivot.columns: pivot["July 2025"] = 0

pivot["Δ Detractors (Jul - Jun)"] = pivot["July 2025"] - pivot["June 2025"]

st.dataframe(pivot.sort_values("Δ Detractors (Jul - Jun)", ascending=False), use_container_width=True)
make_download(pivot, "june_vs_july_detractor_comparison")

# -----------------------------
# NPS by month (not just detractors)
# -----------------------------
st.header("NPS by Month (Region/Course/Batch)")

nps_table = (
    f.groupby(["Region","Courses","Status","Month"], dropna=False)
     .apply(nps_calc)
     .reset_index(name="NPS")
     .sort_values(["Region","Courses","Status","Month"])
)
st.dataframe(nps_table, use_container_width=True)
make_download(nps_table, "nps_by_month_region_course_batch")

# -----------------------------
# Simple charts
# -----------------------------
st.header("Charts")

# Top problem courses (July) by detractor count
st.subheader("Top Problem Courses (July 2025) – Detractor Count")
topj = (
    problems_summary[problems_summary["Month"] == "July 2025"]
    .sort_values("Detractor_Count", ascending=False)
    .head(10)
)
if topj.empty:
    st.info("No detractors in July for the current filter.")
else:
    plt.figure()
    labels = (topj["Region"] + " | " + topj["Courses"]).str.slice(0,70)
    y = topj["Detractor_Count"].values
    plt.barh(labels, y)
    plt.xlabel("Detractor Count")
    plt.ylabel("Region | Course")
    plt.gca().invert_yaxis()
    st.pyplot(plt.gcf(), use_container_width=True)

# Trend bar for a selected triple
st.subheader("June vs July Trend for Selected Group")
g1, g2, g3 = st.columns(3)
with g1:
    t_region = st.selectbox("Trend Region", regions, key="t_region")
with g2:
    t_course = st.selectbox("Trend Course", courses, key="t_course")
with g3:
    t_status = st.selectbox("Trend Status", statuses, key="t_status")

tf = problems_summary.copy()
if t_region != "All": tf = tf[tf["Region"].astype(str) == t_region]
if t_course != "All": tf = tf[tf["Courses"].astype(str) == t_course]
if t_status != "All": tf = tf[tf["Status"].astype(str) == t_status]

if tf.empty:
    st.info("No detractor data for the selected group.")
else:
    tf = tf.pivot(index=["Region","Courses","Status"], columns="Month", values="Detractor_Count").fillna(0)
    if not tf.empty:
        row = tf.iloc[0]
        months = row.index.tolist()
        vals = row.values.tolist()
        plt.figure()
        plt.bar(months, vals)
        plt.ylabel("Detractor Count")
        st.pyplot(plt.gcf(), use_container_width=True)

# -----------------------------
# Raw (standardized) data peek
# -----------------------------
with st.expander("Show standardized raw rows (debug)"):
    st.dataframe(f.sort_values("Created At"), use_container_width=True)
