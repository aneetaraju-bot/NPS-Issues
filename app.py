#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# ---- CONFIG: exact headers in your June/July files ----
JUNE = "NPS June - Detailed NPS comments.csv"
JULY = "NPS July - NPS comments.csv"

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

EXPECTED = [
    "Created At","Feedback","Rating","Vertical","Courses","Region",
    "Status","Is promoter","Is Detractor","No: of Responses","Month"
]

def load_standardize(path: str, mapping: dict, month_label: str) -> pd.DataFrame:
    usecols = list(mapping.values())
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={v:k for k,v in mapping.items()})
    # keep only raw response rows
    df = df[df["Vertical"].notna() & df["Courses"].notna() & df["Region"].notna()].copy()

    # types
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Is promoter"] = pd.to_numeric(df["Is promoter"], errors="coerce").fillna(0).astype("int8")
    df["Is Detractor"] = pd.to_numeric(df["Is Detractor"], errors="coerce").fillna(0).astype("int8")
    df["No: of Responses"] = pd.to_numeric(df["No: of Responses"], errors="coerce").fillna(1).astype("int16")
    df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")

    for c in ["Vertical","Courses","Region","Status"]:
        if c in df:
            df[c] = df[c].astype("category")

    df["Month"] = month_label
    return df[[c for c in EXPECTED if c in df.columns]]

def nps_calc(group: pd.DataFrame) -> float:
    r = group["Rating"]
    total = r.notna().sum()
    if total == 0:
        return np.nan
    promoters = ((r >= 9) & (r <= 10)).sum()
    detractors = ((r >= 0) & (r <= 6)).sum()
    return ((promoters/total) - (detractors/total)) * 100.0

def main():
    base = Path(".")
    june = load_standardize(base / JUNE, JUNE_COLS, "June 2025")
    july = load_standardize(base / JULY, JULY_COLS, "July 2025")
    combined = pd.concat([june, july], ignore_index=True)

    # 1) Regional-wise monthly NPS problems (detractors)
    problems = combined[combined["Is Detractor"] == 1].copy()
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
    problems_summary.to_csv("regional_monthly_nps_problems.csv", index=False)

    # 2) June vs July detractor comparison
    cmp = (
        problems_summary
        .pivot(index=["Region","Courses","Status"], columns="Month", values="Detractor_Count")
        .fillna(0)
        .reset_index()
    )
    if "June 2025" not in cmp.columns: cmp["June 2025"] = 0
    if "July 2025" not in cmp.columns: cmp["July 2025"] = 0
    cmp["Delta_Detractors_Jul_minus_Jun"] = cmp["July 2025"] - cmp["June 2025"]
    cmp.to_csv("june_vs_july_detractor_comparison.csv", index=False)

    # 3) NPS by month (region/course/batch)
    nps = (
        combined.groupby(["Region","Courses","Status","Month"], dropna=False)
        .apply(nps_calc)
        .reset_index(name="NPS")
        .sort_values(["Region","Courses","Status","Month"])
    )
    nps.to_csv("nps_by_month_region_course_batch.csv", index=False)

    # 4) (Optional) Raw combined output for audit
    combined.sort_values("Created At").to_csv("combined_standardized_rows.csv", index=False)

    print("✅ Done.")
    print("  • regional_monthly_nps_problems.csv")
    print("  • june_vs_july_detractor_comparison.csv")
    print("  • nps_by_month_region_course_batch.csv")
    print("  • combined_standardized_rows.csv")

if __name__ == "__main__":
    main()
