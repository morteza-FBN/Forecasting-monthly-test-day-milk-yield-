"""Convert Solar Hijri/Jalali dates in herd files to Gregorian ISO dates and write QA reports."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from mlk_common import ensure_dir, read_csv


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> Tuple[int, int, int]:
    """Convert Jalali date to Gregorian date.

    Algorithm adapted from the widely used jalaali conversion routine. It is included
    here to avoid requiring a nonstandard date-conversion package.
    """
    jy = int(jy) - 979
    jm = int(jm) - 1
    jd = int(jd) - 1
    j_month_days = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]
    g_month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    j_day_no = 365 * jy + jy // 33 * 8 + (jy % 33 + 3) // 4
    for i in range(jm):
        j_day_no += j_month_days[i]
    j_day_no += jd
    g_day_no = j_day_no + 79
    gy = 1600 + 400 * (g_day_no // 146097)
    g_day_no %= 146097
    leap = True
    if g_day_no >= 36525:
        g_day_no -= 1
        gy += 100 * (g_day_no // 36524)
        g_day_no %= 36524
        if g_day_no >= 365:
            g_day_no += 1
        else:
            leap = False
    gy += 4 * (g_day_no // 1461)
    g_day_no %= 1461
    if g_day_no >= 366:
        leap = False
        g_day_no -= 1
        gy += g_day_no // 365
        g_day_no %= 365
    i = 0
    while g_day_no >= g_month_days[i] + (1 if i == 1 and leap else 0):
        g_day_no -= g_month_days[i] + (1 if i == 1 and leap else 0)
        i += 1
    gm = i + 1
    gd = g_day_no + 1
    return gy, gm, gd


def parse_jalali_date(value) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Accept forms such as 1391/11/16, 1391-11-16, or 13911116.
    nums = re.findall(r"\d+", s)
    if len(nums) >= 3:
        jy, jm, jd = map(int, nums[:3])
    elif len(nums) == 1 and len(nums[0]) == 8:
        jy, jm, jd = int(nums[0][:4]), int(nums[0][4:6]), int(nums[0][6:8])
    else:
        return None
    if not (1200 <= jy <= 1500 and 1 <= jm <= 12 and 1 <= jd <= 31):
        return None
    try:
        gy, gm, gd = jalali_to_gregorian(jy, jm, jd)
    except Exception:
        return None
    return f"{gy:04d}-{gm:02d}-{gd:02d}"


def normalize_gregorian(value) -> Optional[str]:
    if pd.isna(value):
        return None
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d")


def convert_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_Gregorian"] = out[col].map(parse_jalali_date)
    return out


def summarize_date_column(df: pd.DataFrame, original_col: str, converted_col: str) -> dict:
    vals = pd.to_datetime(df[converted_col], errors="coerce") if converted_col in df.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "column": original_col,
        "converted_column": converted_col,
        "n_rows": int(len(df)),
        "n_nonmissing_original": int(df[original_col].notna().sum()) if original_col in df.columns else 0,
        "n_converted": int(vals.notna().sum()),
        "gregorian_min": vals.min().strftime("%Y-%m-%d") if vals.notna().any() else "",
        "gregorian_max": vals.max().strftime("%Y-%m-%d") if vals.notna().any() else "",
    }


def run(development: str, external: str, outdir: str) -> None:
    outdir = ensure_dir(outdir)
    reports = []

    dev = read_csv(development)
    dev = convert_columns(dev, ["Date_of_calving", "Date_at_first_calving_days", "PRINITDATE"])
    if "engzdate" in dev.columns and "Date_of_calving_Gregorian" in dev.columns:
        dev["engzdate_ISO"] = dev["engzdate"].map(normalize_gregorian)
        dev["Date_of_calving_match_engzdate"] = dev["Date_of_calving_Gregorian"] == dev["engzdate_ISO"]
    dev_path = outdir / "development_clean_complete_curves_gregorian_dates.csv"
    dev.to_csv(dev_path, index=False)
    for col in ["Date_of_calving", "Date_at_first_calving_days", "PRINITDATE"]:
        if col in dev.columns:
            reports.append(summarize_date_column(dev, col, f"{col}_Gregorian"))
    if "Date_of_calving_match_engzdate" in dev.columns:
        reports.append({
            "column": "Date_of_calving vs engzdate",
            "converted_column": "Date_of_calving_match_engzdate",
            "n_rows": int(len(dev)),
            "n_nonmissing_original": int(dev["engzdate"].notna().sum()),
            "n_converted": int(dev["Date_of_calving_match_engzdate"].sum()),
            "gregorian_min": "matches",
            "gregorian_max": f"{int(dev['Date_of_calving_match_engzdate'].sum())}/{len(dev)}",
        })

    ext = read_csv(external)
    ext = convert_columns(ext, ["Calving_Date"])
    ext_path = outdir / "external_validation_clean_complete_curves_gregorian_dates.csv"
    ext.to_csv(ext_path, index=False)
    if "Calving_Date" in ext.columns:
        reports.append(summarize_date_column(ext, "Calving_Date", "Calving_Date_Gregorian"))

    pd.DataFrame(reports).to_csv(outdir / "date_conversion_QA_report.csv", index=False)
    print(f"Wrote: {dev_path}")
    print(f"Wrote: {ext_path}")
    print(f"Wrote: {outdir / 'date_conversion_QA_report.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    run(args.development, args.external, args.outdir)


if __name__ == "__main__":
    main()
