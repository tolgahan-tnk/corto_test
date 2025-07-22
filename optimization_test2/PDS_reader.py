# -*- coding: utf-8 -*-
"""
Full PDS-label harvesting and time analysis for HRSC IMG files
Author : tolga
Created: Thu Jun 19 18:08:15 2025
Updated: adds full-label export + SR2 subset workbook
"""

import os, re, datetime
import pandas as pd

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
target_directory = r"/home/tt_mmx/corto/PDS_Data"
sr2_suffix      = "SR2.IMG"        # 7-char suffix we want to filter for later
timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --------------------------------------------------------------------------- #
# BASIC I/O CHECK                                                             #
# --------------------------------------------------------------------------- #
if not os.path.exists(target_directory):
    raise FileNotFoundError(f"Target directory does not exist: {target_directory}")
print(f"Scanning directory: {target_directory}")

# --------------------------------------------------------------------------- #
# HELPER: naïve PDS label parser (single-line key=value)                      #
# --------------------------------------------------------------------------- #
_key_val = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")

def parse_pds_label(file_path, max_records=50_000):
    """
    Reads the ASCII header (before 'END') and returns a dict of key-value pairs.
    Handles <CR><LF> junk, removes surrounding quotes, stops at 'END'.
    """
    label = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, raw in enumerate(fh):
            if i > max_records:          # just in case a label is malformed
                break
            line = raw.strip().replace("<CR><LF>", "")
            if line.upper().startswith("END"):
                break
            m = _key_val.match(line)
            if m:
                key, val = m.groups()
                val = val.strip().strip('"').strip("'")
                label[key] = val
    return label

# --------------------------------------------------------------------------- #
# WALK THE TREE AND COLLECT EVERYTHING                                        #
# --------------------------------------------------------------------------- #
records = []
for root, _, files in os.walk(target_directory):
    for fname in files:
        if not fname.endswith(".IMG"):
            continue
        fpath = os.path.join(root, fname)
        lbl   = parse_pds_label(fpath)
        lbl.update({"file_path": fpath, "file_name": fname})
        records.append(lbl)

if not records:
    raise RuntimeError("No IMG files found!")

df = pd.DataFrame(records)
print(f"IMG files found & parsed: {len(df)}")

# --------------------------------------------------------------------------- #
# TIME COLUMNS (START_TIME, STOP_TIME, MEAN_TIME, etc.)                       #
# --------------------------------------------------------------------------- #
for col in ["START_TIME", "STOP_TIME"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if {"START_TIME", "STOP_TIME"} <= set(df.columns):
    df["DURATION_SECONDS"] = (df["STOP_TIME"] - df["START_TIME"]).dt.total_seconds()
    df["MEAN_TIME"]        = df["START_TIME"] + (df["STOP_TIME"] - df["START_TIME"]) / 2
    df["UTC_MEAN_TIME"]    = (
        df["MEAN_TIME"]
          .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
          .str[:-2] + "Z"
    )
else:
    df["DURATION_SECONDS"] = pd.NA
    df["MEAN_TIME"]        = pd.NaT
    df["UTC_MEAN_TIME"]    = pd.NA

df["STATUS"] = (
    df["START_TIME"].notna() & df["STOP_TIME"].notna()
).map({True: "SUCCESS", False: "MISSING_TIME_DATA"})

# --------------------------------------------------------------------------- #
# LAST-7-CHARACTER SUMMARY                                                    #
# --------------------------------------------------------------------------- #
df["last7"] = df["file_name"].str[-7:]
print("\nUnique last-7-char substrings (counts):")
print(df["last7"].value_counts().sort_index().to_string())

# --------------------------------------------------------------------------- #
# EXPORT – FULL COLUMN SET                                                    #
# --------------------------------------------------------------------------- #
excel_main = f"IMG_Time_Analysis_{timestamp}.xlsx"
excel_sr2  = f"IMG_Time_Analysis_filtered_for_SR2_{timestamp}.xlsx"

def to_excel(full_df, filename, main_sheet="IMG Files"):
    # Convert all datetimes to tz-naive for Excel
    for col in full_df.select_dtypes("datetimetz").columns:
        full_df[col] = pd.to_datetime(full_df[col]).dt.tz_localize(None)
    with pd.ExcelWriter(filename, engine="openpyxl") as xls:
        full_df.to_excel(xls, sheet_name=main_sheet, index=False)

        # Quick summary sheet
        summ = {
            "Metric": [
                "File count",
                "Analysis date",
                "Start min",
                "Stop max",
                "Avg duration [s]",
            ],
            "Value": [
                len(full_df),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(full_df["START_TIME"].min().tz_localize(None))
                if "START_TIME" in full_df else "N/A",
                str(full_df["STOP_TIME"].max().tz_localize(None))
                if "STOP_TIME" in full_df else "N/A",
                f"{full_df['DURATION_SECONDS'].mean():.2f}"
                if "DURATION_SECONDS" in full_df else "N/A",
            ],
        }
        pd.DataFrame(summ).to_excel(xls, sheet_name="Summary", index=False)

print(f"\nWriting full workbook: {excel_main}")
to_excel(df.copy(), excel_main)

# --------------------------------------------------------------------------- #
# SR2-ONLY SUBSET                                                             #
# --------------------------------------------------------------------------- #
df_sr2 = df[df["last7"] == sr2_suffix].copy()
if not df_sr2.empty:
    print(f"Writing SR2 subset workbook: {excel_sr2}")
    to_excel(df_sr2, excel_sr2, main_sheet="SR2 IMG Files")
else:
    print("No SR2.IMG files found – subset not created")

# --------------------------------------------------------------------------- #
# SR2: UNIQUE-VALUE SUMMARY FOR EACH COLUMN                                   #
# --------------------------------------------------------------------------- #
if not df_sr2.empty:
    print("\nSR2 subset: per-column unique-value counts (and values if <5):")
    for col in df_sr2.columns:
        uniques = df_sr2[col].dropna().unique()
        count   = len(uniques)
        print(f"- {col}: {count} unique")
        if count < 5:
            print(f"    values: {list(uniques)}")
# %%


print("\nDone!")
