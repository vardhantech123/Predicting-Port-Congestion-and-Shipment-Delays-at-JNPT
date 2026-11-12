"""Loading and cleaning of the JNPT vessel-call dataset."""
import numpy as np
import pandas as pd

from .config import DATA_PATH, MIN_TAT_H, MAX_TAT_H

RENAME = {
    "SN.NO": "sn", "DATE": "report_date", "Port": "port", "Terminal_Name": "term",
    "Vessel_ID": "vessel", "Country (AIS Flag)": "flag", "TEU_Capacity": "teu_cap",
    "Size": "size", "LOA (in meter )": "loa", "Beam (in meter )": "beam",
    "Arrival_DT (Berthed )": "arr", "Departure_DT (Sailed)": "dep",
    "Operation_Commence_Time": "op_start", "Operation_Complete_Time": "op_end",
    "TAT_Hours": "tat_raw", "Handling_Hours": "handling_raw",
    "IMPORT(TEU)": "imp", "EXPORT(TEU)": "exp", "TEU_Handled(TEU)": "teu",
    "Import export ratio": "ier", "Cargo Imbalance Score (CIS)": "cis",
    "Handling Density": "density",
}


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path).rename(columns=RENAME)
    return df.sort_values("arr").reset_index(drop=True)


def clean(df: pd.DataFrame):
    """Return (clean_df, quality_report). Every removal/repair is counted."""
    q = {"rows_raw": int(len(df))}
    df = df.copy()
    # Recompute durations from timestamps (raw Handling_Hours has negative values).
    df["tat"] = (df["dep"] - df["arr"]).dt.total_seconds() / 3600
    df["handling_h"] = (df["op_end"] - df["op_start"]).dt.total_seconds() / 3600
    q["negative_handling_hours_in_raw"] = int((df["handling_raw"] < 0).sum())
    q["missing_teu_cap"] = int(df["teu_cap"].isna().sum())
    q["missing_loa"] = int(df["loa"].isna().sum())
    q["missing_import_or_export"] = int((df["imp"].isna() | df["exp"].isna()).sum())
    q["raw_import_export_ratio_max"] = float(df["ier"].max())

    bad_teu = df["teu"] <= 0
    bad_tat = (df["tat"] < MIN_TAT_H) | (df["tat"] > MAX_TAT_H)
    bad_ops = df["handling_h"] <= 0          # op_end before op_start -> corrupt timestamps
    bad_teu = bad_teu | bad_ops
    q["dropped_corrupt_ops_times"] = int(bad_ops.sum())
    q["dropped_zero_teu"] = int((df["teu"] <= 0).sum())
    q["dropped_tat_out_of_range"] = int((bad_tat & ~bad_teu).sum())
    df = df[~(bad_teu | bad_tat)].reset_index(drop=True)
    q["rows_clean"] = int(len(df))
    q["date_range"] = [str(df["arr"].min()), str(df["arr"].max())]
    q["terminals"] = df["term"].value_counts().to_dict()
    return df, q
