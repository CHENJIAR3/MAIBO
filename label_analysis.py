import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np
def load_all_data(file_paths):
    all_data = []
    for path in tqdm(file_paths):
        df = pd.read_pickle(path)
        df_new = df.drop_duplicates(subset="bp_idx", keep="first")
        tmp = pd.DataFrame({
            "subject": path.split("/")[-1].split(".")[0],
            "HR": df_new["pr_ref"].values,
            "SBP": df_new["sbp_fix"].values,
            "DBP": df_new["dbp_fix"].values,
        })
        all_data.append(tmp)

    return pd.concat(all_data, ignore_index=True)


def overall_statistics(df, variables):
    """对多个变量计算整体统计，返回每个变量一行"""
    rows = []
    for var in variables:
        vals = df[var].dropna()
        rows.append({
            "Variable": var.upper(),
            "Count": len(vals),
            "Mean": round(vals.mean(), 2),
            "SD": round(vals.std(), 2),
            "Min": round(vals.min(), 2),
            "25%": round(np.percentile(vals, 25), 2),
            "50%": round(vals.median(), 2),
            "75%": round(np.percentile(vals, 75), 2),
            "Max": round(vals.max(), 2)
        })
    return pd.DataFrame(rows)
