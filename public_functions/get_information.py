from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import pickle
import datetime
from joblib import Parallel, delayed
from scipy.stats import kruskal, chi2_contingency




POS_BINS = ["lay","sit","stand"]
POS_LABELS = ["lay", "sit", "stand"]

GENDER_BINS = ["Male","Female"]
GENDER_LABELS = [1,2]





def process_single_path(path, mode):
    """单文件处理函数，用于多进程并行"""
    try:
        data = pd.read_pickle(path)
        if data.empty: return None

        # 提取基础信息
        row0 = data.iloc[0]
        weight = row0["weight"]
        height = row0["height"]
        age = row0["age"]
        gender = row0["gender"]
        # 预计算 BMI, 人工登记的可能有误差
        bmi = np.nan
        if 0 < height <= 250:
            bmi = weight / (height / 100) ** 2
            if not (10 <= bmi <= 50): bmi = np.nan

        subset_col = "bp_idx" if mode == "ringbp" else "Record_ID"
        df_unique = data.drop_duplicates(subset=[subset_col], keep="first")
        if mode != "ringbp":
            df_unique = df_unique.dropna(subset=["sbp_fix"])

        sbp = df_unique["sbp_fix"].to_numpy()
        dbp = df_unique["dbp_fix"].to_numpy()
        hr = df_unique["pr_ref"].to_numpy()
        pos = df_unique["position"].to_numpy()
        bp_lvl = df_unique["BP_Level"].to_numpy()

        sbp_v = sbp[(sbp >= 40) & (sbp <= 370) & (~np.isnan(sbp))]
        dbp_v = dbp[(dbp >= 20) & (dbp <= 160) & (~np.isnan(dbp))]


        return {
            "age": age if (0 < age <= 120) else np.nan,
            "bmi": bmi,
            "gender": gender,
            "sbp": sbp_v,
            "dbp": dbp_v,
            "hr": hr,
            "pos": pos,
            "bp_lvl": bp_lvl,
            "ppg_q": ppg_q,
            "raw_counts": len(df_unique)
        }
    except:
        return None
def get_info(allpaths, mode="ringbp",return_result="df"):
    info = defaultdict(float)
    info["subjects"] = len(allpaths)  # 总人数
    info["BP_recordings"] = 0  # 血压记录总数

    # 使用 joblib 进行并行处理 (n_jobs=-1 使用所有核心)
    results = Parallel(n_jobs=-1)(delayed(process_single_path)(p, mode) for p in allpaths)
    results = [r for r in results if r is not None]

    age_all = np.concatenate([[r['age']] for r in results if not np.isnan(r['age'])])
    bmi_all = np.concatenate([[r['bmi']] for r in results if not np.isnan(r['bmi'])])
    gender_all = np.concatenate([[r['gender']] for r in results])

    sbp_all = np.concatenate([r['sbp'] for r in results])
    dbp_all = np.concatenate([r['dbp'] for r in results])
    hr_all = np.concatenate([r['hr'] for r in results])
    pos_all = np.concatenate([r['pos'] for r in results])
    bp_lvl_all = np.concatenate([r['bp_lvl'] for r in results])

    # ==== 整合最终输出信息 ====
    ordered_info = OrderedDict()
    # 基础信息
    ordered_info["总人数, n"] = info["subjects"]
    # ordered_info["有有效信息的人数, n"] = num_valid_subjects
    # ordered_info["血压记录总数, n"] = info["BP_recordings"]
    ordered_info["有效SBP样本数, n"] = len(sbp_all)
    ordered_info["有效DBP样本数, n"] = len(dbp_all)
    # 年龄统计
    ordered_info[
        "年龄（均值±标准差）, years"] = f"{np.round(np.mean(age_all), 2)} ± {np.round(np.std(age_all), 2)}" if not np.isnan(
        info["age_mean"]) else "无有效数据"
    ordered_info["最小年龄, years"] = int(np.mean(age_all))
    ordered_info["最大年龄, years"] = int(np.max(age_all))
    # BMI统计
    ordered_info[
        "BMI（均值±标准差）"] = f"{np.round(np.mean(bmi_all), 2)} ± {np.round(np.std(bmi_all), 2)}" if not np.isnan(
        info["bmi_mean"]) else "无有效数据"
    ordered_info["最小BMI"] = round(np.min(bmi_all), 2)
    ordered_info["最大BMI"] = round(np.max(bmi_all), 2)
    # 性别比例
    if len(gender_all)>0:
        ordered_info["男性比例, %"] = round(np.sum(gender_all==1) / len(gender_all) * 100, 2)
        ordered_info["女性比例, %"] =round(np.sum(gender_all==2) / len(gender_all) * 100, 2)
    # 心率
    ordered_info[
        "心率（均值±标准差）, bpm"] = f"{np.round(np.mean(hr_all), 2)} ± {np.round(np.std(hr_all), 2)}"
    # 血压统计
    ordered_info["SBP（均值±标准差）, mmHg"] = f"{np.round(np.mean(sbp_all), 2)} ± {np.round(np.std(sbp_all), 2)}"
    ordered_info["DBP（均值±标准差）, mmHg"] = f"{np.round(np.mean(dbp_all), 2)} ± {np.round(np.std(dbp_all), 2)}"

    if len(bp_lvl_all) > 0:
        for bp_i in range(1,6):
            ordered_info[f"BP_Level = {bp_i}, %"] = round(np.sum(bp_lvl_all==bp_i) / len(bp_lvl_all) * 100, 2)
    else:
        for bp_i in range(1,6):
            ordered_info[f"BP_Level = {bp_i}, %"] = 0
    ordered_info["SBP ≥ 160 mmHg, %"] = round(np.sum(sbp_all>=160) / len(sbp_all) * 100, 2)
    ordered_info["SBP ≥ 140 mmHg, %"] = round(np.sum(sbp_all>=140) / len(sbp_all) * 100, 2)
    ordered_info["SBP ≤ 100 mmHg, %"] = round(np.sum(sbp_all<=100) / len(sbp_all) * 100, 2)

    ordered_info["DBP ≥ 100 mmHg, %"] = round(np.sum(dbp_all>=100) / len(dbp_all) * 100, 2)
    ordered_info["DBP ≥ 85 mmHg, %"] = round(np.sum(dbp_all>=85)/ len(dbp_all) * 100, 2)
    ordered_info["DBP ≤ 60 mmHg, %"] = round(np.sum(dbp_all<=60) / len(dbp_all) * 100, 2)

    ordered_info["Pos_lay,%"] =   round(np.sum(pos_all=="lay") / len(pos_all) * 100, 2)
    ordered_info["Pos_sit,%"] =   round(np.sum(pos_all=="sit") / len(pos_all) * 100, 2)
    ordered_info["Pos_stand,%"] =   round(np.sum(pos_all=="stand") / len(pos_all) * 100, 2)
    # 转换为DataFrame
    df_basic = pd.DataFrame(list(ordered_info.items()), columns=["统计项", "数值"])
    return df_basic



if __name__ == "__main__":
    dir_path = "/home/cjr/datasets/Ring2Health/shared/"
    result_dir = "../results/"

    allpaths = glob.glob(os.path.join(dir_path, "*.pkl"))
    df = get_info(allpaths)
    print(df)
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    save_path = os.path.join(result_dir, f"数据集划分分析报告_{formatted_date}.xlsx")

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="information", index=False)

