import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
def sort_posture_index(df):
    posture_order = {'lay': 0, 'sit': 1, 'stand': 2}
    def sort_key(x):
        # x 是 tuple
        return (
            len(x), 
            [posture_order[i] for i in x] 
        )
    sorted_index = sorted(df.index, key=sort_key)
    return df.loc[sorted_index]
def posture_coverage_distribution(file_paths):
    subject_posture_set = {}

    for path in tqdm(file_paths):
        df = pd.read_pickle(path)
        s = path.split("/")[-1].split(".")[0]

        posture = pd.Series(df["position"].unique())
        for p in posture:
            if s not in subject_posture_set:
                subject_posture_set[s] = set()
            subject_posture_set[s].add(p)

    combos = [
        tuple(sorted(list(v)))
        for v in subject_posture_set.values()
    ]

    combos = pd.Series(combos)

    count = combos.value_counts()
    ratio = combos.value_counts(normalize=True)
    summary = pd.DataFrame({
        "count(subject)": count,
        "ratio(subject)": ratio
    })

    summary_sorted = sort_posture_index(summary)
    return summary_sorted
