import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scipy.stats import kruskal, mannwhitneyu

QUALITY_KEYS = [
    'ppg_quality(ppg_g_1)', 'ppg_quality(ppg_ga_1)', 'ppg_quality(ppg_r_1)',
    'ppg_quality(ppg_ir_1)', 'ppg_quality(ppg_g_2)', 'ppg_quality(ppg_ga_2)',
    'ppg_quality(ppg_r_2)', 'ppg_quality(ppg_ir_2)'
]
def compute_ppg_quality(file_paths):
    predictions = np.empty((0,len(QUALITY_KEYS)))
    for test_path in tqdm(file_paths):
        data = pd.read_pickle(test_path)
        ppg_quality = np.concatenate(
            [np.mean(np.stack(data[key].values),axis=1)[:,None] for key in QUALITY_KEYS],
            axis=1
        ).astype(np.float32)
        predictions = np.concatenate((predictions, ppg_quality),axis=0)
    return predictions
def mean_std(df):
    return df.mean().round(4).astype(str) + " ± " + df.std().round(4).astype(str)
