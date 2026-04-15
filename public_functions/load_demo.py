import glob

import pandas as pd
from tqdm import tqdm
if __name__ == '__main__':
    save_dir = "/home/cjr/datasets/Ring2Health/shared/"
    file_paths = glob.glob(save_dir+"*.pkl")
    for file_path in tqdm(file_paths):
        data = pd.read_pickle(file_path)
        print(data.keys())
        break