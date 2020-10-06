import pandas as pd
import numpy as np
import os
import pickle
import sys
sys.path.append('..')

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import logging
import configparser
import argparse

import ui_utils
import nlp_tools

arg_parser = argparse.ArgumentParser(description='Run sampling based on config file')
arg_parser.add_argument('--config_path', metavar='config_path', default='./config_meta.cfg',
                        type=str, help='Path to the sampling config file', required=False)
args = arg_parser.parse_args()


# loading config
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(args.config_path)

ROUND_ID = int(config.get('paths', 'ROUND_ID'))
PROJECT_PATH = config.get('paths', 'PROJECT_PATH')
Path(PROJECT_PATH).mkdir(parents=True, exist_ok=True)
SAMPLE_PATH = config.get('paths', 'SAMPLE_PATH')
Path(SAMPLE_PATH).mkdir(parents=True, exist_ok=True)
INPUT_PATH = config.get('paths', 'INPUT_PATH') # input path for CURRENT ROUND
Path(INPUT_PATH).mkdir(parents=True, exist_ok=True)

SCORED_FILE = config.get('data', 'SCORED_FILE')
print(SCORED_FILE)
SAMPLE_SIZE = int(config.get('sampling', 'SAMPLE_SIZE'))
SAMPLING_METHOD = config.get('sampling', 'SAMPLING_METHOD')

# initialize logger
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s: %(levelname)s:: %(message)s')

logfile = f"{SAMPLE_PATH}/sampling_logs.log"
file_handler = logging.FileHandler(logfile, mode='a')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

root_logger.setLevel(logging.INFO)
print = root_logger.info

print(">>> Reading scored dataframe for sampling purpose...")
scores_df = pd.read_csv(SCORED_FILE)

if SAMPLING_METHOD == 'clustering':
    print("Creating clusters based on tag scores...")
    cluster_tsne_viz = f"{SAMPLE_PATH}/clustering_tsne.png"
    scores_df, kmeans_model = ui_utils.kmeans_from_proba(scores_df, cluster_tsne_viz)
    print(">>> Saving trained KMeans model")
    with open(f'{SAMPLE_PATH}/kmeans.pickle', 'wb') as file:
        pickle.dump(kmeans_model, file)

if ROUND_ID > 1:
    print(f">>> This is Round {ROUND_ID}, some records have been tagged already...")
    print(">>> Removing tagged records to avoid doubling efforts...")
    prev_input_file = f"{PROJECT_PATH}/round_{ROUND_ID - 1}/input/input_file.json"
    prev_train_df = pd.read_json(prev_input_file, orient='records')
    tagged_ids = prev_train_df['UID'].values
    scores_df = scores_df.loc[~np.isin(scores_df['UID'], tagged_ids), :].reset_index(drop=True)

if SAMPLING_METHOD == 'clustering':
    print(">>> Generating samples by cluster")
    sample_df = ui_utils.sample_by_cluster(scores_df, SAMPLE_SIZE)
elif SAMPLING_METHOD == 'random':
    print(">>> Generating samples randomly")
    sample_df = ui_utils.sample_by_random(scores_df, SAMPLE_SIZE)

ui_utils.df_to_json_form(sample_df, tag_col='Tags', ui_dir=SAMPLE_PATH,
                ui_filename='/text_tags.json')

with open(f"{SAMPLE_PATH}/sampling_record.cfg", 'w') as file:
    config.write(file)