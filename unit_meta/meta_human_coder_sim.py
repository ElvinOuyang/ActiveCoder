import pandas as pd
import numpy as np
import os
import pickle
import sys
sys.path.append('..')

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import logging
import configparser
import argparse

import ui_utils
import nlp_tools

arg_parser = argparse.ArgumentParser(description='Run human docer sim based on config file')
arg_parser.add_argument('--config_path', metavar='config_path', default='./config_meta.cfg',
                        type=str, help='Path to the human coder config file', required=False)
args = arg_parser.parse_args()

# loading config
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(args.config_path)

ROUND_ID = int(config.get('paths', 'ROUND_ID'))
LABEL_PATH = config.get('paths', 'LABEL_PATH')
Path(LABEL_PATH).mkdir(parents=True, exist_ok=True)
INPUT_PATH = config.get('paths', 'INPUT_PATH')
Path(INPUT_PATH).mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = config.get('data', 'SAMPLE_FILE')
ANSWER_FILE = config.get('coder_sim', 'ANSWER_FILE')


# initialize logger
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s: %(levelname)s:: %(message)s')

logfile = f"{LABEL_PATH}/coder_sim_logs.log"
file_handler = logging.FileHandler(logfile, mode='a')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

root_logger.setLevel(logging.INFO)
print = root_logger.info

print(">>> Reading untagged sample files for coder sim...")
samples_df = pd.read_json(SAMPLE_FILE, orient='records')
answers_df = pd.read_json(ANSWER_FILE, orient='records')

labels_df = ui_utils.coder_sim(samples_df, answers_df)

print(">>> Saving labeled sample files to disk...")
ui_utils.df_to_json_form(labels_df, tag_col='Tags', ui_dir=LABEL_PATH,
                         ui_filename='/text_labeled.json')

if ROUND_ID > 1:
    # append previous input file with current tagged labels
    print(f">>> This is Round {ROUND_ID}, some records have been tagged already...")
    print(">>> Appending tagged records obtain all tagged records for training...")
    prev_input_file = f"{PROJECT_PATH}/round_{ROUND_ID - 1}/input/input_file.json"
    prev_train_df = pd.read_json(prev_input_file, orient='records')
    input_df = pd.concat([prev_train_df, labels_df], ignore_index=True)
else:
    print(">>> This is the first round, input file is the labeled sample file...")
    input_df = labels_df

print(">>> Saving all the labeled sampled from project rounds for training...")
ui_utils.df_to_json_form(input_df, tag_col='Tags', ui_dir=INPUT_PATH,
                         ui_filename='/input_file.json')

with open(f"{LABEL_PATH}/coder_sim_record.cfg", 'w') as file:
    config.write(file)