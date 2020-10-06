import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Level | Level for Humans | Level Description                  
# -------|------------------|------------------------------------ 
#  0     | DEBUG            | [Default] Print all messages       
#  1     | INFO             | Filter out INFO messages           
#  2     | WARNING          | Filter out INFO & WARNING messages 
#  3     | ERROR            | Filter out all messages     

import pandas as pd
import numpy as np
import tensorflow as tf
# the tf settings below ALSO GOVERNS ALL OTHER LOGGERS!
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
import random
import pickle
import spacy
import sys
sys.path.append('..')
from pathlib import Path
import logging
import configparser
import argparse

import ui_utils
import nlp_tools

# allow gpu memory growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


arg_parser = argparse.ArgumentParser(description='Run model scoring based on config file')
arg_parser.add_argument('--config_path', metavar='config_path', default='./al_4_score.cfg',
                        type=str, help='Path to the scoring config file', required=False)
args = arg_parser.parse_args()


config = configparser.ConfigParser()
config.read(args.config_path)

OUTPUT_PATH = config.get('data', 'OUTPUT_PATH')
APPLY_FILE = config.get('data', 'APPLY_FILE')
CLF_THRESHOLD = float(config.get('scoring', 'CLF_THRESHOLD'))
APPLY_BATCH_SIZE = int(config.get('scoring', 'APPLY_BATCH_SIZE'))
SCORE_W_GPU = bool(int(config.get('scoring', 'SCORE_W_GPU')))


SCORED_PATH = f"{OUTPUT_PATH.rstrip('/')}/scored/"
Path(SCORED_PATH).mkdir(parents=True, exist_ok=True)

# initialize logger
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s: %(levelname)s:: %(message)s')

# prints to file
logfile = f"{OUTPUT_PATH.rstrip('/')}/scoring_logs.log"
file_handler = logging.FileHandler(logfile, mode='a')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

root_logger.setLevel(logging.INFO)

print = root_logger.info

print(">>> Using Tensorflow 2 with GPU for this scoring script:")
print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')

print(">>> Loading apply data")
apply_df = pd.read_json(APPLY_FILE, orient='records')

print(">>> Loading pretrained tokenizer...")
with open(f'{OUTPUT_PATH.rstrip("/")}/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)
print('>>> Vocabulary size: {:,}'.format(len(tokenizer.word_index.items())))

print(">>> Loading trained models from model training output folder...")
taggers_dict = nlp_tools.obtain_tag_clfs(OUTPUT_PATH)

print(">>> Scoring the apply data with the trained models...")
labels_dict = nlp_tools.generate_labels_with_model_dict(
    apply_df.Text, tokenizer,
    taggers_dict, batch_size=APPLY_BATCH_SIZE,
    clf_threshold=CLF_THRESHOLD, use_gpu=SCORE_W_GPU)

print(">>> Saving the results to output dir...")
apply_scored_df = pd.concat([apply_df, pd.DataFrame(labels_dict)], axis=1)
apply_scored_df.to_json(f"{SCORED_PATH}scored_output.json", orient='records', indent=2)
with open(f"{OUTPUT_PATH.rstrip('/')}/scoring_record.cfg", 'w') as file:
    config.write(file)