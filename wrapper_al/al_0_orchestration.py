import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')

from pathlib import Path
import logging
import configparser
import argparse
import subprocess
from datetime import datetime
from shutil import copyfile

import ui_utils
import nlp_tools

arg_parser = argparse.ArgumentParser(description='Run orchestration for active learning')
arg_parser.add_argument('--config_path', metavar='config_path', default='./al_0_orchestration_config.cfg',
                        type=str, help='Path to the orchestration config file', required=False)
args = arg_parser.parse_args()

# loading config
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(args.config_path)

TOTAL_ROUNDS = int(config.get('active_learning', 'TOTAL_ROUNDS'))
PROJECT_NOTE = config.get('active_learning', 'PROJECT_NOTE')
APPLY_FILE = config.get('active_learning', 'APPLY_FILE')
RUN_SIM = bool(int(config.get('active_learning', 'RUN_SIM')))

PROJECT_PATH = config.get('active_learning', 'PROJECT_PATH')
Path(PROJECT_PATH).mkdir(parents=True, exist_ok=True)
with open(f"{PROJECT_PATH}/orchestration_record.cfg", 'w') as file:
    config.write(file)

# initialize logger
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s: %(levelname)s:: %(message)s')

logfile = f"{PROJECT_PATH.rstrip('/')}/orchestration_log.log"
file_handler = logging.FileHandler(logfile, mode='a')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

root_logger.setLevel(logging.INFO)
print = root_logger.info

for round_n in range(TOTAL_ROUNDS):
    # create paths for the specific round
    ROUND_ID = round_n + 1
    SAMPLE_PATH = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID}/sample"
    LABEL_PATH = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID}/label"
    INPUT_PATH = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID}/input"
    OUTPUT_PATH = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID}/output"
    CONFIG_PATH = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID}/config"
    Path(CONFIG_PATH).mkdir(parents=True, exist_ok=True)
    if ROUND_ID > 1:
        # scored file is from previous round
        SCORED_FILE = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID - 1}/output/scored/scored_output.json"
    else:
        SCORED_FILE = APPLY_FILE
    INPUT_FILE = f"{PROJECT_PATH.rstrip('/')}/round_{ROUND_ID}/input/input_file.json"
    
    print(f"==========ROUND {ROUND_ID}, STAGE 1.1: Sampling Config==========")
    print(">>> Creating sampling config...")
    conf1_sampling = configparser.ConfigParser()
    conf1_sampling['data'] = {}
    conf1_sampling['data']['ROUND_ID'] = str(ROUND_ID)
    conf1_sampling['data']['APPLY_FILE'] = APPLY_FILE
    conf1_sampling['data']['PROJECT_PATH'] = PROJECT_PATH
    conf1_sampling['data']['SAMPLE_PATH'] = SAMPLE_PATH
    conf1_sampling['data']['SCORED_FILE'] = SCORED_FILE
    conf1_sampling['sampling'] = {}
    conf1_sampling['sampling']['SAMPLE_SIZE'] = config['sampling']['SAMPLE_SIZE']
    conf1_sampling['sampling']['SAMPLING_METHOD'] = config['sampling']['SAMPLING_METHOD']
    conf1_path = f"{CONFIG_PATH.rstrip('/')}/al_1_sampling.cfg"
    print(f">>> Saving sampling config to {conf1_path}")
    with open(conf1_path, 'w') as file:
        conf1_sampling.write(file)

    print(f"==========ROUND {ROUND_ID}, STAGE 1.2: Sampling Execution==========")
    print(f">>> Running subprocess on Sampling for Round {ROUND_ID}...")
    start = datetime.now()
    result = subprocess.run(['python', 'al_1_sampling.py', '--config_path', conf1_path], stdout=subprocess.PIPE, encoding='utf8')
    print(result.stdout)
    print(f"The sampling subprocess was completed within {(datetime.now() - start).seconds:,} seconds")

    if RUN_SIM == True:
        print(f"==========ROUND {ROUND_ID}, STAGE 2.1: Simulated Coding Config==========")
        print(">>> Creating human coder sim config...")
        conf2_coding = configparser.ConfigParser()
        conf2_coding['data'] = {}
        conf2_coding['data']['ROUND_ID'] = str(ROUND_ID)
        conf2_coding['data']['APPLY_FILE'] = APPLY_FILE
        conf2_coding['data']['PROJECT_PATH'] = PROJECT_PATH
        conf2_coding['data']['SAMPLE_PATH'] = SAMPLE_PATH
        conf2_coding['data']['LABEL_PATH'] = LABEL_PATH
        conf2_coding['data']['INPUT_PATH'] = INPUT_PATH
        conf2_coding['coder_sim'] = {}
        conf2_coding['coder_sim']['ANSWER_FILE'] = config['coder_sim']['ANSWER_FILE']
        conf2_path = f"{CONFIG_PATH.rstrip('/')}/al_2_coding_sim.cfg"
        print(f">>> Saving human coder sim config to {conf2_path}")
        with open(conf2_path, 'w') as file:
            conf2_coding.write(file)
        
        print(f"==========ROUND {ROUND_ID}, STAGE 2.2: Simulated Coding Execution==========")
        print(f">>> Running subprocess on Human Coder Sim for Round {ROUND_ID}...")
        start = datetime.now()
        result = subprocess.run(['python', 'al_2_human_coder_sim.py', '--config_path', conf2_path], stdout=subprocess.PIPE, encoding='utf8')
        print(result.stdout)
        print(f"The coding subprocess was completed within {(datetime.now() - start).seconds:,} seconds")
    else:
        print(f"==========ROUND {ROUND_ID}, STAGE 2: Manual Coding Execution==========")
        print(">>> RUN_SIM is False, manual coding is happening...")
        start = datetime.now()
        copyfile(f'{SAMPLE_PATH}/text_tags.json', f'{LABEL_PATH}/text_labeled.json')
        print(">>> Sample file has been copied to label folder for manual tagging!")            
        waiting_status = True
        while waiting_status:
            user_confirmation = input(f">>> Please provide tags to all responses within file {f'{LABEL_PATH}/text_labeled.json'}. Key in Y here when finished to continue: ")
            if user_confirmation == 'Y':
                waiting_status = False
                labels_df = pd.read_json(f'{LABEL_PATH}/text_labeled.json', orient='records')
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
            else:
                print("Please type Y when you complete the tags!")
        print(f">>> The manual coding process was completed within {(datetime.now() - start).seconds:,} seconds")
    
    print(f"==========ROUND {ROUND_ID}, STAGE 3.1: Training Config==========")
    print(">>> Creating training config...")
    conf3_training = configparser.ConfigParser()
    conf3_training['data'] = {}
    conf3_training['data']['ROUND_ID'] = str(ROUND_ID)
    conf3_training['data']['APPLY_FILE'] = APPLY_FILE
    conf3_training['data']['PROJECT_PATH'] = PROJECT_PATH
    conf3_training['data']['INPUT_FILE'] = INPUT_FILE
    conf3_training['data']['OUTPUT_PATH'] = OUTPUT_PATH
    conf3_training['training'] = {}
    conf3_training['training']['MAX_TAGS'] = config['training']['MAX_TAGS']
    conf3_training['training']['BATCH_SIZE'] = config['training']['BATCH_SIZE']
    conf3_training['training']['BUFFER_SIZE'] = config['training']['BUFFER_SIZE']
    conf3_training['training']['EARLY_STOPPING_ROUNDS'] = config['training']['EARLY_STOPPING_ROUNDS']
    conf3_training['training']['DROPOUT_LEVEL'] = config['training']['DROPOUT_LEVEL']
    conf3_training['training']['LEARNING_RATE'] = config['training']['LEARNING_RATE']
    conf3_training['training']['N_LSTM_UNITS'] = config['training']['N_LSTM_UNITS']
    conf3_training['training']['N_FC_NEURONS'] = config['training']['N_FC_NEURONS']
    conf3_training['training']['EMBED_TRAINABLE'] = config['training']['EMBED_TRAINABLE']
    conf3_training['training']['RANDOM_EMBED'] = config['training']['RANDOM_EMBED']
    conf3_path = f"{CONFIG_PATH.rstrip('/')}/al_3_train.cfg"
    print(f">>> Saving training config to {conf3_path}")
    with open(conf3_path, 'w') as file:
        conf3_training.write(file)
    
    print(f"==========ROUND {ROUND_ID}, STAGE 3.2: Training Execution==========")
    print(f">>> Running subprocess on Model Training for Round {ROUND_ID}...")
    start = datetime.now()
    result = subprocess.run(['python', 'al_3_train_models.py', '--config_path', conf3_path], stdout=subprocess.PIPE, encoding='utf8')
    print(result.stdout)
    print(f"The model training subprocess was completed within {(datetime.now() - start).seconds:,} seconds")

    print(f"==========ROUND {ROUND_ID}, STAGE 4.1: Scoring Config==========")
    print(">>> Creating scoring config...")
    conf4_scoring = configparser.ConfigParser()
    conf4_scoring['data'] = {}
    conf4_scoring['data']['ROUND_ID'] = str(ROUND_ID)
    conf4_scoring['data']['APPLY_FILE'] = APPLY_FILE
    conf4_scoring['data']['PROJECT_PATH'] = PROJECT_PATH
    conf4_scoring['data']['OUTPUT_PATH'] = OUTPUT_PATH
    conf4_scoring['scoring'] = {}
    conf4_scoring['scoring']['SCORE_W_GPU'] = config['scoring']['SCORE_W_GPU']
    conf4_scoring['scoring']['CLF_THRESHOLD'] = config['scoring']['CLF_THRESHOLD']
    conf4_scoring['scoring']['APPLY_BATCH_SIZE'] = config['scoring']['APPLY_BATCH_SIZE']
    conf4_path = f"{CONFIG_PATH.rstrip('/')}/al_4_score.cfg"
    print(f">>> Saving scoring config to {conf4_path}")
    with open(conf4_path, 'w') as file:
        conf4_scoring.write(file)
    
    print(f"==========ROUND {ROUND_ID}, STAGE 4.2: Scoring Execution==========")
    print(f">>> Running subprocess on Model Scoring for Round {ROUND_ID}...")
    start = datetime.now()
    result = subprocess.run(['python', 'al_4_score_models.py', '--config_path', conf4_path], stdout=subprocess.PIPE, encoding='utf8')
    print(result.stdout)
    print(f"The model scoring subprocess was completed within {(datetime.now() - start).seconds:,} seconds")

print(">>> Done")
    