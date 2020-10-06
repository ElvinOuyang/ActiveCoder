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
import pickle

import ui_utils

# parse args
arg_parser = argparse.ArgumentParser(description='Run output analytics on completed al runs')
arg_parser.add_argument('project_path', metavar='project_path',
                        type=str, help='Path to active learning output folder')
arg_parser.add_argument('--run_dir', metavar='run_dir', default='.',
                        type=str, help='Path where the active learning script ran from', required=False)
arg_parser.add_argument('--alternative_cutoff', metavar='alternative_cutoff', default='None',
                        type=str, help='The probability cutoff number user defines to overwrite the cutoff from execution', required=False)

args = arg_parser.parse_args()

if args.alternative_cutoff == "None":
    alternative_cutoff = None
else:
    alternative_cutoff = float(args.alternative_cutoff)

# initialize logger
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s: %(levelname)s:: %(message)s')

logfile = os.path.join(args.project_path, 'output_analytics.log')
#logfile = f"{args.project_path}/output_analytics.log"

file_handler = logging.FileHandler(logfile, mode='a')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

root_logger.setLevel(logging.INFO)
print = root_logger.info

print(">>> Loading project outputs...")
parsed_project = ui_utils.MetaProjectWithRounds(
    project_path=args.project_path, rundir=args.run_dir,
    alternative_cutoff=alternative_cutoff)

print(">>> Output processed. Saving flatten model metrics to project folder")
analytics_dir = f"{args.project_path.rstrip('/')}/analytics/"

Path(analytics_dir).mkdir(parents=True, exist_ok=True)
parsed_project.flatten_result_df.to_csv(f"{analytics_dir.rstrip('/')}/modeling_metrics.csv", index=False)

print(">>> Saving loaded output object to project folder")
with open(f"{analytics_dir.rstrip('/')}/metaproject.pickle", 'wb') as file:
    pickle.dump(parsed_project, file)

print(">>> Done")


