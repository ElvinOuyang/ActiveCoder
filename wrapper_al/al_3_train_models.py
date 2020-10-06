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
import subprocess
from tensorboard import program
import argparse

import ui_utils
import nlp_tools

# allow gpu memory growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

arg_parser = argparse.ArgumentParser(description='Run model training for each tagged label')
arg_parser.add_argument('--config_path', metavar='config_path', default='./al_3_train.cfg',
                        type=str, help='Path to the training config file', required=False)
args = arg_parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_path)

MAX_TAGS = int(config.get('training', 'MAX_TAGS'))
BATCH_SIZE = int(config.get('training', 'BATCH_SIZE'))
BUFFER_SIZE = int(config.get('training', 'BUFFER_SIZE'))
INPUT_FILE = config.get('data', 'INPUT_FILE')
OUTPUT_PATH = config.get('data', 'OUTPUT_PATH')
EARLY_STOPPING_ROUNDS = int(config.get('training', 'EARLY_STOPPING_ROUNDS'))
EMBED_TRAINABLE = bool(config.get('training', 'EMBED_TRAINABLE') == 'True')
RANDOM_EMBED = bool(config.get('training', 'RANDOM_EMBED') == 'True')
DROPOUT_LEVEL = float(config.get('training', 'DROPOUT_LEVEL'))
LEARNING_RATE = float(config.get('training', 'LEARNING_RATE'))
N_LSTM_UNITS = int(config.get('training', 'N_LSTM_UNITS'))
N_FC_NEURONS = int(config.get('training', 'N_FC_NEURONS'))

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# initialize logger
root_logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s: %(levelname)s:: %(message)s')

# prints to file
logfile = f"{OUTPUT_PATH.rstrip('/')}/train_logs.log"
file_handler = logging.FileHandler(logfile, mode='a')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

root_logger.setLevel(logging.INFO)

print = root_logger.info

print(">>> Using Tensorflow 2 with GPU for this script:")
print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))

print(">>> Loading training dataframes")
train_df = pd.read_json(INPUT_FILE, orient='records')

train_df = ui_utils.create_tag_columns(train_df)
print(">>> Processed training dataframe with tags:")
print(train_df.head(10))

print(f">>> Displaying the Count of Positive Tags out of {train_df.shape[0]:,} rows")
print(train_df.loc[:, train_df.columns.str.startswith('Tag_')].sum().sort_values(ascending=False))

print(">>> Saving training df for record keeping purpose")
train_df.to_csv(f'{OUTPUT_PATH.rstrip("/")}/train_df.csv', index=False)

print(">>> Tokenize texts...")
tokenizer, text_seqs = nlp_tools.train_tokenizer(train_df)

# For Debugging logs
# exit(0)

print(">>> Saving tokenizer to disk for records")
with open(f'{OUTPUT_PATH.rstrip("/")}/tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file)

print ('>>> Vocabulary size: {:,}'.format(len(tokenizer.word_index.items())))
if not RANDOM_EMBED:
    en_model = spacy.load('en_core_web_lg')

target_names = train_df.loc[:, train_df.columns.str.startswith('Tag_')].sum().sort_values(ascending=False).index.tolist()
print(f">>> {len(target_names)} labels are identified: {', '.join(target_names)}")

if len(target_names) > MAX_TAGS:
    print(f"Too many tags...Only processing the first {MAX_TAGS:,} tags")
    target_names = target_names[:MAX_TAGS]

for target in target_names:
    print("\n---------------------------")
    print(f"Training for target label: {target}".upper())
    print("---------------------------\n")
    target_list = train_df.loc[:, target].tolist()

    print(">>> Spliting data for training and validation")
    test_seqs, test_targets, train_seqs, train_targets = nlp_tools.data_split(
        text_seqs, target_list, test_rate=0.1)
    # flow control: if data split returns empty list, the data does not have enough
    # targets for splitting. Check the function for details
    if len(test_seqs) == 0 and len(train_seqs) == 0:
        print(f">>> WARNING: Target label {target} does not have enough labels to split for testing. Skipping this label!!!")
        continue
    train_ds = nlp_tools.create_balanced_ds(
        train_seqs, train_targets, tokenizer, BATCH_SIZE, BUFFER_SIZE)
    test_ds = nlp_tools.create_balanced_ds(
        test_seqs, test_targets, tokenizer, BATCH_SIZE, BUFFER_SIZE)
    print(">>> tf2 Datasets are ready for modeling")

    print(">>> Creating callbacks for tf2 training")
    TRAINING_OUTPUT_DIR = f'{OUTPUT_PATH.rstrip("/")}/{target}_training/'
    Path(TRAINING_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{TRAINING_OUTPUT_DIR}{target}_ckpt', save_best_only=True,
        save_weights_only=True)
    # using an early stopping rule to stop training when model stops improving in validation
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOPPING_ROUNDS, restore_best_weights=True)
    
    print(">>> Initializing model\n")
    if RANDOM_EMBED:
        print(">>> Generating random text embeddings")
        embed_layer = tf.keras.layers.Embedding(len(tokenizer.word_index.items()) + 1, 300, mask_zero=True)
    else:
        print(">>> Generating text embeddings using Spacy GloVe vectors")
        embed_layer = nlp_tools.get_pretrained_embedding_layer(tokenizer, en_model)


    print(">>> Determine if Embedding Layer is trainable based on parameter")
    # freeze the embeddings layer at early stages of the modeling
    embed_layer.trainable = EMBED_TRAINABLE
    print(">>> Building model structure...")
    # simple bidirectional LSTM network
    model = tf.keras.Sequential([
        embed_layer,
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(N_LSTM_UNITS,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(N_LSTM_UNITS)),
        tf.keras.layers.Dense(N_FC_NEURONS, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT_LEVEL),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    # model compiling settings
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    print(model.summary())
    
    print(">>> Training model\n")
    history = model.fit(train_ds, epochs=2000,
                        steps_per_epoch=10,
                        validation_data=test_ds, 
                        validation_steps=10,
                        callbacks=[
                            model_checkpoint, early_stopper,
                            nlp_tools.get_run_callback(target, OUTPUT_PATH)])
    
    print(">>> Saving model")
    model.save(f"{TRAINING_OUTPUT_DIR}{target}_model.h5")
    print(">>> Evaluating model with test data")
    test_loss, test_acc, test_auc = model.evaluate(test_ds, steps=100)

    print('>>> Test Loss: {}'.format(test_loss))
    print('>>> Test Accuracy: {}'.format(test_acc))
    print('>>> Test AUROC: {}'.format(test_auc))

    print('>>> Plotting charts')
    nlp_tools.plot_graphs(history, 'accuracy', f"{TRAINING_OUTPUT_DIR}learning_curve_accuracy.png", tag_name=target)
    nlp_tools.plot_graphs(history, 'loss', f"{TRAINING_OUTPUT_DIR}learning_curve_loss.png", tag_name=target)
    nlp_tools.plot_graphs(history, 'auc', f"{TRAINING_OUTPUT_DIR}learning_curve_auroc.png", tag_name=target)

with open(f"{OUTPUT_PATH.rstrip('/')}/train_record.cfg", 'w') as file:
    config.write(file)
