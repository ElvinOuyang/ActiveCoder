import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import time
import logging

logger = logging.getLogger(__name__)
print = logger.info

def get_pretrained_embedding_layer(trained_tokenizer, spacy_model):
    """
    function to prepare pretrained embedding layer from spacy glove
    trained_tokenizer: the tokenizer object trained with training text
    spacy_model: the spacy large english model to generate embeddings
    """
    # obtain all vocab from tokenizer to be processed by spacy
    vocabs = " ".join(trained_tokenizer.word_index.keys())
    # create matrix of embeddings from spacy
    doc = spacy_model(vocabs)
    vectors = []
    for token in doc:
        vectors.append(token.vector)
    vectors = np.array(vectors)
    # infer embedding layer shape from vectors
    output_dim = vectors.shape[1]
    # word index starts from 1, thus 0 does not match a word
    input_dim = vectors.shape[0] + 1
    # initiate an embedding layer, get weights and update with embed matrix
    spacy_embed_layer = tf.keras.layers.Embedding(
        input_dim, output_dim, mask_zero=True)
    spacy_embed_layer.build((None, input_dim))
    weights = spacy_embed_layer.get_weights()[0]
    # again, 0 does not match any word, so replacement start from index 1
    weights[1:, :] = vectors
    spacy_embed_layer.set_weights([weights])
    return spacy_embed_layer


def train_tokenizer(train_df):
    """
    train_df: pd.DataFrame. training dataframe that has "Text" and "UID" columns
    """
    texts = [str(text) for text in train_df.Text.tolist()]
    tokenizer = Tokenizer(num_words=100000,
                          lower=True,
                          filters='"#$%&()*+-/:<=>@[\\]^_`{|}~\t\n',
                          char_level=False,
                          oov_token=None)
    tokenizer.fit_on_texts(texts)
    text_seqs = tokenizer.texts_to_sequences(texts)
    return tokenizer, text_seqs


def seq_gen(text_seqs, target_list):
    """
    generator function to be used for tf2 Dataset API
    text_seqs: the list of text id sequence generated by tokenizer.texts_to_sequences()
    target_list: the list of numbers, each number is the classification target of the text
    """
    try:
        assert len(text_seqs) == len(target_list)
    except:
        print("text_seqs should have same length with target_list")
        exit(1)
    while True:
        # select a response
        doc_count = len(text_seqs)
        seq_index = np.random.choice(range(doc_count))
        text_seq = text_seqs[seq_index]
        target_val = target_list[seq_index]
        # generate a random slice
        seq_length = len(text_seq)
        if seq_length <= 1:  # when seq short, use all seq
            yield (text_seq, target_val)
        else:
            indexs = np.random.choice(
                range(seq_length), 2, replace=False)
            #print(f"Indexs: {indexs}")
            start_idx, end_idx = indexs.min(), indexs.max()
            yield (text_seq[start_idx:end_idx], target_val)


def data_split(text_seqs, target_list, test_rate=0.1):
    """
    function to split text seqs and target list into train and test sets
    """
    status = True
    i = 0
    while status:
        test_count = round(len(text_seqs) * test_rate)
        print(f"Out of {len(text_seqs):,} samples, {test_count:,} will be used as test samples")
        test_index = np.random.choice(len(text_seqs), test_count)
        print(test_index)
        train_index = [idx for idx in range(len(text_seqs)) if idx not in test_index]
        text_seqs, target_list = np.array(text_seqs), np.array(target_list)
        test_seqs, test_targets = text_seqs[test_index], target_list[test_index]
        train_seqs, train_targets = text_seqs[train_index], target_list[train_index]
        i += 1
        if test_targets.sum() > 0 and train_targets.sum() > 0:
            print(f"Training Positive Count: {train_targets.sum()}; Testing Positive Count: {test_targets.sum()}")
            status = False
        elif i >= 50:
            print("WARNING: Split still results in single class targets after 50 redos. Skipping this class!!!")
            print("All returned sequences are empty lists!!!")
            status = False
            test_seqs, test_targets, train_seqs, train_targets = [], [], [], []
        else:
            print("Split results in single class targets, redo split!")
            
    return test_seqs, test_targets, train_seqs, train_targets


def create_balanced_ds(text_seqs, target_list, trained_tokenizer,
                       batch_size=50, buffer_size=5000):
    """
    text_seqs: the list of text id sequence generated by tokenizer.texts_to_sequences()
               MUST INCLUDE BOTH POS AND NEG SEQS!
    target_list: the list of numbers, each number is the classification target of the text
    trained_tokenizer: the tokenizer object trained with training text
    batch_size: int. the mini-batch size for the dataset
    buffer_size: int. the amount of preloaded seqs to speed up data feeding for neural nets
    """
    # separate text_seqs and target_list to be pos and neg seqs
    pos_text_seqs = [text_seqs[i] for i in range(len(target_list)) if target_list[i] == 1]
    neg_text_seqs = [text_seqs[i] for i in range(len(target_list)) if target_list[i] == 0]
    # create 2 tf2 dataset, one for pos and one for neg
    pos_ds = tf.data.Dataset.from_generator(
        lambda: seq_gen(pos_text_seqs, [1] * len(pos_text_seqs)),
        output_types=(tf.int32, tf.int32),
        output_shapes=([None,], []))
    neg_ds = tf.data.Dataset.from_generator(
        lambda: seq_gen(neg_text_seqs, [0] * len(neg_text_seqs)),
        output_types=(tf.int32, tf.int32),
        output_shapes=([None,], []))
    # mingle the 2 ds together to form balanced data_ds
    data_ds = tf.data.Dataset.zip((pos_ds, neg_ds)).flat_map(
        lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(tf.data.Dataset.from_tensors(x1)))
    data_ds = data_ds.repeat()\
        .shuffle(buffer_size=buffer_size, seed=0)\
        .padded_batch(batch_size=batch_size, padded_shapes=([None,], []),
                      padding_values=(0, 0))
    return data_ds


def plot_graphs(history, string, filename, tag_name):
    """
    function that plots training history from tf training record
    """
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.title(f"{tag_name.upper()} Learning Curve ({string.upper()})", fontsize=18)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    
def get_run_callback(run_text, output_dir):
    """
    function to generate tf run callback for tensor board
    """
    root_logdir = f'{output_dir.rstrip("/")}/tensorboard_logs/'
    run_id = time.strftime(f'{run_text}_%Y_%m_%d-%H-%M-%S')
    log_path = os.path.join(root_logdir, run_id)
    tensorboad_callback = tf.keras.callbacks.TensorBoard(log_path)
    return tensorboad_callback


def obtain_tag_clfs(train_output_dir, tag_prefix='Tag_'):
    """
    function to load all trained models from training module output
    train_output_dir: str. direct path of the training output
    tag_prefix: str. the prefix to the folders saving each trained model
    =======output=======
    taggers_dict: dict. Dictionary consisting of {tag_name: model} pairs
    """
    dirs = [f.name for f in os.scandir(train_output_dir) if f.is_dir() and f.name.startswith(tag_prefix)]
    taggers_dict = {}
    for dirname in dirs:
        #print(dirname)
        tag_name = dirname.replace(tag_prefix, '').replace('_training', '')
        #print(tag_name)
        model = tf.keras.models.load_model(
            f'{train_output_dir.rstrip("/")}/{dirname.rstrip("/")}/{tag_prefix}{tag_name}_model.h5')
        taggers_dict[tag_name] = model
        print(f"Loaded trained model for {tag_name}!")
    return taggers_dict


def generate_label_with_single_model(
    text_ls, trained_tokenizer, trained_model,
    batch_size=100, clf_threshold=0.8, gpu=True):
    """
    function to load one trained model and predict outputs for a list of texts
    text_ls: list of pure texts to be scored by trained model
    trained_tokenizer: loaded trained tokenizer used by training module
    trained_model: loaded trained model
    batch_size: batch size to be fed for scoring
    clf_threshold: float. between 0 and 1. threshold to assign 1 to a text
    """
    print(f"Input text has {len(text_ls):,} text pieces")
    print("Creating an input text Dataset for binary classfication")
    text_ls = [str(text) for text in text_ls]
    text_ds = tf.data.Dataset.from_generator(
        lambda: iter(trained_tokenizer.texts_to_sequences_generator(text_ls)),
        output_shapes=[None,], output_types=tf.int32)\
        .padded_batch(batch_size=batch_size, padded_shapes=[None,], padding_values=0)
    print("Loading trained classification model")
    print("Generating model scores with trained model")
    #print("WARNING: Scoring is processed using CPUs only!")
    if gpu == True:
        print("Using GPU to score!")
        with tf.device('/GPU:0'):
            pred_proba = trained_model.predict(text_ds).flatten()
    else:
        print("Using CPU to score!")
        with tf.device('/CPU:0'):
            pred_proba = trained_model.predict(text_ds).flatten()
    print(f"Label score thredhold is set to {clf_threshold: .2f}")
    pred_label = (pred_proba > clf_threshold).astype(int).flatten()
    print(f"Generated labels has shape {pred_label.shape}, with {pred_label.sum():,} positives.")
    return pred_proba, pred_label


def generate_labels_with_model_dict(
    text_ls, trained_tokenizer, model_dict,
    batch_size=100, clf_threshold=0.8, use_gpu=True):
    labels_dict = {}
    for tag_name in model_dict:
        print("======================================")
        print(f">>> Scoring texts with trained model for {tag_name.upper()}")
        proba_arr, label_arr = generate_label_with_single_model(
            text_ls=text_ls, trained_tokenizer=trained_tokenizer,
            trained_model=model_dict[tag_name],
            batch_size=batch_size, clf_threshold=clf_threshold,
            gpu=use_gpu)
        labels_dict[f"label_{tag_name}"] = label_arr
        labels_dict[f"proba_{tag_name}"] = proba_arr
        print(f">>> Scoring for {tag_name.upper()} completed!")
    return labels_dict