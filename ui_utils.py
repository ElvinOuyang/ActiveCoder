import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import configparser
from dateutil.parser import parse
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score,\
    recall_score, classification_report, accuracy_score

import logging

logger = logging.getLogger(__name__)
print = logger.info

def multilabel_from_tags(tag_list):
    """
    function to generate pd dataframe for tags based on list of tag strings
    tag_list: the raw list of tags from input. each row is "tag1, tag2, tag3..."
    """
    # turn tag list strings into list for each row
    tag_list = [[tag.strip() for tag in tag_text.split(',')] for tag_text in tag_list]
    # obtain unique tags
    unique_tags = list(set([tag for tags in tag_list for tag in tags]))
    try:
        unique_tags.remove('')
    except:
        print("Unique tags does not have empty situations")
    # create df based on tags
    tag_dict = {}
    for tag in unique_tags:
        tag_dict[f"Tag_{tag}"] = [1 if tag in tags else 0 for tags in tag_list]
    tag_df = pd.DataFrame(tag_dict)
    return tag_df


def create_tag_columns(train_df, tag_col='Tags'):
    """
    function to create tags columns for a training dataframe
    train_df: pd DataFrame of training text and tags
    tag_col: str. Column name of the column that houses the multilabel tags
    """
    tag_list = train_df[tag_col].to_list()
    tag_df = multilabel_from_tags(tag_list)
    train_df = pd.concat([train_df, tag_df], axis=1)
    return train_df


def binary_tag_to_tags(text_df, tag_values):
    """
    +++INPUT+++
    text_df: dataframe with binary tags, fillna with 0
    tag_values: array of tag strings
        example: tag_values = text_df.columns[2:].values
    +++OUTPUT+++
    text_df: with Tags column added containing tags
    """
    tags_list = []
    for row_index in range(len(text_df)):
        selector = text_df.loc[row_index, tag_values].values.astype(bool)
        selected_tags = tag_values[selector]
        tags_string = ", ".join(selected_tags)
        tags_list.append(tags_string)
    text_df['Tags'] = tags_list
    return text_df


def df_to_json_form(sample_df, tag_col='Tags', ui_dir='../input/',
                    ui_filename='text_tags.json'):
    """
    function to save a sampled text df to directory for human tags
    sample_df: pd.DataFrame. Has "Text" and "UID" columns
    tag_col: str. The expected name of the tags column. Blank fields will be
             populated for human input
    ui_dir: str. directory of the human input json form
    ui_filename: str. file name for the human input. should be in json
    """
    try:
        assert "Text" in sample_df.columns
        assert "UID" in sample_df.columns
    except:
        print("Make sure the DF has Text and UID columns!")
        exit(1)
    if tag_col not in sample_df.columns:
        print(f"Column {tag_col} not in columns. Adding empty column for it.")
        sample_df[tag_col] = ''
    sample_df = sample_df.loc[:, ['Text', 'UID', tag_col]]
    print("Saving the sampled texts as JSON for human tags")
    Path(ui_dir).mkdir(parents=True, exist_ok=True)
    sample_df.to_json(f'{ui_dir}{ui_filename}', orient='records', indent=2)
    print("Done")


def kmeans_from_proba(scored_df, tsne_fig_name, score_col_prefix='proba_', random_state=0):
    print("Extracting tag scores and training KMeans for clusters")
    # extract tag scores into np.array
    proba_scores = scored_df.loc[:, scored_df.columns.str.startswith(score_col_prefix)].values
    # fit and extract kmeans clusters
    kmeans = KMeans(n_clusters=proba_scores.shape[1] + 1, random_state=random_state)
    kmeans.fit(proba_scores)
    clusters = kmeans.predict(proba_scores).reshape((-1, 1))
    print("Visualizing tag score-based KMeans clusters with tSNE")
    # visualize the clusters using tsne
    tsne_xy = TSNE(n_components=2).fit_transform(proba_scores)
    visualize_df = pd.DataFrame(
        np.concatenate((tsne_xy, clusters), axis=1), columns=['tsne_1', 'tsne_2', 'cluster_id'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=visualize_df,x='tsne_1',y='tsne_2',hue='cluster_id',
                    legend="full",alpha=0.5, palette='pastel')
    plt.title("KMeans Cluster on TSNE 2D Transformation")
    plt.savefig(tsne_fig_name, bbox_inches='tight')
    plt.close()
    # save cluster info back to scored_df
    print("Saving cluster information back to dataframe")
    scored_df['cluster_id'] = clusters
    return scored_df, kmeans


def sample_by_cluster(scored_df, sample_size, cluster_col='cluster_id', row_key='UID'):
    print("Sampling records based on cluster information...")
    group_sample_n = sample_size // scored_df[cluster_col].nunique()
    sample_df = scored_df.groupby(cluster_col).apply(lambda x: x.sample(n=group_sample_n)).reset_index(drop=True)
    unsampled_count = sample_size - sample_df.shape[0]
    print(f"A total of {sample_df.shape[0]:,} records were sampled based on clusters.")
    if unsampled_count > 0:
        print(f"{unsampled_count:,} remaining records are to be sampled from total population.")
        unsampled_ids = scored_df[row_key][~np.isin(scored_df.UID, sample_df.UID)]
        additional_ids = np.random.choice(unsampled_ids, unsampled_count, replace=False)
        additional_df = scored_df.loc[np.isin(scored_df[row_key], additional_ids), :]
        sample_df = pd.concat([sample_df, additional_df], ignore_index=True)
    sample_df['Tags'] = ''
    return sample_df


def sample_by_random(scored_df, sample_size, cluster_col='cluster_id', row_key='UID'):
    print("Sampling records based on pure randomness...")
    print(f"{sample_size:,} records are to be sampled from total population.")
    sample_ids = np.random.choice(scored_df[row_key], sample_size, replace=False)
    sample_df = scored_df.loc[np.isin(scored_df[row_key], sample_ids), :].reset_index(drop=True)
    sample_df['Tags'] = ''
    return sample_df


def coder_sim(samples_df, answers_df):
    assert "UID" in samples_df.columns
    assert "UID" in answers_df.columns
    assert "Tags" in samples_df.columns
    assert "Tags" in answers_df.columns
    samples_df['Tags'] = answers_df.set_index("UID").loc[samples_df.UID, ['Tags']].values.flatten()
    print("Samples have been tagged using the provided answers dataframe")
    return samples_df


class MetaProject(object):
    def __init__(self, project_path, rundir='./wrapper_al/'):
        """
        Simple MetaProject class to analyze project output
        project_path: path to the project folder of the active learning run
        rundir: the path where the active learning ran, default './wrapper_al/'
        """
        print(">>> Instantiate MetaProject class...")
        self.project_path = project_path
        self.rundir = rundir
        self.cfg_path = os.path.abspath(f'{self.project_path}orchestration_record.cfg')
        self.log_path = os.path.abspath(f'{self.project_path}orchestration_log.log')
        self._load_config()
        self.total_rounds = int(self.config.get('active_learning', 'total_rounds'))
        self.round_sample = int(self.config.get('sampling', 'sample_size'))
        self.total_sample = self.total_rounds * self.round_sample
        # get abspath of the answer file since the exec path of project is different from analytics path
        self.answer_file = os.path.abspath(os.path.join(
            self.rundir, self.config.get('coder_sim', 'answer_file')))
        print(self.answer_file)
        self.max_tags = int(self.config.get('training', 'max_tags'))
        self.run_sim = int(self.config.get('active_learning', 'run_sim'))
        self.run_time = self._parse_log(self.log_path)
        self._gen_tag_sum_df(self.answer_file)
        
    def _load_config(self):
        print(">>> Loading project orchestration config")
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(self.cfg_path)

    def _parse_log(self, log_path):
        """
        Method to parse orchestration log file to obtain run duration in seconds
        """
        print(">>> Parsing project execution run time")
        with open(log_path, 'r') as logfile:
            first_line = logfile.readline()
            for last_line in logfile:
                pass
            try:
                start_time = parse(first_line[:23])
                end_time = parse(last_line[:23])
                run_time = (end_time - start_time).seconds
            except:
                print(">>> Project did not run successfully based on log records!")
                run_time = -1
            return run_time
    
    def _gen_tag_sum_df(self, tag_col='Tag_'):
        """
        Method to generate tag positive ratios of a given DF (stored in JSON format)
        """
        print(">>> Reading full dataset...")
        df = pd.read_json(self.answer_file, orient='records')
        df = create_tag_columns(df)
        self.df = df
        self.total_records = df.shape[0]
        if self.run_sim == 1:
            print(">>> Project ran as simulation...")
            self.answer_tag_sum_df = df.loc[:, df.columns.str.startswith(tag_col)].sum().sort_values(
                ascending=False).reset_index().rename(
                {'index':'Tag_Name', 0: 'Pos_Count'}, axis=1)
            self.answer_tag_sum_df['Pos_Rate'] = self.answer_tag_sum_df.Pos_Count / df.shape[0]
        else:
            print(">>> Project ran in real time with manual coders...")
            self.answer_tag_sum_df = None
    
    def describe(self):
        """
        Method to describe the project with Meta Cfg and Logs
        method only loads attributes of the object
        """
        print(">>> Composing project high level description...")
        self.stmts = []
        self.stmts.append('INTRO\n-------')
        self.stmts.append(f"\nThis Active Learning Run has a round count of {self.total_rounds:,},")
        self.stmts.append(f"and a total of {self.total_sample:,} samples are included for model training.")
        if self.run_sim == 1:
            self.stmts.append("This run is a simulation with known tags already available.")
        else:
            self.stmts.append("This run is an actual application with manual coder input for tags on the fly.")
        self.stmts.append(f"In each round, {int(self.config.get('sampling', 'sample_size')):,} samples are selected as additional training data.")
        self.stmts.append(f"While the first round always runs random sampling to gather the samples,")
        self.stmts.append(f"the second and beyond rounds use {self.config.get('sampling', 'sampling_method')} method.")
        self.stmts.append('\n\nDATA\n-------')
        self.stmts.append(f'\nThe input dataframe has a total of {self.total_records:,} records.')
        if self.answer_tag_sum_df is not None:
            self.stmts.append('The positive rates of each tag in the full answer dataset:')
            self.stmts.append("\n" + self.answer_tag_sum_df.to_string())
        self.stmts.append('\n\nMODELING\n-------')
        self.stmts.append("\nThe training config for each round's Bi-Directional LSTM modeling is as below:")
        for key, value in dict(self.config['training']).items():
            self.stmts.append(f"\n\t{key}: {value}")
        if self.config.get('training', 'random_embed') == 'True':
            self.stmts.append('\nThe text embeddings are randomly initiated 300-length via Tensorflow 2.')
        else:
            self.stmts.append('\nThe text embeddings are GloVe 300-length text embeddings loaded via Spacy.')
        self.stmts.append('\n\nRUNTIME\n-------')
        if self.run_time > 0:
            self.stmts.append(f"\nExecution of the run took {self.run_time / 60:,.2f} minutes to complete")
        else:
            self.stmts.append("Program log file indicates that this run was not successfully executed...")
        self.description = " ".join(self.stmts)
        print(">>> Displaying the description:")
        print(self.description)



class RoundResult(object):
    def __init__(self, round_path, answer_file, proba_cutoff, rundir='./wrapper_al/'):
        self.round_path = os.path.abspath(os.path.join(rundir, round_path))
        print(self.round_path)
        self.config_dir = f"{self.round_path.rstrip('/')}/config/"
        self.sample_dir = f"{self.round_path.rstrip('/')}/sample/"
        self.label_dir = f"{self.round_path.rstrip('/')}/label/"
        self.input_dir = f"{self.round_path.rstrip('/')}/input/"
        self.output_dir = f"{self.round_path.rstrip('/')}/output/"
        self.train_file = f"{self.output_dir.rstrip('/')}/train_df.csv"
        self.scored_file = f"{self.output_dir.rstrip('/')}/scored/scored_output.json"
        self.answer_file = os.path.abspath(os.path.join(rundir, answer_file))
        self.proba_cutoff = proba_cutoff
        self.load_outputs()
        
    def load_outputs(self, proba_prefix='proba_', tag_prefix='Tag_', row_key='UID'):
        # read the round related datasets
        train_df = pd.read_csv(self.train_file)
        scored_df = pd.read_json(self.scored_file, orient='records')
        answer_df = pd.read_json(self.answer_file, orient='records')
        answer_df = create_tag_columns(answer_df)
        # prepare col selectors
        proba_cols = scored_df.columns[scored_df.columns.str.startswith(proba_prefix)].tolist()
        tag_names = [proba_col.replace(proba_prefix, '').strip() for proba_col in proba_cols]
        true_tag_cols = [f"{tag_prefix}{tag}" for tag in tag_names]
        # prepare row selectors
        all_ids = answer_df[row_key].unique()
        train_ids = train_df[row_key].unique()
        test_ids = [uid for uid in all_ids if uid not in train_ids]
        # create 2 dicts for round outputs and results
        round_outputs = {}
        round_results = {}
        for tag_name, proba_col, true_tag_col in zip(tag_names, proba_cols, true_tag_cols):
            round_outputs[tag_name] = {}
            round_results[tag_name] = {}
            # save the y_true, y_pred, and y_proba for train and test runs
            round_outputs[tag_name]['train_y_proba'] = scored_df.loc[scored_df[row_key].isin(train_ids), proba_col].values
            round_outputs[tag_name]['train_y_pred'] = (round_outputs[tag_name]['train_y_proba'] >= self.proba_cutoff).astype(int)
            round_outputs[tag_name]['train_y_true'] = answer_df.loc[answer_df[row_key].isin(train_ids), true_tag_col].values
            round_outputs[tag_name]['test_y_proba'] = scored_df.loc[scored_df[row_key].isin(test_ids), proba_col].values
            round_outputs[tag_name]['test_y_pred'] = (round_outputs[tag_name]['test_y_proba'] >= self.proba_cutoff).astype(int)
            round_outputs[tag_name]['test_y_true'] = answer_df.loc[answer_df[row_key].isin(test_ids), true_tag_col].values

            # calculate train side metrics
            round_results[tag_name]['train_roc_auc'] = roc_auc_score(
                round_outputs[tag_name]['train_y_true'], round_outputs[tag_name]['train_y_proba'])
            round_results[tag_name]['train_f1'] = f1_score(
                round_outputs[tag_name]['train_y_true'], round_outputs[tag_name]['train_y_pred'], zero_division=0)
            round_results[tag_name]['train_precision'] = precision_score(
                round_outputs[tag_name]['train_y_true'], round_outputs[tag_name]['train_y_pred'], zero_division=0)
            round_results[tag_name]['train_recall'] = recall_score(
                round_outputs[tag_name]['train_y_true'], round_outputs[tag_name]['train_y_pred'], zero_division=0)
            round_results[tag_name]['train_cr'] = classification_report(
                round_outputs[tag_name]['train_y_true'], round_outputs[tag_name]['train_y_pred'], zero_division=0)
            round_results[tag_name]['train_f1'] = roc_auc_score(
                round_outputs[tag_name]['train_y_true'], round_outputs[tag_name]['train_y_pred'])
            round_results[tag_name]['train_pos_rate'] = round_outputs[tag_name]['train_y_true'].sum() \
                / round_outputs[tag_name]['train_y_true'].shape[0]
            # calculate test side metrics
            round_results[tag_name]['test_roc_auc'] = roc_auc_score(
                round_outputs[tag_name]['test_y_true'], round_outputs[tag_name]['test_y_proba'])
            round_results[tag_name]['test_f1'] = f1_score(
                round_outputs[tag_name]['test_y_true'], round_outputs[tag_name]['test_y_pred'], zero_division=0)
            round_results[tag_name]['test_precision'] = precision_score(
                round_outputs[tag_name]['test_y_true'], round_outputs[tag_name]['test_y_pred'], zero_division=0)
            round_results[tag_name]['test_recall'] = recall_score(
                round_outputs[tag_name]['test_y_true'], round_outputs[tag_name]['test_y_pred'], zero_division=0)
            round_results[tag_name]['test_cr'] = classification_report(
                round_outputs[tag_name]['test_y_true'], round_outputs[tag_name]['test_y_pred'], zero_division=0)
            round_results[tag_name]['test_f1'] = roc_auc_score(
                round_outputs[tag_name]['test_y_true'], round_outputs[tag_name]['test_y_pred'])
            round_results[tag_name]['test_pos_rate'] = round_outputs[tag_name]['test_y_true'].sum() \
                / round_outputs[tag_name]['test_y_true'].shape[0]
        self.round_outputs = round_outputs
        self.round_results = round_results
    
    def describe_round_metrics(self):
        self.stmts = []
        for tag in self.round_results.keys():
            self.stmts.append(f"==========Tag - {tag.upper()}==========")
            self.stmts.append(f"\n>>> Pos Rate: Train - {self.round_results[tag]['train_pos_rate'] * 100:.2f}%; Test - {self.round_results[tag]['test_pos_rate'] * 100:.2f}%")
            self.stmts.append(f"\n>>> ROC AUC: Train - {self.round_results[tag]['train_roc_auc']:.3f}; Test - {self.round_results[tag]['test_roc_auc']:.3f}\n")
            self.stmts.append("\n>>> Classification Reports:")
            self.stmts.append(f"\n>>> Train:\n {self.round_results[tag]['train_cr']}")
            self.stmts.append(f"\n>>> Test:\n {self.round_results[tag]['test_cr']}")
            self.stmts.append('\n======================================\n')
        self.description = " ".join(self.stmts)
        print(">>> Displaying the description:")
        print(self.description)


class MetaProjectWithRounds(object):
    def __init__(self, project_path, rundir='./wrapper_al/', alternative_cutoff=None):
        """
        Comprehensive Meta Project loader that also loads results of each round
        project_path: path to the project folder of the active learning run
        rundir: the path where the active learning ran, default './wrapper_al/'
        """
        print(">>> Instantiate MetaProject class...")
        self.project_path = project_path
        self.rundir = rundir
        self.cfg_path = os.path.abspath(f'{self.project_path}orchestration_record.cfg')
        self.log_path = os.path.abspath(f'{self.project_path}orchestration_log.log')
        self._load_config()
        self.total_rounds = int(self.config.get('active_learning', 'total_rounds'))
        self.round_sample = int(self.config.get('sampling', 'sample_size'))
        self.total_sample = self.total_rounds * self.round_sample
        # get abspath of the answer file since the exec path of project is different from analytics path
        self.answer_file = os.path.abspath(os.path.join(
            self.rundir, self.config.get('coder_sim', 'answer_file')))
        if alternative_cutoff is not None:
            self.proba_cutoff = alternative_cutoff
        else:
            self.proba_cutoff = float(self.config.get('scoring', 'clf_threshold'))
        self.max_tags = int(self.config.get('training', 'max_tags'))
        self.run_sim = int(self.config.get('active_learning', 'run_sim'))
        self.run_time = self._parse_log(self.log_path)
        self._gen_tag_sum_df(self.answer_file)
        self._load_rounds()
        
    def _load_config(self):
        print(">>> Loading project orchestration config")
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(self.cfg_path)

    def _parse_log(self, log_path):
        """
        Method to parse orchestration log file to obtain run duration in seconds
        """
        print(">>> Parsing project execution run time")
        with open(log_path, 'r') as logfile:
            first_line = logfile.readline()
            for last_line in logfile:
                pass
            try:
                start_time = parse(first_line[:23])
                end_time = parse(last_line[:23])
                run_time = (end_time - start_time).seconds
            except:
                print(">>> Project did not run successfully based on log records!")
                run_time = -1
            return run_time
    
    def _gen_tag_sum_df(self, tag_col='Tag_'):
        """
        Method to generate tag positive ratios of a given DF (stored in JSON format)
        """
        print(">>> Reading full dataset...")
        df = pd.read_json(self.answer_file, orient='records')
        df = create_tag_columns(df)
        self.df = df
        self.total_records = df.shape[0]
        if self.run_sim == 1:
            print(">>> Project ran as simulation...")
            self.answer_tag_sum_df = df.loc[:, df.columns.str.startswith(tag_col)].sum().sort_values(
                ascending=False).reset_index().rename(
                {'index':'Tag_Name', 0: 'Pos_Count'}, axis=1)
            self.answer_tag_sum_df['Pos_Rate'] = self.answer_tag_sum_df.Pos_Count / df.shape[0]
        else:
            print(">>> Project ran in real time with manual coders...")
            self.answer_tag_sum_df = None
    
    def _load_rounds(self):
        print(">>> Loading results for each round...")
        self.rounds = {}
        self.round_results = {}
        self.round_outputs = {}
        self.round_descriptions = {}
        for round_id in range(self.total_rounds):
            config_project_path = self.config.get('active_learning', 'project_path')
            round_path = f"{config_project_path.rstrip('/')}/round_{round_id + 1}/"
            self.rounds[round_id + 1] = RoundResult(
                round_path=round_path, answer_file=self.answer_file, 
                proba_cutoff=self.proba_cutoff, rundir=self.rundir)
            self.round_results[round_id + 1] = self.rounds[round_id + 1].round_results
            self.round_outputs[round_id + 1] = self.rounds[round_id + 1].round_outputs
        self._flatten_results()
    
    def _flatten_results(self):
        self.flatten_result_dict = {}
        self.flatten_result_dict['round_id'] = []
        self.flatten_result_dict['tag_name'] = []

        for round_id, round_result in self.round_results.items():
            for tag_name, model_scores in round_result.items():
                self.flatten_result_dict['round_id'].append(round_id)
                self.flatten_result_dict['tag_name'].append(tag_name)
                for metric_name, metric_value in model_scores.items():
                    if metric_name not in self.flatten_result_dict.keys():
                        self.flatten_result_dict[metric_name] = [metric_value]
                    else:
                        self.flatten_result_dict[metric_name].append(metric_value)
        self.flatten_result_df = pd.DataFrame(self.flatten_result_dict)
    
    def describe(self):
        """
        Method to describe the project with Meta Cfg and Logs
        method only loads attributes of the object
        """
        print(">>> Composing project high level description...")
        self.stmts = []
        self.stmts.append('INTRO\n-------')
        self.stmts.append(f"\nThis Active Learning Run has a round count of {self.total_rounds:,},")
        self.stmts.append(f"and a total of {self.total_sample:,} samples are included for model training.")
        if self.run_sim == 1:
            self.stmts.append("This run is a simulation with known tags already available.")
        else:
            self.stmts.append("This run is an actual application with manual coder input for tags on the fly.")
        self.stmts.append(f"In each round, {int(self.config.get('sampling', 'sample_size')):,} samples are selected as additional training data.")
        self.stmts.append(f"While the first round always runs random sampling to gather the samples,")
        self.stmts.append(f"the second and beyond rounds use {self.config.get('sampling', 'sampling_method')} method.")
        self.stmts.append('\n\nDATA\n-------')
        self.stmts.append(f'\nThe input dataframe has a total of {self.total_records:,} records.')
        if self.answer_tag_sum_df is not None:
            self.stmts.append('The positive rates of each tag in the full answer dataset:')
            self.stmts.append("\n" + self.answer_tag_sum_df.to_string())
        self.stmts.append('\n\nMODELING\n-------')
        self.stmts.append("\nThe training config for each round's Bi-Directional LSTM modeling is as below:")
        for key, value in dict(self.config['training']).items():
            self.stmts.append(f"\n\t{key}: {value}")
        if self.config.get('training', 'random_embed') == 'True':
            self.stmts.append('\nThe text embeddings are randomly initiated 300-length via Tensorflow 2.')
        else:
            self.stmts.append('\nThe text embeddings are GloVe 300-length text embeddings loaded via Spacy.')
        self.stmts.append('\n\nRUNTIME\n-------')
        if self.run_time > 0:
            self.stmts.append(f"\nExecution of the run took {self.run_time / 60:,.2f} minutes to complete")
        else:
            self.stmts.append("Program log file indicates that this run was not successfully executed...")
        self.description = " ".join(self.stmts)
        print(">>> Displaying the description:")
        print(self.description)
        