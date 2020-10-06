"""
This script takes in any text file consisting of text per row and create a sample JSON form compatible with the module
"""

import pandas as pd
import numpy as np
import ui_utils

with open('./input/test_train_input_100.txt') as file:
    texts = list(file.readlines())

text_df = pd.DataFrame({"Text": [text.strip() for text in texts]})
text_df = text_df.reset_index().rename({'index': 'UID'}, axis=1)
ui_utils.df_to_json_form(text_df, tag_col='Tags', ui_dir='./input/',
                         ui_filename='amz_reviews_100.json')