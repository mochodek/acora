import re
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef as mcc_score

import unicodedata
from keras_bert import Tokenizer
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP, TOKEN_UNK

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger('acora.code')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

default_code_stop_delim = r"([\s\t\(\)\[\]{}!@#$%^&*\/\+\-=;:\\\\|`'\"~,.<>/?\n'])"

class CodeTokenizer(Tokenizer):

    def __init__(self,
                 token_dict,
                 token_cls=TOKEN_CLS,
                 token_sep=TOKEN_SEP,
                 token_unk=TOKEN_UNK,
                 pad_index=0,
                 cased=False,
                 code_stop_delim=default_code_stop_delim):
        """Initialize tokenizer.
        :param token_dict: A dict maps tokens to indices.
        :param token_cls: The token represents classification.
        :param token_sep: The token represents separator.
        :param token_unk: The token represents unknown token.
        :param pad_index: The index to pad.
        :param cased: Whether to keep the case.
        """
        
        super(CodeTokenizer, self).__init__(token_dict, 
                                            token_cls,
                                            token_sep,
                                            token_unk,
                                            pad_index,
                                            cased)
        self._code_stop_delim = code_stop_delim
        
        
    def tokenize_training(self, first, second=None):
        """Split text to tokens for BERT pre-training.
        :param first: First text.
        :param second: Second text.
        :return: A list of strings.
        """
        first_tokens = self._tokenize(first)
        second_tokens = self._tokenize(second) if second is not None else None
        return first_tokens, second_tokens

    def _tokenize(self, text):
        """Split text to tokens.
        :param first: First text.
        :param second: Second text.
        :return: A list of strings.
        """
        split_loc = re.split(self._code_stop_delim, text)
        split_loc = list(filter(lambda a: a != '', split_loc))
        
        tokens = []
        for word in split_loc:
            tokens += self._word_piece_tokenize(word)
        return tokens


def generate_code_pairs(file_lines, tokenizer, line_repeat_period=64, logger = None):
    """ Generates pairs of subsequent lines of code that can be used for BERT training."""

    # a line can only appear on each of the sides per given number of samples 
    # to minimize the chance of False negatives pairs while generating 
    left_lines = set()
    right_lines = set()

    no_files = len(file_lines)
    perc = no_files // 10

    code_line_pairs = []
 
    lines_counter = 0
    for i, file in enumerate(file_lines):
        
        if i % perc == 0 and logger is not None:
            logger.info(f"Processing lines in {i+1} / {no_files} of the files.")
            
        for j, left_line in enumerate(file):
            if j + 1 < len(file):
                right_line = file[j+1]
                if left_line not in left_lines and right_line not in right_lines:
                    code_line_pairs.append(tokenizer.tokenize_training(left_line, right_line))
                    left_lines.add(left_line)
                    right_lines.add(right_line)
                
            # handle line period reset
            lines_counter += 1
            if lines_counter == line_repeat_period:
                left_lines = set()
                right_lines = set()
                lines_counter = 0
    
    return code_line_pairs

def load_code_files(data_paths, cols=None, sep=";"):
    """Loads code lines files(either in csv or xlsx format).

    Parameters
    ----------
    training_data_paths : list of str 
        A list of paths to files with the lines data (either csv or xlsx).
    cols : list of str
        list of columns to load from the files.
    sep : str, optional
        A separator used to separate columns in a csv file.
    
    """
    combined_files = []
    for data_path in data_paths:
        logger.info(f"Loading training data from {data_path}")
        if data_path.endswith(".xlsx"):
            code_lines_df = pd.read_excel(data_path)
        elif  data_path.endswith(".csv"):
            code_lines_df = pd.read_csv(data_path, sep=sep)
        else:
            logger.error(f"Unrecognized file format of {data_path}.")
            exit(1)

        if cols is None:
            cols = code_lines_df.columns.tolist()
        
        code_lines_df = code_lines_df[cols]
        combined_files.append(code_lines_df)

    code_lines_all_df = pd.concat(combined_files, axis=0, ignore_index=True, sort=False)
    logger.info(f"Loaded {code_lines_all_df.shape[0]:,} rows and {code_lines_all_df.shape[1]:,} cols...")
    
    return code_lines_all_df


def plot_commented_lines_confusion_matrix(y_pred, y, cm_path,
    figsize=(6,6), cmap="Blues"):
    """Generates and saves a confusion matrix plot to a file.""" 

    cf_matrix = confusion_matrix(y, y_pred)

    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    perc_labs = ["{0:.1%}".format(value) for value in cmn.flatten()]

    group_counts = ["{0:0.0f}\n".format(value) for value in cf_matrix.flatten()]
        
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,perc_labs)]
    box_labels = np.asarray(box_labels).reshape(cmn.shape[0],cmn.shape[1])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cmn, annot=box_labels, fmt='', 
                annot_kws={"fontsize":12},
                xticklabels=["Not commented", "Commented"], 
                yticklabels=["Not commented", "Commented"],
            cmap=cmap,
            linecolor='lightgray', linewidths=0.5,
            square=True,
            cbar=False,
            vmin=0, vmax=1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix for the lines commented on saved to {cm_path}.")
    plt.close()


def report_commented_lines_predictions_accuracy(y_pred, y):
    """ Calculates prediction quality metrics and prints them."""

    commented_lines_acc = accuracy_score(y, y_pred)
    commented_lines_f1 = f1_score(y, y_pred, average="macro")
    commented_lines_precision = precision_score(y, y_pred, average="macro")
    commented_lines_recall = precision_score(y, y_pred, average="macro")
    commented_lines_mcc = mcc_score(y, y_pred)

    logger.info(f"Accuracy = {commented_lines_acc:.2f}")
    logger.info(f"Precision = {commented_lines_precision:.2}")
    logger.info(f"Recall = {commented_lines_recall:.2}")
    logger.info(f"F1-score = {commented_lines_f1:.2}")
    logger.info(f"MCC = {commented_lines_mcc:.2}")