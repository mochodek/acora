import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

import logging

import math

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef as mcc_score

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


logger = logging.getLogger('acora.comments')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

default_purpose_labels = [
    'acknowledgement', 
    'change_request', 
    'discussion_participation', 
    'discussion_trigger', 
    'same_as',
]

default_subject_columns = ["code_design",
	"code_style",
	"code_naming",
	"code_logic",
	"code_io",
	"code_data",
	"code_doc",
	"code_api",
	"compatibility",
	"rule_def",
	"config_commit_patch_review",
	"config_building_installing",
]

def load_comments_files(data_paths, cols=None, sep=";"):
    """Loads review comment files(either in csv or xlsx format).

    Parameters
        ----------
        training_data_paths : list of str 
            A list of paths to files with the reviews data (either csv or xlsx).
        cols : list of str
            list of columns to load from the files.
        sep : str, optional
            A separator used to separate columns in a csv file.
    
    """
    combined_files = []
    for data_path in data_paths:
        logger.info(f"Loading data from {data_path}")
        if data_path.endswith(".xlsx"):
            reviews_df = pd.read_excel(data_path)
        elif  data_path.endswith(".csv"):
            reviews_df = pd.read_csv(data_path, sep=sep)
        else:
            logger.error(f"Unrecognized file format of {data_path}.")
            exit(1)

        if cols is None:
            cols = reviews_df.columns.tolist()
        
        reviews_df = reviews_df[cols]
        combined_files.append(reviews_df)

    reviews_all_df = pd.concat(combined_files, axis=0, ignore_index=True, sort=False)
    logger.info(f"Loaded {reviews_all_df.shape[0]:,} rows and {reviews_all_df.shape[1]:,} cols...")

    return reviews_all_df


class CommentPurposeTransformer(object):
    """Extracts columns related to the comment purpose and transforms them to the form 
       allowing training a BERT classifier."""

    def __init__(self, training_data_df, purpose_column="purpose", 
            purpose_labels=default_purpose_labels):
        """Parameters
        ----------
        training_data_df : pd.DataFrame 
            A data frame containing training data.
        purpose_column : str, optional
            A name of the column with the purpose variable.   
        purpose_labels : list of str, optional
            A list of possible categories for the purpose variable.
        """
        self.purpose_column = purpose_column
        self.purpose_labels = purpose_labels
        self.message_purpose_labels = training_data_df[self.purpose_column].astype(
                CategoricalDtype(self.purpose_labels, False))
        self.message_purpose_labels_cat_mappings = OrderedDict(list(zip(list(range(len(self.message_purpose_labels.cat.categories))), 
                                                                                    self.message_purpose_labels.cat.categories)))
    
    def get_class_labels(self):
        """Returns purpose class labels for each observation in the training dataset."""
        return self.message_purpose_labels

    def transform_class_labels(self, data_df):
        """Returns purpose class labels for each observation in the given dataset."""
        return data_df[self.purpose_column].astype(CategoricalDtype(self.purpose_labels, False)) 

    def encode(self):
        """Returns purpose encoded using one-hot encoding for the training dataset."""
        return pd.get_dummies(self.message_purpose_labels).values

    def transform_encode(self, class_labels):
        """Returns purpose encoded using one-hot encoding for a new dataset."""
        return pd.get_dummies(class_labels).values

    def class_weights(self):
        """Calculates and returns class weights based on their frequency in the training dataset."""
        return class_weight.compute_class_weight('balanced', classes=np.unique(self.message_purpose_labels.cat.codes), 
                y=self.message_purpose_labels.cat.codes)


class CommentSubjectTransformer(object):
    """Extracts columns related to the comment subjects and transforms them to the form 
       allowing training a BERT classifier."""

    def __init__(self, training_data_df, subject_columns=default_subject_columns):
        """Parameters
        ----------
        training_data_df : pd.DataFrame 
            A data frame containing training data.
        subject_columns : list of str, optional
            Names of the columns with the subject variable.   
        """
        self.subject_columns = subject_columns
        
        self.message_subject_types = training_data_df[self.subject_columns].fillna(0.0)
        self.message_subject_types_cat_mappings = dict(list(zip(list(range(len(self.subject_columns))), 
                                                                        self.message_subject_types.columns.tolist())))

    def encode_one_hot_all_subjects(self):
        """Returns subjects encoded using one-hot encoding for the training dataset."""
        return self.message_subject_types

    def transform_encode_one_hot_all_subjects(self, data_df):
        """Returns subjects encoded using one-hot encoding for a new dataset."""
        return data_df[self.subject_columns].fillna(0.0)
        
    def encode_binary_single_subject(self, subject_column):
        """Returns subjects encoded using one-hot encoding for the training dataset."""
        return self.message_subject_types[subject_column]

    def transform_encode_binary_single_subject(self, data_df, subject_column):
        """Returns subjects encoded using one-hot encoding for the training dataset."""
        return data_df[subject_column].fillna(0.0)

    def class_weights(self):
        """Calculates and returns class weights based on their frequency in the training dataset."""
        subject_class_weights = self.message_subject_types.sum().sum()  / (self.message_subject_types.sum(axis=0) * len(self.message_subject_types.columns.tolist()))
        subject_class_weights = subject_class_weights.tolist()

        subject_bin_weights = {}
        subject_bin_per_type_weights = {}

        for i, subject_class in enumerate(self.subject_columns):
            subject_bin_weights[subject_class] = class_weight.compute_class_weight('balanced', classes=self.message_subject_types[subject_class].unique(), 
                                                                                y=self.message_subject_types[subject_class])
            subject_bin_per_type_weights[subject_class] = subject_bin_weights[subject_class] * subject_class_weights[i]

        return subject_bin_weights


def plot_purpose_confusion_matrix(y_pred_purpose, y_purpose, 
        message_purpose_labels_cat_mappings, purpose_cm_path,
        figsize=(6,6), cmap="Blues"):
    """Generates and saves a confusion matrix plot to a file.""" 

    y_pred_purpose_flat = np.argmax(y_pred_purpose, axis=1)
    y_purpose_flat = np.argmax(y_purpose, axis=1)
    
    cf_matrix_all_purpose = confusion_matrix(y_purpose_flat, y_pred_purpose_flat)

    cmn = cf_matrix_all_purpose.astype('float') / cf_matrix_all_purpose.sum(axis=1)[:, np.newaxis]
    perc_labs = ["{0:.1%}".format(value) for value in cmn.flatten()]

    group_counts = ["{0:0.0f}\n".format(value) for value in cf_matrix_all_purpose.flatten()]
        
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,perc_labs)]
    box_labels = np.asarray(box_labels).reshape(cmn.shape[0],cmn.shape[1])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cmn, annot=box_labels, fmt='', 
                annot_kws={"fontsize":12},
                xticklabels=list(message_purpose_labels_cat_mappings.values()), 
                yticklabels=list(message_purpose_labels_cat_mappings.values()),
            cmap=cmap,
            linecolor='lightgray', linewidths=0.5,
            square=True,
            cbar=False,
            vmin=0, vmax=1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(purpose_cm_path)
    logger.info(f"Confusion matrix for the comment purpose saved to {purpose_cm_path}.")
    plt.close()


def plot_subjects_confusion_matrix(y_pred_subject, y_subject, 
        subject_columns, subject_cm_path,
        figsize=(10,20), cmap='Greens'):
    """Generates and saves a confusion matrix plot to a file."""

    y_pred_subject = np.array(y_pred_subject).reshape(len(y_pred_subject),len(y_pred_subject[0])).transpose()
    subject_all_preds = []
    for preds in y_pred_subject:
        subject_all_preds.append([1 if x > 0.5 else 0 for x in preds]) 
    subject_preds_df = pd.DataFrame(subject_all_preds, columns=subject_columns)

    cf_matrix_all_subject = multilabel_confusion_matrix(y_subject, subject_preds_df.values, samplewise=False)

    fig = plt.figure(figsize=figsize)
    cols = math.ceil(float(len(subject_columns)) / 2.0)
    print(cols)
    gs = gridspec.GridSpec(cols, 2, height_ratios=[1]*cols)
    gs.update(hspace=0.4, wspace=0.5)

    for i, cf in enumerate(cf_matrix_all_subject):
        
        row = i // 2
        col = i % 2
        print(row, col)
        ax = plt.subplot(gs[row, col])
        
        cmn = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
        perc_labs = ["{0:.1%}".format(value) for value in cmn.flatten()]
        
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        
        box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,perc_labs)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

        sns.heatmap(cmn, 
                    annot=box_labels, 
                    fmt='', 
                    annot_kws={"fontsize":12},
                    xticklabels=("False", "True"), 
                    yticklabels=("False", "True"),
                cmap=cmap,
                linecolor='lightgray', linewidths=0.5,
                square=True,
                cbar=False,
                vmin=0, vmax=1)
        ax.set_title(subject_columns[i])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig(subject_cm_path)
    logger.info(f"Confusion matrix for the comment subject saved to {subject_cm_path}.")
    plt.close()


def report_comment_predictions_accuracy(y_pred_purpose, y_purpose, y_pred_subject, y_subject, subject_columns):
    """ Calculates prediction quality metrics and prints them."""

    y_pred_purpose_flat = np.argmax(y_pred_purpose, axis=1)
    y_purpose_flat = np.argmax(y_purpose, axis=1)
    
    purpose_acc = accuracy_score(y_purpose_flat, y_pred_purpose_flat)
    purpose_f1 = f1_score(y_purpose_flat, y_pred_purpose_flat, average="macro")
    purpose_precision = precision_score(y_purpose_flat, y_pred_purpose_flat, average="macro")
    purpose_recall = precision_score(y_purpose_flat, y_pred_purpose_flat, average="macro")
    purpose_mcc = mcc_score(y_purpose_flat, y_pred_purpose_flat)

    logger.info(f"Purpose Accuracy = {purpose_acc:.2f}")
    logger.info(f"Purpose Precision = {purpose_precision:.2}")
    logger.info(f"Purpose Recall = {purpose_recall:.2}")
    logger.info(f"Purpose F1-score = {purpose_f1:.2}")
    logger.info(f"Purpose MCC = {purpose_mcc:.2}")

    y_pred_subject = np.array(y_pred_subject).reshape(len(y_pred_subject),len(y_pred_subject[0])).transpose()
    subject_all_preds = []
    for preds in y_pred_subject:
        subject_all_preds.append([1 if x > 0.5 else 0 for x in preds]) 
    subject_preds_df = pd.DataFrame(subject_all_preds, columns=subject_columns)

    for subject in subject_columns:
        subject_acc = accuracy_score(y_subject[subject], subject_preds_df[subject])
        subject_f1 = f1_score(y_subject[subject], subject_preds_df[subject], average="macro")
        subject_precision = precision_score(y_subject[subject], subject_preds_df[subject], average="macro")
        subject_recall = precision_score(y_subject[subject], subject_preds_df[subject], average="macro")
        subject_mcc = mcc_score(y_subject[subject], subject_preds_df[subject])
        logger.info(f"Subject {subject} Accuracy = {subject_acc:.2f}")
        logger.info(f"Subject {subject} Precision = {subject_precision:.2}")
        logger.info(f"Subject {subject} Recall = {subject_recall:.2}")
        logger.info(f"Subject {subject} F1-score = {subject_f1:.2}")
        logger.info(f"Subject {subject} MCC = {subject_mcc:.2}")



def save_comment_predictions_accuracy(y_pred_purpose, y_purpose, 
        y_pred_subject, y_subject, subject_columns, output_file_path, sep, name=None):
    """ Calculates prediction quality metrics and prints them."""

    if name is None:
        name = output_file_path

    with open(output_file_path, 'w', encoding='utf-8', errors='ignore') as out:

        header = ['name', 'p_acc', 'p_prec', 'p_rec', 'p_f1', 'p_mcc']
        for subject in subject_columns:
            header += [f's_{subject}_acc', f's_{subject}_prec', 
                f's_{subject}_rec', f's_{subject}_f1', f's_{subject}_mcc']
        header_line = f"{sep}".join(header)
        out.write(header_line + "\n")

        y_pred_purpose_flat = np.argmax(y_pred_purpose, axis=1)
        y_purpose_flat = np.argmax(y_purpose, axis=1)
        
        purpose_acc = accuracy_score(y_purpose_flat, y_pred_purpose_flat)
        purpose_f1 = f1_score(y_purpose_flat, y_pred_purpose_flat, average="macro")
        purpose_precision = precision_score(y_purpose_flat, y_pred_purpose_flat, average="macro")
        purpose_recall = precision_score(y_purpose_flat, y_pred_purpose_flat, average="macro")
        purpose_mcc = mcc_score(y_purpose_flat, y_pred_purpose_flat)

        line = [name, purpose_acc, purpose_precision, purpose_recall, purpose_f1, purpose_mcc]

        y_pred_subject = np.array(y_pred_subject).reshape(len(y_pred_subject),len(y_pred_subject[0])).transpose()
        subject_all_preds = []
        for preds in y_pred_subject:
            subject_all_preds.append([1 if x > 0.5 else 0 for x in preds]) 
        subject_preds_df = pd.DataFrame(subject_all_preds, columns=subject_columns)

        for subject in subject_columns:
            subject_acc = accuracy_score(y_subject[subject], subject_preds_df[subject])
            subject_f1 = f1_score(y_subject[subject], subject_preds_df[subject], average="macro")
            subject_precision = precision_score(y_subject[subject], subject_preds_df[subject], average="macro")
            subject_recall = precision_score(y_subject[subject], subject_preds_df[subject], average="macro")
            subject_mcc = mcc_score(y_subject[subject], subject_preds_df[subject])
            line += [subject_acc, subject_precision, subject_recall, subject_f1, subject_mcc]

        line = f"{sep}".join([str(x) for x in line])
        out.write(line + "\n")

