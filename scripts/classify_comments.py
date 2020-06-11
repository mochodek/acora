#!/usr/bin/env python

description="""Classify review comments with a trained BERT review-comments classifier."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import numpy as np

from scipy import stats

from collections import Counter

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import keras
    from keras.models import load_model

    from keras_bert import Tokenizer, get_custom_objects

    from keras_radam import RAdam

    import tensorflow as tf

    if tf.__version__.startswith("1."):
        from tensorflow import ConfigProto, Session, set_random_seed
    else:
        from tensorflow.compat.v1 import ConfigProto, Session, set_random_seed
         
    from tensorflow.python.client import device_lib


from acora.vocab import BERTVocab
from acora.comments import default_subject_columns, \
    load_comments_files, CommentPurposeTransformer, CommentSubjectTransformer, \
    plot_purpose_confusion_matrix, plot_subjects_confusion_matrix, \
    report_comment_predictions_accuracy, default_purpose_labels

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':

    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--classify_input_data_paths",
                        help="a list of paths to test data files (either .csv or .xlsx).", 
                        type=str, nargs="+")

    parser.add_argument("--classify_output_data_path", help="a path to a an output file.",
                        type=str, default="./classify-results.xlsx")

    parser.add_argument("--model_trained_path",
                        help="a path to a trained model.", 
                        type=str, required=True)

    parser.add_argument("--bert_vocab_path",
                        help="a path to a file storing the BERT vocabulary.", 
                        type=str, required=True)

    parser.add_argument("--preserve_all_columns", help="to preseve all columns from the existing files (the columns from the first file are used).",
                        action='store_true')

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    parser.add_argument("--message_column", help="a name of the column that stores the comment/message.",
                        type=str, default="message")

    parser.add_argument("--purpose_column", help="a name of the column that stores the decision class for the comment purpose.",
                        type=str, default="purpose")

    parser.add_argument("--purpose_labels", help="a list of possible purpose labels (categories).",
                        type=str, nargs="+", default=default_purpose_labels)

    parser.add_argument("--subject_columns", help="a list of column names that store the decision classes for the comment subjects.",
                        type=str, nargs="+", default=default_subject_columns)

    parser.add_argument("--seq_len", help="a maximum length of a comment (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--random_seed", help="a random seed used to control the process.",
                        type=int, default=102329)



    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    model_trained_path = args['model_trained_path']
    vocab_path = args['bert_vocab_path']
    classify_input_data_paths = args['classify_input_data_paths']
    classify_output_data_path = args['classify_output_data_path']
    sep = args['sep']
    message_column = args['message_column']
    purpose_column = args['purpose_column']
    purpose_labels = args['purpose_labels']
    subject_columns = args['subject_columns']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    random_seed = args['random_seed']
    preserve_all_columns = args['preserve_all_columns']
    
    ######

    if preserve_all_columns:
        cols = None
    else:
        cols = [message_column, ] 

    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if not not_use_gpu and len(gpus) == 0:
        logger.error("You don't have a GPU available on your system, it can affect the performance...")

    import tensorflow as tf
  
    config = ConfigProto( device_count = {'GPU': 0 if not_use_gpu else len(gpus)}, allow_soft_placement = True )
    sess = Session(config=config) 
    
    if tf.__version__.startswith("1."):
            keras.backend.set_session(sess)
    else:
        tf.compat.v1.keras.backend.set_session(sess)

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    logger.info("Initializing a BERT tokenizer...")
    tokenizer = Tokenizer(vocab.token_dict)
    logger.info(f"BERT tokenizer ready, example: 'ACoRA is a nice tool' -> {str(tokenizer.tokenize('ACoRA is a nice tool'))}")

    logger.info("Loading the input data to classify...")
    
    reviews_all_df = load_comments_files(classify_input_data_paths, cols, sep)

    logger.info("Tokenizing messages in the testing dataset...")
    tokenized_all_messages = [tokenizer.encode(str(text), max_len=seq_len)[0] for text in reviews_all_df[message_column].tolist()] 
    x_all = [np.array(tokenized_all_messages), np.zeros_like(tokenized_all_messages)]
  
    logger.info(f"Loading the BERT comment classification model from {model_trained_path}")
    custom_objects = get_custom_objects()
    custom_objects['RAdam'] = RAdam
    model = keras.models.load_model(model_trained_path, 
                                custom_objects=custom_objects)

    np.random.seed(random_seed)
    set_random_seed(random_seed)

    logger.info("Predicting...")
    y_all_pred_purpose, *y_all_pred_subject = model.predict(x_all) 

    
    logger.info("Preparing the output...")

    y_pred_purpose_flat = np.argmax(y_all_pred_purpose, axis=1)
    y_pred_purpose_labels = [purpose_labels[class_idx] for class_idx in y_pred_purpose_flat]
    purpose_preds_df = pd.DataFrame(y_pred_purpose_labels, columns=[purpose_column,])

    y_pred_subject = np.array(y_all_pred_subject).reshape(len(y_all_pred_subject),len(y_all_pred_subject[0])).transpose()
    subject_all_preds = []
    for preds in y_pred_subject:
        subject_all_preds.append([1 if x > 0.5 else 0 for x in preds]) 
    subject_preds_df = pd.DataFrame(subject_all_preds, columns=subject_columns)

    out_df = pd.concat([reviews_all_df, purpose_preds_df, subject_preds_df], axis=1, sort=False)
    if classify_output_data_path.endswith(".xlsx"):
        out_df.to_excel(classify_output_data_path, index=False)
    elif classify_output_data_path.endswith(".csv"):
        out_df.to_csv(classify_output_data_path, index=False, sep=sep)
    else:
        logger.error(f"Unrecognized output file format of the {classify_output_data_path}")

    logger.info(f"Predictions have been saved to {classify_output_data_path}.")