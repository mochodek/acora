#!/usr/bin/env python

description="""Test a BERT model for detecting lines that will be commented on."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)
from pathlib import Path

import pandas as pd
import numpy as np

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    if tf.__version__.startswith("1."):
        os.environ['TF_KERAS'] = '0'
        from tensorflow import ConfigProto, Session, set_random_seed
        import keras
        from keras.models import load_model
    else:
        os.environ['TF_KERAS'] = '1'
        from tensorflow.compat.v1 import ConfigProto, Session, set_random_seed
        import tensorflow.compat.v1.keras as keras
        from tensorflow.compat.v1.keras.models import load_model
         
    from tensorflow.python.client import device_lib


    from keras_bert import get_custom_objects

    from keras_radam import RAdam


from acora.vocab import BERTVocab
from acora.code import CodeTokenizer, load_code_files, \
    plot_commented_lines_confusion_matrix, report_commented_lines_predictions_accuracy


logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--test_lines_paths",
                        help="a list of paths to files with lines to classify (either .csv or .xlsx).", 
                        type=str, nargs="+")

    parser.add_argument("--bert_trained_path",
                        help="a path to a  BERT model trained to detect lines that will be commented on (see train_bert_commented_lines.py)", 
                        type=str, required=True)

    parser.add_argument("--vocab_path",
                        help="a path to the vocabulary txt file.", 
                        type=str, default="./vocab.txt")

    parser.add_argument("--vocab_size",
                        help="a number of entries to use from the vocabulary. If not provided all the entries are included." 
                             " The same vocab_size should be used here as it was used for training the BERT model.", 
                        type=int, default=None)

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    parser.add_argument("--line_column", help="a name of the column that stores the lines.",
                        default="line_contents", type=str)

    parser.add_argument("--seq_len", help="a maximum length of a line (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--decision_column", help="a name of the column that stores the decision (1 - commented, 0 - not commented).",
                        type=str, default="commented")

    parser.add_argument("--lines_commented_cm_test_path", help="a path to a file presenting a confusion matrix for the test.",
                        type=str, default="./lines_commented_test.pdf")
    
    parser.add_argument("--preserve_whitespace",
                        help="whether or not to preserve whitespaces as tokens).", 
                        action='store_true')
    

    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    bert_trained_path = args['bert_trained_path']
    test_lines_paths = args['test_lines_paths']
    vocab_path = args['vocab_path']
    vocab_size = args['vocab_size']
    sep = args['sep']
    line_column = args['line_column']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    decision_column = args['decision_column']
    lines_commented_cm_test_path = args['lines_commented_cm_test_path']
    preserve_whitespace = args['preserve_whitespace']
    
    ######
    
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if not not_use_gpu and len(gpus) == 0:
        logger.error("You don't have a GPU available on your system, it can affect the performance...")
     
    config = ConfigProto( device_count = {'GPU': 0 if not_use_gpu else len(gpus)}, allow_soft_placement = True )
    sess = Session(config=config) 
    keras.backend.set_session(sess)

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path, limit=vocab_size)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    logger.info("Initializing a BERT code tokenizer...")
    tokenizer = CodeTokenizer(vocab.token_dict, cased=True, preserve_whitespace=preserve_whitespace)
    logger.info(f"BERT code tokenizer ready, example: 'bool acoraIs_nice = True;' -> {str(tokenizer.tokenize('bool acoraIs_nice = True;'))}")

    logger.info("Loading test lines data...")
    code_lines_all_df = load_code_files(test_lines_paths, cols=None, sep=sep)
    code_lines_all_df[line_column] = code_lines_all_df[line_column].fillna("")

    logger.info("Tokenizing code lines in the training dataset...")
    tokenized_all_code_lines = [tokenizer.encode(text, max_len=seq_len)[0] for text in code_lines_all_df[line_column].tolist()] 
    x_all = [np.array(tokenized_all_code_lines), np.zeros_like(tokenized_all_code_lines)]
  
    logger.info(f"Loading the trained BERT model from {bert_trained_path}...")
    
    custom_objects = get_custom_objects()
    custom_objects['RAdam'] = RAdam
    model = keras.models.load_model(bert_trained_path, 
                                    custom_objects=custom_objects)

    model.summary(print_fn=logger.info)

    logger.info("Classifying lines...")
    y_all_pred = model.predict(x_all)
    y_all_pred = [1 if y >= 0.5 else 0 for y in y_all_pred]
    y_all = code_lines_all_df[decision_column]

    logger.info("Preparing confusion matrix for the validation data.")
    plot_commented_lines_confusion_matrix(y_all_pred, y_all, lines_commented_cm_test_path)

    logger.info("The accuracy of the predictions on the validation dataset:")   
    report_commented_lines_predictions_accuracy(y_all_pred, y_all)


    logger.info("The process of testing the BERT model for detecting lines that will be commented on has finished.")




