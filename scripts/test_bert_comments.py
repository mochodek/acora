#!/usr/bin/env python

description="""Test BERT trained for classifying review comments on selected datasets."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)

import pandas as pd
import numpy as np

from scipy import stats

from collections import Counter


import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    if tf.__version__.startswith("1."):
        os.environ['TF_KERAS'] = '0'
        from tensorflow import ConfigProto, Session, set_random_seed
        import keras
    else:
        os.environ['TF_KERAS'] = '1'
        from tensorflow.compat.v1 import ConfigProto, Session, set_random_seed
        import tensorflow.compat.v1.keras as keras
         
    from tensorflow.python.client import device_lib


    from keras_bert import Tokenizer, get_custom_objects

    from keras_radam import RAdam



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

    parser.add_argument("--test_data_paths",
                        help="a list of paths to test data files (either .csv or .xlsx).", 
                        type=str, nargs="+")

    parser.add_argument("--model_trained_path",
                        help="a path to a trained model.", 
                        type=str, required=True)

    parser.add_argument("--bert_vocab_path",
                        help="a path to a file storing the BERT vocabulary.", 
                        type=str, required=True)

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

    parser.add_argument("--purpose_cm_test_path", help="a path to a file presenting a confusion matrix for the purpose output.",
                        type=str, default="./comment_purpose_test.pdf")
    
    parser.add_argument("--subject_cm_test_path", help="a path to a file presenting a confusion matrix for the subject output.",
                        type=str, default="./comment_subject_test.pdf")


    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    model_trained_path = args['model_trained_path']
    vocab_path = args['bert_vocab_path']
    test_data_paths = args['test_data_paths']
    sep = args['sep']
    message_column = args['message_column']
    purpose_column = args['purpose_column']
    purpose_labels = args['purpose_labels']
    subject_columns = args['subject_columns']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    random_seed = args['random_seed']
    purpose_cm_test_path = args['purpose_cm_test_path']
    subject_cm_test_path = args['subject_cm_test_path']
    
    ######

    cols = [message_column, purpose_column] + subject_columns

    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if not not_use_gpu and len(gpus) == 0:
        logger.error("You don't have a GPU available on your system, it can affect the performance...")

    config = ConfigProto( device_count = {'GPU': 0 if not_use_gpu else len(gpus)}, allow_soft_placement = True )
    sess = Session(config=config) 
    keras.backend.set_session(sess)

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    logger.info("Initializing a BERT tokenizer...")
    tokenizer = Tokenizer(vocab.token_dict)
    logger.info(f"BERT tokenizer ready, example: 'ACoRA is a nice tool' -> {str(tokenizer.tokenize('ACoRA is a nice tool'))}")

    logger.info("Loading test data...")
    reviews_all_df = load_comments_files(test_data_paths, cols, sep)

    logger.info("Tokenizing messages in the testing dataset...")
    tokenized_all_messages = [tokenizer.encode(text, max_len=seq_len)[0] for text in reviews_all_df[message_column].tolist()] 
    x_all = [np.array(tokenized_all_messages), np.zeros_like(tokenized_all_messages)]

    logger.info("Transforming data for the purpose output variable...")
    purpose_transformer = CommentPurposeTransformer(reviews_all_df, purpose_column, purpose_labels)
    y_all_purpose_labels = purpose_transformer.get_class_labels()
    y_all_purpose = purpose_transformer.encode()
    message_purpose_labels_cat_mappings = purpose_transformer.message_purpose_labels_cat_mappings

    logger.info(f"Distribution of the purpose types: {Counter(y_all_purpose_labels)}")

    logger.info("Transforming data for the subject output variables...")
    subject_transformer = CommentSubjectTransformer(reviews_all_df, subject_columns)
    y_all_subject = subject_transformer.encode_one_hot_all_subjects()
  
    logger.info(f"Loading the BERT comment classification model from {model_trained_path}")
    custom_objects = get_custom_objects()
    custom_objects['RAdam'] = RAdam
    model = keras.models.load_model(model_trained_path, 
                                custom_objects=custom_objects)


    
    np.random.seed(random_seed)
    set_random_seed(random_seed)

    y_all = {"purpose_output": y_all_purpose}
    for i, subject_class in enumerate(subject_columns):
        y_all[f"{subject_class}_output"] = subject_transformer.encode_binary_single_subject(subject_class).values

    
    logger.info("Predicting...")
    y_all_pred_purpose, *y_all_pred_subject = model.predict(x_all) 

    logger.info("Preparing confusion matrix for the comment purpose.")
    plot_purpose_confusion_matrix(y_all_pred_purpose, y_all_purpose, message_purpose_labels_cat_mappings, purpose_cm_test_path)

    
    logger.info("Preparing confusion matrix for the comment subjects.")
    plot_subjects_confusion_matrix(y_all_pred_subject, y_all_subject, subject_columns, subject_cm_test_path)

    logger.info("The accuracy of the predictions on the trainig dataset:")   
    report_comment_predictions_accuracy(y_all_pred_purpose, y_all_purpose, 
                y_all_pred_subject, y_all_subject, subject_columns)


    logger.info("BERT for comment testing has been completed.")