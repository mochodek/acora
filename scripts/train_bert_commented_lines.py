#!/usr/bin/env python

description="""Train a BERT model for detecting lines that will be commented on."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)
import json

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from collections import Counter


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
    from keras_bert.layers.extract import Extract

    from keras_radam import RAdam

from acora.vocab import BERTVocab
from acora.code import CodeTokenizer, load_code_files, plot_commented_lines_confusion_matrix, \
                        report_commented_lines_predictions_accuracy

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--training_data_paths",
                        help="a list of paths to training data files (either .csv or .xlsx).", 
                        type=str, nargs="+")

    parser.add_argument("--bert_pretrained_path",
                        help="a path to a pretrained BERT model trained on code (see pretrain_bert_on_code.py)", 
                        type=str, required=True)

    parser.add_argument("--bert_config_path",
                        help="a path to a BERT config .json file (the same format is used as in the original BERT.).", 
                        type=str, default="./bert_config.json")

    parser.add_argument("--continue_training",
                        help="if set the bert_pretrained has to be already the commented line classifier trained with this script", 
                        action="store_true")

    parser.add_argument("--vocab_path",
                        help="a path to the vocabulary txt file.", 
                        type=str, default="./vocab.txt")

    parser.add_argument("--vocab_size",
                        help="a number of entries to use from the vocabulary. If not provided all the entries are included." 
                             " The same vocab_size should be used here as it was used for pre-training the BERT model.", 
                        type=int, default=None)

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    parser.add_argument("--line_column", help="a name of the column that stores the lines.",
                        default="line_contents", type=str)

    parser.add_argument("--decision_column", help="a name of the column that stores the decision (1 - commented, 0 - not commented).",
                        type=str, default="commented")

    parser.add_argument("--seq_len", help="a maximum length of a line (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--weight_instances", help="to balance the dataset by weighting the instances based on the classes frequency.",
                        action='store_true')

    parser.add_argument("--random_seed", help="a random seed used to control the process.",
                        type=int, default=102329)

    parser.add_argument("--learning_rate", help="a learning rate used to train the model.",
                        type=float, default=2e-4)
    
    parser.add_argument("--batch_size", help="a batch size used for training the model.",
                        type=int, default=32)

    parser.add_argument("--epochs", help="a number of epochs to train the model.",
                        type=int, default=50)

    parser.add_argument("--report_trainig_accuracy", help="to report the accuracy on the training dataset.",
                        action='store_true')

    parser.add_argument("--lines_commented_cm_train_path", help="a path to a file presenting a confusion matrix for training.",
                        type=str, default="./lines_commented_train.pdf")

    parser.add_argument("--model_save_path", help="a path to serialize the trained BERT model.",
                        type=str, default="./bert_lines_commented.h5")

    parser.add_argument("--train_size", help="should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.",
                        type=float, default=1.0)

    parser.add_argument("--lines_commented_cm_train_val_path", help="a path to a file presenting a confusion matrix for validation.",
                        type=str, default="./lines_commented_val.pdf")
    

    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    bert_pretrained_path = args['bert_pretrained_path']
    bert_config_path = args['bert_config_path']
    continue_training = args['continue_training']
    training_data_paths = args['training_data_paths']
    vocab_path = args['vocab_path']
    vocab_size = args['vocab_size']
    sep = args['sep']
    line_column = args['line_column']
    decision_column = args['decision_column']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    weight_instances =  args['weight_instances']
    random_seed = args['random_seed']
    lr = args['learning_rate']
    batch_size = args['batch_size']
    epochs = args['epochs']
    report_trainig_accuracy = args['report_trainig_accuracy']
    lines_commented_cm_train_path = args['lines_commented_cm_train_path']
    model_save_path = args['model_save_path']
    train_size = args['train_size']
    lines_commented_cm_train_val_path = args['lines_commented_cm_train_val_path']
    
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
    tokenizer = CodeTokenizer(vocab.token_dict, cased=True)
    logger.info(f"BERT code tokenizer ready, example: 'bool acoraIs_nice = True;' -> {str(tokenizer.tokenize('bool acoraIs_nice = True;'))}")

    logger.info("Loading training data...")
    code_lines_all_df = load_code_files(training_data_paths, cols=None, sep=sep)
    code_lines_all_df[line_column] = code_lines_all_df[line_column].fillna("")

    if train_size < 1.0:
        ids_train, ids_val, _, _ = train_test_split(code_lines_all_df.index.tolist(), 
                                code_lines_all_df.index.tolist(), train_size=train_size, random_state=random_seed)
        logger.info(f"Selecting {len(ids_train)} out of {code_lines_all_df.shape[0]} comments for training and {len(ids_val)} out of {code_lines_all_df.shape[0]}")
        code_lines_val_df = code_lines_all_df.loc[ids_val, :]
        code_lines_all_df = code_lines_all_df.loc[ids_train, :]

    logger.info("Tokenizing code lines in the training dataset...")
    tokenized_all_code_lines = [tokenizer.encode(text, max_len=seq_len)[0] for text in code_lines_all_df[line_column].tolist()] 
    x_all = [np.array(tokenized_all_code_lines), np.zeros_like(tokenized_all_code_lines)]

    y_all = code_lines_all_df[decision_column]

    logger.info(f"Loading the BERT config from {bert_config_path}")
    with open(bert_config_path, encoding="utf-8", errors="ignore") as f:
        bert_config = json.load(f)

    np.random.seed(random_seed)
    set_random_seed(random_seed)
  
    logger.info(f"Loading the pre-trained BERT model from {bert_pretrained_path}...")
    layer_num = bert_config['num_hidden_layers']

    custom_objects = get_custom_objects()
    custom_objects['RAdam'] = RAdam
    model = keras.models.load_model(bert_pretrained_path, 
                                    custom_objects=custom_objects)

    if not continue_training:
        inputs = model.inputs[:2]
        dense = model.get_layer(f'Encoder-{layer_num}-FeedForward-Norm').output
        dense = Extract(index=0, name="Extract")(dense)
        outputs = [keras.layers.Dense(units=1, activation='sigmoid', name="Output-Commented")(dense)]

        model = keras.models.Model(inputs, outputs)

        model.compile(
            RAdam(learning_rate =lr),
            loss="binary_crossentropy",
            metrics=['accuracy'],
        )

    model.summary(print_fn=logger.info)

    logger.info("Fine-tuning BERT model...")
    if weight_instances:
        labels = np.unique(y_all)
        class_weights = class_weight.compute_class_weight('balanced', labels, y_all)
        class_weights = {labels[i] : class_weights[i] for i in range(len(labels))}
        logger.info(f"Class weights: {str(class_weights)}")

        history = model.fit(
            x_all,
            y_all,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
            class_weight=class_weights
        )
    else:
        history = model.fit(
            x_all,
            y_all,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True
        )

    logger.info(f"Saving the BERT model to {model_save_path}")
    model.save(model_save_path)

    if report_trainig_accuracy:
        y_all_pred = model.predict(x_all)
        y_all_pred = [1 if y >= 0.5 else 0 for y in y_all_pred]

        logger.info("Preparing confusion matrix for training data.")
        plot_commented_lines_confusion_matrix(y_all_pred, y_all, lines_commented_cm_train_path)

        logger.info("The accuracy of the predictions on the trainig dataset:")   
        report_commented_lines_predictions_accuracy(y_all_pred, y_all)

    if train_size < 1.0:
        logger.info("Tokenizing code lines in the validation dataset...")
        tokenized_val_code_lines = [tokenizer.encode(text, max_len=seq_len)[0] for text in code_lines_val_df[line_column].tolist()] 
        x_val = [np.array(tokenized_val_code_lines), np.zeros_like(tokenized_val_code_lines)]
        y_val = code_lines_val_df[decision_column]

        logger.info(f"Distribution of the lines in the validation dataset: {Counter(y_val)}")

        y_val_pred = model.predict(x_val)
        y_val_pred = [1 if y >= 0.5 else 0 for y in y_val_pred]

        logger.info("Preparing confusion matrix for the validation data.")
        plot_commented_lines_confusion_matrix(y_val_pred, y_val, lines_commented_cm_train_val_path)

        logger.info("The accuracy of the predictions on the validation dataset:")   
        report_commented_lines_predictions_accuracy(y_val_pred, y_val)


    logger.info("BERT for detecting lines commented on training has been completed.")




