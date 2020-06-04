#!/usr/bin/env python

description="""Extract lines embeddings using a BERT model."""

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

    import keras
    from keras.models import load_model

    from keras_bert import get_custom_objects
    from keras_bert.layers.extract import Extract

    from keras_radam import RAdam

    import tensorflow as tf

    if tf.__version__.startswith("1."):
        from tensorflow import ConfigProto, Session, set_random_seed
    else:
        from tensorflow.compat.v1 import ConfigProto, Session, set_random_seed
         
    from tensorflow.python.client import device_lib


from acora.vocab import BERTVocab
from acora.code import CodeTokenizer, load_code_files, plot_commented_lines_confusion_matrix, \
                        report_commented_lines_predictions_accuracy
from acora.lamb import Lamb
from acora.code_embeddings import CodeLinesBERTEmbeddingsExtractor

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--lines_paths",
                        help="a list of paths to data files (either .csv or .xlsx).", 
                        type=str, nargs="+")

    parser.add_argument("--bert_path",
                        help="a path to a pretrained BERT model trained on code", 
                        type=str, required=True)

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

    parser.add_argument("--seq_len", help="a maximum length of a line (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--unique_lines", help="to select only unique lines from an input.",
                        action='store_true')

    parser.add_argument("--no_layers", help="the number of layers to include.",
                        default=4, type=int)

    parser.add_argument("--output_file_path", help="a path to a json file containing two lists: lines and their embeddings.",
                        type=str, default="./lines_embeddings.json")
    

    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    bert_path = args['bert_path']
    lines_paths = args['lines_paths']
    vocab_path = args['vocab_path']
    vocab_size = args['vocab_size']
    sep = args['sep']
    line_column = args['line_column']
    no_layers = args['no_layers']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    output_file_path =  args['output_file_path']
    unique_lines =  args['unique_lines']
    
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
    code_lines_all_df = load_code_files(lines_paths, cols=None, sep=sep)
    if unique_lines:
        lines = code_lines_all_df[line_column].fillna("").unique().tolist()
    else:
        lines = code_lines_all_df[line_column].fillna("").tolist()

    logger.info(f"Loading a BERT model from {bert_path}...")
    custom_objects = get_custom_objects()
    custom_objects['RAdam'] = RAdam
    custom_objects['Lamb'] = Lamb
    model = keras.models.load_model(bert_path, 
                                    custom_objects=custom_objects)

    logger.info("Extracting lines embeddings...")
    embeddings_extractor = CodeLinesBERTEmbeddingsExtractor(base_model=model, 
                                                            no_layers=no_layers,
                                                            token_dict=vocab.token_dict)
    lines_embeddings = embeddings_extractor.extract_embeddings(lines)
    logger.info(f"Extracted embeddings for {len(lines_embeddings)} lines, each of size {lines_embeddings[0].shape}.")

    output = [lines, [embeddings.tolist() for embeddings in lines_embeddings]]

    logger.info(f"Saving generated lines and embeddings to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8', errors="ignore") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("The process of extracting lines embeddings has been completed.")




