#!/usr/bin/env python

description="""Detects lines of code that needs reviewer focus providing a very simple web interface."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)
import json

from flask import Flask, render_template, request

from math import nan

import pandas as pd
import numpy as np

from pathlib import Path

import random

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
from acora.comments import default_subject_columns, \
    load_comments_files, default_purpose_labels
from acora.vocab import BERTVocab
from acora.code import CodeTokenizer, load_code_files

from acora.code_similarities import SimilarLinesFinder
from acora.recommend import CodeReviewFocusRecommender
from acora.code_embeddings import CodeLinesBERTEmbeddingsExtractor

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger_similar = logging.getLogger('acora.code_similarities')
logger_similar.setLevel(logging.DEBUG)
logger_similar.addHandler(ch)
logger_recommend = logging.getLogger('acora.recommend')
logger_recommend.setLevel(logging.DEBUG)
logger_recommend.addHandler(ch)



app = Flask(__name__)


@app.route('/', methods=('GET', 'POST'))
def index():
    global graph
    global sess
    global recommender
    review_results = []
    if request.method == 'POST':
        lines = request.form['code'].splitlines()

        with graph.as_default():
            if tf.__version__.startswith("1."):
                keras.backend.set_session(sess)
            else:
                tf.compat.v1.keras.backend.set_session(sess)                                
            review_results = recommender.review(lines)
    print(review_results)
    return render_template('index.html', review_results=review_results)



if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--commented_lines_paths",
                        help="a list of paths to files with the commented lines (either .csv or .xlsx)."
                            " The files need to contain line_contents, purpose of the comment and subjects of the comment."
                            " They will be used as a basis for recommendations.", 
                        type=str, nargs="+")

    parser.add_argument("--commented_lines_embeddings_path",
                        help="a path to the json file storing commented lines and their embeddings "
                            "serving as the basis for recommendations (see lines_to_bert_embeddings.py).", 
                        type=str, default="./lines_with_embeddings.json")

    parser.add_argument("--bert_trained_path",
                        help="a path to a BERT model trained to detect lines that will be commented on", 
                        type=str, required=True)

    parser.add_argument("--bert_pretrained_path",
                        help="a path to a pretrained BERT model on code", 
                        type=str, required=True)

    parser.add_argument("--vocab_path",
                        help="a path to the vocabulary txt file.", 
                        type=str, default="./vocab.txt")

    parser.add_argument("--vocab_size",
                        help="a number of entries to use from the vocabulary. If not provided all the entries are included." 
                             " The same vocab_size should be used here as it was used for training the BERT model.", 
                        type=int, default=None)

    parser.add_argument("--line_column", help="a name of the column that stores the lines.",
                        default="line_contents", type=str)
    
    parser.add_argument("--message_column", help="a name of the column that stores the comment/message.",
                        type=str, default="message")

    parser.add_argument("--purpose_column", help="a name of the column that stores the decision class for the comment purpose.",
                        type=str, default="purpose")

    parser.add_argument("--purpose_labels", help="a list of purpose labels to include (categories).",
                        type=str, nargs="+", default=default_purpose_labels)

    parser.add_argument("--subject_columns", help="a list of column names that store the decision classes for the comment subjects.",
                        type=str, nargs="+", default=default_subject_columns)

    parser.add_argument("--seq_len", help="a maximum length of a line (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--host", help="a host name to serve at.",
                        type=str, default="127.0.0.1")

    parser.add_argument("--port", help="a number of port to run the server on.",
                        type=int, default=8888)

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)
    
    parser.add_argument("--cut_off_percentile", help="a cut_off point to decide whether lines are similar or not. "
                                                "It is the percentile of similarities between the lines from the database " 
                                                "to its most similar lines in the same database.",
                        type=int, default=50)

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--no_layers", help="the number of layers to include while extracting embeddings.",
                        default=4, type=int)

    parser.add_argument("--random_seed", help="a random seed used to control the process.",
                        type=int, default=102329)
    
    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    commented_lines_paths = args['commented_lines_paths']
    commented_lines_embeddings_path = args['commented_lines_embeddings_path']
    bert_trained_path = args['bert_trained_path']
    bert_pretrained_path = args['bert_pretrained_path']
    sep = args['sep']
    port = args['port']
    host = args['host']
    cut_off_percentile = args['cut_off_percentile']
    line_column = args['line_column']
    message_column = args['message_column']
    purpose_column = args['purpose_column']
    purpose_labels = args['purpose_labels']
    subject_columns = args['subject_columns']
    seq_len = args['seq_len']
    vocab_path = args['vocab_path']
    vocab_size = args['vocab_size']
    not_use_gpu = args['not_use_gpu']
    no_layers = args['no_layers']
    random_seed = args['random_seed']
    
    ######

    cols = [line_column, message_column, purpose_column] + subject_columns

    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if not not_use_gpu and len(gpus) == 0:
        logger.error("You don't have a GPU available on your system, it can affect the performance...")

    logger.info("Loading comments data...")
    reviews_all_df = load_comments_files(commented_lines_paths, cols, sep)
    reviews_all_df[line_column] = reviews_all_df[line_column].fillna("")
    logger.info(f"Preserving comments with the purpose: {', '.join(purpose_labels)}")
    reviews_all_df = reviews_all_df[reviews_all_df[purpose_column].isin(purpose_labels)].reset_index(drop=True)
    logger.info(f"The reference database contains {reviews_all_df.shape[0]} comments.")

    logger.info(f"Loading commented lines and their embeddings from {commented_lines_embeddings_path}")
    with open(commented_lines_embeddings_path, encoding="utf-8", errors="ignore") as f:
        lines_database, embeddings_database = json.load(f)
    logger.info(f"Loaded {len(lines_database)} reference lines.")
    
    logger.info(f"Fitting a similarity line finder...")
    np.random.seed(random_seed)
    random.seed(random_seed)
    finder = SimilarLinesFinder(cut_off_percentile=cut_off_percentile, cut_off_sample=200, max_similar=None)
    finder.fit(lines_database, embeddings_database)

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path, limit=vocab_size)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    logger.info("Initializing a BERT code tokenizer...")
    tokenizer = CodeTokenizer(vocab.token_dict, cased=True)
    logger.info(f"BERT code tokenizer ready, example: 'bool acoraIs_nice = True;' -> {str(tokenizer.tokenize('bool acoraIs_nice = True;'))}")

    global graph
    global sess
    config = ConfigProto( device_count = {'GPU': 0 if not_use_gpu else len(gpus)}, allow_soft_placement = True )
    sess = Session(config=config) 
    if tf.__version__.startswith("1."):
        keras.backend.set_session(sess)
        graph = tf.get_default_graph()
    else:
        tf.compat.v1.keras.backend.set_session(sess)
        graph = tf.compat.v1.get_default_graph()
   
    with graph.as_default():
        custom_objects = get_custom_objects()
        custom_objects['RAdam'] = RAdam
        
        logger.info(f"Loading a pre-trained BERT model from {bert_pretrained_path}...")
        recommender = None

        pre_model = keras.models.load_model(bert_pretrained_path, 
                                            custom_objects=custom_objects)

        embeddings_extractor = CodeLinesBERTEmbeddingsExtractor(base_model=pre_model, 
                                                                no_layers=no_layers,
                                                                token_dict=vocab.token_dict)

        logger.info(f"Loading the trained BERT model from {bert_trained_path}...")    
        model = keras.models.load_model(bert_trained_path, 
                                            custom_objects=custom_objects)
        
        recommender = CodeReviewFocusRecommender(classifier=model, 
                            code_tokenizer = tokenizer, 
                            seq_len = seq_len, 
                            embeddings_extractor = embeddings_extractor,
                            similarity_finder = finder,
                            review_comments_df = reviews_all_df,
                            line_column = line_column,
                            purpose_column = purpose_column,
                            subject_columns = subject_columns,
                            message_column = message_column,
                            classify_threshold=0.5)

    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host=host, port=port)






