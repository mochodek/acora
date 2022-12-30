#!/usr/bin/env python

description="""Pre-trains a BERT model on code."""


import argparse
import logging
import os
import json
import gc
import random
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)

import pandas as pd
import numpy as np
import os

from scipy import stats

from collections import Counter

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras.optimizers.experimental import AdamW
    
    from keras_bert import get_model, gen_batch_inputs, get_custom_objects

    from tensorflow.python.client import device_lib


from acora.vocab import BERTVocab

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)


    parser.add_argument("--line_pairs_dir_path",
                        help="a path to the dir where the files with the line pairs are located.", 
                        type=str, default="./code-line-pairs")

    parser.add_argument("--vocab_path",
                        help="a path to the vocabulary txt file.", 
                        type=str, default="./vocab.txt")

    parser.add_argument("--vocab_size",
                        help="a number of entries to use from the vocabulary. If not provided all the entries are included." 
                             " The same vocab_size should be used here as it was used for generating code-line pairs.", 
                        type=int, default=None)

    parser.add_argument("--seq_len", help="a maximum length of a line (in the number of tokens).",
                        type=int, default=128)

    parser.add_argument("--bert_config_path",
                        help="a path to a BERT config .json file (the same format is used as in the original BERT.).", 
                        type=str, default="./bert_config.json")

    parser.add_argument("--model_path",
                        help="a path to a BERT model that you want to continue training.).", 
                        type=str, default=None)

    parser.add_argument("--optimizer",
                        help="an optimizer that will be used to train the model (AdamW).", 
                        type=str, default="AdamW")

    parser.add_argument("--not_use_gpu", help="to forbid using a GPU if available.",
                        action='store_true')

    parser.add_argument("--random_seed", help="a random seed used to control the process.",
                        type=int, default=102329)
    
    parser.add_argument("--batch_size", help="a batch size used for training the model.",
                        type=int, default=32)

    parser.add_argument("--epochs", help="a number of epochs to train the model.",
                        type=int, default=20)

    parser.add_argument("--model_save_path", help="a path to serialize the trained BERT model.",
                        type=str, default="./bert_pretrained_on_code.h5")


    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    line_pairs_dir_path = args['line_pairs_dir_path']
    vocab_path = args['vocab_path']
    vocab_size = args['vocab_size']
    bert_config_path = args['bert_config_path']
    seq_len = args['seq_len']
    not_use_gpu = args['not_use_gpu']
    random_seed = args['random_seed']
    batch_size = args['batch_size']
    epochs = args['epochs']
    model_save_path = args['model_save_path']
    model_path = args['model_path']
    optimizer = args['optimizer']
    if optimizer not in ["AdamW"]:
        logger.error(f"{optimizer} is not a supported optimizer.")
        exit(1)
    
    ######

    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    logger.info(f"GPU devices available: {','.join(gpus)}")
    if not not_use_gpu and len(gpus) == 0:
        logger.error("You don't have a GPU available on your system, it can affect the performance...")

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path, limit=vocab_size)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    if model_path is None:

        logger.info(f"Loading the BERT config from {bert_config_path}")
        with open(bert_config_path, encoding="utf-8", errors="ignore") as f:
            bert_config = json.load(f)

        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        logger.info(f"Creating a model...")
        model = get_model(token_num=vocab.size, #vocab_size
            pos_num=bert_config['max_position_embeddings'],  #max_position_embeddings
            seq_len=seq_len,
            embed_dim=bert_config['hidden_size'], #hidden_size
            transformer_num=bert_config['num_hidden_layers'], #num_hidden_layers
            head_num=bert_config['num_attention_heads'], #num_attention_heads
            feed_forward_dim=bert_config['intermediate_size'], #intermediate_size
            feed_forward_activation=bert_config['hidden_act'], #hidden_act
            dropout_rate=0.1,
            attention_activation=None,
            training=True,
            trainable=True
        )
        #model.name="BERT4Code"

        model.compile(
            optimizer=AdamW(
                learning_rate=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                weight_decay=0.01,
            ),
            loss=keras.losses.sparse_categorical_crossentropy,
        )

        

    else:
        logger.info(f"Loading a model to continue training it...")
        custom_objects = get_custom_objects()
        custom_objects['AdamW'] = AdamW
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    model.summary(print_fn=logger.info)

    def _generator(batch_size, file_paths, token_dict, token_list, seq_len, shuffle=True):  
        line_pairs = []
        line_files_to_read = []
        while True:
            if len(line_pairs) < batch_size:

                if len(line_files_to_read) == 0:      
                    if shuffle:
                        line_files_to_read = random.sample(file_paths, len(file_paths))
                    else:
                        line_files_to_read = [x for x in file_paths]
                    line_pairs = []
                    gc.collect()

                with open(line_files_to_read.pop(), encoding='utf-8') as f:
                    line_pairs.extend(json.load(f))
                
            batch_pairs = line_pairs[:batch_size]
            del line_pairs[:batch_size]
            yield gen_batch_inputs(
                batch_pairs,
                token_dict,
                token_list,
                seq_len=seq_len,
                mask_rate=0.15,
                mask_mask_rate=0.8,
                mask_random_rate=0.1,
                swap_sentence_rate=0.5,
                force_mask=True,
            )

    def calculate_steps_per_epoch(batch_size, file_paths):
        steps = 0
        for file_path in file_paths:
            with open(file_path, encoding='utf-8') as f:
                steps += len(json.load(f))
        return steps // batch_size 
                
    file_paths = glob.glob(os.path.join(line_pairs_dir_path, "*.json"))

    steps_per_epoch = calculate_steps_per_epoch(batch_size, file_paths)
    logger.info(f"Training the model - batch_size = {batch_size}, epochs = {epochs}, steps per epoch = {steps_per_epoch}")
    model.fit_generator(
        generator=_generator(batch_size, file_paths, vocab.token_dict, vocab.token_list, seq_len, shuffle=True),
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs, 
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            keras.callbacks.ModelCheckpoint(model_save_path, monitor='loss', save_best_only=True)
        ],
    )

    logger.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)

    logger.info("Model training process has been completed.")
