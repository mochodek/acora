#!/usr/bin/env python

description="""Generate code line pairs that could be used to pre-train a BERT model on code."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)
import shutil
import json

import numpy as np

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    if tf.__version__.startswith("1."):
        os.environ['TF_KERAS'] = '0'
    else:
        os.environ['TF_KERAS'] = '1'

from acora.vocab import BERTVocab
from acora.code import CodeTokenizer, SignatureCodeTokenizer, generate_code_pairs

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--code_lines_path",
                        help="a path to a .json file with the extracted lines.", 
                        type=str, default=".lines.json")

    parser.add_argument("--vocab_path",
                        help="a path to the vocabulary txt file.", 
                        type=str, default="./vocab.txt")

    parser.add_argument("--vocab_size",
                        help="a number of most common vocab entries to pick. If not provided all the entries are included.", 
                        type=int, default=None)

    parser.add_argument("--output_dir_path",
                        help="a path to the output dir where the files with the line pairs will be stored.", 
                        type=str, default="./code-line-pairs")

    parser.add_argument("--line_pairs_per_file",
                        help="a number of code-line pairs stored in a single file.", 
                        type=int, default=10000)

    parser.add_argument("--line_repeat_period",
                        help="a number of lines per each a given line can be repeated.", 
                        type=int, default=64)

    parser.add_argument("--tokenizer",
                        help="a code tokenizer to be used (either CodeTokenizer or SignatureCodeTokenizer).", 
                        type=str, default="CodeTokenizer")

    parser.add_argument("--omit_whitespace",
                        help="whether or not to omit whitespaces as tokens).", 
                        action='store_true')



    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    code_lines_path = args['code_lines_path']
    vocab_path = args['vocab_path']
    vocab_size = args['vocab_size']
    output_dir_path = args['output_dir_path']
    line_pairs_per_file = args['line_pairs_per_file']
    line_repeat_period = args['line_repeat_period']
    tokenizer = args['tokenizer']
    omit_whitespace = args['omit_whitespace']


    if tokenizer not in ['CodeTokenizer', "SignatureCodeTokenizer"]:
        logger.error(f"{tokenizer} is not a supported tokenizer.")
        exit(1)
    
    ######

    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = BERTVocab.load_from_file(vocab_path, limit=vocab_size)
    logger.info(f"Loaded {vocab.size:,} vocab entries.")

    logger.info(f"Loading code lines from {code_lines_path}")
    with open(code_lines_path, encoding="utf-8", errors="ignore") as f:
        file_lines = json.load(f)


    if os.path.exists(output_dir_path) and os.path.isfile(output_dir_path):
        logger.error(f"The output location {output_dir_path} exists and it is a file.")
        exit(1)

    if os.path.exists(output_dir_path):
        logger.info(f"Removing the output folder {output_dir_path}.")
        try:
            shutil.rmtree(output_dir_path)
        except OSError as e:
            logger.error(f"Error while removing the folder: {e.filename} - {e.strerror}")
            exit(1)

    logger.info(f"Creating the output folder {output_dir_path}.")
    try:
        os.makedirs(output_dir_path)    
    except OSError as e:
        logger.error(f"Error while creating the folder: {e.filename} - {e.strerror}")
        exit(1) 

    logger.info("Initializing a BERT code tokenizer...")
    if tokenizer == 'CodeTokenizer':
        tokenizer = CodeTokenizer(vocab.token_dict, cased=True, preserve_whitespace=not omit_whitespace)
    else:
        tokenizer = SignatureCodeTokenizer(vocab.token_dict, cased=True, preserve_whitespace=not omit_whitespace)
    logger.info(f"BERT code tokenizer ready, example: 'int acoraIs_nice = 1;' -> {str(tokenizer.tokenize('int acoraIs_nice = 1;'))}")

    logger.info("Generating code-line pairs...")
    code_line_pairs = generate_code_pairs(file_lines, tokenizer, line_repeat_period, logger)
    no_lines_pairs = len(code_line_pairs)
    logger.info(f"Generated {no_lines_pairs:,} pairs of lines of code.")

    range_inds = [x for x in np.arange(0, no_lines_pairs, line_pairs_per_file)]
    range_inds.append(no_lines_pairs)
    del range_inds[0]

    logger.info(f"Saving generated lines to .json files.")
    last_id = 0
    for i, range_id in enumerate(range_inds):
        logger.info(f"Saving {range_id-last_id} line pairs to {os.path.join(output_dir_path, f'{i}.json')}")
        with open(os.path.join(output_dir_path, f"{i}.json"), 'w', encoding='utf-8') as f:
            json.dump(code_line_pairs[last_id:range_id], f, ensure_ascii=False, indent=4)
        last_id = range_id


    logger.info("Process of generating code-line pairs has been completed.")






    

    