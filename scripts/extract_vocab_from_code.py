#!/usr/bin/env python

description="""Extracts BERT vocabulary from code."""

import argparse
import logging
import os
import json
from pathlib import Path
from collections import Counter, OrderedDict

from keras_bert import get_base_dict

from acora.vocab import code_vocab_tokenize

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
                        help="a path to the output vocabulary txt file.", 
                        type=str, required="./vocab.txt")


    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    code_lines_path = args['code_lines_path']
    vocab_path = args['vocab_path']
    
    ######

    logger.info(f"Loading code lines from {Path(code_lines_path)}")
    with open(code_lines_path, encoding='utf-8', errors="ignore") as f:
        file_lines = json.load(f)

    if not file_lines:
        logger.error(f"Couldn't load lines from {Path(code_lines_path)}")

    logger.info(f"Extracting tokens from files...")
    tokens = []
    no_files = len(file_lines)
    for i, lines in enumerate(file_lines):
        if i % (no_files//10):
            logger.info(f"Processing {i+1} / {no_files} of the files...")
        for line in lines:
            tokens.extend(code_vocab_tokenize(line))

    logger.info(f"Found {len(tokens):,} in the files.")

    logger.info("Creating BERT vocabulary...")

    token_dict = OrderedDict(get_base_dict())
    chars = [f'##{x}' for x in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_']
    chars += [f'{x}' for x in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_']

    for token in chars:
        if token not in token_dict:
            token_dict[token] = len(token_dict)

    count_tokens = Counter(tokens)
    most_common_tokens = len(count_tokens) # currently we take all of them
    for token in [x[0] for x in count_tokens.most_common(most_common_tokens)]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)

    logger.info(f"Created a vocabulary with {len(token_dict):,} entries.")

    logger.info(f"Saving the vocabulary to {Path(vocab_path)}.")
    with open(vocab_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(f"{str(x)}\n" for x in token_dict.keys())
    
    logger.info("Process of creating a BERT code vocabulary has been completed.")

    