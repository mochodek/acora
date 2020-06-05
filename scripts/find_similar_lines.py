#!/usr/bin/env python

description="""Finds similar lines to those stored in the "database" using line embeddings."""

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger("tensorflow").setLevel(logging.INFO)
import json

import pandas as pd

from pathlib import Path

from acora.code_similarities import SimilarLinesFinder

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger_similar = logging.getLogger('acora.code_similarities')
logger_similar.setLevel(logging.DEBUG)
logger_similar.addHandler(ch)


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--lines_database_path",
                        help="a path to the json file storing lines and their embeddings "
                            "serving as the basis for search (see lines_to_bert_embeddings.py).", 
                        type=str, default="./database_lines_with_embeddings.json")

    parser.add_argument("--lines_path",
                        help="a path to the json file storing lines and their embeddings "
                            "for which similar lines will be searched from the database (see lines_to_bert_embeddings.py).", 
                        type=str, default="./lines_with_embeddings.json")

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    parser.add_argument("--output_file_path", help="a path to an output file storing the base and similar lines (either csv or xlsx).",
                        type=str, default="./similar_lines.xlsx")
    
    parser.add_argument("--cut_off_percentile", help="a cut_off point to decide whether lines are similar or not. "
                                                "It is the percentile of similarities between the lines from the database " 
                                                "to its most similar lines in the same database.",
                        type=int, default=50)

    parser.add_argument("--max_similar", help="the max. number of similar lines returned (None to return all with the cut_of_percentile).",
                        type=int, default=None)
    

    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    lines_database_path = args['lines_database_path']
    lines_path = args['lines_path']
    sep = args['sep']
    output_file_path = args['output_file_path']
    cut_off_percentile = args['cut_off_percentile']
    max_similar = args['max_similar']
    
    ######

    output_file_path_extension = Path(output_file_path).suffix

    if output_file_path_extension not in ['.xlsx', '.csv']:
        logger.error(f"Wrong file type of {output_file_path}. Only csv and xlsx files are supported.")
        exit(1)

    logger.info(f"Loading reference lines and embeddings from {lines_database_path}")
    with open(lines_database_path, encoding="utf-8", errors="ignore") as f:
        lines_database, embeddings_database = json.load(f)
    logger.info(f"Loaded {len(lines_database)} reference lines.")

    logger.info(f"Loading reference lines and embeddings from {lines_path}")
    with open(lines_path, encoding="utf-8", errors="ignore") as f:
        lines, embeddings = json.load(f)
    logger.info(f"Loaded {len(lines)} lines.")
    
    logger.info(f"Fitting a similarity line finder...")
    finder = SimilarLinesFinder(cut_off_percentile=cut_off_percentile, max_similar=max_similar)
    finder.fit(lines_database, embeddings_database)
    similar_lines = finder.query(lines, embeddings)

    logger.info(f"Saving the output to {output_file_path}")
    result_df = pd.DataFrame(similar_lines, columns=["query_line", 'similar_line', 'distance'])

    if output_file_path_extension == '.xlsx':
        result_df.to_excel(output_file_path, index=False)
    else:
        result_df.to_csv(output_file_path, sep=sep, index=False)

    logger.info("The process of finding similar lines has been completed.")




