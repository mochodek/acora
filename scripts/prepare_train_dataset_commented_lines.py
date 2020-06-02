#!/usr/bin/env python

description="""Prepare a dataset to train a BERT for detecting lines that will be commented on.
The dataset is created by combining two files: a file with lines that were commented on, and
a file that contains all lines in such a way that firstly all commented lines are included and then
a more or less the same number of lines will be sampled from the latter file coming from the same review
as the commented lines. As a result, we have a balanced set for training.
""" 

import argparse
import logging
import os

import pandas as pd

from pathlib import Path


logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

default_columns_to_preserve = ['change_id', 'revision_id', 'filename', 'line_contents']

if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--lines_with_comments_path",
                        help="a path to a xlsx or csv file with commented lines.",
                        type=str, required=True)

    parser.add_argument("--lines_path",
                        help="a path to a xlsx or csv file with commented lines.",
                        type=str, required=True)

    parser.add_argument("--output_dataset_path",
                        help="a path where the output dataset is to be stored (either csv or xlsx).",
                        type=str, required=True)

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    parser.add_argument("--ok_to_commented_ratio", 
                        help="a requested ratio between the number of OK lines and lines that were commented on.",
                        type=float, default=1.0)

    parser.add_argument("--line_column", help="a name of the column that stores the lines.",
                        default="line_contents", type=str)

    parser.add_argument("--review_change_column", help="a name of the column that stores the review change id.",
                        default="change_id", type=str)
    
    parser.add_argument("--columns_to_preserve", help="a list of columns to take and save to the output file " 
                        "(should include line and review_change columns).",
                        default=default_columns_to_preserve, type=str, nargs="+")

    
    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    lines_with_comments_path = args['lines_with_comments_path']
    lines_path = args['lines_path']
    sep = args['sep']
    output_dataset_path = args['output_dataset_path']
    ok_to_commented_ratio = args['ok_to_commented_ratio']
    line_column = args['line_column']
    review_change_column = args['review_change_column']
    columns_to_preserve = args['columns_to_preserve']
    
    ######

    lines_with_comments_path_extension = Path(lines_with_comments_path).suffix
    lines_path_extension = Path(lines_path).suffix
    output_dataset_path_extension = Path(output_dataset_path).suffix

    if lines_with_comments_path_extension not in ['.xlsx', '.csv']:
        logger.error(f"Wrong file type of {lines_with_comments_path}. Only csv and xlsx files are supported.")
        exit(1)

    if lines_path_extension not in ['.xlsx', '.csv']:
        logger.error(f"Wrong file type of {lines_path}. Only csv and xlsx files are supported.")
        exit(1)

    if output_dataset_path_extension not in ['.xlsx', '.csv']:
        logger.error(f"Wrong file type of {output_dataset_path}. Only csv and xlsx files are supported.")
        exit(1)

    if lines_with_comments_path_extension == '.xslx':
        lines_with_comments_df = pd.read_excel(lines_with_comments_path)
    else:
        lines_with_comments_df = pd.read_csv(lines_with_comments_path, sep=sep)
    lines_with_comments_df[line_column] = lines_with_comments_df[line_column].fillna("")
    logger.info(f"Loaded {lines_with_comments_df.shape[0]:,} lines with comments from {lines_with_comments_path}")

    if lines_path_extension == '.xlsx':
        lines_df = pd.read_excel(lines_path)
    else:
        lines_df = pd.read_csv(lines_path, sep=sep)
    lines_df[line_column] = lines_df[line_column].fillna("")
    logger.info(f"Loaded {lines_df.shape[0]:,} lines from {lines_path}")

    logger.info(f"Removing empty lines...")
    lines_with_comments_df = lines_with_comments_df[lines_with_comments_df[line_column] != ""]
    logger.info(f"The remaining number of commented lines is {lines_with_comments_df.shape[0]:,}")
    lines_df = lines_df[lines_df[line_column] != ""]
    logger.info(f"The remaining number of lines is {lines_df.shape[0]:,}")

    commented_lines_change_ids = lines_with_comments_df[review_change_column].unique()
    logger.info(f"Commented lines come from {len(commented_lines_change_ids):,} different reviewed changes")

    logger.info(f"Filtering lines only from these changes...")
    lines_df = lines_df[lines_df[review_change_column].isin(commented_lines_change_ids)]
    logger.info(f"The remaining number of lines is {lines_df.shape[0]:,}")

    logger.info("Preparing the dataset...")
    dataset = []
    total_commented_added = 0
    total_ok_added = 0
    for i, change_id in enumerate(commented_lines_change_ids): 

        if i % 100 == 0:
            logger.info(f'Processing {i+1} / {len(commented_lines_change_ids)}: {change_id}')
            
        dataset_change = []
        lines_with_comments_for_change_df = lines_with_comments_df[lines_with_comments_df[review_change_column] == change_id]
        lines_for_change_df = lines_df[lines_df[review_change_column] == change_id]
        
        # TODO: Consider limiting the number of commented lines from a review to up to the same number as non-commented lines.
        #diff_lines = lines_with_comments_for_change_df.shape[0] - lines_for_change_df.shape[0]
        #if diff_lines > 0:
        #    lines_with_comments_for_change_df = lines_with_comments_for_change_df.head(lines_for_change_df.shape[0])

        added_commented_lines = set()
        for idx, commented_line in lines_with_comments_for_change_df.iterrows():
            if commented_line[line_column] not in added_commented_lines:
                dataset.append([commented_line[col] for col in columns_to_preserve] + [1,])
                added_commented_lines.add(commented_line[line_column])
        
        no_commented_lines = len(added_commented_lines)
        total_commented_added += no_commented_lines
        max_no_ok_lines = no_commented_lines * ok_to_commented_ratio
        
        ok_lines_added = 0
        for idx, ok_line in lines_for_change_df.iterrows():
            if ok_lines_added >= max_no_ok_lines and total_ok_added-total_commented_added >= 0:
                break
                
            if ok_line[line_column] not in lines_with_comments_for_change_df[line_column]:
                dataset.append([ok_line[col] for col in columns_to_preserve] + [0,])
                ok_lines_added += 1
                total_ok_added += 1

    dataset_df = pd.DataFrame(dataset, columns=columns_to_preserve + ['commented'])    

    logger.info(f"Lines: {dataset_df.shape[0]:,}")
    logger.info(f"Unique line contents: {dataset_df[line_column].unique().shape[0]:,}")
    counts = dataset_df.groupby('commented').count()
    logger.info(f"Non-commented lines: {counts.loc[0, review_change_column]:,}")
    logger.info(f"Commented lines: {counts.loc[1, review_change_column]:,}")

    counted_unique_line_contents_by_class = dataset_df.groupby([line_column, "commented"]).count()
    counted_unique_lines_by_class= counted_unique_line_contents_by_class.groupby(line_column).count()
    unique_lines_in_multiple_classes = counted_unique_lines_by_class[counted_unique_lines_by_class[review_change_column] > 1].index.tolist()
    ratio_of_lines_with_instances_in_multiple_clases = counted_unique_line_contents_by_class \
                            .loc[unique_lines_in_multiple_classes,:][review_change_column].sum() / dataset_df.shape[0]
    logger.info(f"Number of unique lines both commented and non-commented: {len(unique_lines_in_multiple_classes):,}")
    logger.info(f"Ratio of lines both commented and non-commented to all lines: {ratio_of_lines_with_instances_in_multiple_clases}")

    logger.info(f"Saving the dataset to {output_dataset_path}")
    if output_dataset_path_extension == '.xlsx':
        dataset_df.to_excel(output_dataset_path, index=False)
    else:
        dataset_df.to_csv(output_dataset_path, sep=sep, index=False)

        
        
        

    

    