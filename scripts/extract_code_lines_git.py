#!/usr/bin/env python

description="""Extracts lines of code per file from a Git controlled repository and stores in a json file."""

import argparse
import logging
import os
from pathlib import Path
import json

from pygit2 import Repository, reference_is_valid_name, \
        GIT_SORT_TOPOLOGICAL, GIT_SORT_TIME, GitError

import datetime
from dateutil import parser as date_parser

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

def process_tree(tree, file_extensions, logger):
    """ Recursively exracts code from blobs in a Git tree. """

    result = []
    logger.info(f"Exploring Git tree: {tree.name}")
    for i, obj in enumerate(tree):
        if obj.type_str == 'blob' and Path(obj.name).suffix in file_extensions:
            print(f"Reading the file: {obj.name}")
            try:
                lines = obj.data.decode("utf-8").split("\n")
            except:
                lines = []
            print(f"Extracted {len(lines):,} lines from the  {obj.name}")
            result.append(lines)
        if obj.type_str == 'tree':
            result += process_tree(obj, file_extensions, logger)
    return result
            


if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--output_lines_path",
                        help="a path to an output .json file with the extracted lines.", 
                        type=str, default=".lines.json")

    parser.add_argument("--repo_path",
                        help="a path to local directory of the repository.", 
                        type=str, required=True)

    parser.add_argument("--file_extensions",
                        help="a list of file extensions to be scanned (e.g., .txt, .c, .cpp, .h).", 
                        type=str, nargs="+")

    parser.add_argument("--commit", help="a commit hash if a particular commit is requested.",
                        type=str, default=None)

    parser.add_argument("--last_commit", help="to take the last commit. It is used only if --commit is not provided",
                        action='store_true')

    parser.add_argument("--last_commit_before_date", help="a boundary date to find the last commit before the date."
                        "It is used only if --commit is not provided.",
                        type=str, default=None)


    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    repo_path = args['repo_path']
    output_lines_path = args['output_lines_path']
    file_extensions = args['file_extensions']
    commit_id = args['commit']
    take_last_commit = args['last_commit']
    last_commit_before_date = args['last_commit_before_date']
    if last_commit_before_date is not None:
        first_date = date_parser.parse(last_commit_before_date)
    
    ######

    if os.path.isdir(repo_path):
        try:
            repository = Repository(repo_path)
        except GitError as e:
            logger.error(f"A directory {repo_path} exists but it is not a valid Git repository")
            exit(1)
        logger.info(f"Found a local copy of the repository at {repo_path}.")

    else:
        logger.info(f"A local copy of the repository at {repo_path} doesn't exist. Please, clone the repository.")
        exit(1)

    if commit_id is not None:
        logger.info(f"Found the last commit made {commit_id}")
        commit = repository.get(commit_id)
    elif take_last_commit or last_commit_before_date is not None:
        last_commit = None
        for i, commit in enumerate(repository.walk(repository.head.target, GIT_SORT_TOPOLOGICAL | GIT_SORT_TIME)):
            if take_last_commit:
                break
            commit_time = datetime.datetime.fromtimestamp(commit.commit_time)
            if commit_time < first_date:
                last_commit = commit
                break
        logger.info(f"Found the commit {commit.id} as the one matching the search criteria.")
    else:
        exit(1)

    logger.info(f"Start to process the files of the commit {commit.id}")
    lines = process_tree(commit.tree, file_extensions, logger)

    no_lines = 0
    no_files = 0
    for file_lines in lines:
        no_files += 1
        for line in file_lines:
            no_lines += 1
    logger.info(f"Extracted {no_lines:,} lines from {no_files:,} files")
    

    logger.info(f"Saving the lines to {Path(output_lines_path)}")
    with open(output_lines_path, 'w', encoding='utf-8', errors="ignore") as f:
        json.dump(lines, f, ensure_ascii=False, indent=4)

    logger.info("Lines extraction process has been completed.")