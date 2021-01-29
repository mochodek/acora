#!/usr/bin/env python

description="""Extracts lines of code from the existing directory and stores as json file."""

import argparse
import logging
import os
from pathlib import Path
import json

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--output_lines_path",
                        help="a path to an output .json file with the extracted lines.", 
                        type=str, default="lines.json")

    parser.add_argument("--code_path",
                        help="a path to local directory of the repository.", 
                        type=str, required=True)

    parser.add_argument("--file_extensions",
                        help="a list of file extensions to be scanned (e.g., .txt, .c, .cpp, .h).", 
                        type=str, nargs="+")


    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    code_path = args['code_path']
    output_lines_path = args['output_lines_path']
    file_extensions = args['file_extensions']
    
    ######

    if not os.path.isdir(code_path):
        logger.error(f"A directory {code_path} doesn't exist.")
        exit(1)

    output_file_path_extension = Path(output_lines_path).suffix
    if output_file_path_extension not in ['.json']:
        logger.error(f"Wrong file type of {output_lines_path}. Only json is supported.")
        exit(1)

    src_code_paths = []
    for ext in file_extensions:
        src_code_paths.extend([str(x) for x in Path(code_path).glob(f"**/*{ext}")])
    logger.info(f"Found {len(src_code_paths)} files to be processed.")

    logger.info(f"Starting extracting lines...")
    lines = []
    for code_file_path in src_code_paths:
        logger.info(f"Extracting lines from {code_file_path}")
        rel_path = os.path.relpath(code_file_path, code_path)
        with open(code_file_path, "r", encoding='utf-8', errors="ignore") as f:
            new_lines = f.readlines()
            lines.append(new_lines)
        logger.info(f"Extracted {len(new_lines)} lines from {code_file_path}")



    logger.info(f"Saving the lines to {Path(output_lines_path)}")
    with open(output_lines_path, 'w', encoding='utf-8', errors="ignore") as f:
        json.dump(lines, f, ensure_ascii=False, indent=4)

    logger.info("Lines extraction process has been completed.")