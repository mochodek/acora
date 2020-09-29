#!/usr/bin/env python

description="""Extracts lines of code from the existing directory and stores as xlsx or csv."""

import argparse
import logging
import os
from pathlib import Path
import pandas as pd

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

if __name__ == '__main__':


    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--output_lines_path",
                        help="a path to an output csv or xlsx file with the extracted lines.", 
                        type=str, default="output.xlsx")

    parser.add_argument("--code_path",
                        help="a path to local directory of the repository.", 
                        type=str, required=True)

    parser.add_argument("--file_extensions",
                        help="a list of file extensions to be scanned (e.g., .txt, .c, .cpp, .h).", 
                        type=str, nargs="+")

    parser.add_argument("--sep", help="a seprator used to separate columns in a csv file.",
                        default=";", type=str)

    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    #### Reading run arguments

    code_path = args['code_path']
    output_lines_path = args['output_lines_path']
    file_extensions = args['file_extensions']
    sep = args['sep']
    
    ######

    if not os.path.isdir(code_path):
        logger.error(f"A directory {code_path} doesn't exist.")
        exit(1)

    output_file_path_extension = Path(output_lines_path).suffix
    if output_file_path_extension not in ['.xlsx', '.csv']:
        logger.error(f"Wrong file type of {output_lines_path}. Only csv and xlsx files are supported.")
        exit(1)

    src_code_paths = []
    for ext in file_extensions:
        src_code_paths.extend([str(x) for x in Path(code_path).glob(f"**/*{ext}")])
    logger.info(f"Found {len(src_code_paths)} files to be processed.")

    logger.info(f"Starting extracting lines...")
    filenames = []
    lines = []
    for code_file_path in src_code_paths:
        logger.info(f"Extracting lines from {code_file_path}")
        rel_path = os.path.relpath(code_file_path, code_path)
        with open(code_file_path, "r", encoding='utf-8', errors="ignore") as f:
            new_lines = f.readlines()
            lines.extend(new_lines)
            filenames.extend([rel_path]*len(new_lines))
        logger.info(f"Extracted {len(new_lines)} lines from {code_file_path}")



    logger.info(f"Saving the lines to {Path(output_lines_path)}")
    result_df = pd.DataFrame(dict(filename=filenames, line_contents=lines))

    if output_file_path_extension == '.xlsx':
        result_df.to_excel(output_lines_path, index=False)
    else:
        result_df.to_csv(output_lines_path, sep=sep, index=False)

    logger.info("Lines extraction process has been completed.")