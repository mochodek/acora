#!/usr/bin/env python

description="""Downloads lines of code from a Gerrit instance."""

import argparse
import logging

from acora.gerrit import GerritReviewDataDownloader

logger_gerrit = logging.getLogger('acora.data.gerrit.GerritReviewDataDownloader')
logger_gerrit.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger_gerrit.addHandler(ch)

logger = logging.getLogger(f'acora.{__file__}')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


if __name__ == '__main__':

    logger.info(f"\n#### Running script: {__file__}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("base_url",
                        help="a base URL of the Gerrit's API you want to acccess", type=str)

    parser.add_argument("filename",
                        help="a path to the output csv file", type=str)

    parser.add_argument("base_query",
                        help="a base query to the Gerrit instance to return the changes", type=str)

    parser.add_argument("--sleep_between_pages", help="time to wait between each attempt in seconds.",
                        default=0, type=int)
    
    parser.add_argument("--n", help="a number of changes per request.",
                        default=500, type=int)

    parser.add_argument("--max_queries", help="a maximum number of queries.",
                        default=1000, type=int)
    
    parser.add_argument("--max_fails", help="a maximum number of failed attempts to query.",
                        default=50, type=int)

    parser.add_argument("--sep", help="a seprator used to separate columns in the csv file.",
                        default=";", type=str)

    parser.add_argument("--from_date", help="if provided, only not older changes will be returned.",
                        default=None, type=str)

    parser.add_argument("--to_date", help="if provided, only not newer changes will be returned.",
                        default=None, type=str)

    args = vars(parser.parse_args())
    logger.info(f"Run parameters: {str(args)}")

    base_url = args['base_url']
    filename = args['filename']
    base_query = args['base_query']
    sleep_between_pages = args['sleep_between_pages']
    n = args['n']
    max_queries = args['max_queries']
    max_fails = args['max_fails']
    sep = args['sep']
    from_date = args['from_date']
    to_date = args['to_date']


    downloader = GerritReviewDataDownloader(base_url=base_url,sleep_between_pages=sleep_between_pages)
    downloader.download_lines_to_csv(filename, base_query, n=n, 
                max_queries=max_queries, max_fails=max_fails,
                from_date=from_date, to_date=to_date, sep=sep)


