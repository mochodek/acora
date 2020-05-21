#!/bin/sh

python ../../scripts/download_commented_lines_from_gerrit.py "https://code.wireshark.org/review/" \
     "./data/wireshark_commented_lines_merged.csv" \
     "/changes/?q=status:merged&o=ALL_FILES&o=ALL_REVISIONS&o=DETAILED_LABELS" \
     --sleep_between_pages 1 \
     --n 500 \
     --max_queries 5000 \
     --max_fails 100 \
     --sep "$" \
     --from_date "2015-02-23" \
     --to_date "2020-05-18"

sleep 10