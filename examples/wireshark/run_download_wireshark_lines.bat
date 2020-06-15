python ../../scripts/download_lines_from_gerrit.py "https://code.wireshark.org/review/" ^
     "data\wireshark_lines_merged_20150223_20191231.csv" ^
     "/changes/?q=status:merged&o=ALL_FILES&o=ALL_REVISIONS&o=DETAILED_LABELS" ^
     --sleep_between_pages 1 ^
     --n 500 ^
     --max_queries 10000 ^
     --max_fails 1000 ^
     --sep "$" ^
     --from_date "2015-02-23" ^
     --to_date "2019-12-31"

python ../../scripts/download_lines_from_gerrit.py "https://code.wireshark.org/review/" ^
     "data\wireshark_lines_merged_20200101_20200615.csv" ^
     "/changes/?q=status:merged&o=ALL_FILES&o=ALL_REVISIONS&o=DETAILED_LABELS" ^
     --sleep_between_pages 1 ^
     --n 500 ^
     --max_queries 10000 ^
     --max_fails 1000 ^
     --sep "$" ^
     --from_date "2020-01-01" ^
     --to_date "2020-06-15"

