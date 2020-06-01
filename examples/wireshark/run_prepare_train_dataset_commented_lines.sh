#!/bin/sh

python ../../scripts/prepare_train_dataset_commented_lines.py  \
     --lines_with_comments_path "./data/wireshark_commented_lines_merged.xlsx" \
     --lines_path "./data/wireshark_lines_merged.xlsx" ^
     --output_dataset_path "./data/wireshark_train_detecting_commented_lines.xlsx" \
     --sep  "$" \
     --ok_to_commented_ratio 1.0 \
     --line_column "line_contents" \
     --review_change_column  "change_id"

slepp 10
