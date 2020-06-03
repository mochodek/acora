#!/bin/sh

python ../../scripts/classify_lines.py  --classify_lines_paths \
     "./data/wireshark_comments_test.xlsx" \
     "./data/wireshark_comments_train.xlsx" \
     --bert_trained_path  "./bert_lines_commented.h5" \
     --vocab_path "./data/vocab.txt" \
     --vocab_size 10000 \
     --sep "$" \
     --line_column "LOC" \
     --output_save_path "./data/classify_lines.xlsx"

sleep 10

