#!/bin/sh

python ../../scripts/test_bert_commented_lines.py  --test_lines_paths \
     "./data/wireshark_train_detecting_commented_lines.csv" \
     --bert_trained_path  "bert_lines_commented.h5" \
     --vocab_path "./data/vocab.txt" \
     --vocab_size 10000 \
     --sep "$" \
     --decision_column "commented" \
     --lines_commented_cm_test_path "./lines_commented_test.pdf"

sleep 10