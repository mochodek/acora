#!/bin/sh

python ../../scripts/test_bert_comments.py  --test_data_paths \
     "data/wireshark_comments_test.xlsx" \
     --model_trained_path "bert_on_train_comments.h5" \
     --bert_vocab_path  "../uncased_L-8_H-512_A-8/vocab.txt" \
     --purpose_cm_test_path  "comment_purpose_test.pdf" \
     --subject_cm_test_path  "comment_subject_test.pdf" 
