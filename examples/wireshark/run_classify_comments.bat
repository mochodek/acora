python ../../scripts/classify_comments.py  --classify_input_data_paths ^
     "data\wireshark_comments_test.xlsx" ^
     "data\wireshark_comments_train.xlsx" ^
     --classify_output_data_path "data\classify_comments_results.xlsx" ^
     --model_trained_path "bert_on_train_comments.h5" ^
     --bert_vocab_path  "..\uncased_L-8_H-512_A-8\vocab.txt" ^
     --preserve_all_columns

