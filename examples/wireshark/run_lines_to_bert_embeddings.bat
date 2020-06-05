python ../../scripts/lines_to_bert_embeddings.py  --lines_paths ^
     "data\wireshark_comments_train.xlsx" ^
     --bert_path  "bert_pretrained_on_code.h5" ^
     --vocab_path "data\vocab.txt" ^
     --vocab_size 10000 ^
     --sep "$" ^
     --seq_len 128 ^
     --line_column "line_contents" ^
     --no_layers 4 ^
     --unique_lines ^
     --output_file_path "data\wireshark_comments_train_embeddings.json"

python ../../scripts/lines_to_bert_embeddings.py  --lines_paths ^
     "data\wireshark_comments_test.xlsx" ^
     --bert_path  "bert_pretrained_on_code.h5" ^
     --vocab_path "data\vocab.txt" ^
     --vocab_size 10000 ^
     --sep "$" ^
     --seq_len 128 ^
     --line_column "line_contents" ^
     --no_layers 4 ^
     --unique_lines ^
     --output_file_path "data\wireshark_comments_test_embeddings.json"



