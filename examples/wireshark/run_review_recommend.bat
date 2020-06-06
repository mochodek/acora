python ../../scripts/review_recommend.py  ^
     --input_lines_paths "data\wireshark_comments_test.xlsx" ^
     --commented_lines_paths ^
     "data\wireshark_comments_train.xlsx" ^
     --commented_lines_embeddings_path "data\wireshark_comments_train_embeddings.json" ^
     --bert_trained_path "bert_lines_commented.h5" ^ 
     --bert_pretrained_path "bert_pretrained_on_code.h5" ^ 
     --vocab_path "data\vocab.txt" ^
     --vocab_size 10000 ^
     --output_file_path "data\review_recommend.xlsx" ^
     --line_column "line_contents" ^
     --message_column "message" ^
     --purpose_column "purpose" ^
     --purpose_labels "change_request" "discussion_participation" "discussion_trigger" ^ 
     --sep "$" ^
     --cut_off_percentile 50 ^
     --no_layers 4

