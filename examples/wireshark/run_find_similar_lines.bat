python ../../scripts/find_similar_lines.py  ^
     --lines_database_path "data\wireswireshark_comments_train_embeddingshark_comments_train.json" ^
     --lines_path "data\wireshark_comments_test_embeddings.json" ^
     --output_file_path "data\similar_test_in_train.xlsx" ^
     --sep "$" ^
     --cut_off_percentile 50 ^
     --max_similar 4



