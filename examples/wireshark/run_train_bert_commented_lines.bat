python ../../scripts/train_bert_comments.py  --training_data_paths ^
     "data\wireshark_train_detecting_commented_lines.csv" ^
     --bert_pretrained_path  "bert_pretrained_on_code.h5" ^
     --use_adapter ^
     --vocab_path "data\vocab.txt" ^
     --vocab_size 10000 ^
     --weight_instances ^
     --epochs 10 ^
     --sep "$" ^
     --report_trainig_accuracy ^
     --lines_commented_cm_train_path "lines_commented_train.pdf" ^
     --model_save_path "bert_lines_commented.h5" ^
     --train_size 0.9 ^
     --lines_commented_cm_train_val_path "lines_commented_val.pdf"


