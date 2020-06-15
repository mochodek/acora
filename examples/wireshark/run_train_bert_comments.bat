python ../../scripts/train_bert_comments.py  --training_data_paths ^
     "data\wireshark_comments_train.xlsx" ^
     --bert_pretrained_path  "..\uncased_L-8_H-512_A-8" ^
     --report_comments_lengths ^
     --weight_instances ^
     --epochs  70 ^
     --batch_size 64 ^
     --report_training_accuracy ^
     --purpose_cm_train_path  "comment_purpose_train_w.pdf"  ^
     --subject_cm_train_path  "comment_subject_train_w.pdf"  ^
     --model_save_path  "bert_on_train_comments3.h5" ^
     --train_size  1.0 ^
     --purpose_cm_train_val_path  "comment_purpose_val.pdf" ^
     --subject_cm_train_val_path  "comment_subject_val.pdf" 

