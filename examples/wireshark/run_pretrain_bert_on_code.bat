python ../../scripts/pretrain_bert_on_code.py  ^
     --line_pairs_dir_path "data\line_pairs" ^
     --vocab_path "data\vocab.txt" ^
     --vocab_size 10000 ^
     --bert_config_path "data\bert_config.json" ^
     --seq_len 128 ^
     --batch_size 32 ^
<<<<<<< HEAD
     --epochs 1 ^
=======
     --epochs 10 ^
>>>>>>> 462351b5db2782de34120624dc93491b705f3a84
     --model_save_path "bert_pretrained_on_code.h5" ^
     --use_adapter ^
     --optimizer "Lamb"
