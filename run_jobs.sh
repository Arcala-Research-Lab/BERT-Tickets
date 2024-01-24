# create oneshot pruning checkpoints

# python oneshot.py --weight pre --model glue --rate 0.0 > logs/bert_oneshot_00_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.1 > logs/bert_oneshot_10_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.2 > logs/bert_oneshot_20_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.3 > logs/bert_oneshot_30_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.4 > logs/bert_oneshot_40_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.5 > logs/bert_oneshot_50_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.6 > logs/bert_oneshot_60_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.7 > logs/bert_oneshot_70_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.8 > logs/bert_oneshot_80_nofinetune.log &
# python oneshot.py --weight pre --model glue --rate 0.9 > logs/bert_oneshot_90_nofinetune.log &

# glue_trans command line
# python -u glue_trans.py --dir pre_mrpc --output_dir save_directories/mrpc --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC/ --pruning_steps 10 > logs/bert_mrpc.log &

python -u glue_trans.py --dir pre_mrpc2 --output_dir save_directories/mrpc2 --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC2/ --pruning_steps 10  > logs/bert_mrpc2.log &