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

##############################
# using mrpc model from Intel
# python -u glue_trans.py --dir pre_mrpc2 --output_dir save_directories/mrpc2_new --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC2_new/ --pruning_steps 10  > logs/bert_mrpc2_new.log &
# python -u glue_trans.py --dir pre_mrpc2 --output_dir save_directories/mrpc2_new --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC2_new/ --pruning_steps 10  > logs/bert_mrpc2_new.log &

# python -u glue_trans.py --dir pre_mrpc_distilbert --output_dir save_directories/mrpc2_new --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC2_new/ --pruning_steps 10  > logs/bert_mrpc2_new.log &

# vicl/distilbert-base-uncased-finetuned-mrpc
# python -u glue_trans.py --dir pre_distilbert --output_dir tmp/distilbert --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type distilbert --model_name_or_path distilbert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 42 --checkpoint_dir checkpoints/distilbert/ --pruning_steps 10 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 > logs/distilbert_mrpc.log

python -u glue_trans.py --dir pre_distilbert --output_dir tmp/distilbert --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type distilbert --model_name_or_path distilbert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/distilbert/ --pruning_steps 10 > logs/distilbert_mrpc.log

# done
# python -u glue_trans.py --dir pre_mrpc2 --output_dir save_directories/mrpc2_nofinetune --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 0 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC2_nofinetune/ --pruning_steps 10  > logs/bert_mrpc2_nofinetune.log &
# python -u glue_trans.py --dir pre_mrpc2 --output_dir save_directories/mrpc2_nofinetune --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 0 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MRPC2_nofinetune/ --pruning_steps 10  > logs/bert_mrpc2_nofinetune.log &



# generate pruned checkpoints for MNLI
# yoshitomo-matsubara/bert-base-uncased-mnli
# python -u glue_trans.py --dir pre_mnli --output_dir save_directories/mnli --logging_steps 12271 --task_name MNLI --data_dir glue_data/MNLI --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/MNLI/ --pruning_steps 10  > logs/bert_mnli.log &

# finetuning - no pruning - base bert finetuning - https://huggingface.co/Intel/bert-base-uncased-mrpc
# python -u glue_trans.py --dir pre --output_dir save_directories/mrpc_base2 --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 42 --checkpoint_dir checkpoints/MRPC_base2/ --pruning_steps 1 --no_pruning True --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 > logs/bert_mrpc_base2.log &

# python -u glue_trans.py --dir pre --output_dir save_directories/mrpc_base2 --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --learning_rate 2e-5 --num_train_epochs 5 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 42 --checkpoint_dir checkpoints/MRPC_base2/ --pruning_steps 1 --no_pruning True --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 8 > logs/bert_mrpc_base2.log &

# python -u glue_trans.py --dir pre --output_dir save_directories/mrpc_base3 --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 57 --checkpoint_dir checkpoints/MRPC_base3/ --pruning_steps 1 --no_pruning True --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 8 > logs/bert_mrpc_base3.log &


############
# python -u LT_glue.py --output_dir save_directories/mrpc_lth --logging_steps 36813 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased  --do_train --do_eval   --do_lower_case  --max_seq_length 128   --per_gpu_train_batch_size 32  --learning_rate 2e-5  --num_train_epochs 30  --overwrite_output_dir  --evaluate_during_training   --save_steps 36813  --eval_all_checkpoints   --seed 57  > logs/bert_mrpc_lth.log &

##########

# python -u LT_glue_new.py --dir pre --output_dir tmp/mrpc/LT --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 30 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/LT_MRPC/ --pruning_steps 10 --rewind_epoch 6 --populate_finetune > logs/mrpc_lth.log &

# python -u LT_glue_new.py --dir pre --output_dir tmp/mrpc/LT --logging_steps 12271 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 30 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 5 --checkpoint_dir checkpoints/LT_MRPC/ --pruning_steps 10 --rewind_epoch 6 > logs/mrpc_lth_iterations.log

# running 1/29/24

# python -u LT_glue_new.py --dir pre --output_dir tmp/mrpc/LT2 --logging_steps 36813 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 30 --overwrite_output_dir --evaluate_during_training --save_steps 36813 --eval_all_checkpoints --seed 57 --checkpoint_dir checkpoints/LT_MRPC2/ --pruning_steps 10 --rewind_epoch 6 --populate_finetune > logs/mrpc_lth2.log &
