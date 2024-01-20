# create oneshot pruning checkpoints

# python oneshot.py --weight pre --model glue --rate 0.1 > logs/bert_oneshot_10_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.2 > logs/bert_oneshot_20_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.3 > logs/bert_oneshot_30_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.4 > logs/bert_oneshot_40_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.5 > logs/bert_oneshot_50_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.6 > logs/bert_oneshot_60_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.7 > logs/bert_oneshot_70_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.8 > logs/bert_oneshot_80_nofinetune.log &
python oneshot.py --weight pre --model glue --rate 0.9 > logs/bert_oneshot_90_nofinetune.log &

