
python main.py \
--batch_size 32 \
--test_batch_size 32 \
--save_prefix 230702data \
--do_train 0 \
--do_short 0 \
--seed 1 \
--max_epoch 30 \
--train_path '../POLICE_data/230702data/train_dev.json' \
--dev_path '../POLICE_data/230702data/test.json' \
--test_path '../POLICE_data/230702data/test.json' \