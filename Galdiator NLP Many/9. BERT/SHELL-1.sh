$ export BERT_BASE_DIR= D:\dataset\BERT model\uncased_L-12_H-768_A-12

import pickle
import h5py

$ export TRAINED_CLASSIFIER=./bert_output/model.ckpt-[highest checkpoint number you saw]

$ python bert/run_classifier.py \
--task_name=cola \
--do_predict=true \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=./bert_output/model.ckpt-2673 \
--max_seq_length=128 \
--data_dir=./data \
--output_dir=./bert_output/