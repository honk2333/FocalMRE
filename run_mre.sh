DATASET_NAME="MRE"
BERT_NAME='bert-base-uncased'
VIT_NAME="clip-vit-base-patch32"

CUDA_VISIBLE_DEVICES=3 python run.py \
  --model_name="bert" \
  --bert_name=${BERT_NAME} \
  --vit_name=${VIT_NAME} \
  --dataset_name=${DATASET_NAME} \
  --num_epochs=20 \
  --batch_size=8 \
  --lr=1e-5 \
  --warmup_ratio=0.08 \
  --eval_begin_epoch=1 \
  --seed=123 \
  --do_train \
  --max_seq=128 \
  --aux_size=224 \
  --rcnn_size=224 \
  --aux_num 4 \
  --rcnn_num 3 \
  --save_path="ckpt"
