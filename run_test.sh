DATASET_NAME="MRE"
BERT_NAME='bert-base-uncased'
VIT_NAME="clip-vit-base-patch32"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --model_name="bert" \
  --bert_name=${BERT_NAME} \
  --vit_name=${VIT_NAME} \
  --dataset_name=${DATASET_NAME} \
  --batch_size=1 \
  --seed=1234 \
  --do_test \
  --max_seq=128 \
  --aux_size=224 \
  --rcnn_size=224 \
  --aux_num 4 \
  --rcnn_num 3 \
  --load_path="ckpt/best_model.pth"
