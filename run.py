import os
import argparse
import logging
import sys

sys.path.append("..")

import torch
import numpy as np
import random
# from torchvision import transforms
from torch.utils.data import DataLoader
from models.re_model import REModel

from transformers import CLIPModel
from transformers.models.clip import CLIPProcessor
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from transformers import BertConfig, CLIPConfig, BertModel, AutoModel, AutoConfig
from processor.dataset import MMREProcessor, MMREDataset
from modules.train import BertTrainer

import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    )
logger = logging.getLogger(__name__)


MODEL_CLASS = {
    'bert': (MMREProcessor, MMREDataset),
}

DATA_PATH = {
    'MRE': {'train': 'data/txt/ours_train.txt',
            'dev': 'data/txt/ours_val.txt',
            'test': 'data/txt/ours_test.txt',
            'train_auximgs': 'data/txt/mre_train_dict.pth',  # {data_id : object_crop_img_path}
            'dev_auximgs': 'data/txt/mre_dev_dict.pth',
            'test_auximgs': 'data/txt/mre_test_dict.pth',
            'train_img2crop': 'data/img_detect/train/train_img2crop.pth',
            'dev_img2crop': 'data/img_detect/val/val_img2crop.pth',
            'test_img2crop': 'data/img_detect/test/test_img2crop.pth'
            }
}

IMG_PATH = {
    'MRE': {'train': 'data/img_org/train/',
            'dev': 'data/img_org/val/',
            'test': 'data/img_org/test'}}

AUX_PATH = {
    'MRE': {
        'train': 'data/img_vg/train/crops',
        'dev': 'data/img_vg/val/crops',
        'test': 'data/img_vg/test/crops'
    }
}
re_path = 'data/ours_rel2id.json'


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert', type=str, help="The name of bert.")
    parser.add_argument('--vit_name', default='vit', type=str, help="The name of vit.")
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base', type=str,
                        help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--lr', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--aux_size', default=128, type=int, help="aux size")
    parser.add_argument('--rcnn_size', default=64, type=int, help="rcnn size")
    parser.add_argument('--aux_num', default=3, type=int)
    parser.add_argument('--rcnn_num', default=3, type=int)

    args = parser.parse_args()
    print(args)
    logger.info(args)

    data_path, img_path, ent_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    data_process, dataset_class = MODEL_CLASS[args.model_name]

    set_seed(args.seed)  # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        args.save_path = os.path.join(args.save_path,)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)

    # logdir = "logs/" + args.model_name + "_" + args.dataset_name + "_" + str(args.batch_size) + "_" + str(
    #     args.lr) + args.notes

    # writer = SummaryWriter(logdir="/home/nfs03/wanghk/ckpt/logs/" + formatted_datetime)
    writer = None
    if args.do_train:
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        processor = data_process(data_path, re_path, args.bert_name, clip_processor=clip_processor, )

        train_dataset = dataset_class(processor, args, img_path, ent_path, args.max_seq, args.aux_size, args.rcnn_size, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valid_dataset = dataset_class(processor, args, img_path, ent_path, args.max_seq, args.aux_size, args.rcnn_size, mode='dev')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset = dataset_class(processor, args, img_path, ent_path, args.max_seq, args.aux_size, args.rcnn_size, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer

        # train
        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model
        bert_model = AutoModel.from_pretrained(args.bert_name)

        config = CLIPConfig.from_pretrained(args.vit_name)
        vision_config = config.vision_config
        text_config = AutoConfig.from_pretrained(args.bert_name)

        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert_model.state_dict()

        model = REModel(num_labels, tokenizer, args, vision_config, text_config, clip_model_dict, text_model_dict)
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        trainer = BertTrainer(train_data=train_dataloader, dev_data=valid_dataloader, test_data=test_dataloader,
                              re_dict=re_dict, model=model, args=args, logger=logger, writer=writer)
        trainer.train()
        torch.cuda.empty_cache()
    if args.do_test:
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        processor = data_process(data_path, re_path, args.bert_name, clip_processor=clip_processor, )
        test_dataset = dataset_class(processor, args, img_path, ent_path, args.max_seq, args.aux_size, args.rcnn_size, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer

        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model
        bert_model = AutoModel.from_pretrained(args.bert_name)

        config = CLIPConfig.from_pretrained(args.vit_name)
        vision_config = config.vision_config
        text_config = AutoConfig.from_pretrained(args.bert_name)

        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert_model.state_dict()

        model = REModel(num_labels, tokenizer, args, vision_config, text_config, clip_model_dict, text_model_dict)
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        trainer = BertTrainer(train_data=None, dev_data=None, test_data=test_dataloader,
                              re_dict=re_dict, model=model, args=args, logger=logger, writer=writer)
        trainer.test(-1)

if __name__ == "__main__":
    main()
