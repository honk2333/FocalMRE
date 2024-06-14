import random
import os
import torch
import json
import ast
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, CLIPTokenizer
# from torchvision import transforms
import logging
from transformers.models.clip import CLIPProcessor

logger = logging.getLogger(__name__)


class MMREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, ):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})
        self.clip_processor = clip_processor

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        aux_imgs = {}
        rcnn_imgs = {}
        aux_imgs.update(torch.load(self.data_path[mode + "_auximgs"]))
        rcnn_imgs.update(torch.load(self.data_path[mode + '_img2crop']))

        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class MMREDataset(Dataset):
    def __init__(self, processor, args, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64, mode="train") -> None:
        self.processor = processor
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data/'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)

        self.aux_box_dict = {}
        self.rcnn_box_dict = {}
        self.aux_box_dict.update(torch.load('data/' + mode + '_vg_box_dict.pth'))
        self.rcnn_box_dict.update(torch.load('data/' + mode + '_detect_box_dict.pth'))

        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor

        self.aux_size = aux_size
        self.rcnn_size = rcnn_size

        self.aux_num = args.aux_num
        self.rcnn_num = args.rcnn_num

    def __len__(self):
        return len(self.data_dict['words'])

    def convert(self, box, dw, dh):
        x = float(box[0]) - float(box[2]) / 2.0
        x *= dw
        w = float(box[2]) * dw
        y = float(box[1]) - float(box[3]) / 2.0
        y *= dh
        h = float(box[3]) * dh
        return x, y, x + w, y + h

    def draw_rect(self, ori_image, box, dw, dh, rectangle_color):
        image = ori_image.copy()
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = self.convert(box, dw, dh)
        draw.rectangle([left, top, right, bottom], outline=rectangle_color, width=3)
        return image


    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
            self.data_dict['heads'][idx], self.data_dict['tails'][idx], self.data_dict['imgids'][idx]

        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list) + 1):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            if i < len(word_list):
                extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)

        re_label = self.re_dict[relation]  # label to id

        # org image
        img_path = os.path.join(self.img_path, imgid)
        image = Image.open(img_path).convert('RGB')

        # 单句子编码
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)

        box_imgs = []
        # detected object img
        aux_num = self.aux_num
        aux_img_paths = []
        if item_id in self.data_dict['aux_imgs']:
            aux_img_paths = self.data_dict['aux_imgs'][item_id]
            aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

        for i in range(min(aux_num, len(aux_img_paths))):
            bbox = list(self.aux_box_dict[aux_img_paths[i].split('/')[-1]])
            width, height = image.size
            # Focal Image
            box_imgs.append(self.clip_processor(images=self.draw_rect(image, bbox[:4], width, height, (255, 0, 0)), return_tensors='pt')['pixel_values'].squeeze())

        # padding
        for i in range(aux_num - len(box_imgs)):
            box_imgs.append(torch.zeros((3, 224, 224)))

        assert len(box_imgs) == aux_num

        rcnn_num = self.rcnn_num
        rcnn_img_paths = []
        imgid = imgid.split(".")[0]
        if imgid in self.data_dict['rcnn_imgs']:
            rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
            rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

        for i in range(min(rcnn_num, len(rcnn_img_paths))):
            bbox = list(self.rcnn_box_dict[rcnn_img_paths[i].split('/')[-1]])
            width, height = image.size
            # Focal Image
            box_imgs.append(self.clip_processor(images=self.draw_rect(image, bbox[:4], width, height, (0, 0, 255)), return_tensors='pt')['pixel_values'].squeeze())

        for i in range(rcnn_num + aux_num - len(box_imgs)):
            box_imgs.append(torch.zeros((3, 224, 224)))

        assert len(box_imgs) == rcnn_num + aux_num

        image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
        box_imgs = torch.stack(box_imgs, dim=0)
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, box_imgs
