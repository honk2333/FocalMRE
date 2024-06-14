import sys

from PIL import Image

sys.path.append("..")

import os
import torch
from torch import nn

import torch.nn.functional as F
from .modeling_unimo import UnimoModel


class REModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args, vision_config, text_config, clip_model_dict, bert_model_dict):
        super(REModel, self).__init__()
        self.args = args
        # print(vision_config)
        # print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        # for re
        vision_config.device = args.device
        self.model = UnimoModel(vision_config, text_config, args.aux_num, args.rcnn_num)

        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
            (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))

        self.model.load_state_dict(model_dict)

        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.classifier = nn.Linear(self.text_config.hidden_size * 4, num_labels)

        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            box_imgs=None,
    ):
        bsz = input_ids.size(0)
        head_mask = (input_ids == self.head_start)
        tail_mask = (input_ids == self.tail_start)
        head_idxs = head_mask.nonzero()[:, 1]
        tail_idxs = tail_mask.nonzero()[:, 1]

        output, output2 = self.model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     box_imgs=box_imgs,
                                     return_dict=True, )
        last_hidden_state = output.last_hidden_state
        bsz, sequence_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, hidden_size * 4)  # batch, 3*hidden
        for i in range(bsz):
            head_idx = head_idxs[i].item()
            tail_idx = tail_idxs[i].item()
            head_hidden = output.last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = output.last_hidden_state[i, tail_idx, :].squeeze()
            head_hidden2 = output2.last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden2 = output2.last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden, head_hidden2, tail_hidden2], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits
