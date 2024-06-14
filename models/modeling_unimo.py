from typing import Any, Optional, Tuple
import math

import torch
from torch import nn, Tensor, device
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from .modeling_bert import *
from .modeling_clip import *
from .selective_attention import SelectiveAttention
from .gumbel import gumbel_softmax_sampling


class UnimoModel(nn.Module):
    def __init__(self, vision_config, text_config, aux_num, rcnn_num):
        super(UnimoModel, self).__init__()

        self.projection_dim = 768
        self.text_config = text_config
        self.vision_config = vision_config

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = BertModel(text_config)
        self.vision_model = CLIPVisionModel(vision_config)

        self.v_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.t_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # selective attention
        self.aux_num = aux_num
        self.rcnn_num = rcnn_num
        self.image_num = self.aux_num + self.rcnn_num
        self.num_heads = 1
        self.selective_attns = nn.ModuleList([])
        self.selective_attns.extend([SelectiveAttention(qdim=self.projection_dim, kdim=self.projection_dim, vdim=self.projection_dim,
                                                        attn_dim=self.num_heads * self.projection_dim,
                                                        intermediate_dim=self.num_heads * self.projection_dim,
                                                        output_dim=self.projection_dim,
                                                        num_heads=self.num_heads, attn_drop=0.1) for i in range(self.image_num)])
        # gate
        self.gate_denses = nn.ModuleList([])
        self.gate_denses.extend([nn.Linear(2 * text_config.hidden_size, text_config.hidden_size) for i in range(self.image_num)])
        self.gate_out = nn.Linear(2 * text_config.hidden_size, text_config.hidden_size)
        self.gate_out1 = nn.Linear(2 * text_config.hidden_size, text_config.hidden_size)
        self.gate_out2 = nn.Linear(2 * text_config.hidden_size, text_config.hidden_size)
        self.gate_final = nn.Linear(2 * text_config.hidden_size, text_config.hidden_size)
        self.device = vision_config.device

        # self.t_layer_norm = nn.LayerNorm(self.projection_dim)
        # self.v_layer_norm = nn.LayerNorm(self.projection_dim, 1e-5, True)
        self.t_dropout = nn.Dropout(0.1)
        self.v_dropout = nn.Dropout(0.1)

        self.weights1 = nn.Parameter(torch.ones(1, self.aux_num, 1, 1))
        self.weights2 = nn.Parameter(torch.ones(1, self.rcnn_num, 1, 1))

    def fuse(self, a, b, func):
        merge = torch.cat([a, b], dim=-1)
        gate = torch.sigmoid(func(merge))
        res = (1 - gate) * b + gate * a
        # 计算 gate 的均值
        gate_mean = torch.mean(gate)
        return res

    def fuse_img_feat(self, text, idx, image):
        image = self.v_dropout(image)
        text = self.t_dropout(text)
        output, _map = self.selective_attns[idx](query=text, key=image, value=image, )  # t, b, c
        res = self.fuse(output, text, self.gate_denses[idx])
        return res
        # # 不使用Gate
        # return output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            box_imgs=None,
            obj_imgs=None,
            img_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = text_outputs[0]

        box_imgs_embeds = []  # bsz, image_num, patch_num, hidden_size
        bsz = box_imgs.shape[0]
        assert self.image_num == box_imgs.shape[1]
        for i in range(box_imgs.shape[1]):
            vision_outputs = self.vision_model(
                pixel_values=box_imgs[:, i, :, :, :],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs[0]
            box_imgs_embeds.append(image_embeds)
        xs = []
        for idx in range(0, self.aux_num):
            img = box_imgs_embeds[idx].transpose(0, 1)
            xs.append(self.fuse_img_feat(text_embeds.transpose(0, 1), idx, img).transpose(0, 1))
        xs = torch.stack(xs, dim=1)
        expanded_weights1 = self.weights1.repeat(bsz, 1, xs.size(2), 1)
        sequence_output1 = torch.sum(expanded_weights1 * xs, dim=1)

        xs = []
        for idx in range(self.aux_num, self.image_num):
            img = box_imgs_embeds[idx].transpose(0, 1)
            xs.append(self.fuse_img_feat(text_embeds.transpose(0, 1), idx, img).transpose(0, 1))
        xs = torch.stack(xs, dim=1)
        expanded_weights2 = self.weights2.repeat(bsz, 1, xs.size(2), 1)
        sequence_output2 = torch.sum(expanded_weights2 * xs, dim=1)

        if not return_dict:
            return sequence_output1, sequence_output2

        return BaseModelOutputWithPooling(
            pooler_output=sequence_output1[0],
            last_hidden_state=sequence_output1
        ), BaseModelOutputWithPooling(
            pooler_output=sequence_output2[0],
            last_hidden_state=sequence_output2
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
            self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings
