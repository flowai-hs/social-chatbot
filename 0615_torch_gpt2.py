# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""


import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import gelu

from transformers.configuration_gpt2 import GPT2Config
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer


logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin",
}


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
  """ Load tf checkpoints in a pytorch model
  """
  try:
    import re
    import tensorflow as tf
  except ImportError:
    logger.error(
        "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        "https://www.tensorflow.org/install/ for installation instructions."
    )
    raise
  tf_path = os.path.abspath(gpt2_checkpoint_path)
  logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
  # Load weights from TF model
  init_vars = tf.train.list_variables(tf_path)
  names = []
  arrays = []
  for name, shape in init_vars:
    logger.info("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    arrays.append(array.squeeze())

  for name, array in zip(names, arrays):
    name = name[6:]  # skip "model/"
    name = name.split("/")
    pointer = model
    for m_name in name:
      if re.fullmatch(r"[A-Za-z]+\d+", m_name):
        scope_names = re.split(r"(\d+)", m_name)
      else:
        scope_names = [m_name]
      if scope_names[0] == "w" or scope_names[0] == "g":
        pointer = getattr(pointer, "weight")
      elif scope_names[0] == "b":
        pointer = getattr(pointer, "bias")
      elif scope_names[0] == "wpe" or scope_names[0] == "wte":
        pointer = getattr(pointer, scope_names[0])
        pointer = getattr(pointer, "weight")
      else:
        pointer = getattr(pointer, scope_names[0])
      if len(scope_names) >= 2:
        num = int(scope_names[1])
        pointer = pointer[num]
    try:
      assert pointer.shape == array.shape
    except AssertionError as e:
      e.args += (pointer.shape, array.shape)
      raise
    logger.info("Initialize PyTorch weight {}".format(name))
    pointer.data = torch.from_numpy(array)
  return model




"""
-------------------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
"""
# Self-attention layer 클래스
class Attention(nn.Module):
  
  # 기본 속성
  def __init__(self, nx, n_ctx, config, scale=False):
    super().__init__()
    self.output_attentions = config.output_attentions

    n_state = nx  # in Attention : n_state=768 (nx=n_embd)
    assert n_state % config.n_head == 0  # 768/12=64, 나머지 0
    self.register_buffer("bias", torch.tril(
        torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)) # n_ctx크기로 텐서를 바꾸고, 1로 채운다음 앞에서부터 tril해간다.
    self.n_head = config.n_head # head 개수
    self.split_size = n_state # 스플릿사이즈는 n_state
    self.scale = scale # scale=False

    self.c_attn = Conv1D(n_state * 3, nx) # 어텐션 학습가능한 매트릭스1 : 사이즈 768 * 3(q, k, v 3개), 768
    self.c_proj = Conv1D(n_state, nx)     # 어텐션 학습가능한 매트릭스2 : 사이즈 768, 768
    self.attn_dropout = nn.Dropout(config.attn_pdrop)   # 어텐션 부분 드랍아웃 비율
    self.resid_dropout = nn.Dropout(config.resid_pdrop) # 레지듀얼 커넥션 드랍아웃 비율
    self.pruned_heads = set()  # prune heads는 집합
  
  #prune heads 함수 -> 나중에 필요할 때 사용할 수 있음
  def prune_heads(self, heads):
    if len(heads) == 0:
      return
    mask = torch.ones(self.n_head, self.split_size // self.n_head)
    # Convert to set and emove already pruned heads
    heads = set(heads) - self.pruned_heads
    for head in heads:
      # Compute how many pruned heads are before the head and move the index accordingly
      head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
      mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    index_attn = torch.cat(
        [index, index + self.split_size, index + (2 * self.split_size)])

    # Prune conv1d layers
    self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
    self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

    # Update hyper params
    self.split_size = (self.split_size // self.n_head) * \
        (self.n_head - len(heads))
    self.n_head = self.n_head - len(heads)
    self.pruned_heads = self.pruned_heads.union(heads)
  
  # Self-attention layer 함수
  def _attn(self, q, k, v, attention_mask=None, head_mask=None):
    w = torch.matmul(q, k)   # 쿼리와 키 내적
    if self.scale:
      w = w / math.sqrt(v.size(-1))
    nd, ns = w.size(-2), w.size(-1) # 64, 64
    b = self.bias[:, :, ns - nd: ns, :ns]
    w = w * b - 1e4 * (1 - b)

    if attention_mask is not None:
      # attention 마스크 적용
      w = w + attention_mask

    w = nn.Softmax(dim=-1)(w) # 소프트맥스 계산하여 스코어 출력
    w = self.attn_dropout(w)  # 드랍아웃

    # Mask heads를 할 경우
    if head_mask is not None:
      w = w * head_mask

    outputs = [torch.matmul(w, v)]  # w(스코어)와 밸류를 곱합
    if self.output_attentions:
      outputs.append(w)
    return outputs                  # 아웃풋 출력

  # z1, ..., z12 를 하나의 벡터로 만듬
  def merge_heads(self, x):
    x = x.permute(0, 2, 1, 3).contiguous()  # x를 batch, head, seq_lenth, head_feature 
    new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),) # x shape 변환 -> [batch, head, seq_lenth*head_feature]
    return x.view(*new_x_shape) 
  
  # head를 분리
  def split_heads(self, x, k=False):
    # x.size(-1)은 768이고 이를 12로 나눔
    new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)

    # e.g 12(헤드개수) * 64 shape
    x = x.view(*new_x_shape)
    if k:
      return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
    else:
      return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
  
  # 포워드 함수
  #
  def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
    x = self.c_attn(x) # x는 학습가능한 매트릭스를 통과한 값이고
    query, key, value = x.split(self.split_size, dim=2) # 쿼리, 키, 밸류를 나눈다.
    query = self.split_heads(query) # 쿼리 스플릿
    key = self.split_heads(key, k=True) # 키 스플릿
    value = self.split_heads(value) # 밸류 스플릿
    if layer_past is not None:
      # transpose back cf below
      past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
      key = torch.cat((past_key, key), dim=-1)
      value = torch.cat((past_value, value), dim=-2)
    # transpose to have same shapes for stacking
    present = torch.stack((key.transpose(-2, -1), value))

    attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
    a = attn_outputs[0]

    a = self.merge_heads(a)
    a = self.c_proj(a)
    a = self.resid_dropout(a)

    outputs = [a, present] + attn_outputs[1:]
    return outputs  # a, present, (attentions)

"""
-------------------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
"""

class MLP(nn.Module):
  def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
    super().__init__()
    nx = config.n_embd
    self.c_fc = Conv1D(n_state, nx)
    self.c_proj = Conv1D(nx, n_state)
    self.act = gelu
    self.dropout = nn.Dropout(config.resid_pdrop)

  def forward(self, x):
    h = self.act(self.c_fc(x))
    h2 = self.c_proj(h)
    return self.dropout(h2)


class Block(nn.Module):
  def __init__(self, n_ctx, config, scale=False):
    super().__init__()
    nx = config.n_embd
    self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
    self.attn = Attention(nx, n_ctx, config, scale)
    self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
    self.mlp = MLP(4 * nx, config)

  def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
    output_attn = self.attn(
        self.ln_1(x), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask
    )
    a = output_attn[0]  # output_attn: a, present, (attentions)

    x = x + a
    m = self.mlp(self.ln_2(x))
    x = x + m

    outputs = [x] + output_attn[1:]
    return outputs  # x, present, (attentions)






"""
-------------------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
"""
class GPT2PreTrainedModel(PreTrainedModel):
  """ An abstract class to handle weights initialization and
      a simple interface for downloading and loading pretrained models.
  """

  config_class = GPT2Config
  pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
  load_tf_weights = load_tf_weights_in_gpt2
  base_model_prefix = "transformer"

  def __init__(self, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

  def _init_weights(self, module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)


class GPT2Model(GPT2PreTrainedModel):

  def __init__(self, config):
    super().__init__(config)
    self.output_hidden_states = config.output_hidden_states
    self.output_attentions = config.output_attentions
    self.output_past = config.output_past # 미리 계산된 값 사용
    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
    self.wpe = nn.Embedding(config.n_positions, config.n_embd)
    self.drop = nn.Dropout(config.embd_pdrop)
    self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True)
                            for _ in range(config.n_layer)])
    self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    self.init_weights()

  def get_input_embeddings(self):
    return self.wte

  def set_input_embeddings(self, new_embeddings):
    self.wte = new_embeddings

  def _prune_heads(self, heads_to_prune):
    """ Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
    """
    for layer, heads in heads_to_prune.items():
      self.h[layer].attn.prune_heads(heads)

  def forward(
      self,
      input_ids=None,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
  ):
    if input_ids is not None and inputs_embeds is not None:
      raise ValueError(
          "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      # input_shape은 input_ids의 사이즈
      input_shape = input_ids.size()
      input_ids = input_ids.view(-1, input_shape[-1])
      batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
      batch_size = inputs_embeds.shape[0]
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    if token_type_ids is not None:
      token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
      position_ids = position_ids.view(-1, input_shape[-1])

    if past is None:
      past_length = 0
      past = [None] * len(self.h)
    else:
      past_length = past[0][0].size(-2)
    if position_ids is None:
      device = input_ids.device if input_ids is not None else inputs_embeds.device
      position_ids = torch.arange(
          past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
      position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
      # batch size, 1, 1, to_seq_length(여기서는 1024)
      attention_mask = attention_mask.view(batch_size, -1)# 1024를 배치 사이즈만큼 잘라서 쌓아올림
      attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # 차원 추가
      
      # 마스킹된 값을 0에 가까운 무의미한 값을 만들기위한 계산과정
      attention_mask = attention_mask.to(dtype=next(
          self.parameters()).dtype)  # fp16 compatibility
      attention_mask = (1.0 - attention_mask) * -10000.0

    # head mask가 필요한 경우 사용합니다.
    if head_mask is not None:
      if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(
            0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
      elif head_mask.dim() == 2:
        head_mask = (
            head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # We can specify head_mask for each layer
      head_mask = head_mask.to(
          dtype=next(self.parameters()).dtype
      )  # switch to fload if need + fp16 compatibility
    else:
      head_mask = [None] * self.config.n_layer

    # input_embeds가 없을 때 input_ids로 wte 임베딩한 값을 담습니다.
    if inputs_embeds is None:
      inputs_embeds = self.wte(input_ids) # self.wte = nn.Embedding(config.vocab_size, config.n_embd)
    # position_embeds는 position_ids로 wpe 임베딩
    position_embeds = self.wpe(position_ids)
    # token_type_ids가 있으면 token_type_ids적용하고
    if token_type_ids is not None:
      token_type_embeds = self.wte(token_type_ids)
    # token_type_ids가 없으면 0으로 처리
    else:
      token_type_embeds = 0
    # hidden_state에 inputs_embeds,position_embeds,token_type_embeds를 더한 값을 담습니다.
    hidden_states = inputs_embeds + position_embeds + token_type_embeds
    hidden_states = self.drop(hidden_states)

    # output_shape에 input_shape와 hidden_states의 마지막 차원 사이즈 값을 더해 담습니다. (튜플 1개 요소만 있음)
    output_shape = input_shape + (hidden_states.size(-1),)

    presents = ()
    all_attentions = []
    all_hidden_states = ()
    # h와 past를 각 요소 순서대로 묶고 이를 인덱스로 반복하여 튜플 형식의 자료를 만들고 인덱스 번호 i와 block, layer_past를 튜플 형식으로 담습니다. 즉, h는 block, past는 layer_past
    for i, (block, layer_past) in enumerate(zip(self.h, past)):
      if self.output_hidden_states:
        all_hidden_states = all_hidden_states + \
            (hidden_states.view(*output_shape),)

      outputs = block(
          hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[
              i]
      )

      hidden_states, present = outputs[:2]
      if self.output_past:
        presents = presents + (present,)

      if self.output_attentions:
        all_attentions.append(outputs[2])
    
    # 노말라이제이션, 여기서는 gelu
    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if self.output_hidden_states:  # False
      all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_past:  # True
      outputs = outputs + (presents,)
    if self.output_hidden_states:  # False
      outputs = outputs + (all_hidden_states,)
    if self.output_attentions:  # False
      # let the number of heads free (-1) so we can extract attention even after head pruning
      attention_output_shape = input_shape[:-
                                           1] + (-1,) + all_attentions[0].shape[-2:]
      all_attentions = tuple(t.view(*attention_output_shape)
                             for t in all_attentions)
      outputs = outputs + (all_attentions,)
    # last hidden state, (presents), (all hidden_states), (attentions)
    return outputs


class GPT2LMHeadModel(GPT2PreTrainedModel):

  def __init__(self, config):
    super().__init__(config)
    self.transformer = GPT2Model(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # nn.Linear(768, 50000)
    # 만약 감정임베딩을 이전 layer와 concat한다면 여기에 embedding_layer 추가
    self.init_weights()

  def get_output_embeddings(self):
    return self.lm_head # 여기서는 nn.Linear(768, 50000, bias=False)

  def prepare_inputs_for_generation(self, input_ids, **kwargs):
    # only last token for inputs_ids if past is defined in kwargs
    if "past" in kwargs and kwargs["past"]:
      # last token
      input_ids = input_ids[:, -1].unsqueeze(-1)

    inputs = {"input_ids": input_ids}
    inputs.update(kwargs)
    return inputs

  def forward(
      self,
      input_ids=None,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      # 만약 감정임베딩 추가한다면 여기에 emotion 추가
  ):
    transformer_outputs = self.transformer(
        input_ids,
        past=past,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )
    
    # create emotion_embedding 
    hidden_states = transformer_outputs[0] # + emotion_embedding
    
    lm_logits = self.lm_head(hidden_states)

    # 라벨이 있으면 로스 값 계산
    if labels is not None:
      # Shift so that tokens < n predict n
      # shift_logits = lm_logits[..., :-1, :].contiguous()
      # shift_labels = labels[..., 1:].contiguous()
      shift_logits = lm_logits
      shift_labels = labels
      # Flatten the tokens
      # reduction='none' to calculate ppl
      loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
      loss1 = loss_fct(
          shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

      # ppl
      loss1 = loss1.view(shift_labels.size(0), shift_labels.size(1))
      label_size = torch.sum(shift_labels != -1, dim=1).type(loss1.type())
      loss = torch.sum(loss1) / torch.sum(label_size)
      ppl = torch.exp(torch.mean(
          torch.sum(loss1, dim=1).float() / label_size.float()))
      return loss, ppl
    # (loss, ppl), lm_logits, presents, (all hidden_states), (attentions)
    return (lm_logits,) + transformer_outputs[1:]

"""
-------------------------------------------------------------------------
-------------------------------------------------------------------------
-------------------------------------------------------------------------
"""



class GPT2DoubleHeadsModel(GPT2PreTrainedModel):

  def __init__(self, config):
    super().__init__(config)
    config.num_labels = 1
    self.transformer = GPT2Model(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.multiple_choice_head = SequenceSummary(config)

    self.init_weights()

  def get_output_embeddings(self):
    return self.lm_head

  def forward(
      self,
      input_ids=None,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      mc_token_ids=None,
      lm_labels=None,
      mc_labels=None,
  ):
    transformer_outputs = self.transformer(
        input_ids,
        past=past,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )

    hidden_states = transformer_outputs[0]

    lm_logits = self.lm_head(hidden_states)
    mc_logits = self.multiple_choice_head(
        hidden_states, mc_token_ids).squeeze(-1)

    outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
    if mc_labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
                      mc_labels.view(-1))
      outputs = (loss,) + outputs
    if lm_labels is not None:
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = lm_labels[..., 1:].contiguous()
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(
          shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      outputs = (loss,) + outputs

    # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)
    return outputs
