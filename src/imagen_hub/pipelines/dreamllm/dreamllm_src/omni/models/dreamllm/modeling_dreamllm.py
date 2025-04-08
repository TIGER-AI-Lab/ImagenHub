from typing import Union
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch DreamLLM model."""
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import torch
import torch.backends.cuda
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import ModelOutput

from ...config.instantiate import deep_instantiate
from ...conversation.multimodal import MultimodalContent
from .configuration_dreamllm import DreamLLMConfig
from .modeling_plugins import PipelineImageType
from .tokenization_dreamllm import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_DREAM_END_TOKEN,
    DEFAULT_DREAM_START_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IMAGE_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

from ...utils.fsdp_utils import FSDPMixin
from ...utils.import_utils import _flash_attn_version, is_flash_attn_2_available, is_torch_fx_available
from ...utils.loguru import logger
from ...utils.profiler import pretty_format
from ...utils.pytorch_utils import ALL_LAYERNORM_LAYERS

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


# Copy from modeling_llama.py: start
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)


class DreamLLMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DreamLLMRMSNorm)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)) ** (
                self.dim / (self.dim - 2)
            )
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DreamLLMMLP(nn.Module):
    def __init__(self, config: DreamLLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DreamLLMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DreamLLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Union[torch.Tensor, None]= None,
        position_ids: Union[torch.LongTensor, None]= None,
        past_key_value: Union[tuple[torch.Tensor], None]= None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            # HACK: https://github.com/huggingface/transformers/issues/25065
            dtype_min = torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype)
            attn_weights = torch.max(attn_weights, dtype_min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DreamLLMFlashAttention2(DreamLLMAttention):
    """
    DreamLLM flash attention module. This module inherits from `DreamLLMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Union[torch.LongTensor, None]= None,
        position_ids: Union[torch.LongTensor, None]= None,
        past_key_value: Union[tuple[torch.Tensor], None]= None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # DreamLLMFlashAttention2 attention does not support output_attentions

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (RMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        # HACK: During the inference stage, there may be an inconsistency in dtype between loading the model with fp32 and o_proj.
        # if attn_output.dtype != self.o_proj.weight.dtype:
        #     attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class DreamLLMDecoderLayer(nn.Module):
    def __init__(self, config: DreamLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            DreamLLMAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else DreamLLMFlashAttention2(config=config)
        )
        self.mlp = DreamLLMMLP(config)
        self.input_layernorm = DreamLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Union[torch.Tensor, None]= None,
        position_ids: Union[torch.LongTensor, None]= None,
        past_key_value: Union[tuple[torch.Tensor], None]= None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) :
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or
                `(batch_size, 1, query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DreamLLMPreTrainedModel(PreTrainedModel, FSDPMixin):
    config_class = DreamLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DreamLLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    
    # Flash Attention 2 support
    _supports_flash_attn_2 = True

    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = []

    def init_plugin_modules(self):
        pass

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls, config, torch_dtype: Union[torch.dtype, None]= None, device_map = None
    ) -> PretrainedConfig:
        """
        If you don't know about Flash Attention, check out the official repository of flash attention:
        https://github.com/Dao-AILab/flash-attention
        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this
        specific section of the documentation to learn more about it:
        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models
        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in
        half precision and not ran on CPU.
        If all checks pass, the method will create an attribute in the config `_flash_attn_2_enabled` so that the model
        can initialize the correct attention module
        """
        if not cls._supports_flash_attn_2:
            raise ValueError(
                "The current architecture does not support Flash Attention 2.0. Please open an issue on GitHub to "
                "request support for this architecture: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_2_available():
            raise ImportError(
                "Flash Attention 2 is not available. Please refer to the documentation of https://github.com/Dao-AILab/flash-attention for"
                " installing it. Make sure to have at least the version 2.1.0"
            )
        else:
            is_flash_greater_than_2 = is_flash_attn_2_available(">=", "2.1.0")
            if not is_flash_greater_than_2:
                raise ValueError(
                    f"You need flash_attn package version to be greater or equal than 2.1. Make sure to have that version installed - detected version {_flash_attn_version}"
                )

        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)

        if _is_bettertransformer:
            raise ValueError(
                "Flash Attention 2 and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()"
            )

        if torch_dtype is None:
            logger.warning(
                "You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
            logger.warning(
                "Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes. "
                "No dtype was provided, you should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator."
            )

        if device_map is None:
            if torch.cuda.is_available():
                logger.warning(
                    "You are attempting to use Flash Attention 2.0 with a model initialized on CPU. Make sure to move the model to GPU"
                    " after initializing it on CPU with `model.to('cuda')`."
                )
            else:
                raise ValueError(
                    "You are attempting to use Flash Attention 2.0 with a model initialized on CPU and with no GPU available. "
                    "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                    "or initialising the model on CPU and then moving it to GPU."
                )
        elif (
            device_map is not None
            and isinstance(device_map, dict)
            and ("cpu" in device_map.values() or "disk" in device_map.values())
        ):
            raise ValueError(
                "You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to "
                "initialise the model on a GPU by passing a device_map that contains only GPU devices as keys."
            )
        config._flash_attn_2_enabled = True
        return config


# Copy from modeling_llama.py: end


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Union[tuple[tuple[torch.FloatTensor]], None]= None
    hidden_states: Union[tuple[torch.FloatTensor], None]= None
    attentions: Union[tuple[torch.FloatTensor], None]= None
    additional_log_info: Optional[Dict[str, float]] = None


class DreamLLMModel(DreamLLMPreTrainedModel):
    """DreamLLM Model (https://dreamllm.github.io/)
    """
    def __init__(self, config: DreamLLMConfig):
        super().__init__(config)
        if getattr(self.config, "_flash_attn_2_enabled", False):
            logger.info("DreamLLMModel is using Flash Attention 2")

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DreamLLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = DreamLLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def init_plugin_modules(self):
        for name, init_kwargs in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "embedding":
                logger.info(f"Initialized embedding `{name}` with kwargs:\n{pretty_format(init_kwargs)}")
                setattr(self, name, deep_instantiate(init_kwargs).to(self.device, dtype=self.dtype))

                keys_to_ignore = list(getattr(self, name).state_dict().keys())
                keys_to_ignore = [f"model.{name}.{key}" for key in keys_to_ignore]
                self._keys_to_ignore_on_save.extend(keys_to_ignore)
                logger.info(f"Added the prefix keys of `model.{name}` to the list of keys to ignore on save.")

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = []
        for name, _ in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "embedding":
                ignored_modules += getattr(self, name).fsdp_ignored_modules()
        return ignored_modules

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Union[torch.Tensor, None]= None,
        position_ids: Union[torch.LongTensor, None]= None,
        past_key_values: Union[list[torch.FloatTensor], None]= None,
        inputs_embeds: Union[torch.FloatTensor, None]= None,
        use_cache: Union[bool, None]= None,
        output_attentions: Union[bool, None]= None,
        output_hidden_states: Union[bool, None]= None,
        return_dict: Union[bool, None]= None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
                `past_key_values`).

                If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
                and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
                information on the default strategy.

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
                have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
                of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        log_attentions = self.config.log_attentions
        log_hidden_states = self.config.log_hidden_states

        if log_attentions:
            output_attentions = True

        if getattr(self.config, "_flash_attn_2_enabled", False):
            output_attentions = False
            logger.warning_once(f"Flash Attention 2.0 is enabled. `output_attentions` is set to `False`.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        additional_log_info = {}

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if log_hidden_states:
                additional_log_info[f"hidden_states_layer{idx - 1}"] = torch.max(torch.abs(hidden_states)).cpu().item()

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if log_attentions:
                    additional_log_info[f"attentions_layer{idx}"] = torch.max(torch.abs(layer_outputs[1])).cpu().item()

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if log_hidden_states:
            additional_log_info[f"hidden_states_layer{len(self.layers) - 1}"] = torch.max(torch.abs(hidden_states)).cpu().item()

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, additional_log_info] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            additional_log_info=additional_log_info,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Union[torch.FloatTensor, None]= None,
        images_dm: Union[torch.FloatTensor, None]= None,
        attention_mask: Union[torch.Tensor, None]= None,
        position_ids: Union[torch.LongTensor, None]= None,
        past_key_values: Union[list[torch.FloatTensor], None]= None,
        inputs_embeds: Union[torch.FloatTensor, None]= None,
        use_cache: Union[bool, None]= None,
        output_attentions: Union[bool, None]= None,
        output_hidden_states: Union[bool, None]= None,
        return_dict: Union[bool, None]= None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        # Not update original embedding parameters
        embed_tokens_backup = getattr(self, "embed_tokens_backup", None)
        if embed_tokens_backup is not None:
            logger.warning_once(f"Only update last num_added_tokens {self.num_added_tokens} of the embedding parameters.")
            with torch.no_grad():
                self.embed_tokens.weight[: -self.num_added_tokens] = embed_tokens_backup[: -self.num_added_tokens].data

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        image_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_IMAGE_START_TOKEN]
        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]

        # for evaluation, avoid extract the clip features repeatly
        if (
            (images is not None)
            and (not self.training)
            and (past_key_values is not None)
            and (input_ids is None or torch.where(input_ids.reshape(-1) == image_start_id)[0].shape[0] == 0)
        ):
            images = None

        # replace diffusion query token
        if images_dm is not None and inputs_embeds is not None:
            query_embedding = self.dream_embedding(inputs_embeds.shape[0])
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_query_embedding in zip(input_ids, inputs_embeds, query_embedding):
                dream_start_pos = torch.where(cur_input_ids == dream_start_id)[0]
                cur_new_input_embeds = cur_input_embeds
                for _dream_start_pos in dream_start_pos:
                    cur_new_input_embeds = torch.cat(
                        [
                            cur_new_input_embeds[: _dream_start_pos + 1],
                            cur_query_embedding,
                            cur_new_input_embeds[_dream_start_pos + self.dream_embedding.embed_len + 1 :],
                        ],
                        dim=0,
                    ).contiguous()
                assert cur_new_input_embeds.shape[0] == cur_input_embeds.shape[0]
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # HACK: not elegent, how can we auto select `clip_vision_embedding`
        image_features = self.clip_vision_embedding(images)

        if inputs_embeds is not None:
            if images is not None:
                # replace image token
                new_input_embeds = []
                cur_image_idx = 0
                for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                    image_start_pos = torch.where(cur_input_ids == image_start_id)[0]
                    cur_new_input_embeds = cur_input_embeds
                    for _image_start_pos in image_start_pos:
                        if cur_image_idx >= image_features.shape[0]:
                            print(
                                input_ids.shape,
                                torch.where(input_ids == image_start_id),
                                image_features.shape,
                                images.shape,
                                _image_start_pos,
                                image_start_pos,
                            )
                        if cur_image_idx >= image_features.shape[0]:
                            break
                        cur_image_features = image_features[cur_image_idx]
                        num_patches = cur_image_features.shape[0]
                        assert _image_start_pos + num_patches + 1 <= cur_input_embeds.shape[0]
                        assert cur_image_idx <= image_features.size(0)

                        cur_image_features = cur_image_features.to(device=cur_input_embeds.device)
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_new_input_embeds[: _image_start_pos + 1],
                                cur_image_features,
                                cur_new_input_embeds[_image_start_pos + num_patches + 1 :],
                            ),
                            dim=0,
                        ).contiguous()
                        cur_image_idx += 1
                    assert cur_new_input_embeds.shape[0] == cur_input_embeds.shape[0]
                    new_input_embeds.append(cur_new_input_embeds)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
            elif self.training:
                # dummy image features for unused parameters bug
                inputs_embeds = inputs_embeds + image_features
        else:
            inputs_embeds = image_features.unsqueeze(0)

        return self._forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # `DreamEmbedding`
    def prepare_dream_queries_with_special_token(self, batch_size: int = 1):
        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
        dream_end_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_END_TOKEN]
        dream_special_ids = [dream_start_id, dream_end_id]
        dream_special_embeds = self.embed_tokens(torch.as_tensor([dream_special_ids], device=self.device))
        dream_queries = torch.cat(
            [dream_special_embeds[..., :1, :], self.dream_embedding(), dream_special_embeds[..., 1:, :]], 1
        )
        return dream_queries.repeat(batch_size, 1, 1).to(self.device)


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Union[torch.FloatTensor, None]= None
    logits: torch.FloatTensor = None
    past_key_values: Union[tuple[tuple[torch.FloatTensor]], None]= None
    hidden_states: Union[tuple[torch.FloatTensor], None]= None
    attentions: Union[tuple[torch.FloatTensor], None]= None
    additional_log_info: Optional[Dict[str, float]] = None


class DreamLLMForCausalMLM(DreamLLMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DreamLLMConfig):
        super().__init__(config)
        self.model = DreamLLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loss_weight_lm = config.loss_weight_lm
        self.loss_weight_vm = config.loss_weight_vm

        # initialize weights and apply final processing
        self.post_init()

    def init_plugin_modules(self):
        # define multimodal encoders and decoders
        self.model.init_plugin_modules()
        for name, init_kwargs in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "head":
                logger.info(f"Initialized head `{name}` with kwargs:\n{pretty_format(init_kwargs)}")
                setattr(self, name, deep_instantiate(init_kwargs).to(self.device, dtype=self.dtype))

                keys_to_ignore = list(getattr(self, name).state_dict().keys())
                keys_to_ignore = [f"{name}.{key}" for key in keys_to_ignore]
                self._keys_to_ignore_on_save.extend(keys_to_ignore)
                logger.info(f"Added the prefix keys of `{name}` to the list of keys to ignore on save.")

    def fsdp_ignored_modules(self) -> list:
        ignored_modules = self.model.fsdp_ignored_modules()
        for name, _ in self.config.plugins_init_kwargs.items():
            if self.config.plugins_type[name] == "head":
                ignored_modules += getattr(self, name).fsdp_ignored_modules()
        return ignored_modules

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        tokenizer= None,
        *model_args,
        config = None,
        cache_dir = None,
        ignore_mismatched_sizes= False,
        force_download=False,
        local_files_only= False,
        token = None,
        revision: str = "main",
        use_safetensors = None,
        reset_plugin_model_name_or_path = False,
        **kwargs,
    ):
        """
        Newly add a `tokenizer` argument to support `from_pretrained` method.
        Ensure the consistency of tokenizer and model vocabulary length.
        """
        assert tokenizer is not None, "tokenizer should not be None"

        resume_download = kwargs.get("resume_download", False)
        proxies = kwargs.get("proxies", None)
        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)
        subfolder = kwargs.get("subfolder", "")

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, _ = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )

        # HACK: add all pretrained plugins path if not specified
        for _, init_kwargs in config.plugins_init_kwargs.items():
            if init_kwargs["pretrained_model_name_or_path"] == "none" :
                # load pre-aligned plugin models instead of random initialization during training
                init_kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        

                # use initial `LlamaConfig` or exported `DreamLLMConfig` behind
        model = super(DreamLLMForCausalMLM, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

        if reset_plugin_model_name_or_path:
            config.reset_plugins_init_kwargs()

        if len(tokenizer) > model.config.vocab_size:
            logger.info(
                f"The tokenizer vocabulary size {len(tokenizer)} is larger than the model vocabulary size {model.config.vocab_size}. "
                f"Resizing token embedding of model..."
            )
            model.resize_token_embeddings(len(tokenizer))
        elif len(tokenizer) < model.config.vocab_size:
            logger.warning(
                f"The tokenizer vocabulary size {len(tokenizer)} is smaller than the model vocabulary size {model.config.vocab_size}. "
                f"Carefully check the configuration to avoid potential issues."
            )

        logger.info(f"Now, the tokenizer and model vocabulary sizes are both {len(tokenizer)}.")
        

        # BUG: parameters of pretrained plugin modules will be influenced by `_init_weights` if put this in `__init__` or before `model.resize_token_embeddings`
        model.init_plugin_modules()

        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Union[torch.FloatTensor, None]= None,
        images_dm: Union[torch.FloatTensor, None]= None,
        attention_mask: Union[torch.Tensor, None]= None,
        position_ids: Union[torch.LongTensor, None]= None,
        past_key_values: Union[list[torch.FloatTensor], None]= None,
        inputs_embeds: Union[torch.FloatTensor, None]= None,
        labels: Union[torch.LongTensor, None]= None,
        use_cache: Union[bool, None]= None,
        output_attentions: Union[bool, None]= None,
        output_hidden_states: Union[bool, None]= None,
        return_dict: Union[bool, None]= None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        if input_ids is not None:
            assert (
                input_ids.shape[1] <= self.config.max_position_embeddings
            ), f"the sequence length should be less than model max length {self.config.max_position_embeddings}"

        # assert images is not None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            images=images,
            images_dm=images_dm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        dream_start_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
        dream_end_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_END_TOKEN]
        dream_patch_id = self.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_IMAGE_PATCH_TOKEN]
        bos_id = self.config.special_tokens2ids_dict[DEFAULT_BOS_TOKEN]
        eos_id = self.config.special_tokens2ids_dict[DEFAULT_EOS_TOKEN]

        # Let's train diffusion!
        vm_loss = 0.0
        if self.training and images_dm is not None:
            # get `encoder_hidden_states`
            encoder_hidden_states = []
            for _idx, cur_hidden_states in enumerate(hidden_states):
                cur_input_ids = input_ids[_idx]
                dream_start_pos = torch.where(
                    cur_input_ids
                    == self.model.config.special_tokens2ids_dict["additional_special_tokens"][DEFAULT_DREAM_START_TOKEN]
                )[0]
                for _dream_start_pos in dream_start_pos:
                    if len(encoder_hidden_states) >= images_dm.shape[0]:
                        break
                    encoder_hidden_states.append(
                        cur_hidden_states[_dream_start_pos + 1 : _dream_start_pos + 1 + self.model.dream_embedding.embed_len]
                    )
                    if encoder_hidden_states[-1].shape[0] != self.model.dream_embedding.embed_len:
                        logger.error(_idx, _dream_start_pos, hidden_states.shape)
            encoder_hidden_states = torch.stack(encoder_hidden_states, 0)

            # get `u_encoder_hidden_states` if use classifier free guidance
            u_encoder_hidden_states = None
            if self.stable_diffusion_head.drop_prob is not None:
                u_outputs = self.model(
                    input_ids=torch.tensor(
                        [
                            [bos_id, dream_start_id]
                            + [dream_patch_id] * self.model.dream_embedding.embed_len
                            + [dream_end_id, eos_id]
                        ]
                    ).to(input_ids.device),
                    attention_mask=torch.ones(1, self.model.dream_embedding.embed_len + 4, device=encoder_hidden_states.device),
                    past_key_values=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                u_encoder_hidden_states = u_outputs[0][:, 2 : 2 + self.model.dream_embedding.embed_len, :]
                u_encoder_hidden_states = u_encoder_hidden_states.repeat(encoder_hidden_states.shape[0], 1, 1)

            vm_loss = self.stable_diffusion_head(images_dm, encoder_hidden_states, u_encoder_hidden_states)

        elif self.training:
            # HACK: dummy forward to avoid `find_unused_parameters` error
            vm_loss = self.stable_diffusion_head(images_dm, None, None, self.model.dream_embedding())

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        lm_loss = 0.0
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            valid_labels =(shift_labels !=-100).bool()
            if valid_labels.sum() > 0:
                lm_loss = (loss_fct(shift_logits, shift_labels) * valid_labels).sum() / valid_labels.sum()
            else:
                lm_loss = loss_fct(shift_logits, shift_labels).mean()

        if self.config.loss_scale_schedule == "l1_norm":
            loss_scale = self.loss_weight_lm + self.loss_weight_vm
        elif self.config.loss_scale_schedule == "l2_norm":
            loss_scale = math.sqrt(self.loss_weight_lm**2 + self.loss_weight_vm**2)
        else:
            loss_scale = 1

        if self.training and images is not None and np.isnan(lm_loss.cpu().item()):
            logger.warning("lm_loss is NaN!")
            loss = vm_loss * self.loss_weight_vm + lm_loss * 0.0
        elif self.training and images_dm is not None and np.isnan(vm_loss.cpu().item()):
            logger.warning("vm_loss is NaN!")
            loss = vm_loss * 0.0 + lm_loss * self.loss_weight_lm
        else:
            # For comprehension_only or creation_only, set another loss weight to 0
            loss = vm_loss * self.loss_weight_vm + lm_loss * self.loss_weight_lm
        loss = loss / loss_scale

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        additional_log_info = {
            "lm_loss": lm_loss.cpu().item() if not isinstance(lm_loss, float) else lm_loss,
            "vm_loss": vm_loss.cpu().item() if not isinstance(vm_loss, float) else vm_loss,
        }
        additional_log_info.update(outputs.additional_log_info)
        if len(additional_log_info) == 0:
            additional_log_info = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            additional_log_info=additional_log_info,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.pop("images", None),
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),)
        return reordered_past

    # Diffusion utility functions
    def check_inputs(
        self,
        tokenizer,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                "The `tokenizer` argument should be a `PreTrainedTokenizerBase` instance. Make sure to pass the tokenizer as `tokenizer=your_tokenizer`."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, MultimodalContent) and not isinstance(prompt, list)
        ):
            raise ValueError(f"`prompt` has to be of type `str`, `MultimodalContent` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_prompt_embeds(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt,
        device: Union[str, torch.device],
    ):
        if isinstance(prompt, str) or isinstance(prompt, MultimodalContent):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        if isinstance(prompt, str) or isinstance(prompt, list):
            text_inputs = tokenizer(prompt, padding=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(device)
            text_input_attention_mask = text_inputs.attention_mask.to(device)

            images_input_pt = None

        elif isinstance(prompt, MultimodalContent):
            # FIXME: noqa, not debug yet
            image_placeholder_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            image_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_START_TOKEN)
            image_patch_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
            image_end_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_END_TOKEN)

            text_inputs = tokenizer(prompt.text)
            text_input_ids = text_inputs.input_ids

            text_input_ids_start = torch.where(torch.as_tensor(text_input_ids) == image_placeholder_id)[0].tolist()
            text_input_ids_start = [-1] + text_input_ids_start
            text_input_ids_end = text_input_ids_start[1:] + [len(text_input_ids)]

            image_patch_ids = [image_start_id] + [image_patch_id] * 256 + [image_end_id]

            text_input_ids_list = []
            for start_index, end_index in zip(text_input_ids_start, text_input_ids_end):
                text_input_ids_list.append(text_input_ids[start_index + 1 : end_index])

            text_input_ids = text_input_ids_list[0]

            for i in range(1, len(text_input_ids_list)):
                text_input_ids = text_input_ids + image_patch_ids + text_input_ids_list[i]

            text_input_ids = torch.as_tensor([text_input_ids], device=device)
            text_input_attention_mask = torch.ones(batch_size, text_input_ids.shape[1], device=device)

            images_input_pt = self.model.clip_vision_embedding.processor.preprocess(
                prompt.mm_content_list[0],
                return_tensors="pt",
            )["pixel_values"]
            images_input_pt = images_input_pt.to(device)

        text_out = self.forward(
            input_ids=text_input_ids,
            images=images_input_pt,
            attention_mask=text_input_attention_mask,
            use_cache=True,
        )

        dream_query = self.model.prepare_dream_queries_with_special_token(batch_size=batch_size)

        dream_query_attention_mask = torch.ones(batch_size, dream_query.shape[1], device=device)
        attention_mask = torch.cat([text_input_attention_mask, dream_query_attention_mask], dim=1)

        out = self.forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=text_out.past_key_values,
            inputs_embeds=dream_query,
            use_cache=True,
            output_hidden_states=True,
        )

        prompt_embeds = out.hidden_states[-1][:, 1:-1, :]

        return prompt_embeds

    def encode_prompt(
        self,
        tokenizer,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Union[torch.FloatTensor, None]= None,
        negative_prompt_embeds: Union[torch.FloatTensor, None]= None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        if prompt is not None and isinstance(prompt, str) or isinstance(prompt, MultimodalContent):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self.get_prompt_embeds(tokenizer, prompt, device)

        prompt_embeds_dtype = self.dtype

        prompt_embeds = prompt_embeds.to(device, dtype=prompt_embeds_dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: list[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds = self.get_prompt_embeds(tokenizer, uncond_tokens, device)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def stable_diffusion_pipeline(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt= None,
        height: Union[int, None]= None,
        width: Union[int, None]= None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt = None,
        num_images_per_prompt: Union[int, None]= 1,
        eta: float = 0.0,
        generator= None,
        latents= None,
        prompt_embeds= None,
        negative_prompt_embeds= None,
        output_type= "pil",
        callback = None,
        callback_steps: int = 1,
        cross_attention_kwargs = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(tokenizer, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            tokenizer,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # NOTE: 4. DreamLLMForCausalMLM only acts as a text encoder and leaves the rest to the stable diffusion head
        return self.stable_diffusion_head.pipeline(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
        )

    @torch.no_grad()
    def controlnet_pipeline(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt = None,
        image: PipelineImageType = None,
        height: Union[int, None]= None,
        width: Union[int, None]= None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt = None,
        num_images_per_prompt: Union[int, None]= 1,
        eta: float = 0.0,
        generator = None,
        latents: Union[torch.FloatTensor, None]= None,
        prompt_embeds: Union[torch.FloatTensor, None]= None,
        negative_prompt_embeds: Union[torch.FloatTensor, None]= None,
        output_type = "pil",
        callback = None,
        callback_steps: int = 1,
        cross_attention_kwargs = None,
        controlnet_conditioning_scale: Union[float, list[float]]= 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, list[float]]= 0.0,
        control_guidance_end: Union[float, list[float]]= 1.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(tokenizer, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            tokenizer,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # NOTE: 4. DreamLLMForCausalMLM only acts as a text encoder and leaves the rest to the stable diffusion head
        return self.controlnet_head.pipeline(
            image=image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )