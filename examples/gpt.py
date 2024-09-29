import argparse
import dataclasses
import os
import pathlib
import typing

import torch
import torch.nn as nn
import numpy as np

import ipdb


class GPT(nn.Module):
    def __init__(
        self,
        head_num,
        size_per_head,
        vocab_size,
        start_id,
        end_id,
        layer_num,
        max_seq_len: int,
        tensor_para_size: int,
        pipeline_para_size: int,
        lib_path: typing.Union[str, pathlib.Path],
        inference_data_type: str,
        inter_size: int = 0,
        # gpt_variant_params
        layernorm_eps: float = 1e-6,
        layernorm_type: typing.Literal[
            "pre_layernorm", "post_layernorm"
        ] = "pre_layernorm",
        activation_type: str = "Gelu",
        gpt_with_moe: bool = False,
        expert_num: int = 0,
        moe_k: int = 0,
        moe_layer_index: typing.List = [],
        has_positional_encoding: bool = True,
        has_pre_decoder_layernorm: bool = False,
        has_post_decoder_layernorm: bool = True,
        has_adapters: bool = False,
        adapter_inter_size: int = 0,
        use_attention_linear_bias: bool = False,
        int8_mode: int = 0,
        weights_data_type: typing.Union[str, np.dtype] = np.float32,
        shared_contexts_ratio: float = 1.0,
    ):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        self.inter_size = (
            inter_size if inter_size != 0 else 4 * self.head_num * self.size_per_head
        )

        # gpt_variant_params
        self.layernorm_eps = layernorm_eps
        self.layernorm_type = layernorm_type
        self.activation_type = activation_type
        self.gpt_with_moe = gpt_with_moe
        self.expert_num = expert_num
        self.moe_k = moe_k
        self.moe_layer_index = moe_layer_index
        self.has_positional_encoding = has_positional_encoding
        self.has_pre_decoder_layernorm = has_pre_decoder_layernorm
        self.has_post_decoder_layernorm = has_post_decoder_layernorm
        self.has_adapters = has_adapters
        self.adapter_inter_size = adapter_inter_size
        self.use_attention_linear_bias = use_attention_linear_bias

        # multi-gpu params
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.use_sparse_gemm = False
        self.build_model = False
        self.int8_mode = int8_mode
        self.weights_data_type = weights_data_type
        self.shared_contexts_ratio = shared_contexts_ratio

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert (
            head_num % tensor_para_size == 0
        ), "head_num must be a multiple of tensor_para_size."
        assert (
            layer_num % pipeline_para_size == 0
        ), "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(lib_path))

        # Prepare weights
