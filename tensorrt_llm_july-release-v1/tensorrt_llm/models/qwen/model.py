import math
from collections import OrderedDict
from typing import Optional, Tuple, List

import tensorrt as trt
import numpy as np

from ...logger import logger
#添加通用层
from ..._common import default_net,default_trtnet, precision
from ..._utils import pad_vocab_size, str_dtype_to_trt
#添加变量的定义
from ...functional import (RaggedTensor, Tensor, assertion, expand_mask,
                           gather_last_token_logits, is_gated_activation,
                           non_gated_version, shape , silu, arange, outer, expand,gather,
                            cos, sin, slice, concat, view, unsqueeze, constant ,_create_tensor)
#添加具体的层
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, InflightBatchingParam, LayerNorm,
                       PositionEmbeddingType, PromptTuningEmbedding, RmsNorm, QWenAttention)
from ...module import Module, ModuleList
from ...quantization import QuantMode

from ...plugin import  _TRT_LLM_PLUGIN_NAMESPACE as TRT_LLM_PLUGIN_NAMESPACE

#进行模型的搭建

apply_rotary_emb_func = None
rms_norm = None
flash_attn_unpadded_func = None

#直接调用已有的函数
#进行旋转编码模块构建 未构建完成，可能需要进行plugin的加入，直接替换这个算子 ，纯粹生成旋转位置坐标
# class RotaryEmbedding(Module):
#     def __init__(self, dim, base=10000) -> None:
#         super().__init__()
#         self.dim = dim
#         self.base = base
#         self.inv_freq = constant(1.0 / (base ** (np.arange(0, dim, 2).astype(np.float16) / dim)))
#         self._rotary_pos_emb_cache = None
#         self._seq_len_cached = 0
#         self._ntk_alpha_cached = 1.0

#     def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
#         seqlen = max_seq_len + offset
#         if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
#             base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
#             self.inv_freq = constant(1.0 / (base ** (np.arange(0, self.dim, 2).astype(np.float16) / self.dim)))
#             # 1.0 / (
#             #     base
#             #     ** (arange(0, self.dim, 0)
#             #         / self.dim
#             #     )
#             # )
#             self._seq_len_cached = max(2 * seqlen, 16)
#             self._ntk_alpha_cached = ntk_alpha
#             seq = constant(np.arange(0, self._seq_len_cached).astype(np.float16))
#             # seq = arange(concat(seq, 0), self._seq_len_cached,trt.float16)

#             freqs = outer(seq, self.inv_freq)
            
#             emb = concat([freqs, freqs], dim=-1)

#             #此处需要重排张量
#             emb = unsqueeze(emb, 1)
#             emb = unsqueeze(emb, 0)

#             cos_, sin_ = cos(emb), sin(emb)
#             self._rotary_pos_emb_cache = [cos_, sin_]   

#     def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
#         self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
#         cos_, sin_ = self._rotary_pos_emb_cache
#         #对tensor进行切片操作,需要使用slice算子
#         return [slice(cos_,concat(cos_ ,offset), concat(cos_, max_seq_len)), slice(sin_,concat(sin_ ,offset), concat(sin_, max_seq_len))] 

def identity_op(tensor: Tensor) -> Tensor:
    input_tensor = tensor.trt_tensor
    # Create a plugin instance.
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        'Identity', '1', TRT_LLM_PLUGIN_NAMESPACE)
    assert plugin_creator is not None
    pfc = trt.PluginFieldCollection([])
    plugin = plugin_creator.create_plugin("identity", pfc)
    layer = default_trtnet().add_plugin_v2([input_tensor], plugin)
    return _create_tensor(layer.get_output(0), layer)


class QWenMLP(Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 bias,
                 dtype) -> None:
        super().__init__()
        self.w1 = ColumnLinear(hidden_size, intermediate_size // 2, bias=bias,gather_output=False,dtype = dtype)
        self.w2 = ColumnLinear(hidden_size, intermediate_size // 2, bias=bias,gather_output=False,dtype = dtype)
        ff_dim_in = intermediate_size // 2
        self.c_proj = ColumnLinear(ff_dim_in, hidden_size, bias=bias,dtype = dtype)
    
    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output

class QWenBlock(Module):
    def __init__(self, 
                 hidden_size,
                 num_attention_heads,
                 max_position_embeddings,
                 num_layers,
                 dtype=None,
                 apply_query_key_layer_scaling=False,
                 attention_mask_type=AttentionMaskType.bidirectional,
                 hidden_act = 'relu',
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 quant_mode=QuantMode(0),
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=False,
                 bf16=False,
                 multi_query_mode=False,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        # self.hidden_size = hidden_size
        # self.num_attention_heads = num_attention_heads
        # self.max_position_embeddings = max_position_embeddings
        # self.num_layers = num_layers
        # self.dtype = dtype
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_mask_type = attention_mask_type
        self.position_embedding_type = position_embedding_type
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.bf16 = bf16
        # self.kv_channels = kv_channels
        
        self.ln_1 = RmsNorm(
            hidden_size,
            dtype = dtype
        )
        #此处采用通用的注意力机制
        self.attention = QWenAttention(
            hidden_size,
            num_attention_heads,
            max_position_embeddings,
            num_layers,
            apply_query_key_layer_scaling,
            dtype=dtype,
            attention_mask_type=attention_mask_type,
            position_embedding_type=position_embedding_type,
            neox_rotary_style=True,
            rotary_embedding_percentage=rotary_embedding_percentage,
            bias=bias,
            multi_query_mode=multi_query_mode,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache()
        ) 
        self.ln_2 = RmsNorm(
            hidden_size,
            dtype = dtype
        )

        self.mlp = QWenMLP(
            hidden_size = hidden_size,
            intermediate_size = inter_size,
            bias = bias,
            dtype = dtype
        )

        self.ln_test = ColumnLinear(hidden_size,
                        3 * hidden_size,   
                        bias=not bias,
                        dtype=dtype,
                        tp_group=tp_group,
                        tp_size=tp_size,
                        gather_output=False)

    def forward(
        self,
        hidden_states: RaggedTensor,
        position_embedding,
        attention_mask=None,
        past_key_value=None,
        sequence_length=None,
        past_key_value_length=None,
        masked_tokens=None,
        use_cache=False,
        cache_indirection=None,
        kv_cache_block_pointers=None,
        inflight_batching_args=None,
        past_key_value_pointers=None
    ):
        assert isinstance(hidden_states, RaggedTensor) #判断输入是否为可变长度的张量
        #解压 RaggedTensor，因为某些层（例如 MLP、LayerNorm）只需要数据张量
        input_lengths = hidden_states.row_lengths
        max_input_length = hidden_states.max_row_length
        hidden_states = hidden_states.data

        # residual = hidden_states

        layernorm_output = self.ln_1(hidden_states)

        attention_output = self.attention(
            RaggedTensor.from_row_lengths(layernorm_output, input_lengths,
                                          max_input_length),
            position_embedding,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            masked_tokens=masked_tokens,
            use_cache=use_cache,
            cache_indirection=cache_indirection,
            kv_cache_block_pointers=kv_cache_block_pointers,
            inflight_batching_args=inflight_batching_args,
            past_key_value_pointers=past_key_value_pointers
        )
        if use_cache:
            attention_output, presents = attention_output

        # # register as model output
        # # ------------------------------------------------------
        # self.register_network_output('attention_output', attention_output.data)
        # # ------------------------------------------------------

        hidden_states = hidden_states + attention_output.data
        
        # residual = hidden_states

        layernorm_output = self.ln_2(hidden_states)

        mlp_output = self.mlp(layernorm_output)

        hidden_states = hidden_states + mlp_output

        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                                input_lengths,
                                                max_input_length)

        if use_cache:
            return (hidden_states, presents)
        
        return hidden_states

class QWenModel(Module):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype=None,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 apply_query_key_layer_scaling=False,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0),
                 multi_query_mode=False,
                 use_prompt_tuning=False,
                 use_parallel_embedding=False):
        super().__init__()

        self.half_head_size = hidden_size // num_heads // 2    #64 
        #以下对应RotaryEmbedding
        self.position_embedding_cos = Embedding(max_position_embeddings,
                                                self.half_head_size*2,
                                                dtype=dtype)  # shape=8192,128
        self.position_embedding_sin = Embedding(max_position_embeddings,
                                                self.half_head_size*2,
                                                dtype=dtype)


        self.wte = Embedding(
            vocab_size, 
            hidden_size,
            dtype=dtype)

        self.layers = ModuleList([
            QWenBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                max_position_embeddings=max_position_embeddings,
                num_layers=num_layers,
                dtype=dtype,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                attention_mask_type=AttentionMaskType.causal,
                hidden_act=hidden_act,
                position_embedding_type=position_embedding_type,
                rotary_embedding_percentage=rotary_embedding_percentage,
                multi_query_mode=multi_query_mode,
                tp_group=tensor_parallel_group,
                tp_size=tensor_parallel,
                inter_size=inter_size,
                bias=bias,
                quant_mode=quant_mode) for _ in range(num_layers)
        ])

        self.ln_f = RmsNorm(
            hidden_size,
            dtype=dtype
        )

    def forward(self,
                input_ids,
                position_ids,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                attention_mask=None,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None,
                inflight_batching_args=None):
        
        hidden_states = self.wte(input_ids.data)

        batch_size = shape(input_ids.data, 0)
        input_len = shape(input_ids.data, 1)

        hidden_states = self.wte(input_ids.data)
        position_embedding_cos = self.position_embedding_cos(position_ids) #  Previously calculated 
        position_embedding_sin = self.position_embedding_sin(position_ids) #  Previously calculated 

        position_embedding_cos0, position_embedding_cos1 = position_embedding_cos.split(
            1, dim=1)
        position_embedding_sin0, position_embedding_sin1 = position_embedding_sin.split(
            1, dim=1)

      
        position_embedding_cos0 = position_embedding_cos0.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))
        position_embedding_cos1 = position_embedding_cos1.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))
        position_embedding_sin0 = position_embedding_sin0.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))
        position_embedding_sin1 = position_embedding_sin1.view(
            concat([batch_size, input_len, 1, self.half_head_size*2]))

        position_embedding = [
            position_embedding_cos0, position_embedding_cos1,
            position_embedding_sin0, position_embedding_sin1
        ]

        # # register as model output
        # # ------------------------------------------------------
        # self.register_network_output('wte', hidden_states)
        # # ------------------------------------------------------

        # position_ids = None
        # kv_seq_len = hidden_states.size()[1]
        # if past_key_value is None:
        #     past_key_value = tuple([None] * len(self.layers))

        #     kv_seq_len += past_key_value[0][0].shape[1]

        if use_cache: 
            presents = []   

        # if attention_mask is not None: #生成注意力掩码
        #     attention_mask = expand_mask(attention_mask,
        #                                  shape(input_ids.data, -1))
            
        hidden_states = RaggedTensor.from_row_lengths(hidden_states,
                                                      input_ids.row_lengths,
                                                      input_ids.max_row_length)

        for idx, (layer, past, pointers) in enumerate(
                zip(self.layers, past_key_value, kv_cache_block_pointers)):
            hidden_states = layer(
                hidden_states,
                position_embedding,
                past_key_value=past,
                sequence_length=sequence_length,
                past_key_value_length=past_key_value_length,
                masked_tokens=masked_tokens,
                use_cache=use_cache,
                attention_mask=attention_mask,
                cache_indirection=cache_indirection,
                kv_cache_block_pointers=pointers,
                inflight_batching_args=inflight_batching_args,
                past_key_value_pointers=None if inflight_batching_args is None
                else inflight_batching_args.past_key_value_pointers[idx])

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]       

        # hidden_states = self.ln_f(hidden_states.data)
        # if use_cache:
        #     return (hidden_states, tuple(presents))
        # return hidden_states

        hidden_states=hidden_states.data
        hidden_states_clone=identity_op(hidden_states)
        hidden_states_final = self.ln_f(hidden_states_clone)    

        if use_cache:
            return (hidden_states_final, tuple(presents))
        return hidden_states_final


class QWenLMHeadModel(QWenModel):

    def __init__(self,
                 num_layers,
                 num_heads,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 max_position_embeddings,
                 dtype,
                 tensor_parallel=1,
                 tensor_parallel_group=None,
                 apply_query_key_layer_scaling=False,
                 position_embedding_type=PositionEmbeddingType.learned_absolute,
                 rotary_embedding_percentage=1.0,
                 inter_size=None,
                 bias=True,
                 quant_mode=QuantMode(0),
                 multi_query_mode=False,
                 use_prompt_tuning=False,
                 share_embedding_table=False,
                 use_parallel_embedding=False):

        if isinstance(dtype, str):
            self._kv_dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self._kv_dtype = dtype
        self._dtype = self._kv_dtype
        if quant_mode.has_int8_kv_cache():
            self._kv_dtype = str_dtype_to_trt('int8')

        self.quant_mode = quant_mode

        self._num_layers = num_layers
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._tensor_parallel = tensor_parallel
        self._multi_query_mode = multi_query_mode
        self._use_prompt_tuning = use_prompt_tuning
        
        super().__init__(num_layers = num_layers, num_heads = num_heads, hidden_size=hidden_size, vocab_size=vocab_size,
                         hidden_act=hidden_act, max_position_embeddings=max_position_embeddings, dtype=dtype,
                         tensor_parallel=tensor_parallel, tensor_parallel_group=tensor_parallel_group,
                         apply_query_key_layer_scaling=apply_query_key_layer_scaling, position_embedding_type=position_embedding_type,
                         rotary_embedding_percentage=rotary_embedding_percentage, inter_size=inter_size, bias=bias,
                         quant_mode=quant_mode, multi_query_mode=multi_query_mode, use_prompt_tuning=use_prompt_tuning,
                         use_parallel_embedding=use_parallel_embedding)
        
        vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel)

        share_weight = None

        if share_embedding_table == True:
            share_weight = self.embedding.vocab_embedding.weight

        self.lm_head = ColumnLinear(hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=tensor_parallel_group,
                                    tp_size=tensor_parallel,
                                    gather_output=True,
                                    share_weight=share_weight)

    def forward(self,
                input_ids,position_ids,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                masked_tokens=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                cache_indirection=None,
                kv_cache_block_pointers=None,
                prompt_embedding_table=None,
                prompt_tasks=None,
                prompt_vocab_size=None,
                inflight_batching_args=None
    ):
        assert last_token_ids is not None, "Expecting last token ids to be not None"
        
        hidden_states = super().forward(
            input_ids,position_ids, past_key_value, sequence_length,
            past_key_value_length, masked_tokens, use_cache, attention_mask,
            cache_indirection, kv_cache_block_pointers, prompt_embedding_table,
            prompt_tasks, prompt_vocab_size, inflight_batching_args)

        if use_cache:
            hidden_states, presents = hidden_states

        # hidden_states = gather_last_token_logits(
        #     hidden_states, last_token_ids,
        #     default_net().plugin_config.remove_input_padding)
        

        # only calculate logits for the last token
        # [batch_size, seqlen, hidden_size] -> [batch_size, hidden_size]
        last_token_ids = last_token_ids.view(
            concat([shape(last_token_ids, 0), 1, 1]))
        last_token_ids = expand(
            last_token_ids,
            concat([shape(last_token_ids, 0), 1,
                    shape(hidden_states, 2)]))
        last_token_ids = last_token_ids - 1
        hidden_states = gather(
            hidden_states, dim=1, indices=last_token_ids).view(
                concat([shape(hidden_states, 0),
                        shape(hidden_states, 2)]))

        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        lm_logits =identity_op(lm_logits)
        lm_logits.mark_output('logits', str_dtype_to_trt('float32'))
        # out_inter.mark_output('inter', str_dtype_to_trt('float32'))

        if use_cache:
            for i, present in enumerate(presents):
                present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            return (lm_logits, presents)

        return lm_logits


    def prepare_inputs(self,
                       max_batch_size,
                       max_input_len,
                       max_new_tokens,
                       use_cache,
                       max_beam_width: int = 1,
                       paged_kv_cache: bool = False,
                       tokens_per_block: int = 64,
                       prompt_embedding_table_size: int = 128):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
            ranges of the dimensions of when using TRT dynamic shapes.

            @return: a list contains values which can be fed into the self.forward()
        '''

        # Prepare inputs
        head_size = self._hidden_size // self._num_heads
        num_heads = self._num_heads // self._tensor_parallel
        num_heads_kv = 1 if self._multi_query_mode else num_heads
        max_len = max_input_len + max_new_tokens
        bb_range = [
            1, (max_batch_size * max_beam_width + 1) // 2,
            max_batch_size * max_beam_width
        ]
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        beam_width_range = [1, (max_beam_width + 1) // 2, max_beam_width]
        inlen_range = [1, 1, max_input_len]
        max_len_range = [0, (max_len + 1) // 2, max_len]
        mask_len_range = [1, (max_len + 1) // 2 + 1, max_len + 1]
        num_tokens_range = [
            1, max_batch_size * max_beam_width,
            max(max_input_len * max_batch_size, max_beam_width * max_batch_size)
        ]
        if paged_kv_cache:
            blocks_range = [
                math.ceil((bb_range[0] * max_len_range[0]) / tokens_per_block),
                math.ceil((bb_range[1] * max_len_range[1]) / tokens_per_block),
                math.ceil((bb_range[2] * max_len_range[2]) / tokens_per_block)
            ]
            # NOTE(nkorobov): we multiply max_blocks_per_seq by 2 because plugin expects pointers as int64,
            # but TRT does not support int64. Thus, we emulate int64 with doubled int32.
            max_blocks_per_seq_range = [
                2 * math.ceil(max_len_range[0] / tokens_per_block),
                2 * math.ceil(max_len_range[1] / tokens_per_block),
                2 * math.ceil(max_len_range[2] / tokens_per_block)
            ]
        p_embedding_range = [
            1, prompt_embedding_table_size // 2, prompt_embedding_table_size
        ]

        past_key_value = []
        past_key_value_pointers = []
        sequence_length = None
        past_key_value_length = None
        masked_tokens = None
        attention_mask = None
        inflight_batching_args = None
        cache_indirection = None
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_inflight_batching_gpt_attention_plugin = default_net(
        ).plugin_config.inflight_batching_gpt_attention_plugin

        if remove_input_padding:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', [1]),
                                   ('num_tokens', [num_tokens_range]),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size', [1]),
                                      ('num_tokens', [num_tokens_range]),
                                  ]))
            # position_ids = None
        else:
            input_ids = Tensor(name='input_ids',
                               dtype=trt.int32,
                               shape=[-1, -1],
                               dim_range=OrderedDict([
                                   ('batch_size', [bb_range]),
                                   ('input_len', [inlen_range]),
                               ]))
            position_ids = Tensor(name='position_ids',
                                  dtype=trt.int32,
                                  shape=[-1, -1],
                                  dim_range=OrderedDict([
                                      ('batch_size', [bb_range]),
                                      ('input_len', [inlen_range]),
                                  ]))
            # position_ids = None

        prompt_embedding_table = None
        tasks = None
        prompt_vocab_size = None
        if self._use_prompt_tuning:
            prompt_embedding_table = Tensor(name='prompt_embedding_table',
                                            dtype=self._dtype,
                                            shape=[-1, self._hidden_size],
                                            dim_range=OrderedDict([
                                                ('prompt_embedding_table_size',
                                                 [p_embedding_range]),
                                                ('hidden_size',
                                                 [self._hidden_size]),
                                            ]))
            tasks = Tensor(name='tasks',
                           dtype=trt.int32,
                           shape=[-1],
                           dim_range=OrderedDict([
                               ('batch_size', [bb_range]),
                           ]))
            prompt_vocab_size = Tensor(name='prompt_vocab_size',
                                       dtype=trt.int32,
                                       shape=[1],
                                       dim_range=OrderedDict([('size', [1])]))

        for i in range(self._num_layers):
            if not paged_kv_cache:
                kv_dim_range = OrderedDict([
                    ('batch_size', [bb_range]),
                    ('kv', [2]),
                    ('num_heads', [num_heads_kv]),
                    ('past_key_len', [max_len_range]),
                    ('head_size', [head_size]),
                ])
                kv = Tensor(name=f'past_key_value_{i}',
                            dtype=self._kv_dtype,
                            shape=[-1, 2, num_heads_kv, -1, head_size],
                            dim_range=kv_dim_range)
                past_key_value.append(kv)

                # TODO(kaiyu): Remove this when TRT fix the named dimension
                if not remove_input_padding:
                    assertion(shape(input_ids, 0) == shape(kv, 0), 'batch size')
            else:
                kv_dim_range = OrderedDict([
                    ('blocks', [blocks_range]),
                    ('kv', [2]),
                    ('num_heads', [num_heads_kv]),
                    ('tokens_per_block', [tokens_per_block]),
                    ('head_size', [head_size]),
                ])
                # (2, blocks, kv_num_heads, tokens_per_block, head_size)
                kv = Tensor(
                    name=f'past_key_value_{i}',
                    dtype=self._kv_dtype,
                    shape=[-1, 2, num_heads_kv, tokens_per_block, head_size],
                    dim_range=kv_dim_range)
                past_key_value.append(kv)

            if use_inflight_batching_gpt_attention_plugin:
                kv = Tensor(
                    name=f'past_key_value_pointers_{i}',
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size=[bs_range],
                                          pointer_width=[2]))

                past_key_value_pointers.append(kv)

        if use_gpt_attention_plugin or use_inflight_batching_gpt_attention_plugin:
            past_key_value_length = Tensor(
                name='past_key_value_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('past_key_value_length',
                                        [max_len_range])]),
            )

        if use_gpt_attention_plugin:
            sequence_length = Tensor(
                name='sequence_length',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size', [bb_range])]),
            )
            masked_tokens = Tensor(name='masked_tokens',
                                   dtype=trt.int32,
                                   shape=[-1, -1],
                                   dim_range=OrderedDict([
                                       ('batch_size', [bb_range]),
                                       ('max_seq_len', [max_len_range]),
                                   ]))
        elif not use_inflight_batching_gpt_attention_plugin:
            attention_mask = Tensor(name='attention_mask',
                                    dtype=trt.int32,
                                    shape=[-1, -1],
                                    dim_range=OrderedDict([
                                        ('batch_size', [bb_range]),
                                        ('mask_len', [mask_len_range]),
                                    ]))

        input_lengths = Tensor(name='input_lengths',
                               dtype=trt.int32,
                               shape=[-1],
                               dim_range=OrderedDict([('batch_size', [bb_range])
                                                      ]))

        max_input_length = Tensor(name='max_input_length',
                                  dtype=trt.int32,
                                  shape=[-1],
                                  dim_range=OrderedDict([('max_input_len',
                                                          [inlen_range])]))

        last_token_ids = Tensor(name='last_token_ids',
                                dtype=trt.int32,
                                shape=[-1],
                                dim_range=OrderedDict([
                                    ('batch_size', [bb_range]),
                                ]))
        input_ids_ragged = RaggedTensor.from_row_lengths(
            input_ids, input_lengths, max_input_length)

        if not use_inflight_batching_gpt_attention_plugin:
            cache_indirection = Tensor(name='cache_indirection',
                                       dtype=trt.int32,
                                       shape=[-1, -1, -1],
                                       dim_range=OrderedDict([
                                           ('batch_size', [bs_range]),
                                           ('beam_width', [beam_width_range]),
                                           ('max_seq_len', [max_len_range]),
                                       ]))

        if use_inflight_batching_gpt_attention_plugin:
            inflight_batching_args = InflightBatchingParam(
                # [nbReq]
                host_input_lengths=Tensor(
                    name='host_input_lengths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range])),
                # [nbReq]
                host_beam_widths=Tensor(
                    name='beam_widths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range])),
                # [nbReq, 2]
                cache_indir_pointers=Tensor(
                    name='cache_indir_pointers',
                    dtype=trt.int32,
                    # 2 INT32s for representing a single INT64 pointer
                    shape=[-1, 2],
                    dim_range=OrderedDict(batch_size=[bs_range],
                                          pointer_width=[2])),
                # [nbReq]
                host_req_cache_max_seq_lengths=Tensor(
                    name='req_cache_max_seq_lengths',
                    dtype=trt.int32,
                    shape=[-1],
                    dim_range=OrderedDict(batch_size=[bs_range])),
                max_input_length=max_input_len,
                max_beam_width=max_beam_width,
                use_int8_kv_cache=self.quant_mode.has_int8_kv_cache(),
                past_key_value_pointers=past_key_value_pointers)

        kv_cache_block_pointers_list = []
        for i in range(self._num_layers):
            if paged_kv_cache:
                kv_cache_block_pointers = Tensor(
                    name=f'kv_cache_block_pointers_{i}',
                    dtype=trt.int32,
                    shape=[-1, -1, 2, -1],
                    dim_range=OrderedDict([
                        ('batch_size', [bs_range]),
                        ('beam_width', [beam_width_range]),
                        ('kv', [2]),
                        ('max_blocks_per_seq', [max_blocks_per_seq_range]),
                    ]))
            else:
                kv_cache_block_pointers = None
            kv_cache_block_pointers_list.append(kv_cache_block_pointers)

        return (input_ids_ragged, position_ids, past_key_value, sequence_length,
                past_key_value_length, masked_tokens, True, last_token_ids,
                attention_mask, cache_indirection, kv_cache_block_pointers_list,
                prompt_embedding_table, tasks, prompt_vocab_size,
                inflight_batching_args)

