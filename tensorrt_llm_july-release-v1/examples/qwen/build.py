import argparse
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.models import smooth_quantize, weight_only_quantize 
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

from tensorrt_llm.models.qwen.model import  QWenLMHeadModel

import tensorrt as trt
#在此定义了从保存的文件中取出权重的部分
from weight import load_from_ft, qwen_parse_ft_config, check_embedding_share  # isort:skip

MODEL_NAME = "qwen"


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=51200)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_positions', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--hidden_act', type=str, default='swiglu')
    parser.add_argument(
        '--rotary_pct',
        type=float,
        default=0.0,
        help="Setting this to a value > 0.0 (and <= 1.0) activates RoPE.")
    parser.add_argument('--inter_size', type=int, default=None)
    parser.add_argument('--no_bias', action="store_false")
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='qwen_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument(
        "--multi_query_mode",
        "-mq",
        default=False,
        action='store_true',
        help=
        "Whether this model uses multi-query attention mechanism (default: False)"
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')

    # Arguments related to the quantization of the model.
    parser.add_argument(
        '--use_smooth_quant',
        default=False,
        action="store_true",
        help=
        'Use the SmoothQuant method to quantize activations and weights for the various GEMMs.'
        'See --per_channel and --per_token for finer-grained quantization options.'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help=
        'Seed to use when initializing the random number generator for torch.')
    parser.add_argument(
        '--paged_kv_cache',
        action="store_true",
        default=False,
        help=
        'By default we use contiguous KV cache. By setting this flag you enable paged KV cache'
    )
    parser.add_argument('--tokens_per_block',
                        type=int,
                        default=64,
                        help='Number of tokens per block in paged KV cache')
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        type=int,
        default=0,
        help='Setting to a value > 0 enables support for prompt tuning.')
    parser.add_argument(
        '--use_inflight_batching',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32'],
        help=
        "Activates attention plugin for inflight batching. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_lookup_plugin',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32'],
        help=
        "Activates the lookup plugin which enables tensor-parallel embedding. It is also required for embedding table and language modeling weight sharing."
    )

    args = parser.parse_args(args)
    logger.set_level(args.log_level)

    args.bias = not args.no_bias
    if args.inter_size is None:
        args.inter_size = 4 * args.n_embd

    if args.model_dir is not None:
        logger.info(f"Setting model configuration from {args.model_dir}.")
        #从这里取出配置文件.ini的配置信息
        n_embd, n_head, n_layer, n_positions, vocab_size, _, hidden_act, rotary_pct, bias, inter_size, multi_query_mode, dtype, prompt_num_tasks, prompt_max_vocab_size = qwen_parse_ft_config(
            Path(args.model_dir) / "config.ini")
        args.n_embd = n_embd
        args.n_head = n_head
        args.n_layer = n_layer
        args.n_positions = n_positions
        args.vocab_size = vocab_size
        args.hidden_act = hidden_act
        args.rotary_pct = rotary_pct
        args.bias = bias
        args.dtype = dtype
        args.inter_size = inter_size
        args.multi_query_mode = multi_query_mode
    plugins_args = [
        'use_gpt_attention_plugin', 'use_gemm_plugin', 'use_layernorm_plugin',
        'use_inflight_batching', 'use_lookup_plugin'
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"plugin_arg is None, setting it as {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token,
                                                     args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(args.dtype) 
    # Decide if we can share the embedding table between
    # the lookup OP and the logits calculation OP
    share_embedding_table = False
    if args.use_lookup_plugin and args.model_dir is not None:
        share_embedding_table = check_embedding_share(args.model_dir)

    if share_embedding_table and (not args.use_gemm_plugin):
        logger.warning(
            f'Sharing embedding tables between OPs requires using GEMM plugin. Otherwise, you might fail to see the engine size reduction.'
        )

    # Initialize Module
    tensorrt_llm_qwen = QWenLMHeadModel(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        inter_size=args.inter_size,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        position_embedding_type=PositionEmbeddingType.learned_absolute
        if args.rotary_pct == 0.0 else PositionEmbeddingType.rope,
        rotary_embedding_percentage=args.rotary_pct,
        dtype=kv_dtype,
        tensor_parallel=args.world_size,  # TP only
        tensor_parallel_group=list(range(args.world_size)),  # TP only
        apply_query_key_layer_scaling=builder_config.
        apply_query_key_layer_scaling,
        quant_mode=args.quant_mode,
        bias=args.bias,
        multi_query_mode=args.multi_query_mode,
        use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
        share_embedding_table=share_embedding_table)

    if args.use_smooth_quant:
        tensorrt_llm_qwen = smooth_quantize(tensorrt_llm_qwen, args.quant_mode)
    elif args.use_weight_only:
        tensorrt_llm_qwen = weight_only_quantize(tensorrt_llm_qwen,
                                                args.quant_mode)

    if args.model_dir is not None:
        load_from_ft(tensorrt_llm_qwen, args.model_dir, rank, args.world_size,
                     args.dtype, share_embedding_table, args.use_lookup_plugin)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache()
    if args.use_inflight_batching:
        network.plugin_config.set_inflight_batching_gpt_attention_plugin(
            dtype=args.use_inflight_batching)

    # Quantization plugins.
    if args.use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
        network.plugin_config.set_layernorm_quantization_plugin(
            dtype=args.dtype)
        # FIXME(nkorobov)
        # See https://nvbugs/4164762
        # See https://nvbugs/4174113
        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    elif args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)

    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)
    assert not (args.use_lookup_plugin
                and args.max_prompt_embedding_table_size > 0
                ), "Lookup plugin isn't compatible with prompt tuning right now"
    
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_qwen.named_parameters())

        # Forward
        inputs = tensorrt_llm_qwen.prepare_inputs(
            args.max_batch_size,
            args.max_input_len,
            args.max_output_len,
            False,
            args.max_beam_width,
            paged_kv_cache=args.paged_kv_cache,
            tokens_per_block=args.tokens_per_block,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size)

        outputs = tensorrt_llm_qwen(*inputs)
        # 标记为 TRT 网络输出
        # ---------------------------- ------------------------------------
        for k, v in tensorrt_llm_qwen.named_network_outputs():
            network._mark_output(v, k,
                                 tensorrt_llm.str_dtype_to_trt(args.dtype))
        # -----------------------------------------------------------------

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node) #设置torch的可使用gpu数量
    tensorrt_llm.logger.set_level(args.log_level) #设置logger等级
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    builder = Builder()
    cache = None
    apply_query_key_layer_scaling = False 
    for cur_rank in range(args.world_size): #设置工作空间大小 有几个卡
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=args.timing_cache if cache is None else cache,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers=args.n_layer, 
            num_heads=args.n_head,
            hidden_size=args.n_embd, #隐藏层size
            vocab_size=args.vocab_size, #神经网络模型词汇表大小
            hidden_act=args.hidden_act, #隐藏层中神经元的激活函数
            max_position_embeddings=args.n_positions, #max_position_embeddings
            apply_query_key_layer_scaling=apply_query_key_layer_scaling, #是否对其进行缩放
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            int8=(args.quant_mode.has_act_and_weight_quant()
                  or args.quant_mode.has_int8_kv_cache()),
            opt_level=args.builder_opt,
            multi_query_mode=args.multi_query_mode,
            paged_kv_cache=args.paged_kv_cache,
            tokens_per_block=args.tokens_per_block,
            use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
            use_parallel_embedding=bool(args.use_lookup_plugin))

        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,  #给生成的模型取名字
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(
            builder_config, os.path.join(args.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."


def run_build(args=None):
    args = parse_arguments(args) #设置参数

    if args.random_seed is not None: #设置随机数种子
        torch.manual_seed(args.random_seed)

    logger.set_level(args.log_level)
    tik = time.time()
    if args.parallel_build and args.world_size > 1 and \
            torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False #此处使用一台机器进行构建
        logger.info('Serially build TensorRT engines.')
        build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {args.world_size} engines: {t}')


if __name__ == '__main__':
    run_build()
