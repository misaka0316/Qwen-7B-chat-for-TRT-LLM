'''
Convert huggingface GPT model. Use https://huggingface.co/Qwen/Qwen-7B-Chat as demo.
'''

import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import re

import numpy as np

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import AutoTokenizer
from transformers.models.qwen.modeling_qwen import QWenBlock
from utils.convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "Qwen"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="Qwen",
            type=str,
            help="Specify qwen variants to convert checkpoints correctly",
            choices=["Qwen", "santacoder", "starcoder"])
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        return ProgArgs(**vars(parser.parse_args(args)))

@torch.no_grad()
def smooth_gpt_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        print(name)
        print(module)
        # if not isinstance(module, QWenBlock):
        pattern = r'transformer\.h\.\d+'
        if not re.match(pattern, name):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight.T,
                               scales[layer_name]["x"], module.ln_1.weight,
                               module.ln_1.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

        # fc1
        layer_name = name + ".mlp.w1"
        smoother = smooth_gemm(module.mlp.w1.weight.T,
                               scales[layer_name]["x"], module.ln_2.weight,
                               module.ln_2.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.w1.weight.abs().max(dim=0)[0]


# SantaCoder separates Q projection from KV projection
def concat_qkv_weight_bias(q, hf_key, hf_model):
    kv = hf_model.state_dict()[hf_key.replace("q_attn", "kv_attn")]
    return torch.cat([q, kv], dim=-1)


# StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = ["c_attn", "c_proj", "w1", "w2"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def gpt_to_ft_name(orig_name):
    global_weights = {
        # "transformer.wpe.weight": "model.wpe",
        "transformer.wte.weight": "model.wte",
        "transformer.ln_f.bias": "model.final_layernorm.bias",
        "transformer.ln_f.weight": "model.final_layernorm.weight",
        "lm_head.weight": "model.lm_head.weight"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        # "transformer.ln_1.bias": "input_layernorm.bias",
        "transformer.ln_1.weight": "input_layernorm.weight",
        "transformer.attn.c_attn.bias": "attention.query_key_value.bias",
        "transformer.attn.c_attn.weight": "attention.query_key_value.weight",
        # "transformer.attn.q_attn.weight": "attention.query.weight",
        # "transformer.attn.q_attn.bias": "attention.query.bias",
        # "transformer.attn.kv_attn.weight": "attention.key_value.weight",
        # "transformer.attn.kv_attn.bias": "attention.key_value.bias",
        "transformer.attn.c_proj.bias": "attention.dense.bias",
        "transformer.attn.c_proj.weight": "attention.dense.weight",
        # "transformer.ln_2.bias": "post_attention_layernorm.bias",
        "transformer.ln_2.weight": "post_attention_layernorm.weight",
        # "transformer.mlp.w1.bias": "mlp.dense_w1_to_4h.bias",
        "transformer.mlp.w1.weight": "mlp.dense_w1_to_4h.weight",
        # "transformer.mlp.w2.bias": "mlp.dense_w2_to_4h.bias",
        "transformer.mlp.w2.weight": "mlp.dense_w2_to_4h.weight",
        "transformer.mlp.c_proj.bias": "mlp.dense_4h_to_h.bias",
        "transformer.mlp.c_proj.weight": "mlp.dense_4h_to_h.weight",
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_gpt_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = True if args.model in ["santacoder", "starcoder"
                                              ] else False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)


    # add weight of position embedding  decode max length is 2048 保存旋转位置编码
    nMaxSL = 8192
    base=10000.0
    inv_freq = 1.0 / (base ** (np.arange(0, 128, 2,dtype=np.float32) / 128))
    # inv_freq = 10**(-1 / 16 * np.arange(0, 64, 2, dtype=np.float32))
    valueTable = np.matmul(
        np.arange(nMaxSL, dtype=np.float32).reshape(-1, 1),
        np.concatenate([inv_freq, inv_freq],
                       axis=0).reshape(1, -1)).reshape(nMaxSL,
                                                       len(inv_freq) * 2)  # shape is [2048,64] the relate is for postions
    # valueTable=rearrange(valueTable, "n d -> 1 n 1 d")
    cos = np.cos(valueTable) #[:,:64]
    cos = cos.astype(storage_type).tofile(saved_dir /
                                                   "model.cosTable.weight.bin")
    
    sin = np.sin(valueTable)#[:,:64]

    sin = sin.astype(storage_type).tofile(saved_dir /
                                                   "model.sinTable.weight.bin")
    print("Save model.cosTable.weight.bin")
    print("Save model.sinTable.weight.bin")



    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat",
                                                 device_map="auto",
                                                 trust_remote_code=True,fp16=True)
    # model = QWenLMHeadModel.from_pretrained(checkpoint_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
    # print(model)
    #此处加不加评估模式都可以，我们只取出它的参数

    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
        act_range = capture_activation_range(
            model, tokenizer, dataset)
        if args.smoothquant is not None:
            smooth_gpt_model(model, act_range, args.smoothquant)

    config = configparser.ConfigParser()
    config["qwen"] = {}
    for key in vars(args):
        config["qwen"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["qwen"][k] = f"{v}"
    config["qwen"]["storage_dtype"] = args.storage_type
    config["qwen"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)
    
    #确定转换后的参数类型，此处为   fp16 || bf16
    storage_type = str_dtype_to_torch(args.storage_type)

    global_ft_weights = [
        "model.wte", "model.final_layernorm.bias",
        "model.final_layernorm.weight", "model.lm_head.weight"
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    #取模型参数，根据参数取张量的形状
    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name: 
            continue
        ft_name = gpt_to_ft_name(name)
        if args.model == "starcoder":
            param = transpose_weights(name, param)
        # 此处将除开transformer重复部分进行权重保存
        if ft_name in global_ft_weights:
            torch_to_numpy(param.to(storage_type).cpu()).tofile(saved_dir / f"{ft_name}.bin")
        else:
            if 'q_attn' in name:
                param = concat_qkv_weight_bias(param, name, model)
                ft_name = ft_name.replace("query", "query_key_value")
            #此处并未使用多查询模式
            local_dim = model.transformer.h[0].attn.embed_dim if multi_query_mode else None            
            split_and_save_weight(
                0, saved_dir, infer_tp, ft_name, param.to(storage_type),
                 storage_type, act_range.get(name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": multi_query_mode,
                     "local_dim": local_dim
                 })
            
def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================") #此处打印要转换所需的参数
    hf_gpt_converter(args)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn") #torch设置多线程的方法
    run_conversion(ProgArgs.parse()) #
    print("ok.")