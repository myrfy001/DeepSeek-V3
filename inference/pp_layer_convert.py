import os
import math
import shutil
from glob import glob
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm
from safetensors.torch import safe_open, save_file
from tqdm import tqdm, trange

from convert import mapping




def extract_single_layer_param(hf_ckpt_path, save_path, layer_id):
    assert layer_id < 62, "deepseek v3 only have 61 layers"

    torch.set_num_threads(8)
    state_dict = {}

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if f"model.layers.{layer_id}." not in name:
                    continue

                param: torch.Tensor = f.get_tensor(name)
                
                name_parts = name.split(".")
                assert len(name_parts) > 3, f"name parts should at least has the form model.layers.x.***, but got: {name_parts}"
                name = ".".join(name_parts[3:])

                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)

                state_dict[name] = param

    os.makedirs(save_path, exist_ok=True)

    save_file(state_dict, os.path.join(save_path, f"pp_model_layer{layer_id}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--layer-id", type=int, required=False)
    args = parser.parse_args()
    assert args.layer_id is None or args.layer_id <= 61, "layer id must <= 61"
    
    if args.layer_id is None:
        for layer_id in tqdm(range(61)):
            extract_single_layer_param(args.hf_ckpt_path, args.save_path, layer_id)
    else:
        extract_single_layer_param(args.hf_ckpt_path, args.save_path, args.layer_id)

