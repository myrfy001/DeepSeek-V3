import os
import math
import shutil
import json
import sys
import time
from glob import glob
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

from kernel import act_quant, weight_dequant, fp8_gemm
from safetensors.torch import safe_open, save_file
from tqdm import tqdm, trange
from safetensors.torch import load_model, save_model

import model
from model import Block, ModelArgs, Linear, precompute_freqs_cis
import gc
import objgraph
import tracemalloc




def forward_test(ckpt_path, config_path, layer_id):
    assert layer_id < 62, "deepseek v3 only have 61 dense layers"

    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    
    Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    with torch.device("cuda"):
        model_inst = Block(layer_id, args)

    start_pos = 0
    seqlen = 1
    batch_size = 2
    freqs_cis_table = precompute_freqs_cis(args).to("cuda")
    freqs_cis = freqs_cis_table[start_pos:start_pos+seqlen]
    h = torch.randn((batch_size, seqlen, args.dim), device="cuda")

    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device).triu_(1)

    load_model(model_inst, os.path.join(ckpt_path, f"pp_model_layer{layer_id}.safetensors"))

    t0 = tlast = time.time()
    

    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         record_shapes=True,
    #         with_stack=True
    #     ) as prof:
    #         with record_function("model_inference"):
    #             for iter_idx in range(5):
    #                 # h = torch.randn((batch_size, seqlen, args.dim), device="cuda")
    #                 model(h, start_pos, freqs_cis, mask)
    #                 t1 = time.time() - tlast
    #                 print(f"[{iter_idx}], delta = {t1}")
    #                 tlast = time.time()
    #         time.sleep(2)
    # prof.export_chrome_trace("trace.json")  # 可导入Chrome://tracing


    s1 = set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                s1.add(id(obj))
        except: pass
    

    model_inst(h, start_pos, freqs_cis, mask)
    for iter_idx in range(1000):
       
        h = torch.randn((batch_size, seqlen, args.dim), device="cuda")
        model_inst(h, start_pos, freqs_cis, mask)
        t1 = time.time() - tlast
        print(f"[{iter_idx}], delta = {t1}")
        tlast = time.time()

    
    # import pdb
    # objs = []
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             if id(obj) not in s1:
    #                 objs.append(obj)
    #         #         # pdb.set_trace()
    #         #         # obj_id = id(obj)
    #         #         # obj_type = type(obj)
    #         #         # obj_size = obj.size()
    #         #         # print(f"{obj_id}, {obj_type}, {obj_size}, {sys.getrefcount(obj)}, \n")
    #         #         gc.collect()
    #         #         print(f"{sys.getrefcount(obj)}, \n")
                    
                    
    #     except: pass

    


    # gc.collect()
    
    # for obj in objs:
    #     obj_id = id(obj)
    #     obj_type = type(obj)
    #     obj_size = obj.size()
    #     print(f"{obj_id}, {obj_type}, {obj_size}, {sys.getrefcount(obj)}, \n")

    
    


    t2 = time.time() - t0
    print(f"total delta = {t2}")






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    args = parser.parse_args()
    assert args.layer_id is None or args.layer_id <= 61, "layer id must <= 61"

    forward_test(args.ckpt_path, args.config_path, args.layer_id)

    
    
    

