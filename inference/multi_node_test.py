from argparse import ArgumentParser
import json
import os
import time
import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer
from model import Block, Linear, ModelArgs, ParallelEmbedding, RMSNorm, ColumnParallelLinear, precompute_freqs_cis
from torch import nn
from typing import List
from generate import sample

class TransformerPP(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs, ckpt_path):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        self.prev_hop_rank = (rank - 1) % args.n_layers
        self.next_hop_rank = (rank + 1)  % args.n_layers
        self.layer_id = rank
        self.args = args

        super().__init__()

        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank)

        with self.device:
            self.model_inst = Block(self.layer_id, args)        
        

        
        if self.layer_id == 0:
            with self.device:
                self.embed_inst = ParallelEmbedding(args.vocab_size, args.dim)
                self.norm = RMSNorm(args.dim)
                self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())


        self.max_seq_len = args.max_seq_len

        
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0, is_warmup: bool = False):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        batch_size = tokens.size(0)
        seqlen = tokens.size(1)
        
        
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)


        if is_warmup:
            if self.layer_id == 0:
                h = self.embed_inst(tokens)
                h = self.model_inst(h, start_pos, freqs_cis, mask)
                h = self.norm(h)[:, -1]
                logits = self.head(h)
                return logits
            else:
                h = torch.zeros((batch_size, seqlen, self.args.dim), device=tokens.device)
                h = self.model_inst(h, start_pos, freqs_cis, mask)
                return None
            

        if self.layer_id == 0:
            h = self.embed_inst(tokens)
            print(f"{time.time()} layer{self.layer_id} got init tokens")
        else:
            h = torch.zeros((batch_size, seqlen, self.args.dim), device=tokens.device)

            print(f"{time.time()} layer{self.layer_id} start waiting recv tokens from {self.prev_hop_rank}")
            recv_ret = dist.recv(h, self.prev_hop_rank)
            # The following line is very important, otherwise, layer 1 will recv without blocking anymore(i.e., it seems a bug, layer 0 doesn't send anything, but layer 1 can recv from layer 0) 
            torch.cuda.default_stream().synchronize()
            print(f"{time.time()} layer{self.layer_id} has recv tokens from {self.prev_hop_rank}, recv ret value is {recv_ret}, begin calc")

        h = self.model_inst(h, start_pos, freqs_cis, mask)

        print(f"{time.time()} layer{self.layer_id} finish layer calc, start_pos={start_pos}, seqlen={seqlen}")
        print(f"{time.time()} layer{self.layer_id} begin send to next layer(layer{self.next_hop_rank})")
        dist.send(h, self.next_hop_rank)
        print(f"{time.time()} layer{self.layer_id} finish send to next layer(layer{self.next_hop_rank})")
        

        if self.layer_id == 0:
            print(f"{time.time()} layer{self.layer_id} start waiting result from {self.prev_hop_rank}")
            dist.recv(h, self.prev_hop_rank)

            h = self.norm(h)[:, -1]
            logits = self.head(h)
            return logits

        return None


@torch.inference_mode()
def generate(
    model: TransformerPP,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    is_warmup = False
) -> List[List[int]]:

    rank = int(os.environ["RANK"])
    layer_id = rank
 
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=model.device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=model.device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=model.device)
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, is_warmup)
        print(f"{time.time()} layer{layer_id} has finished one forward calc")
        if layer_id == 0:
            if temperature > 0:
                next_token = sample(logits, temperature)
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
            prev_pos = cur_pos
            if finished.all():
                break
        else:
            prev_pos = cur_pos

    print(f"{time.time()} layer{layer_id} stop generate")
    if layer_id == 0:
        completion_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            completion_tokens.append(toks)
        
        print(f"{time.time()} layer{layer_id} return completion_tokens={completion_tokens}")
        return completion_tokens
    else:
        return None
    

@torch.inference_mode()
def main():
    assert torch.distributed.is_available(), "torch.dist is not enabled"
    assert torch.distributed.is_nccl_available(), "nccl is not enabled"

    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    cli_args = parser.parse_args()
    
    with open(cli_args.config_path) as f:
        args = ModelArgs(**json.load(f))
    

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    layer_id = rank

    print(f"layer{layer_id} args={args}")

    if rank >= args.n_layers:
        print(f"{time.time()} layer{layer_id} exit because it is beyond may layer(which is {args.n_layers})")
        exit(0)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.n_layers)
    
    print (f"layer{layer_id} dist.get_world_size() = {dist.get_world_size()}")

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16

    with device:
        model_inst = TransformerPP(args, cli_args.ckpt_path)
    
    # Note: load_module must not run in `with device` context, otherwise, GPU memory will overflow.
    # it seems that in the with device context, `load_model` function will try to alloc another copy of weights in GPU which
    # leads to OOM. Even if you try to set `load_model`'s device argument to "cpu", it still not work, maybe a bug.    
    load_model(model_inst.model_inst, os.path.join(cli_args.ckpt_path, f"pp_model_layer{layer_id}.safetensors"))
    if layer_id == 0:
        load_model(model_inst.embed_inst, os.path.join(cli_args.ckpt_path, f"pp_model_layer_embed.safetensors"))
        load_model(model_inst.norm, os.path.join(cli_args.ckpt_path, f"pp_model_layer_norm.safetensors"))
        load_model(model_inst.head, os.path.join(cli_args.ckpt_path, f"pp_model_layer_head.safetensors"))
    
    tokenizer = AutoTokenizer.from_pretrained(cli_args.ckpt_path)

    # warm up
    print(f"{time.time()} layer{layer_id} start warm up")
    generate_ret = generate(model_inst, [tokenizer.encode("DeepSeek")], 2, -1, 1., is_warmup=True)
    if layer_id == 0:
        tokenizer.decode(generate_ret[0])
    print(f"{time.time()} layer{layer_id} finished warm up")

    # dist.barrier()
    time.sleep(1)
    os.system("mkdir -p /data/mmh/logs")
    with open(f"/data/mmh/logs/layer-{layer_id}.log", "w") as fo:
        pass

    print(f"{time.time()} layer{layer_id} pass first barrier")
    
    max_new_tokens: int = 10
    temperature: float = 1.0
    messages = []
    messages.append({"role": "user", "content": "你好"})
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    

    print(f"{time.time()} layer {layer_id} started...")
    # dist.barrier()
    time.sleep(1)


    completion_tokens = generate(model_inst, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
    
    if layer_id == 0:

        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for completion in completions:
            print(f"{time.time()} Completion:", completion)
            print()


@torch.inference_mode()
def barrier_test():
    assert torch.distributed.is_available(), "torch.dist is not enabled"
    assert torch.distributed.is_nccl_available(), "nccl is not enabled"

    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    cli_args = parser.parse_args()
    
    with open(cli_args.config_path) as f:
        args = ModelArgs(**json.load(f))
    

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    layer_id = rank

    print(f"layer{layer_id} args={args}")

    if rank >= args.n_layers:
        print(f"{time.time()} layer{layer_id} exit because it is beyond may layer(which is {args.n_layers})")
        exit(0)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.n_layers)
    
    print (f"layer{layer_id} dist.get_world_size() = {dist.get_world_size()}")

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16



    print(f"{time.time()} layer{layer_id} before barrier")

    dist.barrier()
    os.system("mkdir -p /data/mmh/logs")
    with open(f"/data/mmh/logs/layer-{layer_id}.log", "w") as fo:
        pass

    print(f"{time.time()} layer{layer_id} pass first barrier")

if __name__ == "__main__":
    barrier_test()




    