import os
import time

import numpy as np

from base import SaprotBaseModel
from transformers import EsmTokenizer
import torch
import argparse
import sys

def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  foldseek_verbose: bool = False) -> dict:
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    seq_dict = {}
    name = os.path.basename(path)
    
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] =  combined_seq
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict

def structurembedding(combined_seq, device, config_path):
    config = {
        "task": "base",
        "config_path": config_path,
        "load_pretrained": True,
    }
    
    Saprot = SaprotBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    Saprot.to(device)
    
    inputs = tokenizer(combined_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    embeddings = Saprot.get_hidden_states(inputs) 
    return embeddings




