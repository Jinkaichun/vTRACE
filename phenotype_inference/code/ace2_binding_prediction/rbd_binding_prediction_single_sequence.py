import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm import pretrained
from tqdm.auto import tqdm
from structure_embeddiing import *  
from utils import set_seed         

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--seq', type=str)
parser.add_argument('--esm_model_path', type=str, default='../../model_trained/evoRBD_ba5_trained.pth')
parser.add_argument('--binding_model_path', type=str, default='../../model_trained/ace2binding_trained.pth')
parser.add_argument('--pdb_path', type=str, default='configs/6M0J.cif')
parser.add_argument('--foldseek', type=str, default='configs/foldseek')
parser.add_argument('--config_path', type=str, default='configs/saprot_model')
args = parser.parse_args()

class OptimizedRbdMLP(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=256):
        super().__init__()
        self.rbd_attn = nn.Linear(in_dim, 1)
        self.ace_attn = nn.Linear(in_dim, 1)
        self.rbd_norm = nn.BatchNorm1d(in_dim)
        self.ace_norm = nn.BatchNorm1d(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_emd, ace_embedding):
        attn_r = self.rbd_attn(seq_emd)
        w_r = F.softmax(attn_r, dim=1)
        rbd_pooled = torch.sum(seq_emd * w_r, dim=1)
        rbd_pooled_norm = self.rbd_norm(rbd_pooled)
        
        ace = ace_embedding.unsqueeze(0).expand(seq_emd.size(0), -1, -1)
        attn_a = self.ace_attn(ace)
        w_a = F.softmax(attn_a, dim=1)
        ace_pooled = torch.sum(ace * w_a, dim=1)
        ace_pooled_norm = self.ace_norm(ace_pooled)

        inter = torch.cat([rbd_pooled_norm, ace_pooled_norm], dim=-1)
        out = self.net(inter).squeeze(-1)
        return out

def load_esm_model(model_path, device):
    print(f"Loading ESM model from {model_path}...")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    sd = torch.load(model_path, map_location='cpu')
    sd = {k.replace('module.esm_model.', ''): v for k, v in sd.items()}
    esm_model.load_state_dict(sd, strict=False)
    esm_model.to(device)
    esm_model.eval()
    return esm_model, alphabet

def load_binding_model(model_path, device):
    print(f"Loading Binding MLP from {model_path}...")
    sd = torch.load(model_path, map_location='cpu')
    if 'net.0.weight' in sd:
        hidden_dim = sd['net.0.weight'].shape[0]
        print(f"Auto-detected hidden_dim: {hidden_dim}")
    else:
        hidden_dim = 256
        print(f"Warning: Could not auto-detect hidden_dim, using default {hidden_dim}")

    model = OptimizedRbdMLP(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running inference on {device}")

    all_seqs = [args.seq]
    print(f"Total mutant_seq to predict: {len(all_seqs)}")

    print("Generating structure embedding for ACE2...")
    combined_seq = get_struc_seq(args.foldseek, args.pdb_path, ["A"])["A"]
    with torch.no_grad():
        structure_embedding = structurembedding(combined_seq, device, args.config_path)[0]
    structure_embedding = structure_embedding.to(device)

    esm_model, alphabet = load_esm_model(args.esm_model_path, device)
    binding_model = load_binding_model(args.binding_model_path, device)
    
    batch_converter = alphabet.get_batch_converter()
    
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_seqs), args.batchsize), desc="Inferring"):
            batch_seqs = all_seqs[i : i + args.batchsize]
            labels, strs, toks = batch_converter([(None, s) for s in batch_seqs])
            toks = toks.to(device)
            results = esm_model(toks, repr_layers=[esm_model.num_layers], return_contacts=False)
            token_representations = results["representations"][esm_model.num_layers][:, 1:-1, :]
            
            bind_pred = binding_model(token_representations, structure_embedding)
            
            if isinstance(bind_pred, torch.Tensor):
                bind_pred = bind_pred.cpu().numpy()
            all_preds.extend(bind_pred.tolist())

    print(f"Predicted binding affinity for the input sequence: {all_preds[0]}")

if __name__ == "__main__":
    main()