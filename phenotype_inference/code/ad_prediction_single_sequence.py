import torch
import argparse
import torch.nn as nn
from esm import pretrained

class SequenceInteractionModel(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.esm_model = esm_model.eval()        
        for p in self.esm_model.parameters():   
            p.requires_grad = False

    def forward(self, seq_tokens):
        out = self.esm_model(seq_tokens, repr_layers=[self.esm_model.num_layers])
        rep = out["representations"][self.esm_model.num_layers][:, 1:-1, :]   
        rep = rep.flatten(start_dim=1)    
        return rep

def get_args():
    parser = argparse.ArgumentParser(description='Predict antigenic distance between two RBD sequences')
    parser.add_argument('--seq1', type=str, required=True, help='First RBD sequence')
    parser.add_argument('--seq2', type=str, required=True, help='Second RBD sequence')
    parser.add_argument('--loadmodel', type=str, default='wt', choices=['wt', 'ba5'])
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading ESM model and {args.loadmodel} trained evoRBD weights...")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    model = SequenceInteractionModel(esm_model)
    
    loadmodelpath = f'phenotype_inference/model_trained/evoRBD_{args.loadmodel}_trained.pth'
    try:
        sd = torch.load(loadmodelpath, map_location='cpu')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
    except FileNotFoundError:
        print(f"Warning: Model file {loadmodelpath} not found. Using raw ESM-2.")

    data = [("seq1", args.seq1), ("seq2", args.seq2)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        embeddings = model(batch_tokens)
        dist = torch.cdist(embeddings[0:1], embeddings[1:2], p=1) * 0.1
        result = dist.item()

    print("-" * 30)
    print(f"Sequence 1: {args.seq1}")
    print(f"Sequence 2: {args.seq2}")
    print(f"Predicted Antigenic Distance: {result:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    main()