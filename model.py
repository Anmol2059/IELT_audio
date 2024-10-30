import torch
import torch.nn as nn
import fairseq
from torch.nn.modules.transformer import _get_clones
from torch import Tensor
from ielt import *

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
    def __init__(self, config, emb_size=128, heads=4, kernel_size=16, n_encoders=1, 
                 vote_perhead=24, total_num=126, cam=True, dsm=True, fix=True, assess=False):
        super(MyConformer, self).__init__()
        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.encoder = IELTEncoder(config, vote_perhead=vote_perhead, cam=cam, dsm=dsm, 
                                   fix=fix, total_num=total_num, assess=assess)

    def forward(self, x, device):
        x = x + self.positional_emb[:, :x.size(1), :]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])
        x, xc = self.encoder(x, test_mode=False)
        return x, xc

class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = 'xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MyConformer(
            emb_size=args.emb_size,
            n_encoders=args.num_encoders,
            heads=args.heads,
            kernel_size=args.kernel_size,
            config=args.config
        )
        self.head = nn.Linear(args.emb_size, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_alpha = args.loss_alpha

    def forward(self, x, labels=None):
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat).unsqueeze(dim=1)
        x = self.selu(self.first_bn(x)).squeeze(dim=1)
        x, xc = self.conformer(x, self.device)

        complement_logits = self.head(xc)
        probability = self.softmax(complement_logits)
        assist_logit = probability * self.head.weight.sum(-1)
        part_logits = self.head(x) + assist_logit

        if labels is None:
            return part_logits

        loss_fct = CrossEntropyLoss() if args.smooth_value == 0 else LabelSmoothing(args.smooth_value)
        loss_p = loss_fct(part_logits.view(-1, 2), labels.view(-1))
        loss_c = loss_fct(complement_logits.view(-1, 2), labels.view(-1))
        loss = (1 - self.loss_alpha) * loss_p + self.loss_alpha * loss_c
        return part_logits, loss

# Main script to test model
import argparse

class Config:
    def __init__(self):
        self.num_layers = 6
        self.hidden_size = 128
        self.mlp_dim = 512
        self.num_heads = 4
        self.dropout_rate = 0.1
        self.att_dropout = 0.1

parser = argparse.ArgumentParser(description="Test Model with Wav2Vec and IELT Conformer")
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--num_encoders', type=int, default=1)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--loss_alpha', type=float, default=0.4)
parser.add_argument('--smooth_value', type=float, default=0.1)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
args.config = config

model = Model(args, device).to(device)
dummy_input = torch.randn(2, 16000, 1).to(device)

with torch.no_grad():
    output = model(dummy_input)
    print("Model output:", output)

probabilities = torch.softmax(output, dim=-1)
predictions = torch.argmax(probabilities, dim=-1)
print("Probabilities:", probabilities)
print("Predicted Classes:", predictions)
