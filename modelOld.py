import torch
import torch.nn as nn
import fairseq
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones
from torch import Tensor
from model import *

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
    def __init__(self, config, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1, 
                 vote_perhead=24, total_num=126, cam=True, dsm=True, fix=True, assess=False):
        super(MyConformer, self).__init__()
        self.dim_head = int(emb_size / heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders
        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))

        # Replace Conformer blocks with IELTEncoder
        self.encoder = IELTEncoder(config, update_warm=500, vote_perhead=vote_perhead, cam=cam, dsm=dsm, 
                                   fix=fix, total_num=total_num, assess=assess)

    def forward(self, x, device):  # x shape [bs, tiempo, frecuencia]
        x = x + self.positional_emb[:, :x.size(1), :]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])  # [bs, 1+tiempo, emb_size]
        
        # Run through IELTEncoder and receive x and xc
        x, xc = self.encoder(x, test_mode=False)
        
        return x, xc



class SSLModel(nn.Module): #W2V
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = 'xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + Conformer')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MyConformer(
            emb_size=args.emb_size,
            n_encoders=args.num_encoders,
            heads=args.heads,
            kernel_size=args.kernel_size,
            config=args.config  # Pass config if required
        )
        self.head = nn.Linear(args.emb_size, 2)  # Assuming num_classes = 2
        self.softmax = nn.Softmax(dim=-1)
        self.loss_alpha = args.loss_alpha  # Set from args

    def forward(self, x, labels=None):
        # Extract features from Wav2Vec model
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs, frame_number, feat_out_dim)
        x = x.unsqueeze(dim=1)  # Add channel dimension (bs, 1, frame_number, emb_size)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)

        # Run through the conformer and get x and xc
        x, xc = self.conformer(x, self.device)

        # Logit Assist logic using xc
        complement_logits = self.head(xc)
        probability = self.softmax(complement_logits)
        weight = self.head.weight
        assist_logit = probability * weight.sum(-1)
        part_logits = self.head(x) + assist_logit

        # Return for test mode (when labels are None)
        if labels is None:
            return part_logits

        # Training mode with loss calculation
        if args.smooth_value == 0:
            loss_fct = CrossEntropyLoss()
        else:
            loss_fct = LabelSmoothing(args.smooth_value)

        loss_p = loss_fct(part_logits.view(-1, 2), labels.view(-1))
        loss_c = loss_fct(complement_logits.view(-1, 2), labels.view(-1))
        alpha = self.loss_alpha
        loss = (1 - alpha) * loss_p + alpha * loss_c

        return part_logits, loss



# test case
import torch
import argparse

# Define arguments for the model
parser = argparse.ArgumentParser(description="Test Model with Wav2Vec and IELT Conformer")
parser.add_argument('--emb_size', type=int, default=128, help="Embedding size")
parser.add_argument('--num_encoders', type=int, default=1, help="Number of encoder layers")
parser.add_argument('--heads', type=int, default=4, help="Number of attention heads")
parser.add_argument('--kernel_size', type=int, default=16, help="Convolution kernel size")
parser.add_argument('--loss_alpha', type=float, default=0.4, help="Loss alpha for complement loss")
parser.add_argument('--smooth_value', type=float, default=0.1, help="Smoothing value for LabelSmoothing")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume config is defined elsewhere, set a dummy config if needed
config = None  # Replace with your actual config as needed
args.config = config

# Initialize the model
model = Model(args, device).to(device)

# Create a dummy input tensor (batch_size=2, seq_len=16000, channels=1)
dummy_input = torch.randn(2, 16000, 1).to(device)

# Run a forward pass
with torch.no_grad():  # Optional: avoids tracking gradients for testing
    output = model(dummy_input)
    print("Model output:", output)
