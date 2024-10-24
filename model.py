import time

import numpy as np
from scipy import ndimage
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from modules import *
# from models.vit import get_b16_config

class InterEnsembleLearningTransformer(nn.Module):
	def __init__(self, config, img_size=448, num_classes=2, dataset='cub', smooth_value=0.,
	             loss_alpha=0.4, cam=True, dsm=True, fix=True, update_warm=500,
	             vote_perhead=24, total_num=126, assess=False):
		super(InterEnsembleLearningTransformer, self).__init__()
		self.assess = assess
		self.smooth_value = smooth_value
		self.num_classes = num_classes
		self.loss_alpha = loss_alpha
		self.cam = cam
        # anmol changed 1
        # self.embeddings = Embeddings(config, img_size=img_size)
		self.embeddings = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

		self.encoder = IELTEncoder(config, update_warm, vote_perhead, dataset, cam, dsm,
		                           fix, total_num, assess)
		self.head = Linear(config.hidden_size, num_classes)
		self.softmax = Softmax(dim=-1)

	def forward(self, x, labels=None):
		test_mode = False if labels is not None else True
        # anmol changed 2
		x = self.embeddings(x).last_hidden_state
		if self.assess:
			x, xc, assess_list = self.encoder(x, test_mode)
		else:
			x, xc = self.encoder(x, test_mode)

		if self.cam:
			complement_logits = self.head(xc)
			probability = self.softmax(complement_logits)
			weight = self.head.weight
			assist_logit = probability * (weight.sum(-1))
			part_logits = self.head(x) + assist_logit
		else:
			part_logits = self.head(x)

		if self.assess: # remove assess branch 
			return part_logits, assess_list

		elif test_mode:
			return part_logits

		else:
			if self.smooth_value == 0:
				loss_fct = CrossEntropyLoss()
			else:
				loss_fct = LabelSmoothing(self.smooth_value)

			if self.cam: # keep this condition & include the loss inside Conformer model 
				loss_p = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
				loss_c = loss_fct(complement_logits.view(-1, self.num_classes), labels.view(-1))
				alpha = self.loss_alpha
				loss = (1 - alpha) * loss_p + alpha * loss_c
			else:
				loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
			return part_logits, loss

	def get_eval_data(self):
		return self.encoder.select_num

	def load_from(self, weights):
		with torch.no_grad():
			nn.init.zeros_(self.head.weight)
			nn.init.zeros_(self.head.bias)

			for bname, block in self.encoder.named_children():
				for uname, unit in block.named_children():
					if not bname.startswith('key') and not bname.startswith('clr'):
						if uname == '12':
							uname = '11'
						unit.load_from(weights, n_block=uname)


class MultiHeadVoting(nn.Module):
    def __init__(self, config, vote_perhead=24, fix=True):
        super(MultiHeadVoting, self).__init__()
        self.fix = fix
        self.num_heads = config.num_heads
        self.vote_perhead = vote_perhead

        if self.fix:
            self.kernel = torch.tensor([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]], device='cuda').unsqueeze(0).unsqueeze(0).half()
            self.conv = F.conv2d  # Use conv2d as in the original code
        else:
            self.conv = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x, select_num=None, last=False):
        B, seq_len = x.shape[0], x.shape[1]  # Adapt for sequence length
        select_num = self.vote_perhead if select_num is None else select_num
        count = torch.zeros((B, seq_len), dtype=torch.int, device='cuda').half()

        # Tuan Suggestion to modify the score calculation to slice correctly for audio input
        score = x[:, 1:x.shape[1], 0]  # Removed spatial dimension handling for audio
        _, select = torch.topk(score, self.vote_perhead, dim=-1)
        select = select.reshape(B, -1)

        for i, b in enumerate(select):
            bincount_result = torch.bincount(b, minlength=seq_len)
            count[i, :seq_len] += bincount_result[:seq_len]

        if not last:
            count = self.enhace_local(count)

        patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
        patch_idx += 1
        return patch_idx[:, :select_num], count

    def enhace_local(self, count):
        B, total_length = count.shape[0], count.shape[1]
        H = int(np.sqrt(total_length))

        #
        if H * H == total_length:
            count = count.reshape(B, 1, H, H)  # Reshape for 2D convolution
            if self.fix:
                count = self.conv(count, self.kernel, stride=1, padding=1).reshape(B, -1)
            else:
                count = self.conv(count).reshape(B, -1)
        else:
            # Use conv1d when total_length doesn't form a perfect square
            kernel_1d = self.kernel.view(1, 1, -1)
            count = F.conv1d(count.unsqueeze(1), kernel_1d, stride=1, padding=1).reshape(B, -1)

        return count




class CrossLayerRefinement(nn.Module):
	def __init__(self, config, clr_layer):
		super(CrossLayerRefinement, self).__init__()
		self.clr_layer = clr_layer
		self.clr_norm = LayerNorm(config.hidden_size, eps=1e-6)

	def forward(self, x, cls):
		out = [torch.stack(token) for token in x]
		out = torch.stack(out).squeeze(1)
		out = torch.cat((cls, out), dim=1)
		out, weights = self.clr_layer(out)
		out = self.clr_norm(out)
		return out, weights


class IELTEncoder(nn.Module): # same level with MyConformer class 
	def __init__(self, config, update_warm=500, vote_perhead=24, dataset='cub',
	             cam=True, dsm=True, fix=True, total_num=126, assess=False):
		super(IELTEncoder, self).__init__()
		self.assess = assess
		self.warm_steps = update_warm # remove this warm 
		self.layer = nn.ModuleList()
		self.layer_num = config.num_layers
		self.vote_perhead = vote_perhead
		self.dataset = dataset
		self.cam = cam
		self.dsm = dsm

		for _ in range(self.layer_num - 1):
			self.layer.append(Block(config, assess=self.assess))

		if self.dataset == 'dog' or self.dataset == 'nabrids': # remove this condition 
			self.layer.append(Block(config, assess=self.assess))
			self.clr_layer = self.layer[-1]
			if self.cam:
				self.layer.append(Block(config, assess=self.assess))
				self.key_layer = self.layer[-1]
		else:
			self.clr_layer = Block(config)
			if self.cam:
				self.key_layer = Block(config)

		if self.cam:
			self.key_norm = LayerNorm(config.hidden_size, eps=1e-6)

		self.patch_select = MultiHeadVoting(config, self.vote_perhead, fix)

		self.total_num = total_num
		## for CUB and NABirds
		self.select_rate = torch.tensor([16, 14, 12, 10, 8, 6, 8, 10, 12, 14, 16], device='cuda') / self.total_num # make the list of rate equal to number of Conformer encoder 
		## for Others
		# self.select_rate = torch.ones(self.layer_num-1,device='cuda')/(self.layer_num-1)

		self.select_num = self.select_rate * self.total_num
		self.clr_encoder = CrossLayerRefinement(config, self.clr_layer)
		self.count = 0

	def forward(self, hidden_states, test_mode=False):
		if not test_mode:
			self.count += 1
		B, N, C = hidden_states.shape
		complements = [[] for i in range(B)]
		class_token_list = []
		if self.assess:
			layer_weights = []
			layer_selected = []
			layer_score = []

		for t in range(self.layer_num - 1):
			layer = self.layer[t]
			select_num = torch.round(self.select_num[t]).int()
			hidden_states, weights = layer(hidden_states)
			select_idx, select_score = self.patch_select(weights, select_num)
			for i in range(B):
				complements[i].extend(hidden_states[i, select_idx[i, :]])
			class_token_list.append(hidden_states[:, 0].unsqueeze(1))
			if self.assess:
				layer_weights.append(weights)
				layer_score.append(select_score)
				layer_selected.extend(select_idx)
		cls_token = hidden_states[:, 0].unsqueeze(1)

		clr, weights = self.clr_encoder(complements, cls_token)
		sort_idx, _ = self.patch_select(weights, select_num=24, last=True)

		if not test_mode and self.count >= self.warm_steps and self.dsm:
			layer_count = self.count_patch(sort_idx)
			self.update_layer_select(layer_count)

		class_token_list = torch.cat(class_token_list, dim=1)

		if not self.cam:
			return clr[:, 0], None
		else:
			out = []
			for i in range(B):
				out.append(clr[i, sort_idx[i, :]])
			out = torch.stack(out).squeeze(1)
			out = torch.cat((cls_token, out), dim=1)
			out, _ = self.key_layer(out)
			key = self.key_norm(out)

		if self.assess:
			assess_list = [layer_weights, layer_selected, layer_score, sort_idx]
			return key[:, 0], clr[:, 0], assess_list
		else:
			return key[:, 0], clr[:, 0]

# !pip install ml-collections
if __name__ == '__main__':
	start = time.time()
	config = get_b16_config()
	net = InterEnsembleLearningTransformer(config).cuda()
    # anmol changed 8
	x = torch.rand(4, 16000, device='cuda')  # 1-second audio samples at 16kHz
	y = net(x)
	print(y.shape)
