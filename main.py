import argparse
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import Dataset_train, Dataset_eval
from model import InterEnsembleLearningTransformer  # Imported the new model
from transformers import Wav2Vec2Model
from utils import reproducibility
from utils import read_metadata
from model import InterEnsembleLearningTransformer
from modules import *
import numpy as np
from tqdm import tqdm

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_batch = len(dev_loader)
    
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            target = torch.LongTensor(batch_y).to(device)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # Forward pass through the model
            part_logits, _ = model(batch_x)

            pred = part_logits.max(1)[1]
            correct += pred.eq(target).sum().item()
            batch_loss = criterion(part_logits, batch_y)
            val_loss += (batch_loss.item() * batch_size)
    
    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print(f"Validation Accuracy: {test_accuracy}% | Validation Loss: {val_loss}")
    return val_loss

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []
    text_list = []

    with torch.no_grad():
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            part_logits, _ = model(batch_x)
            batch_score = part_logits[:, 1].data.cpu().numpy().ravel()

            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

    for f, cm in zip(fname_list, score_list):
        text_list.append(f'{f} {cm}')
    
    with open(save_path, 'a+') as fh:
        for i in range(0, len(text_list), 500):
            batch = text_list[i:i+500]
            fh.write('\n'.join(batch) + '\n')
    
    print(f'Scores saved to {save_path}')

def train_epoch(train_loader, model, optimizer, device, epoch):
    num_total = 0.0
    model.train()

    # Objective function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    pbar = tqdm(train_loader)
    for i, (batch_x, batch_y) in enumerate(pbar):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass
        part_logits, _ = model(batch_x)

        # Loss calculation
        batch_loss = criterion(part_logits, batch_y)

        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}: cls_loss {batch_loss.item()}")
    
    return batch_loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepfake Detection with InterEnsembleLearningTransformer')
    
    # Dataset
    parser.add_argument('--database_path', type=str, default='ASVspoof_database/', help='Dataset directory')
    parser.add_argument('--protocols_path', type=str, default='ASVspoof_database/', help='Protocols directory')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # Model parameters
    parser.add_argument('--smooth_value', type=float, default=0.0, help='Label smoothing value')
    parser.add_argument('--loss_alpha', type=float, default=0.4, help='Alpha for weighted loss calculation')
    parser.add_argument('--vote_perhead', type=int, default=24, help='Number of votes per head in transformer')
    
    # Model save path
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # Training
    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']), help='Training mode')
    
    # Rawboost Parameters
        ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    
    args = parser.parse_args()
    args.track = 'LA'
    
    reproducibility(args.seed, args)
    
    track = args.track
    n_mejores = args.n_mejores_loss
    
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Initialize the model
    model = InterEnsembleLearningTransformer(config=None, num_classes=2, smooth_value=args.smooth_value, loss_alpha=args.loss_alpha, vote_perhead=args.vote_perhead).to(device)
    
    # Set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training set
    label_trn, files_id_train = read_metadata(os.path.join(args.protocols_path, f'LA/{track}_cm_protocols/{track}_cm.train.trn.txt'), is_eval=False)
    print(f'Number of training trials: {len(files_id_train)}')
    train_set = Dataset_train(args, list_IDs=files_id_train, labels=label_trn, base_dir=os.path.join(args.database_path, 'LA', 'train'), algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=10, shuffle=True, drop_last=True)
    
    # Validation set
    labels_dev, files_id_dev = read_metadata(os.path.join(args.protocols_path, f'LA/{track}_cm_protocols/{track}_cm.dev.trl.txt'), is_eval=False)
    print(f'Number of validation trials: {len(files_id_dev)}')
    dev_set = Dataset_train(args, list_IDs=files_id_dev, labels=labels_dev, base_dir=os.path.join(args.database_path, 'LA', 'dev'), algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=8, num_workers=10, shuffle=False)
    
    # Training loop
    epoch = 0
    best_loss = float('inf')
    if args.train:
        while epoch < args.num_epochs:
            print(f'######## Epoch {epoch} ########')
            train_epoch(train_loader, model, optimizer, device, epoch)
            val_loss = evaluate_accuracy(dev_loader, model, device)
            
            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join('models', 'best_model.pth'))
                print(f"New best model saved at epoch {epoch}")
            
            epoch += 1
        print(f'Total epochs: {epoch}')


print('######## Eval ########')
if args.average_model:
    sdl = []
    model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
    print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
    sd = model.state_dict()
    for i in range(1, args.n_average_model):
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
        sd2 = model.state_dict()
        for key in sd:
            sd[key] = (sd[key] + sd2[key])
    for key in sd:
        sd[key] = (sd[key]) / args.n_average_model
    model.load_state_dict(sd)
    torch.save(model.state_dict(), os.path.join(best_save_path, 'avg_5_best_{}.pth'.format(i)))
    print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
else:
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth')))
    print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

# Evaluation for multiple tracks
eval_tracks = ['LA', 'DF']
if args.comment_eval:
    model_tag = model_tag + '_{}'.format(args.comment_eval)

for tracks in eval_tracks:
    score_file = 'Scores/{}/{}.txt'.format(tracks, model_tag)
    if not os.path.exists(score_file):
        prefix = 'ASVspoof_{}'.format(tracks)
        prefix_2019 = 'ASVspoof2019.{}'.format(tracks)
        prefix_2021 = 'ASVspoof2021.{}'.format(tracks)

        file_eval = read_metadata(os.path.join(args.protocols_path, '{}/{}_cm_protocols/{}.cm.eval.trl.txt'.format(tracks, prefix, prefix_2021)), is_eval=True)
        print('Number of evaluation trials:', len(file_eval))
        
        eval_set = Dataset_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path, '{}/ASVspoof2021_{}_eval/'.format(tracks, tracks)), track=tracks)
        produce_evaluation_file(eval_set, model, device, score_file)
    else:
        print('Score file already exists: {}'.format(score_file))
