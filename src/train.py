# adapted from https://github.com/HUAWEI-Theory-Lab/deepqueuenet/blob/e84fc9bf09260e2a1bb586aa5c2024e346858569/train1000.py#L74
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import copy

from ptm import deepPTM 
from dataset import TracesDataset 

pytorch_seed = 0
torch.manual_seed(pytorch_seed)



batch_size = 64 * 2 * 2

lr = 1e-3


#### training config ####################

identifier = "default"

save_base_dir = f"saved/{identifier}"

model_dir = "{}/saved_model".format(save_base_dir)
saved_model_name = "best_model.pt"

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

#########################################
# base_dir = "./DeepQueueNet-synthetic-data/data" # this path may need to be changed/unsure of how we want to segment data
# # if we are saving the data separately:  
# train_data_dir = base_dir+"_train"
# valid_data_dir = base_dir+"_valid"


# TODO: Retreive right directory from the directory instead of explicit naming the file
train_data_dir = ['./data/processed_data/train_processed.csv']
valid_data_dir = ['./data/processed_data/test_processed.csv']


writer = SummaryWriter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create update lr function

train_epoch_losses = []
eval_epoch_losses = []
total_batch_num = 0
iter_num = 0

cached_loss = "Init"


def save_model(model, out_file):
    torch.save(model.state_dict(), model_dir + "/" + out_file)


def train_epochs(model, train_dl, valid_dl, epochs=10, start_label=0):
    model = model.to(device)
    optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr = lr
                                )
    loss_func = torch.nn.MSELoss()
    eval_loss_func = copy.deepcopy(loss_func)
    for i in range(epochs):
        with tqdm(train_dl, unit="batch") as tepoch:
            model.train()
            batch_num = 0
            sum_of_loss = 0
            epoch_batch_num = 0 # total number of batches seen so far in epoch
            
            for batch_x, batch_y in tepoch:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                current_batch_size = batch_x.shape[0]
                tepoch.set_description(f"Epoch {i}")
                epoch_batch_num += current_batch_size
                out = model(batch_x)
                loss = loss_func(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()     
                sum_of_loss += loss.item() 
                tepoch.set_postfix(avg_loss=sum_of_loss/(batch_num+1))
    
        writer.add_scalar("Loss/train", sum_of_loss/epoch_batch_num, i+ start_label)
        valid_avg_loss = valid(model, eval_loss_func, valid_dl, eval_epoch_losses)
        writer.add_scalar("Loss/valid", valid_avg_loss, i+ start_label)

def valid(model, eval_loss_func, validation_loader, eval_epoch_losses):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        eval_epoch_batch_num = len(validation_loader)
        for batch_x, batch_y in validation_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            loss = eval_loss_func(out, batch_y)
            eval_loss += loss.item()

        epoch_avg_loss = eval_loss / eval_epoch_batch_num
        eval_epoch_losses.append(epoch_avg_loss)

        ### save the model for the epoch with best eval loss
        if epoch_avg_loss == min(eval_epoch_losses):
            save_model(model, saved_model_name)
    return epoch_avg_loss

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train_ds = TracesDataset(train_data_dir,
                             n_timesteps=42)
    valid_ds = TracesDataset(valid_data_dir,
                             n_timesteps=42)
    train_dl = DataLoader(train_ds, batch_size = batch_size,
                            shuffle = True,
                            num_workers = 0, 
                            # pin_memory = True,
                            # sampler = SubsetRandomSampler(sample(dataset_indices, batch_size * batch_num_per_epoch))
                            )
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = deepPTM(in_feat=train_ds.num_feat)
    train_epochs(model, train_dl=train_dl, valid_dl=valid_dl, epochs=500, start_label=0)

                    
    
    # mp.set_sharing_strategy('file_system')
    # train()
    # print("FINISHED ...")

