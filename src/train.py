# adapted from https://github.com/HUAWEI-Theory-Lab/deepqueuenet/blob/e84fc9bf09260e2a1bb586aa5c2024e346858569/train1000.py#L74
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import copy

from ptm import deepPTM, load_model_from_ckpt
from dataset import TracesDataset 

pytorch_seed = 0
torch.manual_seed(pytorch_seed)


def save_model(model, saved_model_pth):
    torch.save(model.state_dict(), saved_model_pth)


def train_epochs(model, lr, train_dl, valid_dl, epochs=10, start_label=0):
    train_epoch_losses = []
    eval_epoch_losses = []
    total_batch_num = 0
    iter_num = 0

    cached_loss = "Init"

    model = model.to(device)
    optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr = lr
                                )
    loss_func = torch.nn.MSELoss(reduction="mean")
    eval_loss_func = copy.deepcopy(loss_func)

    valid_avg_loss = valid(model, eval_loss_func, valid_dl, eval_epoch_losses)
    writer.add_scalar("Loss/valid", valid_avg_loss, start_label)

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
                epoch_batch_num += 1
                out = model(batch_x)
                loss = loss_func(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()     
                sum_of_loss += loss.item() 
                batch_num = batch_num+1
                tepoch.set_postfix(avg_loss=sum_of_loss/batch_num)
    
        if i < 10 or i % 5:
            writer.add_scalar("Loss/train", sum_of_loss/epoch_batch_num, i+ start_label)
            valid_avg_loss = valid(model, eval_loss_func, valid_dl, eval_epoch_losses)
            writer.add_scalar("Loss/valid", valid_avg_loss, i + start_label + 1)

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
            save_model(model, saved_model_pth)
    return epoch_avg_loss

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--train_config", "-c",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    args = arg_parser.parse_args()
    
    specs = json.load(open(os.path.join(args.train_config)))

    #### training config ####################

    identifier = "default"

    if not "save_pth" in specs:
        save_base_dir = f"saved/{identifier}"
        model_dir = "{}/saved_model".format(save_base_dir)
    else:
        model_dir = os.path.join(specs["save_pth"], specs["exp_name"] + "_lr_{}_steps_{}".format(specs["train_lr"],specs["n_timesteps"]))

    saved_model_name = "best_model.pt"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    saved_model_pth = os.path.join(model_dir, saved_model_name)


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    summary_dir = "runs_ts"
    writer = SummaryWriter(os.path.join(summary_dir, specs["exp_name"] + "_lr_{}_steps_{}_".format(specs["train_lr"],specs["n_timesteps"]) + current_time))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create update lr function
    data_specs = specs["data_specs"]
    model_specs = specs["model_specs"]
    n_timesteps = specs["n_timesteps"]
    lr = specs["train_lr"]

    # Get data
    train_data = [data_specs["train_data_pth"]] if not isinstance(data_specs["train_data_pth"], list) else data_specs["train_data_pth"]
    val_data = [data_specs["val_data_pth"]] if not isinstance(data_specs["val_data_pth"], list) else data_specs["val_data_pth"]

    mp.set_start_method('spawn')
    train_ds = TracesDataset(train_data,
                             n_timesteps=n_timesteps,
                             use_norm_time=specs['use_norm_time'])
    valid_ds = TracesDataset(val_data,
                             n_timesteps=n_timesteps,
                             use_norm_time=specs['use_norm_time'])
    train_dl = DataLoader(train_ds,
                          batch_size=specs['batch_size'],
                          shuffle = True,
                          num_workers = 4, 
                          # pin_memory = True,
                          # sampler = SubsetRandomSampler(sample(dataset_indices, batch_size * batch_num_per_epoch))
                          )
    valid_dl = DataLoader(valid_ds,
                          batch_size=specs['batch_size'],
                          shuffle=False)

    model = deepPTM(in_feat=train_ds.num_feat,
                    lstm_config=model_specs["lstm_config"],
                    attn_config=model_specs["attn_config"],
                    time_steps=n_timesteps,
                    use_norm_time=specs['use_norm_time']
                    )
    if 'trained_model_pth' in specs:
        saved_model_pth = specs['trained_model_pth']
        load_model_from_ckpt(model, specs['trained_model_pth'])
    
    train_epochs(model, lr=lr,
                 train_dl=train_dl, valid_dl=valid_dl, epochs=specs["n_epochs"], start_label=0)

                    
    
    # mp.set_sharing_strategy('file_system')
    # train()
    # print("FINISHED ...")

