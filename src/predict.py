import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import os, sys
from torch.utils.data import DataLoader
import argparse
import json

from dataset import HistogramTracesDataset
from dataset import TracesDataset 
from ptm import deepPTM, load_model_from_ckpt

def predict(model, save_csv_path):    
    model.eval()

    # Create data structures to store all data
    features = [] # we aren't currently including features in the generated csv
    predictions = []
    labels = []
    with torch.no_grad():
        # iterate through data loader
        for batch_x, batch_y in dataloader:            
            # Move to devices
            batch_x= batch_x.to(device)
            batch_y = batch_y.to(device)

            # features.append(batch_x)
            labels.append(batch_y)

            # Predict
            batch_predictions = model(batch_x)

            # Append predictions to list
            batch_predictions = batch_predictions.cpu().numpy()
            predictions.append(batch_predictions)

        # Concatenate into single arrays
        # features = np.concatenate(features, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions.flatten() # flatten so each row is per-packet instead of per-batch
        labels = np.concatenate(labels, axis=0)
        labels = labels.flatten()

        # Create data frame
        df = pd.DataFrame(list(zip(labels, predictions)),
               columns =['actual', 'predictions'])   
        df.to_csv(save_csv_path)

    
    

if __name__ == "__main__":
    import argparse
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", "-c",
        required=True,
        help="Config file designating file paths.",
    )
    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.config)))
    data_specs = specs["data_specs"]
    model_specs = specs["model_specs"]
    n_timesteps = specs["n_timesteps"]
    save_csv_path = specs["save_csv_path"]

    # csv_paths points to processed validation data
    csv_paths = [data_specs["val_data_pth"]] if not isinstance(data_specs["val_data_pth"], list) else data_specs["val_data_pth"]
    
    # Create data set
    dataset = HistogramTracesDataset(
        csv_paths=csv_paths,
        n_timesteps=n_timesteps)
 
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=64,
        num_workers=8)

    model = deepPTM(in_feat=dataset.num_feat,
                    lstm_config=model_specs["lstm_config"],
                    attn_config=model_specs["attn_config"],
                    time_steps=n_timesteps,
                    use_norm_time=specs['use_norm_time']
                    )

    mp.set_start_method('spawn')    
    model = model.to(device)
    # load model
    if 'trained_model_pth' in specs:
        saved_model_pth = specs['trained_model_pth']
        load_model_from_ckpt(model, specs['trained_model_pth'])

    predict(model, save_csv_path)

    


 
    
 

 
    
 
    