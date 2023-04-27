import torch
import numpy as np
import pandas as pd
import os, sys
import argparse
import json

from dataset import HistogramTracesDataset
def predict(model, csv_paths, save_csv_path):
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c",
        required=True,
        help="Config file designating file paths.",
    )
    
    specs = json.load(open(os