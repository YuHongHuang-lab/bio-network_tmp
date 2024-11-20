import yaml
import logging
import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from dataset.bio_dataset import BioDataset
from model.GraphAutoencoder import HeteroGraphAutoencoder
from train import train
from test import test

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_config('config.yaml')
dataset = BioDataset(config=config["dataset"])

