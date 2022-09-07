import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn



from trainer import Trainer

params={
    "model_params":{"de_norm":nn.Sequential(nn.ConvTranspose2d(512,512,kernel_size=3,stride=2),
                                            # nn.BatchNorm2d(512),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2),
                                            # nn.BatchNorm2d(512),
                                            nn.ReLU()),
                                            "beta":0.01,
                                            "latent_dim":512,},
    "optimizer_name":"adam",
    "optimizer_params":{"lr":1e-3},
    "train_test_val_split":{"train":0.8,"test":0.1,"val":0.1},
    "train_batch_size":16,
    "test_batch_size":16,
    "val_batch_size":16,
    "val_freq":5,
    "seed":42,
    "epochs":100,
    "save_path":"checkpoints"
}

if __name__=="__main__":
    wandb.init(project="Greek-Vase-VAE", entity="m6481")
    wandb.config = params
    trainer=Trainer(params)
    trainer.train()

