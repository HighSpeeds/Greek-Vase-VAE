import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from dataloader import VaseDataset
from model import VAE
from VGGLoss import VGGPerceptualLoss

class Trainer:
    def __init__(self,param):
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_params=param["model_params"]
        self.optimizer_name=param["optimizer_name"]
        self.optimizer_params=param["optimizer_params"]

        self.train_test_val_split=param["train_test_val_split"]
        self.train_batch_size=param["train_batch_size"]
        self.test_batch_size=param["test_batch_size"]
        self.val_batch_size=param["val_batch_size"]
        self.seed=param["seed"]
        self.epochs=param["epochs"]

        self.seed()
        self.init_model()
        self.init_optimizer()
        self.init_dataloader()
        self.init_recon_loss()

    def seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def init_model(self,checkpoint=None):
        self.model=VAE(**self.model_params)
        self.model.to(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
    
    def init_optimizer(self):
        if self.optimizer_name=="adam":
            self.optimizer=optim.Adam(self.model.parameters(),**self.optimizer_params)
        elif self.optimizer_name=="sgd":
            self.optimizer=optim.SGD(self.model.parameters(),**self.optimizer_params)
        elif self.optimizer_name=="adagrad":
            self.optimizer=optim.Adagrad(self.model.parameters(),**self.optimizer_params)

    def init_dataloader(self):
        self.dataset=VaseDataset(path_to_tsv="export2AFECA4C997C412A93A30CCF60896F16.tsv",path_to_data="Data/Images")

        self.train_size=int(len(self.dataset)*self.train_test_val_split["train"])
        self.test_size=int(len(self.dataset)*self.train_test_val_split["test"])
        self.val_size=int(len(self.dataset)*self.train_test_val_split["val"])

        self.train_dataset,self.test_dataset,self.val_dataset=torch.utils.data.random_split(self.dataset,[self.train_size,self.test_size,self.val_size])
        self.train_dataloader=torch.utils.data.DataLoader(self.train_dataset,batch_size=self.train_batch_size,shuffle=True)
        self.test_dataloader=torch.utils.data.DataLoader(self.test_dataset,batch_size=self.test_batch_size,shuffle=True)
        self.val_dataloader=torch.utils.data.DataLoader(self.val_dataset,batch_size=self.val_batch_size,shuffle=True)

    def init_recon_loss(self):
        self.recon_loss=VGGPerceptualLoss().to(self.device)
    
    def loss_func(self,x,x_recon,mu,logvar):
        recon_loss=self.recon_loss(x_recon,x)
        KLD_loss=self.model.kld_loss(mu,logvar)
        return recon_loss+KLD_loss,recon_loss,KLD_loss

    
    def train_one_epoch(self):
        self.model.train()
        train_loss=0
        for batch_idx,(data,_) in enumerate(self.train_dataloader):
            data=data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch,mu,logvar=self.model(data)


            loss,recon_loss,KLD_loss=self.loss_func(data,recon_batch,mu,logvar)
            loss.backward()
            self.optimizer.step()
            train_loss+=loss.item()
            if batch_idx%100==0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))
        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(self.train_dataloader.dataset)))
    
