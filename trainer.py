import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import os

from dataloader import VaseDataset
from model import VAE
from VGGLoss import VGGPerceptualLoss
from meter import train_meter,val_meter

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
        self.val_freq=param["val_freq"]
        self.seed=param["seed"]
        self.epochs=param["epochs"]
        self.save_path=param["save_path"]

        self.set_seed()
        self.init_model()
        self.init_optimizer()
        self.init_dataloader()
        self.init_recon_loss()

    def set_seed(self):
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
        self.val_size=len(self.dataset)-self.train_size-self.test_size

        self.train_dataset,self.test_dataset,self.val_dataset=torch.utils.data.random_split(self.dataset,[self.train_size,self.test_size,self.val_size])
        self.train_dataloader=torch.utils.data.DataLoader(self.train_dataset,batch_size=self.train_batch_size,shuffle=True)
        self.test_dataloader=torch.utils.data.DataLoader(self.test_dataset,batch_size=self.test_batch_size,shuffle=True)
        self.val_dataloader=torch.utils.data.DataLoader(self.val_dataset,batch_size=self.val_batch_size,shuffle=True)

    def init_recon_loss(self):
        self.recon_loss=VGGPerceptualLoss().to(self.device)
    
    def loss_func(self,x,x_recon,mu,logvar):
        recon_loss=self.recon_loss(x_recon,x)
        KLD_loss=self.model.kld_loss(mu,logvar)
        return recon_loss+self.model.beta*KLD_loss,recon_loss,KLD_loss

    
    def train_one_epoch(self,epoch):
        self.model.train()
        meter=train_meter(epoch)
        for batch_idx,data in tqdm.tqdm(enumerate(self.train_dataloader)):
            data=data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch,mu,logvar=self.model(data)


            loss,recon_loss,KLD_loss=self.loss_func(data,recon_batch,mu,logvar)
            loss.backward()
            self.optimizer.step()
            meter.update(loss.item(),recon_loss.item(),KLD_loss.item())

        print(meter)
        meter.wandb_log()
    
    def val(self,epoch,loader=None,latent_space_save_path=None,image_save_path=None):

        if loader is None or loader=="val":
            loader=self.val_dataloader
        elif loader=="test":
            loader=self.test_dataloader
        elif loader=="train":
            loader=self.train_dataloader
        else:
            loader=loader


        self.model.eval()
        meter=val_meter(epoch)
        with torch.no_grad():
            for batch_idx,data in enumerate(loader):
                data=data.to(self.device)
                recon_batch,mu,logvar=self.model(data)
                loss,recon_loss,KLD_loss=self.loss_func(data,recon_batch,mu,logvar)
                meter.update(loss.item(),recon_loss.item(),KLD_loss.item(),mu,data,recon_batch)
        if loader==self.val_dataloader:
            print(meter)
            meter.wandb_log()
        if latent_space_save_path:
            meter.plot_latent_space(save_path=latent_space_save_path)
        if image_save_path:
            meter.plot_images(save_path=image_save_path)

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            if epoch%self.val_freq==0:
                save_path=self.save_path+"/"+str(epoch)
                os.makedirs(self.save_path+"/"+str(epoch),exist_ok=True)
                print("Val on val set")
                self.val(epoch,loader="val",latent_space_save_path=save_path+"/latent_space_val.png",image_save_path=save_path+"/images_val.png")
                print("Val on train set")
                self.val(epoch,loader="train",latent_space_save_path=save_path+"/latent_space_train.png",image_save_path=save_path+"/images_train.png")
                torch.save(self.model.state_dict(),save_path+"/model.pt")
        

        
        
    
