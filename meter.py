import wandb
import matplotlib.pyplot as plt
import numpy as np
from tsnecuda import TSNE

class train_meter:
    def __init__(self,epoch):
        self.loss=0
        self.recon_loss=0
        self.kl_loss=0
        self.count=0
        self.epoch=epoch

    def log(self, loss, recon_loss, kl_loss):
        self.loss+=loss
        self.recon_loss+=recon_loss
        self.kl_loss+=kl_loss
        self.count+=1

    def __str__(self):
        return "Epoch: {}, Loss: {}, Recon Loss: {}, KL Loss: {}".format(self.epoch, self.loss/self.count, self.recon_loss/self.count, self.kl_loss/self.count)

    def wandb_log(self):
        wandb.log({"loss": self.loss/self.count, "recon_loss": self.recon_loss/self.count, "kl_loss": self.kl_loss/self.count})
    
class val_meter:
    def __init__(self,epoch):
        self.loss=0
        self.recon_loss=0
        self.kl_loss=0
        self.count=0
        self.latent_spaces=[]
        self.epoch=epoch

        self.ground_truth_images=[]
        self.recon_images=[]

    def log(self, loss, recon_loss, kl_loss,latent_space,ground_truth_images,recon_images):
        self.loss+=loss
        self.recon_loss+=recon_loss
        self.kl_loss+=kl_loss
        self.latent_spaces.append(latent_space.to('cpu').detach().numpy())

        self.ground_truth_images.append(ground_truth_images.to('cpu').detach().numpy())
        self.recon_images.append(recon_images.to('cpu').detach().numpy())
        self.count+=1

    def plot_latent_space(self,save_path):
        latent_space=np.concatenate(self.latent_spaces)
        latent_space_embedded=TSNE(n_components=2).fit_transform(latent_space)

        plt.scatter(latent_space[:,0],latent_space[:,1])
        plt.savefig(save_path)

        wandb.log({save_path[save_path.rfind('/')+1:]: plt})

    def plot_images(self,n_plot=4,save_path=""):
        #randomly pick out 4 images
        n_images=len(self.ground_truth_images)
        indices=np.random.choice(n_images,n_plot,replace=False)
        ground_truth_images=np.concatenate([self.ground_truth_images[i] for i in indices])
        recon_images=np.concatenate([self.recon_images[i] for i in indices])
        #plot out in a grid
        fig,ax=plt.subplots(n_plot,2,figsize=(5*n_plot,10))
        col=["Ground Truth","Recon"]
        for a, col in zip(ax[0], cols):
            a.set_title(col)

        for i in range(n_plot):
            ax[i,0].imshow(ground_truth_images[i],cmap='gray')
            ax[i,1].imshow(recon_images[i],cmap='gray')
        wandb.log({save_path[save_path.rfind("/")+1:]: plt})
        plt.savefig(save_path)

    def __str__(self):
        return "Epoch: {}, Val Loss: {}, Val Recon Loss: {}, Val KL Loss: {}".format(self.epoch, self.loss/self.count, self.recon_loss/self.count, self.kl_loss/self.count)

    def wandb_log(self):
        wandb.log({"val_loss": self.loss/self.count, "val_recon_loss": self.recon_loss/self.count, "val_kl_loss": self.kl_loss/self.count})



