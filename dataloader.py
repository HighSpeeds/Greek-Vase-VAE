import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import glob
import PIL
from data_creator import load_and_clean
import tqdm

class VaseDataset(data.Dataset):
    """
    Custom dataset class for the vase dataset.
    """
    def __init__(self, path_to_tsv,path_to_data, reshape_size=(224,224)):
        """
        Initializes the dataset.
        """
        self.path_to_tsv = path_to_tsv
        self.path_to_data = path_to_data
        self.reshape_size = reshape_size

        self.df = load_and_clean(self.path_to_tsv)
        self.images=glob.glob(self.path_to_data+"/*/*.jpe")
        #remove bad images
        self.images.remove('Data/Images/AA477AA6-13B6-4FB1-8438-BA0E614AEEC3/0.jpe')

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):

        #import the image
        image = PIL.Image.open(self.images[idx])
        #resize the image and pad
        image = PIL.ImageOps.pad(image, self.reshape_size,color=image.getpixel((0,0)))
        #convert to grayscale
        image = PIL.ImageOps.grayscale(image)
        #convert to tensor
        image = transforms.ToTensor()(image)
        return image

#testing
if __name__ == "__main__":
    dataset=VaseDataset(path_to_tsv="export2AFECA4C997C412A93A30CCF60896F16.tsv",path_to_data="Data/Images")
    for i in tqdm.tqdm(range(len(dataset))):
        dataset[i]
