import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import tqdm
import re
import requests
from bs4 import BeautifulSoup
import shutil
import os

def load_and_clean(path_to_tsv):
    """
    Loads the data from the tsv file and cleans it.
    """
    df = pd.read_csv(path_to_tsv, sep='\t')
    #drop measurments column because it is mostly nan
    df.drop(columns=['Measurements'], inplace=True)
    #parse the URI column to get only the url
    df['URI'] = df['URI'].apply(lambda x: x[x.find("'http")+1:x.find("');")])
    return df


def save_images(vase_url,save_path="Data/Images/"):
    """
    Saves the images from the vase url.
    """
    vase_id = vase_url.split('/')[-1]
    save_path+="/"+vase_id
    os.makedirs(save_path, exist_ok=True)
    #get the html from the url
    html = requests.get(vase_url).text
    #parse the html
    soup = BeautifulSoup(html, 'html.parser')
    #get the image url from the html
    img_tags = soup.find_all('img')
    #get the image from the url
    urls = [img['src'] for img in img_tags if '.jpe' in img['src']]
    for i,image in enumerate(urls):
        ## Set up the image URL and filename
        image_url= 'https://www.beazley.ox.ac.uk'+image

        filename = f"{save_path}/{i}.jpe"

        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            
        else:
            print('Image Couldn\'t be retreived')

if __name__=="__main__":
    df=load_and_clean('export2AFECA4C997C412A93A30CCF60896F16.tsv')
    print(df.head())
    print(df.shape)
    print(df['URI'][0])
    #cut only the top 2000 rows
    df=df.head(2000)
    # os.makedirs('Data/Images', exist_ok=True)
    with mp.Pool(mp.cpu_count()) as p:
        #p.map(save_images, df['URI'])
        r = list(tqdm.tqdm(p.imap(save_images, df['URI']), total=df.shape[0]))
    print("Done")