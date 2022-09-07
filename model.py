from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
    def forward(self, x):
        return x

class decoderBlock(nn.Module):
    """expands input by 2x"""
    def __init__(self, in_channels, out_channels,num_layers=2):
        super(decoderBlock, self).__init__()
        
        layers=[nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()]
        for i in range(num_layers-1):
            layers.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        
        self.block=nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view([-1]+list(self.shape))

class VAE(nn.Module):
    def __init__(self, de_norm:nn.Module,latent_dim=32, in_channels=1, out_channels=1,encoder="ResNet18",reshape_size=(512,1),
                 decoderBlock_channels=[512,256,128,64],decoderBlock_layers=[4,4,4,4],beta=1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.make_encoder(encoder)

        self.fcMu = nn.Linear(512, latent_dim)
        self.fcSigma=nn.Linear(512, latent_dim)

        decoder_layers=[nn.Linear(latent_dim,reshape_size[0]),Reshape(reshape_size),de_norm]
        Blockin_channels=reshape_size[0]
        for i,decoderBlock_layer in enumerate(decoderBlock_layers):
            decoder_layers.append(decoderBlock(Blockin_channels,decoderBlock_channels[i],decoderBlock_layer))
            Blockin_channels=decoderBlock_channels[i]
        decoder_layers.append(nn.ConvTranspose2d(Blockin_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        decoder_layers.append(nn.Tanh())
        self.decoder=nn.Sequential(*decoder_layers)

        self.beta=beta
    
        
    def make_encoder(self,encoder):
        if encoder=="ResNet18":
            self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            self.encoder.fc = identity()
            self.encoder.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            print("No encoder found")

    def encode(self,x):
        return self.encoder(x)
            
    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    
    def forward(self,x):
        x=self.encode(x)
        mu=self.fcMu(x)
        logvar=self.fcSigma(x)
        z=self.reparameterize(mu,logvar)
        return self.decode(z),mu,logvar

    def kld_loss(self,mu,logvar):
        kld_loss= torch.mean(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp(), dim = 1),dim=0)
        return kld_loss
    

        