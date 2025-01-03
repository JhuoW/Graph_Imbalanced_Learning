import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self,encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, conditional_size=0) -> None:
        super(VAE, self).__init__()

        if conditional:
            assert conditional_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size)

    def forward(self, x, condition):
        means, log_var = self.encoder(x, condition)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, condition)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return means + eps * std
    
    def inference(self, z, condition):
        recon_x = self.decoder(z, condition)
        return recon_x
    

class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, conditional_size) -> None:
        super(Encoder, self).__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += conditional_size   #  因为输入是中心节点和它的邻居
        
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
             self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
             self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        
        self.lin_means = nn.Linear(layer_sizes[-1], latent_size)
        self.lin_log_var = nn.Linear(layer_sizes[-1], latent_size)


    def forward(self, x, condition): # 要重构x
        c = torch.mean(condition, dim=0)
        x_c = torch.cat((x, c), dim = -1)

        h = self.MLP(x_c)
        means = self.lin_means(h)   # 基于节点和其邻居生成的均值 
        log_vars = self.lin_log_var(h) # 基于节点和其另据生成的方差

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):
        '''
        layer_size = [256, D]
        '''
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size + conditional_size
        
        # mlp: (input_size, layer_sizes[0])  (layer_sizes[0], layer_size[1])
        for i, (in_size, out_size) in enumerate(zip([input_size]+ layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
    
    def forward(self, z, condition):
        c = torch.mean(condition, dim=0)
        x_c = torch.cat((z,c), dim = -1)

        h = self.MLP(x_c)

        return h
        

