import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import kl_divergence

from auton_survival.models.dsm.dsm_torch import DeepSurvivalMachinesTorch
from auton_survival.models.dsm import DeepSurvivalMachines

class DeepSurvivalMachinesVAETorch(DeepSurvivalMachinesTorch):
    """A Torch implementation of Deep Survival Machines model with VAE.
  This is an implementation of Deep Survival Machines model with VAE in torch.
  It inherits from the DeepSurvivalMachinesTorch class in autonlab and reused
  some code in it. The class extend the DSM model to be a Categorical VAE model.
  In addition to compute survival rate, this model also contains logics form
  computing VAE loss, and we will use the total loss to train the model.
  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  n: int
      The number of underlying parametric distributions.
  latent_dim: int
      The dimension of latention variable
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  dist: str
      Choice of the underlying survival distributions.
      Only 'Weibull' is supposed to use for our model
  tau: float
      The tempurature used for Gumbel-Softmax sampling
  temperature: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.
  """

    def __init__(self, inputdim, k, n=3, latent_dim=20, layers=None, dist='Weibull',
                 temperature=1000., tau=1., discount=1.0, optimizer='Adam',
                 risks=1):
        super(DeepSurvivalMachinesVAETorch, self).__init__(
            inputdim, k, layers, dist, temperature, discount, optimizer, risks)

        self.latent_dim = latent_dim  # latent variable demension
        self.n = n  # no. of latent variable distributions
        self.tau = tau

        last_dim = layers[-1] if self.layers else inputdim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(last_dim, self.latent_dim * self.n, bias=True),
            nn.ReLU()
        )

        # Decoder (Reverse of Encoder & Embedding)
        modules = [nn.Linear(self.latent_dim * self.n, last_dim, bias=False)]
        prev_dim = last_dim
        for hidden in self.layers[:-1][::-1] + [inputdim]:
            modules.append(nn.ReLU6())
            modules.append(nn.Linear(prev_dim, hidden, bias=False))
            prev_dim = hidden
        self.decoder = nn.Sequential(*modules)

    def forward(self, x, risk='1'):
        # Create embedding of the input features
        embed_x = self.embedding(x)
        batch = x.shape[0]

        # Gumbel-Softmax to get latent variable
        enc = self.encoder(embed_x)
        self.hidden = enc = enc.view(enc.size(0), self.n, self.latent_dim)
        self.latent = F.gumbel_softmax(enc, tau=self.tau)

        # Reshape the latent layer & Reverse the embedding to reconstruct x hat
        x_hat = self.decoder(self.latent.view(-1, self.n * self.latent_dim))

        self.vae_loss = self.loss_fn(x, x_hat)

        return (
            self.act(self.shapeg[risk](embed_x)) + self.shape[risk].expand(batch, -1),
            self.act(self.scaleg[risk](embed_x)) + self.scale[risk].expand(batch, -1),
            self.gate[risk](embed_x) / self.temp,
        )

    def get_shape_scale(self, risk='1'):
        return (self.shape[risk], self.scale[risk])

    def loss_fn(self, x, x_hat):
        """
        Total Loss = Reconstruction Loss + KL Divergence
        x = input to forward()
        x_hat = output of forward()
        Reconstruction Loss = binary cross entropy between inputs and outputs
        KL Divergence = KL Divergence between gumbel softmax distributions with
                        self.hidden and uniform log-odds
        """

        # Reconstruction Loss
        MSE = F.mse_loss(x_hat, x)
        MSE = MSE.mean()

        q_y = F.log_softmax(self.hidden, dim=-1)  # convert hidden layer values to probabilities
        posterior_dist = torch.distributions.Categorical(logits=q_y)
        prior_dist = torch.distributions.Categorical(probs=torch.ones_like(q_y) / self.latent)

        KL = kl_divergence(posterior_dist, prior_dist)
        KL = KL.sum(-1)

        # total loss = reconstruction loss + KL Divergence
        loss = MSE + torch.mean(KL)

        return loss

class DSMVAE(DeepSurvivalMachines):
    def __init__(self, k=3, n=3, latent_dim=20, layers=None, distribution="Weibull",
                 temperature=1000., tau=1., discount=1.0, random_seed=0):
        super(DSMVAE, self).__init__(k, layers, distribution, temperature, discount, random_seed)
        self.tau = tau
        self.latent_dim = latent_dim
        self.n = n

    def _gen_torch_model(self, inputdim, optimizer, risks):
        """Helper function to return a torch model."""
        return DeepSurvivalMachinesVAETorch(inputdim,
                                            k=self.k,
                                            n=self.n,
                                            latent_dim=self.latent_dim,
                                            layers=self.layers,
                                            dist=self.dist,
                                            temperature=self.temp,
                                            tau=self.tau,
                                            discount=self.discount,
                                            optimizer=optimizer,
                                            risks=risks)