import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Sequence, Callable

from flax import linen as nn  # Linen API



#####################
### ARCHITECTURES ###
#####################

class BaseNeuralnet(nn.Module):
    """Abstract base class. Needs layer sizes and activation function used"""
    layer_sizes: Sequence[int]
    act_func: Callable = nn.relu
    
    def setup(self):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError 
    
class MLP(BaseNeuralnet):
    """Basic multi-layer perceptron: a feedforward neural network with multiple Dense layers."""

    def setup(self):
        self.layers = [nn.Dense(n) for n in self.layer_sizes]

    @nn.compact
    def __call__(self, x: Array):
        """_summary_

        Args:
            x (Array): Input data of the neural network.
        """

        for layer in self.layers[:-1]:
            # Apply the linear part of the layer's operation
            x = layer(x)
            # Apply the given activation function
            x = self.act_func(x)

        x = self.layers[-1](x) # for the output layer only apply the linear part
        return x

class Encoder(nn.Module):
    layer_sizes: Sequence[int]
    act_func: Callable = nn.relu

    def setup(self):
        self.mu_layers = [nn.Dense(n) for n in self.layer_sizes]
        self.logvar_layers = [nn.Dense(n) for n in self.layer_sizes]

    @nn.compact
    def __call__(self, y: Array):

        mu = y.copy()
        for layer in self.mu_layers[:-1]:
            mu = layer(mu)
            mu = self.act_func(mu)
        mu = self.mu_layers[-1](mu)

        logvar = y.copy()
        for layer in self.logvar_layers[:-1]:
            logvar = layer(logvar)
            logvar = self.act_func(logvar)
        logvar = self.logvar_layers[-1](logvar)
        return mu, logvar

class Decoder(MLP):

    @nn.compact
    def __call__(self, z: Array):
        for layer in self.layers[:-1]:
            # Apply the linear part of the layer's operation
            z = layer(z)
            # Apply the given activation function
            z = self.act_func(z)

        z = self.layers[-1](z) # for the output layer only apply the linear part
        return z


class CVAE(nn.Module):
    """Conditional Variational Autoencoder consisting of an Encoder and a Decoder."""
    hidden_layer_sizes: Sequence[Int] # used for both the encoder and decoder
    output_size: Int
    z_dim: Int = 20

    def setup(self):
        self.encoder = Encoder([*self.hidden_layer_sizes, self.z_dim])
        self.decoder = Decoder(layer_sizes = [*self.hidden_layer_sizes[::-1], self.output_size], act_func=nn.relu)
    
    def __call__(self, y: Array, x: Array, z_rng: jax.random.PRNGKey):
        y = jnp.concatenate([y, x.copy()], axis = -1)
        mu, logvar = self.encoder(y)
    
        # Reparametrize
        std = jnp.exp(0.5* logvar)
        eps = jax.random.normal(z_rng, logvar.shape)
        z = mu + eps * std

        z_x = jnp.concatenate([z, x.copy()], axis = -1)
        reconstructed_y = self.decoder(z_x)
        return reconstructed_y, mu, logvar

class CNN(nn.Module):
    """Convolutional Neural Network"""
    dense_layer_sizes: Sequence[Int]
    kernel_sizes: Sequence[Int]
    conv_layer_sizes: Sequence[Int]
    output_shape: tuple[Int, Int]
    spatial: Int = 32
    act_func: Callable = nn.relu

    def setup(self):
        if self.dense_layer_sizes[-1] != self.conv_layer_sizes[0]:
            raise ValueError(f"Final dense layer must be equally large as first convolutional layer.")
        if self.conv_layer_sizes[-1] != 1: 
            raise ValueError(f"Last convolutional layer must be of size 1 to predict 2D array.")

        self.dense_layers = [nn.Dense(n) for n in self.dense_layer_sizes[:-1]]
        self.dense_layers += (nn.Dense(self.dense_layer_sizes[-1] * self.spatial**2), )  # the last dense layer should create an array that can be reshaped into spatial and chanel parts
        self.conv_layers = [nn.Conv(features = f, kernel_size = (k,k)) for f, k in zip(self.conv_layer_sizes, self.kernel_sizes)]

    def __call__(self, x: Array):
        # Apply the dense layers
        for layer in self.dense_layers:
            x = layer(x)
            x = self.act_func(x)
        
        x = x.reshape((-1, self.spatial, self.spatial, self.dense_layer_sizes[-1]))
        for layer in self.conv_layers[:-1]:
            x = layer(x)
            x = self.act_func(x)
        
        x = self.conv_layers[-1](x) # only apply convolution part of last convolutional layer
        x = x[:,:,:,0]
        x = jax.image.resize(x, shape = (x.shape[0], *self.output_shape), method = "bilinear") # resize the NN output to the desired output
        return x