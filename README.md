# fiesta 🎉

`fiesta`: **F**ast **I**nference of **E**lectromagnetic **S**ignals and **T**ransients with j**A**x

![fiesta logo](docs/fiesta_logo.jpeg)

**NOTE:** `fiesta` is currently under development. We have some basic documentation available under `./docs`. Feel free to contact us for any questions.

## Installation

`fiesta` can be installed from pip via 
```
pip install fiesta
```

Alternatively you can install it directly from source by cloning 
```
git clone git@github.com:nuclear-multimessenger-astronomy/fiestaEM.git
```
and then run 
```
pip install -e .
```
in the cloning directory. Note, that by default only the cpu version of `jax` is installed. If you want to use GPU acceleration, install `jax[cuda12]` as indicated on the [`jax` webpage](https://docs.jax.dev/en/latest/installation.html#installation).



## Training surrogate models

To train your own surrogate models, have a look at some of the example scripts in the repository for inspiration. You can find them under `./surrogates/GRB/` and `./surrogates/KN/` in the respective model folders. The example section on training is currently work in progress. 

## Examples

We have example scripts for running an inference on AT2017gfo + GRB170817A. They can be found in `./examples/inference/`. We also plan to add an example section on training surrogates in the future.

## Acknowledgements

The logo was created by [ideogram AI](https://ideogram.ai/). 
