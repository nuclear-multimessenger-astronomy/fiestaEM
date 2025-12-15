# fiesta 🎉

`fiesta`: **F**ast **I**nference of **E**lectromagnetic **S**ignals and **T**ransients with j**A**x

![fiesta logo](docs/fiesta_logo.jpeg)

**NOTE:** `fiesta` is currently under development. We have some basic documentation available under `./docs`. Feel free to contact us for any questions.

## Installation

`fiesta` can be installed from pip via 
```
pip install fiestaEM
```

Alternatively you can install it directly from source by cloning 
```
git clone git@github.com:nuclear-multimessenger-astronomy/fiestaEM.git
```
and then run 
```
pip install -e .
```
in the cloning directory. 

Note, that by default only the cpu version of `jax` is installed. If you want to use GPU acceleration, run 
```
pip install fiestaEM[gpu]
```
or install `jax[cuda12]` as indicated on the [`jax` webpage](https://docs.jax.dev/en/latest/installation.html#installation) manually.

Also, due to the file size limit on pypi, the pypi distribution only contains the most important built-in surrogates. If you want all built-in surrogates, we recommend *editable* installation from source or to download the `.pkl` files manually and store them in the `surrogates` folder of the site-package.
You can check which built-in surrogates are available by running 
```
python -c "from fiesta.inference.lightcurve_model import list_built_in_surrogates; list_built_in_surrogates()"
```


## Training surrogate models

To train your own surrogate models, have a look at some of the example scripts in the repository for inspiration. You can find them under `./surrogates/GRB/` and `./surrogates/KN/` in the respective model folders. The example section on training is currently work in progress. 

## Examples

The `./examples/training/` directory contains scripts that show how to use the `fiesta` API to train a flux density surrogate either with a CVAE architecture or with a simple feed-forward NN.
Mock training data is provided as well, but note that the data set is reduced significantly and thus you will not be able to get a decent surrogate from it. 
If you want to use our training data, please contact us so we can figure out a way how to deliver the heavy files (> 10GB) to you.
We also have example scripts for running an inference on AT2017gfo + GRB170817A. They can be found in `./examples/inference/`. 
Note that for all of these example scripts it is highly recommended to use GPU-acceleration, since otherwise the runtime will rather long.

## Acknowledgements

The logo was created by [ideogram AI](https://ideogram.ai/). 
