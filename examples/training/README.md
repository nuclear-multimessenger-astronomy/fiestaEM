This directory contains two example scripts on how to train a surrogate that predicts the spectral flux density. 
One script uses a simple feed-forward neural network, the other the cVAE architecture.

Training data is provided in the ``data`` directory, however it is only heavily reduced version of the training data set for the afterglowpy tophat jet model. 
The full data file is several GB and can be made available upon request to hauke.koehn@uni-potsdam.de.

Since the dataset is insufficient to train a valid surrogate, these scripts should be seen as mere test cases that show which methods and objects to use for the training process.
The training scripts for the built-in surrogates can be found under the ``surrogates`` directory in the main repo.