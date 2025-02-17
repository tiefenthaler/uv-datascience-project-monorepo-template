# Lit Auto Encoder

This package is a side package which is part of a monorepo setup within the "uv-datascience-project-monorep-template" project.
The package is meant to be a stand alone package that can be used in other projects, but which is maintained within the monorepo.
The reason for this is to have a single source of truth for all packages and to have a single place to manage all dependencies, including the dependencies of the packages themselves.
The main intension for this set up is to facilitate collaboration, development and maintainability of multiple packages in the monorepo.

This package contains the code for training an autoencoder using PyTorch Lightning. The project includes the following components:

## Custom Code in src Folder

The `src` folder contains the custom code for the machine learning project. The main components include:

### lit_auto_encoder.py

This file defines the `LitAutoEncoder` class, which is a LightningModule for an autoencoder using PyTorch Lightning. The `LitAutoEncoder` class includes:

1. An `__init__` method to initialize the encoder and decoder.
2. A `training_step` method to define the training loop.
3. A `configure_optimizers` method to set up the optimizer.

### train_autoencoder.py

This file defines the training function `train_litautoencoder` to initialize and train the model on the MNIST dataset using PyTorch Lightning.

## Tests

The `tests` folder contains unit tests for the custom code in the `src` folder. The tests include:

