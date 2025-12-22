# LogP Factor Learning

## Overview

This project investigates the application of deep learning techniques to predict the lipophilicity (logP) of chemical compounds. Specifically, it proposes a **Contextual Atom Scalar Multi-Layer Perceptron (MLP)** model designed to learn the atomic contributions to the global logP value. Unlike traditional additive models that assign fixed values to atom types, this approach posits that an atom's contribution is dynamically influenced by its molecular context.

## Methodology

### Model Architecture: Contextual Atom Scalar MLP

The core architecture, defined in `src/mlp_regressor/mlp.py`, operates on the premise that the lipophilicity of a molecule is the sum of its atomic contributions. However, these contributions are modeled as functions of both the local atomic environment and the global molecular structure.

The model consists of three primary components:

1.  **Atom Encoder:** A feed-forward neural network that projects raw atom features into a latent representation.
2.  **Molecular Context Module:** A global pooling mechanism (mean pooling) that aggregates individual atom encodings into a single vector representing the entire molecule. This vector is further processed to extract high-level molecular features.
3.  **Scalar Predictor:** A final network that predicts a scalar value for each atom. It takes as input the concatenation of the local atom encoding and the broadcasted global molecular context. This ensures that the predicted contribution of each atom is conditioned on the entire molecule.

The final predicted logP is calculated as the sum of these atomic scalars.

### Dataset

The project utilizes the Lipophilicity dataset (`data/lipophilicity.csv`), consisting of experimental logP values for a diverse set of molecules. The data is split into training, validation, and testing sets (`data/splits/`).

### Evaluation

The model's performance is evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Crucially, the model is benchmarked against the standard RDKit (Crippen-Wildman) logP calculation. Analysis is stratified by molecular weight to assess the model's robustness on larger, more complex molecules (MW > 500 Da), where traditional methods often exhibit higher variance.

## Project Structure

*   `main.py`: The primary entry point for training and evaluating the model.
*   `src/mlp_regressor/`: Contains the model definition (`mlp.py`), data handling (`data.py`), and training logic (`training.py`).
*   `src/mp_graph/`: Contains graph-based featurization and potential graph neural network implementations.
*   `src/utils/`: Utility functions for visualization and data processing.
*   `data/`: Stores the raw dataset and processed splits.
*   `checkpoints/`: Directory for saving trained model artifacts.

## Usage

It is recommanded to follow the jupyter notebook at `notebooks/` to train the model and evaluate the performance in interactive way.

Dependencies are listed in `environment.yml` (for Conda) and `requirements.txt` (for pip).
