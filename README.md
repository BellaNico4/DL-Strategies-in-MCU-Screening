# Deep Learning Strategies for Labeling and Accuracy Optimization in Microcontroller Performance Screening

## Introduction

This repository contains the implementation of the deep learning-based framework described in the paper "Deep Learning Strategies for Labeling and Accuracy Optimization in Microcontroller Performance Screening" by Nicolo' Bellarmino, Riccardo Cantoro, Martin Huch, Tobias Kilian, Ulf Schlichtmann, and Giovanni Squillero. Published in 2024 in the *Transactions on Computer-Aided Design*. 

Our framework leverages Semi-Supervised Learning (SSL) and Transfer Learning to optimize the performance screening of microcontrollers (MCUs). The approach reduces the prediction error and minimizes the need for labeled samples, enhancing the efficiency of the MCU characterization phase and data collection.

## Installation

To use the models and run the example provided, please follow these steps:

1. Clone this repository:
    ```sh
    git clone https://github.com/BellaNico4/TRANSFER_LEARNING_JOURNAL_PAPER
    cd TRANSFER_LEARNING_JOURNAL_PAPER
    ```

2. Create and activate a virtual environment with conda:
    ```sh
    conda create --name <env> --file requirements.txt
    conda activate <env>
    ```

## Usage

The notebook (*smons_transfer_learning*) provide a minimal working example on artificial data sample. Target y is a linear combination of polynomial transformation of the input features X.

### Example 1: Performance Prediction for Product A1

This example demonstrates how to train a deep feature extractor to predict the performance of MCU product A1 and save the checkpoint.

### Example 2: Transfer Learning on A2

This example shows the application of transfer learning to predict the performance of a new MCU product A2 using a pre-trained model on product A1.

### Example 3: Transfer Learning and features selection

Benchmark of several ML models on A2, with class to select random or specific columns in order to use the pretrained models.


## Model Details

The framework is designed to:
- Reduce the number of labeled samples required by up to a factor of six.
- Use deep neural networks as feature extractors.
- Leverage transfer learning to adapt models to new products and families.
- Minimize prediction error and guardband, enhancing process yield.

For more details on the model architecture and training process, refer to the paper [Deep Learning Strategies for Labeling and Accuracy Optimization in Microcontroller Performance Screening](https://doi.org/XXXXXX).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this code in your research, please cite the paper:

```
@article{bellarmino2024,
  title={Deep Learning Strategies for Labeling and Accuracy Optimization in Microcontroller Performance Screening},
  author={Nicolo' Bellarmino and Riccardo Cantoro and Martin Huch and Tobias Kilian and Ulf Schlichtmann and Giovanni Squillero},
  journal={Transactions on Computer-Aided Design},
  volume={X},
  number={Y},
  pages={Z},
  year={2024},
  publisher={IEEE}
}
```

## Contact

For any questions or comments, please contact Nicolo' Bellarmino at [nicolo.bellarmino@polito.it].
```