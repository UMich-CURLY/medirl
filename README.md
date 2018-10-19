# vehicle-motion-forecasting
## Work in progress.

This repository contains the training and inference code used in our paper *[Integrating kinematics and environment context into deep inverse reinforcement learning for predicting off-road vehicle trajectories](https://arxiv.org/abs/1810.07225)*. This work proposes a inverse reinforcement learning based framework that infers the reward structure and forecasts the vehicle's motion.

[![video](https://img.youtube.com/vi/nuJjIdEEDBk/0.jpg)](https://www.youtube.com/watch?v=nuJjIdEEDBk)

## Installation
### conda
We recommend using *[conda](https://conda.io/docs/)* to install dependencies with the `environment.yml` provided in this repository.
```bash
conda env create -f environment.yml
source activate vehicle_motion_forecasting
```

### pip
You can also use `pip` to install dependencies with the `requirements.txt` provided.
```bash
pip install -r requirements.txt
```

## Inference
We provide the trained weights and example data for inference. Please check `demo.ipynb`.
```bash
jupyter notebook demo.ipynb
```

## Training
Training examples will be made available later after we open source the dataset.

## Citation
Please consider citing the corresponding publication.
