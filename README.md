# Setup
Need to run the following command to install the `neuralodes` package in edit mode.

```
pip install -e .
```

# Cityscapes Benchmarking 

To run the benchmark on the Cityscapes dataset, you need to create a directory named dataset in the Cityscapes directory, download the dataset from: https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs by pressing the "Download" button, into the dataset directory, and unzip the folder. Then to train the Resnet-UNet you need to run python3 train.py --Model ResNetUNet and to train teh ODEUnet you need to use the option --Model ODEUnet. 

The UNet architecture has been taken from the repo https://github.com/usuyama/pytorch-unet

The Cityscapes benchmarking skeleton was taken from: https://www.kaggle.com/code/levusik/semantic-segmentation-pytorch-unet-enet-esnet/notebook