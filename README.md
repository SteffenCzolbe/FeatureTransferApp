# FeatureTransferApp
Image editing based on a Variational Autoencoder.

For usage and explanation consult this youtube video:
[![Video](https://img.youtube.com/vi/uszj2MOLY08/hqdefault.jpg)](https://youtu.be/uszj2MOLY08)


# Installation
Install guide for ubuntu based systems. Windows/Mac can deviate slighty.

## Create and activate virtual enviroment (optional but recommended)
This will keep changes to your python installation contained in the virtual enviroment.

```
python3 -m venv venv-feature-transfer
source venv-feature-transfer/bin/activate
```

## Install
Some libraries used require additional build tools. Install them with
```
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev
```

Install python packages listed with

```
pip3 install -r requirements.txt
```

## Execute
Start the program with
```
python3 main.py
```
If you used a virtual environment to install the dependencies, you need to activate it with `source venv-feature-transfer/bin/activate` before each program start.

# Add your own Images
To add your own images, place them in the directory `additional_images`. All common image file formats are accepted. Images should only contain one face. On program startup, the images are preprocessed. Sucessfully processed images are placed in the directory `additional_images_preprocessed`. If no face / more than one face is found in the image, it is ignored.
