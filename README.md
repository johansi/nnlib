# nnlib
This library implements some ready to use deep learning leaners in Pytorch for various use cases, mostly in the field of computer vision.

Current implemented learners:

- 2D landmark detection learner with heatmaps. Various backends and loss functions are implemented
- image classification learner
- image regression learner

 There are also some other functionalities around deep learning, i.e.:

 - automatic batchsize estimation
 - tools for neural net data preparation
 - image manipulation tools
 - different backends for inferencing (Pytorch, TensorRT)
 - tools for communication with SPS

# dependencies
- jupyter lab
- pytorch
- torchvision
- opencv
- fastprogressbar
- numba
- scipy
- matplotlib
- ipywidgets
- numpy
- tensorrt (only for tensorrt fast inference) -> for installation see official documents from nvidia
- Pillow 
- pynvml 
- scikit-image 
- snap7 (only for communication with SPS)
 
# installation
0. Library works best with linux based os (like ubuntu). Windows isn´t the best with Pytorch, some problems with the dataloaders from Pytorch.
1. Download installer of [anaconda](https://www.anaconda.com/products/individual) for linux 
2. Make the script executable with `chmod +x Anaconda3-2021.11-Linux-x86_64.sh`
3. Install it with `./Anaconda3-2021.11-Linux-x86_64.sh`.
4. Make new environment with `conda create --name nnlib python=3.7`
5. Activate environment `conda activate nnlib`
6. Install dependencies with following command `conda install -c fastai -c pytorch -c anaconda -c numba -c conda-forge jupyterlab pytorch torchvision opencv fastprogress numba scipy matplotlib ipywidgets numpy pillow pynvml scikit-image ipykernel`
7. Go to the root folder of the library and install it with `python setup.py install`
8. Now you can use the library
