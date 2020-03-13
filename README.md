# ST-ReIDNet
![](https://i.imgur.com/BrNvwt0.jpg)
[[paper]](https://) 

Hsuan-I Ho, Minho Shim, and Dongyoon Wee, "Learning from Dances: Pose-Invariant Re-Identification for Multi-Person Tracking", ICASSP 2020.

This is the offical Pytorch implementation for our work. Our proposed framework is able to tackle the issue of losing tracks in multi-person tracking by performing pose-invariant human re-ID. Please also refer to our newly collected dance videos re-ID dataset [DanceReID](https://github.com/azuxmioy/DanceReID).

## Prerequisites

```
pip3 install torch torchvision Pillow opencv-python \
     matplotlib visdom scikit-image scikit-learn dominate h5py scipy==1.1.0
```

**Note**: We implemented our code under version 1.0.1. Subtle modifications might be needed if your Pytorch version is not matched.
 
## Datasets preparation

In our experiment, we perform analysis and comparison on [Market1501](http://www.liangzheng.org/Project/project_reid.html), [DukeMTMC-ReID](https://github.com/layumi/DukeMTMC-reID_evaluation), and our proposed [DanceReID](https://github.com/azuxmioy/DanceReID). The datasets have been preprocessed and followed the implementation in [open-reid](https://github.com/Cysu/open-reid). Please follow the step to prepare each dataset respectively.

### DanceReID
See [DanceReID](https://github.com/azuxmioy/DanceReID) repository for preparing the data.

### Market1501 & DukeMTMC-ReID
We exploit the same data used in [FD-GAN](https://github.com/yxgeee/FD-GAN#datasets), please refer to their repository.


## Running the code

### Running baseline model

The baseline model (ResNet-50) use only color images as input, pose information is not required.
```
CUDA_VISIBLE_DEVICES=0,1 python3 baseline.py \
    -b 128 -j 8 -d DanceReid --split 0 --combine-trainval \
    -a resnet50 --emb-type Single \
    --soft-margin --lambda-tri 1.0 --lambda-cla 1.0 \
    --lr 5e-5 --epochs 100 --step-size 40 --eval-step 10 \
    --inst-mode \
    --last-stride 1 \
    --use-bn \
    --test-bn \
    --label-smoothing \
    --eraser \
    --data-dir /path/to/dataset/directory \
    --logs-dir /path/to/saving/directory
```

We've implemented several training options of the baseline model:
* **combine-trainval**: using valiation data in the training stage
* **emb-type**: either using a classifier or siamese net as re-ID classifier. Default is classifier(Single).
* **soft-margin**: using soft margin when computing triplet loss. If not using this option, one should specify **margin** value.
* **inst-mode**: sampling triplet in a instance-based sampling manner. If not using this flag, spatial temproal sampling will be perform.
* **last-stride**: modify the last stride number of ResNet-50 backbone before global pooling.
* **use-bn**: using batch norm after feature extraction.
* **test-bn**: using bathnormed features for evaluation.
* **label-smoothing**: use label smoothing for computing classification loss.
* **eraser**: erase random rectangle at training images for data augmentation. 


### Training ST-ReIDNet 

Our ST-ReIDNet use both color images and pose information for training. We note that only color images are available during inference.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py \
    -b 24 -j 8 -d DanceReid --name (your_session_name) \
    --lr 2e-5 --stage 0 --inter-rate 0.7 --skip-frame 10 \
    --niter 50 --niter-decay 50 --save-step 50 --eval-step 10 \
    --display-port 6060 --display-freq 5000 --display-id 1 --display-single-pane-ncols 5 \
    --pose-aug gauss --smooth-label --soft-margin --batch-hard --emb-type Single \
    --last-stride 1 --emb-smooth --eraser --mask \
    --lambda-recon 10.0 --lambda-tri 1.0 --lambda-class 1.0 \
    --lambda-d 0.1 --lambda-dp 0.1 \
    --dataroot /path/to/dataset/directory \
    --checkpoints /path/to/saving/directory
```
Similar to the baseline model, we have implemented several training options.

* **last-stride**: modify the last stride number of ResNet-50 backbone before global pooling.
* **emb-smooth**: use label smoothing for computing classification loss.
* **eraser**: erase random rectangle at training images for data augmentation. 
* **mask**: generate a skeleton mask for the image recovery purpose. The pixels within the mask would have a higher weight of the reconstruction loss. 
* **visualization**: by setting`--display-id` > 0, you can visualize the generated results via vidsom. Note that you need to open a new command line prompt and run `python3 -m visdom.server -port=(your_port)` before running the main program, where the port number should be identical to `--display-port`.


## Acknowledgements

Our code is built on top of the great projects [open-reid](https://github.com/Cysu/open-reid) and [FD-GAN](https://github.com/yxgeee/).