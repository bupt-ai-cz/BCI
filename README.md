# BCI: Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix ![visitors](https://visitor-badge.glitch.me/badge?page_id=bupt-ai-cz.BCI)

## News
- ⚡(2022-4-26): We have released the code of PyramidPix2pix.
---

![datasetview_github](imgs/datasetpreview.png)

---
## Setup
### 1)Envs
- Linux
- Python>=3.6
- CPU or NVIDIA GPU + CUDA CuDNN

Install python packages
```
git clone https://github.com/bupt-ai-cz/BCI
cd PyramidPix2pix
pip install -r requirements.txt
```
### 2)Prepare dataset
- File structure
```
PyramidPix2pix
  ├──datasets
       ├── BCI
             ├──train
             |    ├── 00000_train_1+.png
             |    ├── 00001_train_3+.png
             |    └── ...
             └──test
                  ├── 00000_test_1+.png
                  ├── 00001_test_2+.png
                  └── ...
   
```
### 3)Train
Train at full resolution(1024*1024): 
```
python train.py --dataroot ./datasets/BCI --model pix2pix --direction AtoB --netG resnet_9blocks --batch_size 2 --preprocess none --gpu_ids 0
```
### 4)Test
Test at full resolution(1024*1024): 
```
python test.py --dataroot ./datasets/BCI --model pix2pix --direction AtoB --netG resnet_9blocks --preprocess none --gpu_ids 0 --num_test 1000
```
