# PSCC-Net
This repo contains the official codes for our paper:

### PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization
[Xiaohong Liu](https://jhc.sjtu.edu.cn/~xiaohongliu/), [Yaojie Liu](https://yaojieliu.github.io/), [Jun Chen](http://www.ece.mcmaster.ca/~junchen/), [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)

![plot](./architecture.png)

Accepted in _IEEE Transactions on Circuits and Systems for Video Technology_
___

## Training Dataset Generation
Since there is no standard IMDL dataset for training, a synthetic dataset is built to train and validate our PSCCNet. This dataset includes four classes. More details and download link can be found below.

Download: [Baidu Cloud](https://pan.baidu.com/s/1jbpPNp7UtnKo9OnZeEm1Yw), password: js74

### 1. Splicing
The splicing dataset contains two parts. One is the content-aware splicing, and the other one is the random-mask splicing.
For the former one, MS COCO dataset is used, where one annotated region is randomly
selected per image, and pasted into a different image after
several transformations. For the latter one, the images from KCMI, VISION, and Dresden are randomly selected as donor and target images. 
where the Bezier curve is adopted to generate random contours.
In total, we generate 76,583 content-aware splicing images, and 40,000 random-mask splicing ones.

### 2. Copy-move
For copy-move dataset, we uncompress the [USCISI-CMFD](https://github.com/isi-vista/BusterNet/tree/master/Data/USCISI-CMFD-Small) and generate the corresponding binary masks instead of the given tenary masks.
In total, we have 100,000 copy-moved images for training.

### 3. Removal
For removal dataset, the inpating method, [RFR-Net](https://github.com/jingyuanli001/RFR-Inpainting), is used to fill one
annotated region that is randomly removed, where the MS COCO dataset provides the source images.
In total, we generate 78,246 removal images.

### 4. Authentic
For authentic dataset, we simply select images from the MS COCO dataset. Therefore, we have 81,910 images in this class.

___
Codes and models will be released soon.

