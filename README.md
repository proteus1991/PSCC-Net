# PSCC-Net
This repo contains the official codes for our paper:

### PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization
[Xiaohong Liu](https://jhc.sjtu.edu.cn/~xiaohongliu/), [Yaojie Liu](https://yaojieliu.github.io/), [Jun Chen](http://www.ece.mcmaster.ca/~junchen/), [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)

![plot](./architecture.png)

Accepted in _IEEE Transactions on Circuits and Systems for Video Technology_
___

## Training Dataset Generation
Since there is no standard IMDL dataset for training, a synthetic dataset is built to train and validate our PSCC-Net. This dataset includes four classes. More details are shown as follows. The codes for dataset generation are provided in ```dataset_generation/```, and the download link can be found below.

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
## Test Dataset Illustration
In total, *six* datasets are used to test the PSCC-Net. 
For localization, we use Columbia, Coverage, CASIA, NIST16, and IMD20. 
As for detection, we build a dataset named CASIA-D. 
We will provide more details below.

### 1. Columbia

It consists of 180 spliced images (Download link: [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)). Note that you need to generate the binary splicing mask by your own from the provided tricolor masks.
This dataset is only used to test the pre-trained model, and not *split*.

### 2. Coverage
It consists of 100 copy-moved images (Download link: [Coverage](https://github.com/wenbihan/coverage)). 
To test the pre-trained model, all images in this dataset are used.
As for fine-tuning, it is split into 75/25 for training and testing. The name list of test images in fine-tuning is saved in ```dataset/test/Coverage```.

### 3. CASIA
It has two versions namely, CASIA v1.0 and v2.0 (Download link: [v1.0](https://github.com/namtpham/casia1groundtruth), [v2.0](https://github.com/namtpham/casia2groundtruth)), composed of spliced and copymoved images, where v1.0 has 921 images and v2.0 has 5123 images. For pre-trained model, we sum up images in both v1.0 and v2.0 for testing. 
To fine-tune the pre-trained model, v2.0 is used for training, and v1.0 is used for testing.

### 4. NIST16
To download the NIST16 dataset, we need to first complete the license agreement via signing up on this [website](https://mfc.nist.gov/).
Then, we can download the dataset via the provided link, usually sent by email.
For pre-trained model, we use in total 564 forged images for testing. To fine-tune the pre-trained model, it is split into 404/160 for training and testing.
The needed name list is provided in ```dataset/test/NIST16```.

### 5. IMD20
It is composed of 2010 real-life forged images collected from Internet (Download link: [IMD20](http://staff.utia.cas.cz/novozada/db/)). All images are used to test the pre-trained model, and no fine-tuning is adopted.

### 6. CASIA-D
Since there is no standard dataset for testing the detection performance, we build a dataset named CASIA-D that consists of 1842 images with 50% forged and 50% pristine from CASIA v1.0 and v2.0.
The name list of these figures can be found in ```dataset/test/CASIA-D/```.

___
## Testing
To test the PSCC-Net, simply run ```test.py```. It will probe the images in the ```sample``` folder that contains 6 authentic images and 6 forged images including splicing, copy-move, and removal manipulations.

> You can also put other images in this folder for testing.
___
## Training
For training, you need to first download the generated training dataset and put them into the ```dataset``` folder correspondingly. We provide the ```README``` file in each sub-folder for a guidance.
Subsequently, run ```train.py``` to retrain the PSCC-Net.

> All hyper-parameters about training are stored in ```utils/config.py```. You may modify them to accommodate your system. Besides, the weights of PSCC-Net are stored in the ```checkpoint``` folder.

___
## Citations
If PSCC-Net helps your research or work, please kindly cite our paper. The following is a BibTeX reference.
```
@article{liu2022pscc,
  title={PSCC-Net: Progressive spatio-channel correlation network for image manipulation detection and localization},
  author={Liu, Xiaohong and Liu, Yaojie and Chen, Jun and Liu, Xiaoming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
```


