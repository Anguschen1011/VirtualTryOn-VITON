# Virtual Try-On Using VITON Dataset  

## Environments  
```
conda env create -f environment.yaml
```
```
conda activate vton
```

## 1. Data Preprocessing
Due to flaws in the original method for generating 'agnostic', which stem from issues within the dataset itself, "Self Correction Human Parsing" [[Github]](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) was used to reprocess the human images for human parsing.  

Additionally, since there were some errors in the original cloth-mask detection, "SAM 2: Segment Anything in Images and Videos" [[Github]](https://github.com/facebookresearch/segment-anything-2) using the sam2_hiera_large model was employed to regenerate the cloth masks.

If you would like to change the data, weights, output path or other settings,   
you can find them in ```config.py```.

## 2. Setting Data's Root 
Get a .pkl file contain data's path 
```
python 001_make_Dataset.py
```  

## 3. Training
Start training 
```
python 002_train_gmm.py
```

## 4. Testing 
Start testing
```
python 003_test.py
```

## 5. Evaluation
```
python 004_evaluation.py
```

```
=======================
SSIM score  :  0.7907
LPIPS score :  0.2130
FID score   : 12.7824
=======================
```  

## 6. Hardware
The model architectures proposed in this study are implemented using the PyTorchDL framework, and training is conducted on hardware featuring an Intel® Core™ i7-12700 CPU and Nvidia RTX 3060 12GB graphics processing unit (GPU).

# References： 
> * 柯良頴, 夏至賢, 許良亦, 陳麒安, "Virtual Try-on Based on Composable Sequential Appearance Flow," ITAC, 2024.
> [[Virtual Try-on Based on Composable Sequential Appearance Flow]](https://github.com/Anguschen1011/VirtualTryOn-VITON-V1)
> * P. Li, Y. Xu, Y. Wei and Y. Yang, "Self-Correction for Human Parsing," _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 44, no. 6, pp. 3260-3271, 2022.
>[[Self Correction Human Parsing]](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)  
> * N. Ravi, et al., "Sam 2: Segment anything in images and videos," _ArXiv_, 2024.
>[[StyleGAN-Human]](https://github.com/facebookresearch/segment-anything-2)
