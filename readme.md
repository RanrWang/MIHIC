 # "MIHIC dataset benchmark source codes"
 `In this project, we can perform the baseline experiments based on our self-built dataset.`
 
 ## Part 1: Dataset introduction 
 The dataset includes two components: train and test, each of which is consisted of seven categories tissue patches.
 
 All the patches size is 128*128, extracted from tissue chips of Non-small cell lung cancer patients in Liaoning cancer hospital&institute.
 

## Part 2:  Main program introduction

### train.py

Used to train several general models based on self-defined model.
    #   models：
    #   vgg16、vgg16_bn
    #   mobilenetv2
    #   googlenet
    #   resnet50
    #   vit_b_16
    #   swin_transformer_small swin_transformer_tiny
    #   swin_transformer_V2_tiny
    
All the img_size in model training and testing is (128,128). '_make_divisible function', it is used to ensure that a given value is divisible by a specified diviso.

### test.py

Used to calculate the relevant evaluation metrics of models corresponding to 'train.py'


### train_timm.py

Train more general models using the pre-trained model in timm library.
    # models:
    
    # swin_tiny_patch4_window7_224  equal to swin_transformer_tiny in train.py, but it doesn't include '_make_divisible function', 	    so we resize img_size to (224,224) to train or test.
    # coat_tiny: coat_tiny. batchsize=512. img_size can be set as (128,128) or (224,224).
    The other version perform '_make_divisible function', which is used to ensure that a given value.
    is divisible by a specified divisor, so img_size in the other version is (128,128),window_size=7.
    # convnext_tiny: convnext_tiny. batchsize=512. img_size=128. But it can also be resize to (224,224).
    # convit_tiny: convit_tiny. batchsize=512. sgd. img_size is resize to (224,224). The other parameter is default.
    # efficientnet_b0: efficientnet_b0. batchsize=512.  img_size is (128,128). But it can also be resize to (224,224).
    # hrnet_w18: hrnet_w18. batchsize=256. img_size=128.  But it can also be resize to (224,224).
    # cait: cait_s24_224. batchsize=128. img_size=224. The training time of this model is too long, so it has not been trained.
    # crossvit_tiny: crossvit_tiny_240. batchsize=512. img_size=128. 
    # crossvit_tiny_224: crossvit_tiny_240. batchsize=512. img_size=224.
    # twins_pcpvt_base: twins_papvt_base. batchsize=512. img_size can be set as (128,128)(batchsize=512) or (224,224)(batchsize=256).
    # twins_svt_base: twins_svt_base. img_size can be set as (128,128)(batchsize=512) or (224,224)(batchsize=256).
    '''
    
### test_timm.py
Used to calculate the relevant evaluation metrics of models corresponding to 'train_timm.py'

### b_test.py
Used to process the whole tissue core directly. Predict the categories of each non-overlapping patches and overlay the corresponding colors. 
  
### totxt.py
To generate the txt of train and test path.
  
  
  
  
  
