## Caffe for PVANET
by Kye-Hyeon Kim, Yeongjae Cheon, Sanghoon Hong, Byungseok Roh, Minje Park (Intel Imaging and Camera Technology)

### Introduction

This repository is a fork from [BVLC/caffe](https://github.com/BVLC/caffe). Some modifications have been made to run PVANET with Caffe:
- Implemented a new learning rate scheduling based on plateau detection.
- Implemented proposal layer for both CPU and GPU versions.
- Implemented NMS for both CPU and GPU versions.
- Copied RoI pooling layer and smoothed L1 loss layer from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- Applied a hotfix for 'average_loss'

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  git checkout ssd
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `$HOME/data/`
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

3. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
  # and save temporary evaluation results in:
  #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
  # It should reach 77.* mAP at 120k iterations.
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz).

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples/ssd/score_ssd_pascal.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples/ssd/ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out [`examples/ssd_detect.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) or [`examples/ssd/ssd_detect.cpp`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_detect.cpp) on how to detect objects using a SSD model. Check out [`examples/ssd/plot_detections.py`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/plot_detections.py) on how to plot detection results output by ssd_detect.cpp.

5. To train on other dataset, please refer to data/OTHERDATASET for more details. We currently add support for COCO and ILSVRC2016. We recommend using [`examples/ssd.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) to check whether the new dataset is prepared correctly.

### Models
We have provided the latest models that are trained from different datasets. To help reproduce the results in [Table 6](https://arxiv.org/pdf/1512.02325v4.pdf), most models contain a pretrained `.caffemodel` file, many `.prototxt` files, and python scripts.

1. PASCAL VOC models:
   * 07+12: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz), [SSD512*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_512x512.tar.gz)
   * 07++12: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712Plus_SSD_300x300.tar.gz), [SSD512*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712Plus_SSD_512x512.tar.gz)
   * COCO<sup>[1]</sup>: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300_coco.tar.gz), [SSD512*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_512x512_coco.tar.gz)
   * 07+12+COCO: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300_ft.tar.gz), [SSD512*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_512x512_ft.tar.gz)
   * 07++12+COCO: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712Plus_SSD_300x300_ft.tar.gz), [SSD512*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712Plus_SSD_512x512_ft.tar.gz)

2. COCO models:
   * trainval35k: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_300x300.tar.gz), [SSD512*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_coco_SSD_512x512.tar.gz)

3. ILSVRC models:
   * trainval1: [SSD300*](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ILSVRC2016_SSD_300x300.tar.gz), [SSD500](http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_ilsvrc15_SSD_500x500.tar.gz)

<sup>[1]</sup>We use [`examples/convert_model.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/convert_model.ipynb) to extract a VOC model from a pretrained COCO model.
