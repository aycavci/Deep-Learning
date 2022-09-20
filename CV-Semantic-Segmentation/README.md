The software is copied from https://github.com/HRNet/HRNet-Semantic-Segmentation. 
The pre-trained model is copied from https://github.com/HRNet/HRNet-Image-Classification

# Quick start
## Install
1. We have used PyTorch=1.11.0 with cuda 11.3
2. Install dependencies: pip install -r requirements.txt

## Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset

Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── gtFine
│   ├── test
│   ├── train
│   └── val
│
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├
├── list
│   └── cityscapes
│       ├── test.lst
│       ├── trainval.lst
│       └── val.lst
````

# Changes made to the code
- Using PyTorch for training and validation on a single GPU
- Remove the class weights by train_dataset.class_weights = None
- The epoch number is added to the filename when saving checkpoints
- The checkpoint with the highest epoch number of loaded on startup

# Folders
- pre_trained contains the pretrained model on the ImageNet
- trained_models contains the final models, both the model trained from scratch, and finetuned.
- experiments contains the experiment settings for training the finetune model, and the model from scratch
- output for training with corresponding log files are in the output folder. In the output folder is a subfolder called logToGraph. When the file plotTrainingResults.py in the root folder is run, the log files in the subfolder are plotted. 

# Run experiments
## Train the models
- python3 -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg experiments/CS_W18_smallv1_Scratch.yaml
- python3 -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg experiments/CS_W18_smallv1_finetune.yaml

## Validate on the validation set
- python3 tools/test.py --cfg experiments/CS_W18_smallv1_finetune.yaml DATASET.TEST_SET list/cityscapes/val.lst TEST.MODEL_FILE trained_models/trained_model_finetune.pth TEST.FLIP_TEST True
- python3 tools/test.py --cfg experiments/CS_W18_smallv1_Scratch.yaml DATASET.TEST_SET list/cityscapes/val.lst TEST.MODEL_FILE trained_models/trained_model_scratch.pth TEST.FLIP_TEST True

# Predict on the test images 
- python3 tools/test.py --cfg experiments/CS_W18_smallv1_finetune.yaml DATASET.TEST_SET list/cityscapes/test.lst TEST.MODEL_FILE trained_models/trained_model_finetune.pth TEST.FLIP_TEST True
- python3 tools/test.py --cfg experiments/CS_W18_smallv1_Scratch.yaml DATASET.TEST_SET list/cityscapes/test.lst TEST.MODEL_FILE trained_models/trained_model_scratch.pth TEST.FLIP_TEST True